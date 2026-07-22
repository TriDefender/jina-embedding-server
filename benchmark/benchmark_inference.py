"""
Jina Embedding & Reranker — PyTorch Inference Benchmark / Bottleneck Finder

Standalone profiling harness that loads both models exactly like `jina_server.py`
and breaks down the PyTorch inference cost for:

  - /v1/embeddings  (jina-embeddings-v5-text-small, Qwen3-0.6B + LoRA)
  - /v1/rerank      (jina-reranker-v3, Qwen3-0.6B CausalLM + projector)

Phases:
  A. Environment & model loading + torch.compile warmup timing
  B. Embeddings per-stage timing + batch_size / seq_len sweep
  C. Rerank per-stage timing + doc_count / block_size sweep
  D. torch.profiler deep-dive on representative workloads (top kernels)
  E. Eager-vs-compiled and bf16-vs-fp32 comparison
  F. Bottleneck findings report (console + JSON)

Usage:
  uv run python benchmark_inference.py                 # full benchmark
  uv run python benchmark_inference.py --phase embed   # only embeddings
  uv run python benchmark_inference.py --quick         # fewer iterations
  uv run python benchmark_inference.py --no-compile    # eager only
  uv run python benchmark_inference.py --list-tests    # show plan, no load

Outputs:
  - Console report (always)
  - benchmark_results.json (timing tables)
  - profile_embed_*.json / profile_rerank_*.json (chrome://tracing traces)
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# =============================================================================
# Thread configuration — MUST be set before torch import (mirrors jina_server.py)
# =============================================================================
MAX_THREADS = 24
os.environ["OMP_NUM_THREADS"] = str(MAX_THREADS)
os.environ["MKL_NUM_THREADS"] = str(MAX_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(MAX_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(MAX_THREADS)
os.environ["TORCH_INTEROP_THREADS"] = "8"

import torch  # noqa: E402  (must come after OMP env vars above)

torch.set_num_threads(MAX_THREADS)

# =============================================================================
# Configuration (mirrors jina_server.py)
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MODELS_DIR = os.path.join(PROJECT_ROOT, "jinaai")
EMBEDDING_MODEL_PATH = os.path.join(MODELS_DIR, "jina-embeddings-v5-text-small")
RERANKER_MODEL_PATH = os.path.join(MODELS_DIR, "jina-reranker-v3")

EMBED_DIM_EXPECTED = 1024
RERANKER_PROJECTOR_DIM = 512

# =============================================================================
# Timing helpers
# =============================================================================


@dataclass
class TimingResult:
    label: str
    samples_ms: List[float] = field(default_factory=list)

    @property
    def median_ms(self) -> float:
        return statistics.median(self.samples_ms) if self.samples_ms else 0.0

    @property
    def mean_ms(self) -> float:
        return statistics.fmean(self.samples_ms) if self.samples_ms else 0.0

    @property
    def p95_ms(self) -> float:
        if len(self.samples_ms) < 2:
            return self.median_ms
        s = sorted(self.samples_ms)
        idx = max(0, int(len(s) * 0.95) - 1)
        return s[idx]

    @property
    def min_ms(self) -> float:
        return min(self.samples_ms) if self.samples_ms else 0.0

    @property
    def max_ms(self) -> float:
        return max(self.samples_ms) if self.samples_ms else 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "n": len(self.samples_ms),
            "median_ms": round(self.median_ms, 3),
            "mean_ms": round(self.mean_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
        }


@contextmanager
def stage_timer(results: Dict[str, TimingResult], label: str):
    """Context manager that records one sample for a given stage label."""
    t0 = time.perf_counter()
    yield
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    results.setdefault(label, TimingResult(label=label)).samples_ms.append(elapsed_ms)


def run_repeated(
    n_iter: int,
    n_warmup: int,
    fn: Callable[[], Any],
) -> Tuple[Any, List[float]]:
    """Run `fn` n_warmup + n_iter times; return last result + per-iter ms list."""
    last = None
    timings: List[float] = []
    for i in range(n_warmup + n_iter):
        gc.collect()
        t0 = time.perf_counter()
        last = fn()
        # CPU ops are blocking by default; no explicit sync needed.
        timings.append((time.perf_counter() - t0) * 1000.0)
    return last, timings[n_warmup:]


# =============================================================================
# Test data — realistic varied texts
# =============================================================================

SHORT_QUERIES = [
    "What is machine learning?",
    "How do neural networks work?",
    "Best practices for Python error handling",
    "Explain gradient descent",
    "What is RAG?",
    "Compare REST vs GraphQL",
    "How does attention work in transformers?",
    "What is fine-tuning?",
    "Explain LoRA adapters",
    "What is BM25 retrieval?",
]

MEDIUM_DOCS = [
    "Machine learning is a branch of artificial intelligence that enables systems to learn from data and improve over time without being explicitly programmed. It includes supervised, unsupervised, and reinforcement learning approaches.",
    "Neural networks are computational models inspired by the human brain. They consist of interconnected nodes (neurons) organized in layers. Deep learning refers to networks with many hidden layers that can learn complex hierarchical representations.",
    "PyTorch error handling should follow the principle of explicit is better than implicit. Use try/except blocks judiciously, prefer raising specific exception types over generic Exception, and always log the original traceback when re-raising in production code.",
    "Gradient descent is an optimization algorithm that iteratively adjusts model parameters to minimize a loss function. Variants include stochastic gradient descent (SGD), mini-batch GD, and adaptive methods like Adam that maintain per-parameter learning rates.",
    "Retrieval-Augmented Generation (RAG) combines a retriever and a generator. The retriever fetches relevant documents from a corpus using embeddings and similarity search, then the generator (LLM) conditions on the retrieved context to produce grounded answers.",
    "REST is an architectural style using HTTP methods (GET, POST, PUT, DELETE) on resources identified by URIs. GraphQL is a query language for APIs that lets clients request exactly the fields they need via a single endpoint. REST is simpler; GraphQL is more flexible.",
    "The attention mechanism computes weighted averages of value vectors where weights are derived from query-key similarity. Self-attention lets each token attend to every other token. Multi-head attention runs several attention subspaces in parallel and concatenates their outputs.",
    "Fine-tuning adapts a pre-trained model to a specific task by continuing training on task-specific labeled data. Full fine-tuning updates all parameters; parameter-efficient methods like adapters, LoRA, and prefix tuning update only a small subset while freezing the backbone.",
    "LoRA (Low-Rank Adaptation) injects trainable low-rank decomposition matrices into existing linear layers. Instead of updating the original weight matrix W, it learns W + BA where B and A are small rank-r matrices. This drastically reduces trainable parameters while preserving quality.",
    "BM25 is a probabilistic ranking function used by search engines to score document relevance to a query. It extends TF-IDF with length normalization and term frequency saturation. BM25 remains a strong sparse retrieval baseline and is often combined with dense retrieval.",
]

LONG_DOCS = [
    # ~300-500 tokens: long technical passages
    (
        "Transformers have become the dominant architecture for natural language processing tasks since their introduction in 2017. "
        "The key innovation is the self-attention mechanism, which allows the model to consider the entire input sequence simultaneously "
        "rather than processing it sequentially as recurrent architectures do. This parallelism enables much more efficient training "
        "on modern hardware like GPUs and TPUs, while the attention mechanism itself provides a flexible way to model long-range "
        "dependencies that have historically been challenging for sequence models. The original transformer follows an encoder-decoder "
        "structure for sequence-to-sequence tasks like translation, but subsequent work has shown that decoder-only architectures "
        "like GPT are remarkably effective for language modeling and generation, while encoder-only models like BERT excel at "
        "classification and understanding tasks. Recent embedding models built on the decoder-only foundation, such as the Jina v5 "
        "family, leverage the last-token pooling strategy and LoRA-based task adapters to produce high-quality task-specific "
        "embeddings with a single shared backbone. The computational cost of full attention is O(n^2) in sequence length, which "
        "has driven research into sparse attention patterns, linear attention variants, and sliding-window mechanisms. Despite "
        "these alternatives, full attention remains the most widely used in practice because the quality trade-offs of approximate "
        "methods are often unacceptable for downstream tasks, particularly those requiring fine-grained semantic understanding."
    ),
    (
        "Retrieval-Augmented Generation (RAG) systems combine the strengths of parametric and non-parametric memory. The parametric "
        "memory is encoded in the weights of a language model trained on massive text corpora, while the non-parametric memory is "
        "an external index of documents that can be searched at inference time. A typical RAG pipeline consists of three components: "
        "an embedder that maps text to dense vectors, a vector database that supports efficient nearest-neighbor search over those "
        "vectors, and a language model that consumes the retrieved passages as additional context. The embedder is often a "
        "fine-tuned transformer that produces sentence-level or passage-level embeddings in a high-dimensional continuous space. "
        "Dense embedding retrieval complements traditional sparse retrieval methods like BM25; production systems frequently use "
        "hybrid retrieval, combining the lexical matching strength of sparse methods with the semantic matching strength of dense "
        "methods. After retrieving a candidate set, a reranker model is typically applied to score query-passage pairs with higher "
        "fidelity than first-stage retrieval embeddings allow. Modern listwise rerankers process multiple passages in a single "
        "forward pass, computing attention between the query and all candidate passages jointly, which captures cross-passage "
        "interactions that pointwise rerankers miss. This listwise approach yields significant quality gains on benchmarks like "
        "BEIR and MIRACL while remaining computationally tractable for sub-billion-parameter models."
    ),
]

RERANKER_QUERY = (
    "What are the practical trade-offs between dense embedding retrieval and sparse BM25 retrieval "
    "in a production search system, and how can they be effectively combined?"
)

RERANKER_DOCS: List[str] = []
# Build 64 docs from existing passages (varied relevance to query)
for i in range(6):
    RERANKER_DOCS.extend(MEDIUM_DOCS)
RERANKER_DOCS.extend([LONG_DOCS[0], LONG_DOCS[1]] * 4)  # 8 long docs
# Pad to 64
while len(RERANKER_DOCS) < 64:
    RERANKER_DOCS.append(MEDIUM_DOCS[len(RERANKER_DOCS) % len(MEDIUM_DOCS)])
RERANKER_DOCS = RERANKER_DOCS[:64]


# =============================================================================
# Phase A: Environment & model loading
# =============================================================================


def gather_env_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "cpu_count": os.cpu_count(),
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "bf16_supported": False,
        "bf16_matmul_native": False,
    }
    # Detect BF16 support (Zen 4+ has AVX512_BF16; AMD R9 9900X supports it)
    try:
        caps = torch.backends.cpu.get_cpu_capability()
        info["bf16_supported"] = caps in ["AVX512", "AVX2"]
        # True if torch can run bf16 matmul natively (no fp32 upcast)
        info["bf16_matmul_native"] = caps == "AVX512"
    except Exception:
        pass
    # Try py-cpuinfo for finer detail
    try:
        import cpuinfo

        ci = cpuinfo.get_cpu_info()
        info["cpu_brand"] = ci.get("brand_raw", "Unknown")
        flags = ci.get("flags", [])
        info["has_avx512_bf16"] = "avx512_bf16" in flags
        info["has_avx512_vnni"] = "avx512vnni" in flags
        info["has_amx_bf16"] = "amx_bf16" in flags
        info["has_avx512f"] = "avx512f" in flags
    except Exception:
        info["cpu_brand"] = platform.processor() or "Unknown"
    return info


def print_env(env: Dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print("Phase A: Environment")
    print("=" * 70)
    for k, v in env.items():
        print(f"  {k:<28} {v}")


def load_embedding_model(
    compiled: bool, dtype: torch.dtype = torch.bfloat16
) -> Tuple[Any, float]:
    """Load embedding model exactly like jina_server.py; return (model, load_seconds)."""
    from sentence_transformers import SentenceTransformer

    t0 = time.perf_counter()
    model = SentenceTransformer(
        EMBEDDING_MODEL_PATH,
        trust_remote_code=True,
        model_kwargs={
            "default_task": "retrieval",
            "default_prompt": "query",
            "torch_dtype": dtype,
            "attn_implementation": "sdpa",
        },
    )
    model.to(torch.device("cpu"))
    if compiled:
        try:
            model = torch.compile(model, dynamic=True, mode="max-autotune")
        except Exception as e:
            print(f"  [WARN] torch.compile(embedding) failed: {e}")
    load_s = time.perf_counter() - t0
    return model, load_s


def load_reranker_model(
    compiled: bool, dtype: torch.dtype = torch.bfloat16
) -> Tuple[Any, float]:
    """Load reranker exactly like jina_server.py; return (model, load_seconds)."""
    from transformers import AutoModel

    t0 = time.perf_counter()
    model = AutoModel.from_pretrained(
        RERANKER_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    model.eval()
    if compiled:
        try:
            model = torch.compile(model, mode="max-autotune", dynamic=True)
        except Exception as e:
            print(f"  [WARN] torch.compile(reranker) failed: {e}")
    model.to(torch.device("cpu"))
    load_s = time.perf_counter() - t0
    return model, load_s


def warmup_embedding_model(model: Any, n_iter: int = 3) -> float:
    """Pre-warm to trigger torch.compile; return warmup seconds."""
    t0 = time.perf_counter()
    for _ in range(n_iter):
        model.encode(
            ["warmup text for embedding model compilation"],
            task="retrieval",
            prompt_name="query",
            normalize_embeddings=True,
            convert_to_numpy=False,
        )
    return time.perf_counter() - t0


def warmup_reranker_model(model: Any, n_iter: int = 2) -> float:
    t0 = time.perf_counter()
    for _ in range(n_iter):
        try:
            model.rerank(
                query="warmup query",
                documents=["dummy document one", "dummy document two"],
                top_n=1,
            )
        except Exception:
            pass
    return time.perf_counter() - t0


# =============================================================================
# Phase B: Embeddings per-stage timing + sweeps
# =============================================================================


def _resolve_inner_model(model: Any) -> Any:
    """Drill down through SentenceTransformer + custom_st to reach the inner model."""
    inner = model
    # torch.compile wraps in OptimizedModule (older) or _DynamoWrapper (newer)
    for attr in ("_orig_mod", "_modules",):
        if hasattr(inner, attr) and attr == "_orig_mod":
            inner = getattr(inner, "_orig_mod")
    # SentenceTransformer has ._first_module() returning custom_st.Transformer
    if hasattr(inner, "_first_module"):
        inner = inner._first_module()
        # custom_st.Transformer stores the PeftMixedModel in .model
        if hasattr(inner, "model"):
            inner = inner.model
    return inner


def bench_embedding_stage_timing(
    model: Any,
    texts: List[str],
    n_iter: int = 5,
    n_warmup: int = 2,
) -> Dict[str, TimingResult]:
    """Break down embeddings into 4 stages using the actual forward path."""
    results: Dict[str, TimingResult] = {}

    # Reach the custom_st.Transformer to access tokenizer + inner PeftMixedModel
    inner_st = model
    if hasattr(inner_st, "_orig_mod"):
        inner_st = inner_st._orig_mod
    if hasattr(inner_st, "_first_module"):
        transformer_module = inner_st._first_module()
    else:
        transformer_module = inner_st
    tokenizer = transformer_module.tokenizer
    inner_model = transformer_module.model
    max_length = transformer_module.max_seq_length

    for i in range(n_warmup + n_iter):
        gc.collect()
        # Stage 1: tokenization
        t0 = time.perf_counter()
        batch = tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        results.setdefault("1_tokenize", TimingResult("1_tokenize")).samples_ms.append(
            (time.perf_counter() - t0) * 1000.0
        )

        device = next(inner_model.parameters()).device
        # Stage 2: tensor transfer (CPU->CPU, mostly free; included for completeness)
        t0 = time.perf_counter()
        batch_dev = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        results.setdefault(
            "2_to_device", TimingResult("2_to_device")
        ).samples_ms.append((time.perf_counter() - t0) * 1000.0)

        # Adapter switch (same as custom_st.forward)
        t0 = time.perf_counter()
        inner_model.set_adapter("retrieval")
        inner_model.eval()
        results.setdefault(
            "3_set_adapter", TimingResult("3_set_adapter")
        ).samples_ms.append((time.perf_counter() - t0) * 1000.0)

        # Stage 3: forward pass (the big one)
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = inner_model(**batch_dev)
            hidden = outputs.last_hidden_state
        results.setdefault(
            "4_forward", TimingResult("4_forward")
        ).samples_ms.append((time.perf_counter() - t0) * 1000.0)

        # Stage 4: last-token pooling
        t0 = time.perf_counter()
        with torch.no_grad():
            mask = batch_dev.get("attention_mask")
            seq_lens = mask.sum(dim=1) - 1
            pooled = hidden[torch.arange(hidden.shape[0]), seq_lens]
        results.setdefault(
            "5_pool_last_token", TimingResult("5_pool_last_token")
        ).samples_ms.append((time.perf_counter() - t0) * 1000.0)

        # Stage 5: L2 normalization
        t0 = time.perf_counter()
        with torch.no_grad():
            emb = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        results.setdefault(
            "6_normalize", TimingResult("6_normalize")
        ).samples_ms.append((time.perf_counter() - t0) * 1000.0)

        # Stage 6: .tolist() (used in API response building)
        t0 = time.perf_counter()
        _ = [e.tolist() for e in emb]
        results.setdefault(
            "7_to_list", TimingResult("7_to_list")
        ).samples_ms.append((time.perf_counter() - t0) * 1000.0)

    # Total wall-clock via model.encode() end-to-end
    def _end_to_end():
        return model.encode(
            texts,
            task="retrieval",
            prompt_name="query",
            normalize_embeddings=True,
            convert_to_numpy=False,
        )

    _, e2e_timings = run_repeated(n_iter=n_iter, n_warmup=n_warmup, fn=_end_to_end)
    results["0_end_to_end"] = TimingResult(
        label="0_end_to_end", samples_ms=e2e_timings
    )
    return results


def bench_embedding_batch_sweep(
    model: Any, n_iter: int = 3, n_warmup: int = 1
) -> List[Dict[str, Any]]:
    """Sweep batch_size and report throughput."""
    batch_sizes = [1, 4, 8, 16, 32, 64]
    out: List[Dict[str, Any]] = []
    for bs in batch_sizes:
        texts = (MEDIUM_DOCS * ((bs // len(MEDIUM_DOCS)) + 1))[:bs]
        _, timings = run_repeated(
            n_iter=n_iter,
            n_warmup=n_warmup,
            fn=lambda t=texts: model.encode(
                t,
                task="retrieval",
                prompt_name="query",
                normalize_embeddings=True,
                convert_to_numpy=False,
                batch_size=max(bs, 8),
            ),
        )
        median_ms = statistics.median(timings)
        out.append(
            {
                "batch_size": bs,
                "n_texts": bs,
                "median_ms": round(median_ms, 2),
                "p95_ms": round(sorted(timings)[max(0, int(len(timings) * 0.95) - 1)], 2),
                "texts_per_sec": round(bs / (median_ms / 1000.0), 2),
                "ms_per_text": round(median_ms / bs, 2),
            }
        )
    return out


def bench_embedding_seqlen_sweep(
    model: Any, n_iter: int = 3, n_warmup: int = 1
) -> List[Dict[str, Any]]:
    """Sweep sequence-length buckets."""
    buckets = [
        ("short~10tok", SHORT_QUERIES[:8]),
        ("medium~50tok", MEDIUM_DOCS),
        ("long~200tok", LONG_DOCS * 5),  # 10 long docs
    ]
    out: List[Dict[str, Any]] = []
    for label, texts in buckets:
        # Pad/truncate to consistent length per bucket
        if not texts:
            continue
        _, timings = run_repeated(
            n_iter=n_iter,
            n_warmup=n_warmup,
            fn=lambda t=texts: model.encode(
                t,
                task="retrieval",
                prompt_name="query",
                normalize_embeddings=True,
                convert_to_numpy=False,
                batch_size=16,
            ),
        )
        median_ms = statistics.median(timings)
        out.append(
            {
                "bucket": label,
                "n_texts": len(texts),
                "median_ms": round(median_ms, 2),
                "ms_per_text": round(median_ms / len(texts), 2),
            }
        )
    return out


# =============================================================================
# Phase C: Rerank per-stage timing + sweeps
# =============================================================================


def bench_rerank_stage_timing(
    model: Any,
    query: str,
    documents: List[str],
    block_size: int = 125,
    n_iter: int = 3,
    n_warmup: int = 1,
) -> Dict[str, TimingResult]:
    """Break down rerank into stages mirroring JinaForRanking.rerank()."""
    results: Dict[str, TimingResult] = {}

    inner = model
    if hasattr(inner, "_orig_mod"):
        inner = inner._orig_mod

    # Ensure tokenizer loaded
    inner._ensure_tokenizer()
    tokenizer = inner._tokenizer
    device = next(inner.parameters()).device
    special_tokens = inner.special_tokens

    # Mirror rerank() truncation step
    def _truncate_stage():
        return inner._truncate_texts(query, documents, max_query_length=512, max_doc_length=2048)

    last_q, last_docs, last_doc_lens, last_q_len = _truncate_stage()

    for i in range(n_warmup + n_iter):
        gc.collect()

        # Stage 1: truncate texts (per-doc tokenization to measure length)
        t0 = time.perf_counter()
        q_t, docs_t, doc_lens_t, q_len_t = _truncate_stage()
        results.setdefault(
            "1_truncate", TimingResult("1_truncate")
        ).samples_ms.append((time.perf_counter() - t0) * 1000.0)

        # Mirror the block formation logic from rerank()
        max_length = tokenizer.model_max_length
        length_capacity = max_length - 2 * q_len_t
        blocks: List[List[str]] = []
        current: List[str] = []
        for length, doc in zip(doc_lens_t, docs_t):
            current.append(doc)
            length_capacity -= length
            if len(current) >= block_size or length_capacity <= 2048:
                blocks.append(current)
                current = []
                length_capacity = max_length - 2 * q_len_t
        if current:
            blocks.append(current)

        # Stage 2: build prompt strings
        # Access format_docs_prompts_func via the model's dynamically-loaded module
        # (jinaai/ is not a Python package; transformers loads modeling.py via trust_remote_code)
        if not hasattr(inner, "_format_prompt_fn"):
            model_module = sys.modules.get(type(inner).__module__)
            if model_module and hasattr(model_module, "format_docs_prompts_func"):
                inner._format_prompt_fn = model_module.format_docs_prompts_func
            else:
                import importlib.util

                spec = importlib.util.spec_from_file_location(
                    "jina_reranker_v3_modeling",
                    os.path.join(RERANKER_MODEL_PATH, "modeling.py"),
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                inner._format_prompt_fn = mod.format_docs_prompts_func
        format_fn = inner._format_prompt_fn

        t0 = time.perf_counter()
        prompts = [
            format_fn(
                q_t, block, instruction=None, special_tokens=special_tokens, no_thinking=True
            )
            for block in blocks
        ]
        results.setdefault(
            "2_build_prompt", TimingResult("2_build_prompt")
        ).samples_ms.append((time.perf_counter() - t0) * 1000.0)

        # Stage 3: tokenize full prompts (one per block)
        t0 = time.perf_counter()
        tokenized_blocks = [
            tokenizer(text=[p], padding=True, padding_side="left", return_tensors="pt").to(
                device
            )
            for p in prompts
        ]
        results.setdefault(
            "3_tokenize", TimingResult("3_tokenize")
        ).samples_ms.append((time.perf_counter() - t0) * 1000.0)

        # Stage 4: forward pass per block
        t0_forward_total = time.perf_counter()
        per_block_forward_ms: List[float] = []
        all_scores = []
        for batch in tokenized_blocks:
            t0 = time.perf_counter()
            outputs = inner(**batch)
            per_block_forward_ms.append((time.perf_counter() - t0) * 1000.0)
            all_scores.append(outputs.scores.view(-1).cpu().float().numpy())
        results.setdefault(
            "4_forward_total", TimingResult("4_forward_total")
        ).samples_ms.append((time.perf_counter() - t0_forward_total) * 1000.0)
        results["4_forward_per_block"] = TimingResult(
            label="4_forward_per_block", samples_ms=per_block_forward_ms
        )

        # Stage 5: score aggregation (numpy ops, cheap)
        t0 = time.perf_counter()
        import numpy as np

        scores_concat = np.concatenate(all_scores)
        _ = np.argsort(scores_concat)[::-1]
        results.setdefault(
            "5_score_aggregate", TimingResult("5_score_aggregate")
        ).samples_ms.append((time.perf_counter() - t0) * 1000.0)

    # End-to-end via model.rerank()
    def _e2e():
        return inner.rerank(query=query, documents=documents, top_n=min(10, len(documents)))

    _, e2e_timings = run_repeated(n_iter=n_iter, n_warmup=n_warmup, fn=_e2e)
    results["0_end_to_end"] = TimingResult("0_end_to_end", e2e_timings)
    return results


def bench_rerank_doc_count_sweep(
    model: Any, n_iter: int = 2, n_warmup: int = 1
) -> List[Dict[str, Any]]:
    """Sweep number of documents to rerank."""
    doc_counts = [4, 16, 32, 64]
    out: List[Dict[str, Any]] = []
    inner = model
    if hasattr(inner, "_orig_mod"):
        inner = inner._orig_mod

    for n_docs in doc_counts:
        docs = RERANKER_DOCS[:n_docs]
        # Set block_size on the model (server does this via reranker_lock)
        if hasattr(inner, "_block_size"):
            inner._block_size = 125
        _, timings = run_repeated(
            n_iter=n_iter,
            n_warmup=n_warmup,
            fn=lambda d=docs: inner.rerank(query=RERANKER_QUERY, documents=d, top_n=min(5, len(d))),
        )
        median_ms = statistics.median(timings)
        out.append(
            {
                "n_docs": n_docs,
                "median_ms": round(median_ms, 2),
                "docs_per_sec": round(n_docs / (median_ms / 1000.0), 2),
                "ms_per_doc": round(median_ms / n_docs, 2),
            }
        )
    return out


def bench_rerank_block_size_sweep(
    model: Any, n_iter: int = 2, n_warmup: int = 1
) -> List[Dict[str, Any]]:
    """Sweep block_size (docs per prompt)."""
    block_sizes = [8, 16, 32, 64, 125]
    out: List[Dict[str, Any]] = []
    inner = model
    if hasattr(inner, "_orig_mod"):
        inner = inner._orig_mod

    n_docs = 64
    docs = RERANKER_DOCS[:n_docs]
    for bs in block_sizes:
        if hasattr(inner, "_block_size"):
            inner._block_size = bs
        _, timings = run_repeated(
            n_iter=n_iter,
            n_warmup=n_warmup,
            fn=lambda d=docs: inner.rerank(query=RERANKER_QUERY, documents=d, top_n=min(5, len(d))),
        )
        median_ms = statistics.median(timings)
        out.append(
            {
                "block_size": bs,
                "n_docs": n_docs,
                "median_ms": round(median_ms, 2),
                "docs_per_sec": round(n_docs / (median_ms / 1000.0), 2),
            }
        )
    return out


# =============================================================================
# Phase D: torch.profiler deep-dive
# =============================================================================


def profile_embedding_workload(
    model: Any, out_trace: str, n_iter: int = 3
) -> Dict[str, Any]:
    """Run torch.profiler on embedding encode; export chrome trace; return top kernels."""
    texts = MEDIUM_DOCS * 3  # 30 texts
    # Warmup
    for _ in range(2):
        model.encode(
            texts,
            task="retrieval",
            prompt_name="query",
            normalize_embeddings=True,
            convert_to_numpy=False,
        )

    activities = [torch.profiler.ProfilerActivity.CPU]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(n_iter):
            model.encode(
                texts,
                task="retrieval",
                prompt_name="query",
                normalize_embeddings=True,
                convert_to_numpy=False,
            )

    try:
        prof.export_chrome_trace(out_trace)
    except Exception as e:
        print(f"  [WARN] trace export failed: {e}")

    # Top kernels by CPU time
    key_averages = prof.key_averages()
    table = key_averages.table(
        sort_by="cpu_time_total", row_limit=25, header=True
    )
    # Extract top entries as dict (parse key_averages directly)
    top_ops = []
    for evt in sorted(key_averages, key=lambda e: e.cpu_time_total, reverse=True)[:15]:
        top_ops.append(
            {
                "key": evt.key,
                "cpu_time_total_ms": round(evt.cpu_time_total / 1000.0, 3),
                "cpu_memory_mb": round(evt.cpu_memory_usage / (1024 * 1024), 3),
                "self_cpu_time_total_ms": round(evt.self_cpu_time_total / 1000.0, 3),
                "calls": evt.count,
            }
        )
    return {"top_ops": top_ops, "trace_file": out_trace, "table": table}


def profile_rerank_workload(
    model: Any, out_trace: str, n_iter: int = 2
) -> Dict[str, Any]:
    """Run torch.profiler on rerank; export chrome trace; return top kernels."""
    inner = model
    if hasattr(inner, "_orig_mod"):
        inner = inner._orig_mod

    docs = RERANKER_DOCS[:32]
    # Warmup
    for _ in range(1):
        inner.rerank(query=RERANKER_QUERY, documents=docs, top_n=5)

    activities = [torch.profiler.ProfilerActivity.CPU]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(n_iter):
            inner.rerank(query=RERANKER_QUERY, documents=docs, top_n=5)

    try:
        prof.export_chrome_trace(out_trace)
    except Exception as e:
        print(f"  [WARN] trace export failed: {e}")

    key_averages = prof.key_averages()
    table = key_averages.table(
        sort_by="cpu_time_total", row_limit=25, header=True
    )
    top_ops = []
    for evt in sorted(key_averages, key=lambda e: e.cpu_time_total, reverse=True)[:15]:
        top_ops.append(
            {
                "key": evt.key,
                "cpu_time_total_ms": round(evt.cpu_time_total / 1000.0, 3),
                "cpu_memory_mb": round(evt.cpu_memory_usage / (1024 * 1024), 3),
                "self_cpu_time_total_ms": round(evt.self_cpu_time_total / 1000.0, 3),
                "calls": evt.count,
            }
        )
    return {"top_ops": top_ops, "trace_file": out_trace, "table": table}


# =============================================================================
# Phase E: A/B comparisons
# =============================================================================


def compare_dtypes_embedding(
    n_iter: int = 3, n_warmup: int = 1
) -> Dict[str, Any]:
    """Compare bf16 vs fp32 for embeddings (eager mode)."""
    out: Dict[str, Any] = {}
    texts = MEDIUM_DOCS
    for dtype_name, dtype in [("bf16", torch.bfloat16), ("fp32", torch.float32)]:
        print(f"    loading embedding in {dtype_name} (eager)...")
        m, load_s = load_embedding_model(compiled=False, dtype=dtype)
        warmup_embedding_model(m, n_iter=1)
        _, timings = run_repeated(
            n_iter=n_iter,
            n_warmup=n_warmup,
            fn=lambda _m=m: _m.encode(
                texts,
                task="retrieval",
                prompt_name="query",
                normalize_embeddings=True,
                convert_to_numpy=False,
            ),
        )
        out[dtype_name] = {
            "load_seconds": round(load_s, 2),
            "median_ms": round(statistics.median(timings), 2),
            "texts_per_sec": round(len(texts) / (statistics.median(timings) / 1000.0), 2),
        }
        del m
        gc.collect()
    out["speedup_bf16_vs_fp32"] = round(
        out["fp32"]["median_ms"] / out["bf16"]["median_ms"], 2
    )
    return out


def compare_compile_embedding(n_iter: int = 3) -> Dict[str, Any]:
    """Compare eager vs compiled for embeddings."""
    out: Dict[str, Any] = {}
    texts = MEDIUM_DOCS
    # eager
    print("    loading embedding eager...")
    m_eager, _ = load_embedding_model(compiled=False)
    warmup_embedding_model(m_eager, n_iter=1)
    _, t_eager = run_repeated(
        n_iter=n_iter,
        n_warmup=1,
        fn=lambda _m=m_eager: _m.encode(
            texts,
            task="retrieval",
            prompt_name="query",
            normalize_embeddings=True,
            convert_to_numpy=False,
        ),
    )
    out["eager_median_ms"] = round(statistics.median(t_eager), 2)
    del m_eager
    gc.collect()
    # compiled
    print("    loading embedding compiled...")
    m_compiled, _ = load_embedding_model(compiled=True)
    print("    warming up compiled (may take minutes)...")
    warmup_embedding_model(m_compiled, n_iter=3)
    _, t_compiled = run_repeated(
        n_iter=n_iter,
        n_warmup=1,
        fn=lambda _m=m_compiled: _m.encode(
            texts,
            task="retrieval",
            prompt_name="query",
            normalize_embeddings=True,
            convert_to_numpy=False,
        ),
    )
    out["compiled_median_ms"] = round(statistics.median(t_compiled), 2)
    out["speedup"] = round(out["eager_median_ms"] / out["compiled_median_ms"], 2)
    del m_compiled
    gc.collect()
    return out


def compare_thread_counts_embedding(n_iter: int = 3) -> Dict[str, Any]:
    """Sweep torch num_threads to find optimal."""
    out: Dict[str, Any] = {"data_points": []}
    thread_counts = [1, 4, 8, 12, 24]
    print("    loading embedding eager for thread sweep...")
    m, _ = load_embedding_model(compiled=False)
    warmup_embedding_model(m, n_iter=1)
    texts = MEDIUM_DOCS
    for nt in thread_counts:
        torch.set_num_threads(nt)
        _, timings = run_repeated(
            n_iter=n_iter,
            n_warmup=1,
            fn=lambda _m=m: _m.encode(
                texts,
                task="retrieval",
                prompt_name="query",
                normalize_embeddings=True,
                convert_to_numpy=False,
            ),
        )
        out["data_points"].append(
            {
                "num_threads": nt,
                "median_ms": round(statistics.median(timings), 2),
                "texts_per_sec": round(len(texts) / (statistics.median(timings) / 1000.0), 2),
            }
        )
    # restore
    torch.set_num_threads(MAX_THREADS)
    return out


# =============================================================================
# Phase F: Findings report
# =============================================================================


def derive_findings(
    env: Dict[str, Any],
    embed_stages: Optional[Dict[str, TimingResult]],
    embed_batch: Optional[List[Dict[str, Any]]],
    rerank_stages: Optional[Dict[str, TimingResult]],
    rerank_doc: Optional[List[Dict[str, Any]]],
    embed_profile: Optional[Dict[str, Any]],
    rerank_profile: Optional[Dict[str, Any]],
    dtype_cmp: Optional[Dict[str, Any]],
    compile_cmp: Optional[Dict[str, Any]],
    thread_cmp: Optional[Dict[str, Any]],
) -> List[str]:
    """Derive textual findings from measured data."""
    findings: List[str] = []

    # --- Environment-based findings ---
    if env.get("has_avx512_bf16"):
        findings.append(
            "[ENV] CPU has AVX512_BF16. PyTorch can run bf16 matmul natively "
            "(no fp32 emulation). bf16 should be roughly 2x faster than fp32 here."
        )
    elif env.get("bf16_matmul_native"):
        findings.append(
            "[ENV] CPU has AVX512 but no AVX512_BF16. bf16 matmul may internally upcast to fp32, "
            "negating the dtype advantage. Consider int8 quantization for further speedup."
        )
    else:
        findings.append(
            "[ENV] CPU lacks AVX512. bf16 may be emulated; expect significant matmul overhead."
        )

    # --- Embedding stage findings ---
    if embed_stages:
        e2e = embed_stages.get("0_end_to_end")
        fwd = embed_stages.get("4_forward")
        tok = embed_stages.get("1_tokenize")
        adapt = embed_stages.get("3_set_adapter")
        tolist = embed_stages.get("7_to_list")
        if e2e and fwd:
            pct = (fwd.median_ms / e2e.median_ms) * 100 if e2e.median_ms else 0
            findings.append(
                f"[EMBED] Forward pass dominates: {fwd.median_ms:.1f}ms / {e2e.median_ms:.1f}ms "
                f"end-to-end ({pct:.1f}%). This is the matmul-bound transformer compute."
            )
        if tok and e2e and tok.median_ms > 0.05 * e2e.median_ms:
            findings.append(
                f"[EMBED] Tokenization is non-trivial: {tok.median_ms:.1f}ms "
                f"({(tok.median_ms / e2e.median_ms) * 100:.1f}% of e2e). For tiny batches or "
                "short texts, tokenization overhead matters."
            )
        if adapt and adapt.median_ms > 1.0:
            findings.append(
                f"[EMBED] set_adapter() per call costs {adapt.median_ms:.2f}ms. If task is stable "
                "across requests, hoist this outside the hot path."
            )
        if tolist and tolist.median_ms > 0.10 * e2e.median_ms:
            findings.append(
                f"[EMBED] .tolist() conversion costs {tolist.median_ms:.1f}ms "
                f"({(tolist.median_ms / e2e.median_ms) * 100:.1f}% of e2e). For API responses, "
                "consider base64-encoded raw bytes instead of JSON float arrays."
            )

    # --- Embedding batch sweep findings ---
    if embed_batch and len(embed_batch) >= 2:
        # Find best throughput point
        best = max(embed_batch, key=lambda r: r.get("texts_per_sec", 0))
        worst = min(embed_batch, key=lambda r: r.get("texts_per_sec", 0))
        findings.append(
            f"[EMBED] Throughput scales with batch size: best={best['texts_per_sec']} texts/s "
            f"at bs={best['batch_size']}; worst={worst['texts_per_sec']} texts/s at bs={worst['batch_size']}. "
            "Larger batches amortize Python/tokenization overhead and improve CPU utilization."
        )
        # Diminishing returns check
        if len(embed_batch) >= 3:
            large_bs = embed_batch[-1]
            mid_bs = embed_batch[len(embed_batch) // 2]
            if large_bs["ms_per_text"] > mid_bs["ms_per_text"] * 0.9:
                findings.append(
                    "[EMBED] ms/text stops improving (or worsens) at the largest batch — "
                    "you may be hitting memory-bandwidth or thread-saturation limits."
                )

    # --- Rerank stage findings ---
    if rerank_stages:
        e2e = rerank_stages.get("0_end_to_end")
        fwd_total = rerank_stages.get("4_forward_total")
        fwd_per_block = rerank_stages.get("4_forward_per_block")
        tok = rerank_stages.get("3_tokenize")
        build = rerank_stages.get("2_build_prompt")
        if fwd_total and e2e:
            pct = (fwd_total.median_ms / e2e.median_ms) * 100 if e2e.median_ms else 0
            findings.append(
                f"[RERANK] Forward pass (sum across blocks) dominates: {fwd_total.median_ms:.1f}ms "
                f"/ {e2e.median_ms:.1f}ms ({pct:.1f}%). Each forward is a full causal LM pass over "
                "query+docs."
            )
        if fwd_per_block and fwd_per_block.samples_ms:
            n_blocks = len(fwd_per_block.samples_ms)
            mean_block = statistics.fmean(fwd_per_block.samples_ms)
            findings.append(
                f"[RERANK] {n_blocks} sequential forward passes; mean {mean_block:.1f}ms/block. "
                "Each block is batch_size=1 — there is NO cross-prompt batching. "
                "Major opportunity: batch multiple prompts in parallel."
            )
        if tok and tok.median_ms > 0.05 * (e2e.median_ms if e2e else 999):
            findings.append(
                f"[RERANK] Tokenization of full prompts costs {tok.median_ms:.1f}ms. "
                "Long prompts (query + N docs) make tokenization non-trivial."
            )
        if build and build.median_ms > 5.0:
            findings.append(
                f"[RERANK] Prompt string building costs {build.median_ms:.1f}ms — "
                "f-string concatenation in a hot path is surprisingly expensive."
            )

    # --- Rerank doc-count scaling ---
    if rerank_doc and len(rerank_doc) >= 2:
        first = rerank_doc[0]
        last = rerank_doc[-1]
        ratio_docs = last["n_docs"] / first["n_docs"]
        ratio_time = last["median_ms"] / first["median_ms"] if first["median_ms"] else 0
        findings.append(
            f"[RERANK] Doc-count scaling: {first['n_docs']}→{last['n_docs']} docs "
            f"({ratio_docs:.1f}x) takes {ratio_time:.2f}x longer. "
            f"{'Sublinear' if ratio_time < ratio_docs else 'Superlinear'} scaling."
        )

    # --- Profile-based findings ---
    if embed_profile and embed_profile.get("top_ops"):
        top = embed_profile["top_ops"][0]
        findings.append(
            f"[PROFILE-EMBED] Hottest op: '{top['key']}' at {top['cpu_time_total_ms']}ms "
            f"({top['calls']} calls). See top_ops table for full breakdown."
        )
    if rerank_profile and rerank_profile.get("top_ops"):
        top = rerank_profile["top_ops"][0]
        findings.append(
            f"[PROFILE-RERANK] Hottest op: '{top['key']}' at {top['cpu_time_total_ms']}ms "
            f"({top['calls']} calls)."
        )

    # --- dtype comparison ---
    if dtype_cmp and "speedup_bf16_vs_fp32" in dtype_cmp:
        sp = dtype_cmp["speedup_bf16_vs_fp32"]
        if sp >= 1.5:
            findings.append(
                f"[CMP] bf16 is {sp}x faster than fp32 for embeddings — keep bf16."
            )
        elif sp >= 1.05:
            findings.append(
                f"[CMP] bf16 is only {sp}x faster than fp32. Native bf16 matmul may be missing; "
                "verify AVX512_BF16 is being used."
            )
        else:
            findings.append(
                f"[CMP] bf16 is NOT faster than fp32 ({sp}x). bf16 path is likely emulated; "
                "consider staying on fp32 or moving to int8."
            )

    # --- compile comparison ---
    if compile_cmp and "speedup" in compile_cmp:
        sp = compile_cmp["speedup"]
        if sp >= 1.2:
            findings.append(
                f"[CMP] torch.compile gives {sp}x speedup — keep compiled mode."
            )
        elif sp >= 0.95:
            findings.append(
                f"[CMP] torch.compile gives negligible speedup ({sp}x). "
                "Compile overhead may not pay off for serving dynamic-shape inputs."
            )
        else:
            findings.append(
                f"[CMP] torch.compile is SLOWER ({sp}x) — likely due to dynamic shapes "
                "triggering frequent recompilation. Consider mode='reduce-overhead' or static shapes."
            )

    # --- thread sweep ---
    if thread_cmp and thread_cmp.get("data_points"):
        dp = thread_cmp["data_points"]
        best = max(dp, key=lambda r: r["texts_per_sec"])
        worst = min(dp, key=lambda r: r["texts_per_sec"])
        findings.append(
            f"[CMP] Optimal thread count: {best['num_threads']} threads "
            f"({best['texts_per_sec']} texts/s). Worst: {worst['num_threads']} threads "
            f"({worst['texts_per_sec']} texts/s). "
            f"Oversubscription beyond physical cores typically hurts matmul."
        )

    # Add code-review findings (static, derived from inspecting the modeling files)
    findings.extend(
        [
            "[STATIC] custom_st.py calls set_adapter(task) on every forward. If task is constant "
            "across many requests, batch them or memoize adapter state.",
            "[STATIC] JinaForRanking.rerank() calls _compute_single_batch per block with batch=1. "
            "Multiple blocks could be padded and run in parallel as a single batched forward "
            "(requires equal-length padding across prompts).",
            "[STATIC] JinaForRanking.forward() sets output_hidden_states=True but uses only the last "
            "layer. Returning all 28 layers wastes memory allocation; if patched, use "
            "output_hidden_states=False and read the final hidden state directly from the model.",
            "[STATIC] JinaForRanking.forward() reassigns self.lm_head = nn.Identity() on every call. "
            "Hoist this to __init__ to avoid per-call overhead (minor but pointless).",
            "[STATIC] Server uses reranker_lock + per-request _block_size mutation. This serializes "
            "rerank calls even with multiple worker threads; consider one model instance per worker.",
            "[STATIC] Rerank does .cpu().float().numpy() per block — on CPU this is a dtype upcast + "
            "memcpy. Could batch the conversions after all blocks complete.",
        ]
    )
    return findings


# =============================================================================
# Result printing
# =============================================================================


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_stage_timings(stages: Dict[str, TimingResult], title: str) -> None:
    print_section(title)
    # Compute total of stages (excluding end-to-end) for percentage
    stage_keys = [k for k in stages if not k.startswith("0_")]
    total_median = sum(stages[k].median_ms for k in stage_keys)
    e2e = stages.get("0_end_to_end")
    e2e_ms = e2e.median_ms if e2e else total_median
    print(f"\n  {'Stage':<22} {'median':>10} {'mean':>10} {'p95':>10} {'% of e2e':>10}")
    print(f"  {'-' * 22} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")
    if e2e:
        print(
            f"  {'END-TO-END':<22} {e2e.median_ms:>10.2f} {e2e.mean_ms:>10.2f} "
            f"{e2e.p95_ms:>10.2f} {'100.0%':>10}"
        )
        print(f"  {'-' * 22} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")
    for k in stage_keys:
        s = stages[k]
        pct = (s.median_ms / e2e_ms) * 100 if e2e_ms else 0
        print(
            f"  {s.label:<22} {s.median_ms:>10.2f} {s.mean_ms:>10.2f} "
            f"{s.p95_ms:>10.2f} {pct:>9.1f}%"
        )
    summed_pct = (total_median / e2e_ms) * 100 if e2e_ms else 0
    print(f"\n  Sum of stages: {total_median:.2f}ms ({summed_pct:.1f}% of end-to-end)")
    if summed_pct < 90:
        print(
            f"  ⚠ Only {summed_pct:.1f}% accounted for — "
            f"the remaining {100 - summed_pct:.1f}% is Python/scheduler/framework overhead."
        )


def print_sweep_table(rows: List[Dict[str, Any]], title: str) -> None:
    print_section(title)
    if not rows:
        print("  (no data)")
        return
    cols = list(rows[0].keys())
    widths = [max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols]
    header = "  " + "  ".join(c.rjust(w) for c, w in zip(cols, widths))
    print(header)
    print("  " + "  ".join("-" * w for w in widths))
    for r in rows:
        print("  " + "  ".join(str(r.get(c, "")).rjust(w) for c, w in zip(cols, widths)))


def print_profile_top(profile: Dict[str, Any], title: str) -> None:
    print_section(title)
    if not profile:
        print("  (no data)")
        return
    print(f"\n  Trace: {profile.get('trace_file', '?')}")
    print("\n  Top 15 ops by total CPU time:\n")
    print(f"  {'Op key':<55} {'total ms':>10} {'self ms':>10} {'calls':>8}")
    print(f"  {'-' * 55} {'-' * 10} {'-' * 10} {'-' * 8}")
    for op in profile.get("top_ops", []):
        key = op["key"][:55]
        print(
            f"  {key:<55} {op['cpu_time_total_ms']:>10.2f} "
            f"{op['self_cpu_time_total_ms']:>10.2f} {op['calls']:>8}"
        )


def print_findings(findings: List[str]) -> None:
    print_section("Phase F: Bottleneck Findings")
    for i, f in enumerate(findings, 1):
        print(f"\n  {i}. {f}")


# =============================================================================
# CLI / main
# =============================================================================


PHASES = ["env", "embed", "rerank", "profile", "compare", "all"]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Jina embedding & reranker PyTorch inference benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--phase",
        choices=PHASES,
        default="all",
        help="Which phase(s) to run (default: all)",
    )
    p.add_argument("--quick", action="store_true", help="Fewer iterations, skip slow parts")
    p.add_argument(
        "--no-compile",
        action="store_true",
        help="Skip torch.compile (load eager models only)",
    )
    p.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip rerank phases (only run embeddings)",
    )
    p.add_argument(
        "--no-embed",
        action="store_true",
        help="Skip embedding phases (only run rerank)",
    )
    p.add_argument(
        "--output",
        default=os.path.join(SCRIPT_DIR, "benchmark_results.json"),
        help="JSON output file (default: <script_dir>/benchmark_results.json)",
    )
    p.add_argument(
        "--list-tests",
        action="store_true",
        help="Print the benchmark plan and exit (no models loaded)",
    )
    return p


def print_plan() -> None:
    print(__doc__)
    print("\nBenchmark plan:")
    print("  A. Environment + model loading + warmup timing")
    print("  B. Embeddings: stage breakdown + batch/seq sweep")
    print("  C. Rerank: stage breakdown + doc_count/block_size sweep")
    print("  D. torch.profiler traces for top kernels")
    print("  E. Eager vs compiled / bf16 vs fp32 / thread count sweep")
    print("  F. Findings report (console + JSON)")


def main() -> int:
    args = build_argparser().parse_args()

    if args.list_tests:
        print_plan()
        return 0

    n_iter_main = 2 if args.quick else 5
    n_iter_sweep = 1 if args.quick else 3
    n_iter_profile = 2 if args.quick else 3
    use_compile = not args.no_compile

    run_embed = not args.no_embed and args.phase in ("embed", "all")
    run_rerank = not args.no_rerank and args.phase in ("rerank", "all")
    run_profile = args.phase in ("profile", "all")
    run_compare = args.phase in ("compare", "all")
    # If phase is specifically env, only run env
    only_env = args.phase == "env"

    # ----- Phase A: environment -----
    env = gather_env_info()
    print_env(env)

    results: Dict[str, Any] = {"env": env}

    if only_env:
        _save_results(results, args.output)
        return 0

    # ----- Load models -----
    print_section("Phase A: Model Loading")
    embed_model = None
    rerank_model = None

    if run_embed or run_profile or run_compare:
        print(f"\n  Loading embedding model (compiled={use_compile})...")
        embed_model, embed_load_s = load_embedding_model(compiled=use_compile)
        print(f"  [OK] embedding loaded in {embed_load_s:.1f}s")
        print(f"  Embedding dim: {_get_embed_dim(embed_model)}")
        if use_compile:
            print("  Warming up compiled embedding (may take minutes)...")
            warmup_s = warmup_embedding_model(embed_model, n_iter=3 if not args.quick else 1)
            print(f"  [OK] embedding warmed up in {warmup_s:.1f}s")
        else:
            warmup_embedding_model(embed_model, n_iter=1)
        results["embed_load_seconds"] = round(embed_load_s, 2)

    if run_rerank or (run_profile and not args.no_rerank):
        print(f"\n  Loading reranker model (compiled={use_compile})...")
        rerank_model, rerank_load_s = load_reranker_model(compiled=use_compile)
        print(f"  [OK] reranker loaded in {rerank_load_s:.1f}s")
        if use_compile:
            print("  Warming up compiled reranker (may take minutes)...")
            warmup_s = warmup_reranker_model(rerank_model, n_iter=2 if not args.quick else 1)
            print(f"  [OK] reranker warmed up in {warmup_s:.1f}s")
        results["rerank_load_seconds"] = round(rerank_load_s, 2)

    # ----- Phase B: embeddings -----
    embed_stages = None
    embed_batch = None
    embed_seqlen = None
    if run_embed:
        print_section("Phase B: Embeddings — stage breakdown")
        embed_stages = bench_embedding_stage_timing(
            embed_model,
            texts=MEDIUM_DOCS,
            n_iter=n_iter_main,
            n_warmup=2 if not args.quick else 1,
        )
        print_stage_timings(embed_stages, "Phase B.1: Embeddings stage timings (10 medium docs)")
        results["embed_stages"] = {k: v.as_dict() for k, v in embed_stages.items()}

        print_section("Phase B.2: Embeddings — batch size sweep")
        embed_batch = bench_embedding_batch_sweep(
            embed_model, n_iter=n_iter_sweep, n_warmup=1
        )
        print_sweep_table(embed_batch, "Batch size sweep")
        results["embed_batch_sweep"] = embed_batch

        print_section("Phase B.3: Embeddings — sequence length sweep")
        embed_seqlen = bench_embedding_seqlen_sweep(
            embed_model, n_iter=n_iter_sweep, n_warmup=1
        )
        print_sweep_table(embed_seqlen, "Sequence length sweep")
        results["embed_seqlen_sweep"] = embed_seqlen

    # ----- Phase C: rerank -----
    rerank_stages = None
    rerank_doc = None
    rerank_block = None
    if run_rerank:
        print_section("Phase C: Rerank — stage breakdown")
        try:
            rerank_stages = bench_rerank_stage_timing(
                rerank_model,
                query=RERANKER_QUERY,
                documents=RERANKER_DOCS[:32],
                block_size=32,
                n_iter=n_iter_main,
                n_warmup=1,
            )
            print_stage_timings(rerank_stages, "Phase C.1: Rerank stage timings (32 docs, block=32)")
            results["rerank_stages"] = {k: v.as_dict() for k, v in rerank_stages.items()}
        except Exception as e:
            print(f"  [WARN] rerank stage timing failed: {e}")
            import traceback

            traceback.print_exc()

        print_section("Phase C.2: Rerank — doc count sweep")
        try:
            rerank_doc = bench_rerank_doc_count_sweep(
                rerank_model, n_iter=n_iter_sweep, n_warmup=1
            )
            print_sweep_table(rerank_doc, "Doc count sweep")
            results["rerank_doc_sweep"] = rerank_doc
        except Exception as e:
            print(f"  [WARN] rerank doc sweep failed: {e}")

        print_section("Phase C.3: Rerank — block size sweep")
        try:
            rerank_block = bench_rerank_block_size_sweep(
                rerank_model, n_iter=n_iter_sweep, n_warmup=1
            )
            print_sweep_table(rerank_block, "Block size sweep")
            results["rerank_block_sweep"] = rerank_block
        except Exception as e:
            print(f"  [WARN] rerank block sweep failed: {e}")

    # ----- Phase D: profiler -----
    embed_profile = None
    rerank_profile = None
    if run_profile:
        if embed_model is not None:
            print_section("Phase D.1: torch.profiler — embedding workload")
            trace_file = os.path.join(SCRIPT_DIR, "profile_embed_trace.json")
            embed_profile = profile_embedding_workload(
                embed_model, trace_file, n_iter=n_iter_profile
            )
            print_profile_top(embed_profile, "Top kernels: embeddings (30 medium docs)")
            results["embed_profile"] = {
                "top_ops": embed_profile["top_ops"],
                "trace_file": embed_profile["trace_file"],
            }
        if rerank_model is not None and not args.no_rerank:
            print_section("Phase D.2: torch.profiler — rerank workload")
            trace_file = os.path.join(SCRIPT_DIR, "profile_rerank_trace.json")
            try:
                rerank_profile = profile_rerank_workload(
                    rerank_model, trace_file, n_iter=n_iter_profile
                )
                print_profile_top(rerank_profile, "Top kernels: rerank (32 docs)")
                results["rerank_profile"] = {
                    "top_ops": rerank_profile["top_ops"],
                    "trace_file": rerank_profile["trace_file"],
                }
            except Exception as e:
                print(f"  [WARN] rerank profiling failed: {e}")
                import traceback

                traceback.print_exc()

    # ----- Phase E: comparisons -----
    dtype_cmp = None
    compile_cmp = None
    thread_cmp = None
    if run_compare and not args.quick:
        print_section("Phase E.1: dtype comparison (bf16 vs fp32) — embeddings")
        try:
            dtype_cmp = compare_dtypes_embedding(n_iter=3, n_warmup=1)
            print(f"  bf16: {dtype_cmp['bf16']['median_ms']}ms ({dtype_cmp['bf16']['texts_per_sec']} texts/s)")
            print(f"  fp32: {dtype_cmp['fp32']['median_ms']}ms ({dtype_cmp['fp32']['texts_per_sec']} texts/s)")
            print(f"  Speedup: {dtype_cmp['speedup_bf16_vs_fp32']}x")
            results["dtype_cmp"] = dtype_cmp
        except Exception as e:
            print(f"  [WARN] dtype comparison failed: {e}")

        if use_compile:
            print_section("Phase E.2: compile comparison (eager vs compiled) — embeddings")
            try:
                compile_cmp = compare_compile_embedding(n_iter=3)
                print(f"  eager:    {compile_cmp['eager_median_ms']}ms")
                print(f"  compiled: {compile_cmp['compiled_median_ms']}ms")
                print(f"  Speedup:  {compile_cmp['speedup']}x")
                results["compile_cmp"] = compile_cmp
            except Exception as e:
                print(f"  [WARN] compile comparison failed: {e}")

        print_section("Phase E.3: thread count sweep — embeddings")
        try:
            thread_cmp = compare_thread_counts_embedding(n_iter=3)
            print_sweep_table(thread_cmp["data_points"], "Thread count sweep")
            results["thread_cmp"] = thread_cmp
        except Exception as e:
            print(f"  [WARN] thread sweep failed: {e}")

    elif run_compare and args.quick:
        print("\n  [SKIP] Phase E (comparisons) skipped in --quick mode")

    # ----- Phase F: findings -----
    findings = derive_findings(
        env=env,
        embed_stages=embed_stages,
        embed_batch=embed_batch,
        rerank_stages=rerank_stages,
        rerank_doc=rerank_doc,
        embed_profile=embed_profile,
        rerank_profile=rerank_profile,
        dtype_cmp=dtype_cmp,
        compile_cmp=compile_cmp,
        thread_cmp=thread_cmp,
    )
    print_findings(findings)
    results["findings"] = findings

    # Save JSON
    _save_results(results, args.output)

    print_section("Done")
    print(f"\n  Results saved to: {os.path.abspath(args.output)}")
    if embed_profile:
        print(f"  Embedding trace:  {embed_profile['trace_file']}")
    if rerank_profile:
        print(f"  Rerank trace:     {rerank_profile['trace_file']}")
    print("\n  Open traces in chrome://tracing for kernel-level timeline view.")
    return 0


def _get_embed_dim(model: Any) -> int:
    try:
        return model.get_sentence_embedding_dimension()
    except Exception:
        return -1


def _save_results(results: Dict[str, Any], path: str) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    except Exception as e:
        print(f"  [WARN] failed to save results JSON: {e}")


if __name__ == "__main__":
    sys.exit(main())
