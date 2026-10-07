"""
In-process micro-benchmark for optimization variants.

Loads the embedding + reranker models exactly like jina_server.py, then
measures candidate optimizations one by one. Everything runs in one process
so numbers are directly comparable.

Usage:
  uv run python bench/microbench_variants.py [sections...]
  sections: lora hidden states shapes threads firstcall compile
"""
from __future__ import annotations

import copy
import gc
import os
import statistics
import sys
import time

MAX_THREADS = max(1, int(os.environ.get("JINA_TORCH_THREADS", "12")))
os.environ["OMP_NUM_THREADS"] = str(MAX_THREADS)
os.environ["MKL_NUM_THREADS"] = str(MAX_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(MAX_THREADS)

import torch

torch.set_num_threads(MAX_THREADS)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB_PATH = os.path.join(PROJECT_ROOT, "jinaai", "jina-embeddings-v5-text-small")
RER_PATH = os.path.join(PROJECT_ROOT, "jinaai", "jina-reranker-v3")
DTYPE = os.environ.get("JINA_MODEL_DTYPE", "bfloat16")

MEDIUM_DOCS = [
    "Machine learning is a branch of artificial intelligence that enables systems to learn from data and improve over time without being explicitly programmed.",
    "Neural networks are computational models inspired by the human brain, consisting of interconnected nodes organized in layers that learn hierarchical representations.",
    "Gradient descent is an optimization algorithm that iteratively adjusts parameters to minimize a loss function, with variants like SGD and Adam.",
    "Retrieval-Augmented Generation combines a retriever and a generator; the retriever fetches documents using embeddings and the generator conditions on them.",
    "REST is an architectural style using HTTP methods on resources identified by URIs, while GraphQL lets clients request exactly the fields they need.",
    "The attention mechanism computes weighted averages of value vectors where weights come from query-key similarity across multiple heads.",
    "LoRA injects low-rank decomposition matrices into linear layers, learning W + BA instead of updating W, drastically reducing trainable parameters.",
    "BM25 is a probabilistic ranking function scoring document relevance with term frequency saturation and length normalization.",
    "Fine-tuning adapts pre-trained models on task-specific data; parameter-efficient methods update small subsets of parameters.",
    "Deep learning refers to networks with many hidden layers that learn complex representations of data such as images, audio and text.",
]


def t_ms(fn, n_iter=5, n_warmup=2):
    """Return median ms over n_iter calls after n_warmup."""
    for _ in range(n_warmup):
        fn()
    ts = []
    for _ in range(n_iter):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts), ts


def load_embedding():
    from sentence_transformers import SentenceTransformer

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[DTYPE]
    model = SentenceTransformer(
        EMB_PATH,
        trust_remote_code=True,
        model_kwargs={
            "default_task": "retrieval",
            "default_prompt": "query",
            "dtype": dtype,
            "attn_implementation": "sdpa",
        },
    )
    model.to(device=torch.device("cpu"), dtype=dtype)
    return model


def load_reranker():
    from transformers import AutoModel

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[DTYPE]
    model = AutoModel.from_pretrained(
        RER_PATH, trust_remote_code=True, dtype=dtype, attn_implementation="sdpa"
    )
    model.to(device=torch.device("cpu"), dtype=dtype)
    model.eval()
    return model


def get_inner(st_model):
    """Return (transformer_module, peft_model, tokenizer)."""
    t = st_model._first_module()
    return t, t.model, t.tokenizer


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
def bench_lora_vs_merged(model):
    """Compare unmerged PEFT forward vs per-task merged plain-Qwen3 forward."""
    section("LoRA: unmerged PEFT vs merged plain model")
    transformer, peft_model, tokenizer = get_inner(model)

    def forward_peft(batch):
        with torch.no_grad():
            peft_model.set_adapter("retrieval")
            return peft_model(**batch)

    batch = tokenizer(
        [f"Query: {t}" for t in MEDIUM_DOCS],
        return_tensors="pt", padding=True, truncation=True, max_length=128,
    )
    med_peft, _ = t_ms(lambda: forward_peft(batch))
    print(f"  unmerged PEFT forward (10 texts, ~64 tok): {med_peft:.2f} ms")

    # Merge a copy for the retrieval task
    t0 = time.perf_counter()
    merged = copy.deepcopy(peft_model).merge_and_unload(adapter_names=["retrieval"])
    merge_s = time.perf_counter() - t0
    print(f"  merge_and_unload copy took {merge_s:.2f} s")

    def forward_merged(b):
        with torch.no_grad():
            return merged(**b)

    med_merged, _ = t_ms(lambda: forward_merged(batch))
    print(f"  merged plain forward (same batch):          {med_merged:.2f} ms")
    print(f"  --> speedup {med_peft / med_merged:.3f}x")

    # small batch=1 case (serial latency regime)
    b1 = tokenizer(["Query: What is machine learning?"], return_tensors="pt")
    med_peft1, _ = t_ms(lambda: forward_peft(b1))
    med_merged1, _ = t_ms(lambda: forward_merged(b1))
    print(f"  unmerged PEFT forward (1 text):  {med_peft1:.2f} ms")
    print(f"  merged plain forward (1 text):   {med_merged1:.2f} ms  ({med_peft1/med_merged1:.3f}x)")

    # verify equivalence
    with torch.no_grad():
        peft_model.set_adapter("retrieval")
        o1 = peft_model(**batch).last_hidden_state
        o2 = merged(**batch).last_hidden_state
    diff = (o1.float() - o2.float()).abs().max().item()
    print(f"  max abs diff merged vs peft: {diff:.5f}")
    return merged


def bench_hidden_states(reranker):
    """Reranker forward: output_hidden_states=True vs False."""
    section("Reranker: output_hidden_states=True (current) vs False")
    tok = reranker._ensure_tokenizer()
    tok = reranker._tokenizer

    inner = reranker.model  # Qwen3Model
    for n_docs in (8, 16, 48):
        docs = (MEDIUM_DOCS * 8)[:n_docs]
        prompt_parts = [
            "<|im_start|>system\nYou are a search relevance expert.<|im_end|>\n<|im_start|>user\n",
            f"I will provide you with {len(docs)} passages. Rank them by relevance to query: What is ML?\n",
        ]
        prompt_parts += [f'<passage id="{i}">\n{d}<|embed_token|>\n</passage>\n' for i, d in enumerate(docs)]
        prompt_parts.append("<query>\nWhat is ML?<|rerank_token|>\n</query><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        prompt = "".join(prompt_parts)
        batch = tok(text=[prompt], padding=True, padding_side="left", return_tensors="pt")

        n_tok = batch["input_ids"].shape[1]
        print(f"\n  prompt tokens: {n_tok} ({n_docs} docs)")

        def fwd_hs():
            with torch.no_grad():
                return reranker(**batch)

        med_hs, _ = t_ms(fwd_hs, n_iter=3, n_warmup=1)
        print(f"  current forward (hidden_states=True, use_cache=False): {med_hs:.2f} ms")

        def fwd_direct():
            with torch.no_grad():
                out = inner(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                )
                return out.last_hidden_state

        med_direct, _ = t_ms(fwd_direct, n_iter=3, n_warmup=1)
        print(f"  direct Qwen3Model (no hidden_states list):              {med_direct:.2f} ms")
        print(f"  --> speedup {med_hs / med_direct:.3f}x")


def bench_shape_cache(model):
    """Measure cold-shape penalty: same batch size, new seq length."""
    section("Cold-shape penalty (first call at a new seq length)")
    transformer, peft_model, tokenizer = get_inner(model)

    def run_at_len(n_tok):
        ids = torch.ones((1, n_tok), dtype=torch.long)
        mask = torch.ones((1, n_tok), dtype=torch.long)
        with torch.no_grad():
            peft_model.set_adapter("retrieval")
            peft_model(input_ids=ids, attention_mask=mask)

    run_at_len(64)  # warm base
    lengths = [16, 32, 48, 64, 96, 128, 129, 192, 256, 384, 512]
    for L in lengths:
        t0 = time.perf_counter()
        run_at_len(L)
        cold = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        run_at_len(L)
        warm = (time.perf_counter() - t0) * 1000
        print(f"  seq={L:>4}: first={cold:8.2f} ms   second={warm:8.2f} ms   penalty={cold - warm:7.2f} ms")


def bench_first_call_per_task(model):
    """Measure first-call latency per task after only warming 'retrieval'."""
    section("First-call penalty per task (only retrieval warmed)")
    transformer, peft_model, tokenizer = get_inner(model)
    texts = [f"Query: {t}" for t in MEDIUM_DOCS[:4]]
    batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)

    with torch.no_grad():
        peft_model.set_adapter("retrieval")
        peft_model(**batch)  # warm retrieval

    for task in ["retrieval", "text-matching", "clustering", "classification", "retrieval"]:
        with torch.no_grad():
            t0 = time.perf_counter()
            peft_model.set_adapter(task)
            peft_model(**batch)
        first = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        with torch.no_grad():
            peft_model.set_adapter(task)
            peft_model(**batch)
        second = (time.perf_counter() - t0) * 1000
        print(f"  task={task:<15} first={first:7.2f} ms  second={second:7.2f} ms")


def bench_compile(model):
    """torch.compile default mode on PEFT model with fixed shape."""
    section("torch.compile (mode=default) fixed-shape probe")
    transformer, peft_model, tokenizer = get_inner(model)
    batch = tokenizer(
        [f"Query: {t}" for t in MEDIUM_DOCS[:4]],
        return_tensors="pt", padding=True, truncation=True, max_length=128,
    )
    with torch.no_grad():
        peft_model.set_adapter("retrieval")
        peft_model(**batch)
    med_eager, _ = t_ms(lambda: (peft_model.set_adapter("retrieval"), peft_model(**batch)) and None, n_iter=3, n_warmup=1)
    print(f"  eager: {med_eager:.2f} ms")

    t0 = time.perf_counter()
    compiled = torch.compile(peft_model, dynamic=False)
    with torch.no_grad():
        compiled(**batch)
    compile_s = time.perf_counter() - t0
    print(f"  compile (1 shape): {compile_s:.1f} s")
    med_comp, _ = t_ms(lambda: compiled(**batch), n_iter=3, n_warmup=1)
    print(f"  compiled: {med_comp:.2f} ms  ({med_eager / med_comp:.3f}x vs eager)")


def bench_threads(model):
    """Thread sweep on merged/plain forward."""
    section("Thread sweep (PEFT forward, 10 medium texts)")
    transformer, peft_model, tokenizer = get_inner(model)
    batch = tokenizer(
        [f"Query: {t}" for t in MEDIUM_DOCS],
        return_tensors="pt", padding=True, truncation=True, max_length=128,
    )
    results = {}
    for nt in [4, 6, 8, 12, 16, 24]:
        torch.set_num_threads(nt)
        with torch.no_grad():
            peft_model.set_adapter("retrieval")
            peft_model(**batch)
        med, _ = t_ms(lambda: peft_model(**batch), n_iter=3, n_warmup=1)
        results[nt] = med
        print(f"  threads={nt:>2}: {med:8.2f} ms")
    torch.set_num_threads(MAX_THREADS)
    best = min(results, key=results.get)
    print(f"  best: {best} threads ({results[best]:.2f} ms)")


SECTIONS = {
    "lora": bench_lora_vs_merged,
    "hidden": bench_hidden_states,
    "shapes": bench_shape_cache,
    "firstcall": bench_first_call_per_task,
    "compile": bench_compile,
    "threads": bench_threads,
}


def main():
    requested = [a for a in sys.argv[1:] if a in SECTIONS]
    if not requested:
        requested = list(SECTIONS)
    print(f"torch {torch.__version__} | threads={torch.get_num_threads()} | dtype={DTYPE}")
    print(f"cpu capability: {torch.backends.cpu.get_cpu_capability()}")

    need_emb = any(s in ("lora", "shapes", "firstcall", "compile", "threads") for s in requested)
    need_rer = "hidden" in requested

    emb = load_embedding() if need_emb else None
    rer = load_reranker() if need_rer else None

    for s in requested:
        try:
            if s == "hidden":
                SECTIONS[s](rer)
            else:
                SECTIONS[s](emb)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  [FAIL] section {s}: {e}")


if __name__ == "__main__":
    main()
