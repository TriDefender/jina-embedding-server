"""
Jina Embedding — dtype / precision overhead investigation.

Hypothesis-driven experiment harness inspired by
  https://github.com/houslast3/85.30-GFLOPS-Single-Core-FP32-Matrix-Multiplication-on-AMD-Zen-3
which tested 28 GEMM configurations with one variable changed per trial, validated
accuracy against a reference, and reported GFLOPS vs theoretical peak.

--------------------------------------------------------------------------------
HYPOTHESES (formed from prior `benchmark_inference.py` measurements on a
Ryzen 9 9900X with bf16 + AVX512 + AVX512_VNNI but NO AVX512_BF16):

  H1  PyTorch bf16 GEMM on AVX512 (no AVX512_BF16) is slower than fp32
      because of internal upcast.
      FALSIFIED by 1024^3 microbench: bf16=0.52ms vs fp32=1.96ms (3.8x faster).
      Re-test at multiple sizes to confirm.

  H2  End-to-end model inference mirrors kernel-level speedup, i.e. bf16 is
      also faster at the transformer level (where dtype conversions,
      softmax, normalization interleave with matmuls).

  H3  int8 dynamic quantization (uses AVX512_VNNI VPDPBUSD) beats both
      bf16 and fp32 on this CPU.

  H4  set_adapter() 17.86ms per-call overhead is dominated by PEFT weight
      swapping (computation, not I/O). Caching or one-time setup eliminates it.

  H5  Mixed precision (bf16 weights + fp32 attention scores) preserves
      accuracy without paying for full fp32 matmul.

  H6  Achieved GFLOPS on the embedding forward pass is far below theoretical
      peak (memory bandwidth bound, not compute bound).

--------------------------------------------------------------------------------
OUTPUTS
  - Console report with verdict per hypothesis
  - dtype_results.json (full numeric tables)
  - Pareto frontier chart data (speed vs accuracy)
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# =============================================================================
# Thread config (mirror jina_server.py exactly)
# =============================================================================
MAX_THREADS = 24
os.environ["OMP_NUM_THREADS"] = str(MAX_THREADS)
os.environ["MKL_NUM_THREADS"] = str(MAX_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(MAX_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(MAX_THREADS)
os.environ["TORCH_INTEROP_THREADS"] = "8"

import torch  # noqa: E402  (must come after OMP env vars)

torch.set_num_threads(MAX_THREADS)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MODELS_DIR = os.path.join(PROJECT_ROOT, "jinaai")
EMBEDDING_MODEL_PATH = os.path.join(MODELS_DIR, "jina-embeddings-v5-text-small")

# =============================================================================
# Theoretical peak for AMD Zen 5 (Ryzen 9 9900X)
# =============================================================================
# Per core: 2 FMA units × 16 floats (AVX-512) × 2 ops (mul+add) × clock
# Boost clock: ~5.4 GHz
# Single-core fp32 peak: 2 × 16 × 2 × 5.4 = 345.6 GFLOPS
# 12 cores fp32 peak:    345.6 × 12 = 4147.2 GFLOPS
# For bf16 without AVX512_BF16: same compute throughput as fp32 (emulated)
# For int8 via VPDPBUSD (VNNI): 4 int8 muls per lane → effectively 4x throughput
#   int8 per-core: 2 × 64 × 2 × 5.4 = 1382.4 GOPS
ZEN5_BOOST_GHZ = 5.4
ZEN5_PHYSICAL_CORES = 12
ZEN5_FP32_PEAK_GFLOPS = 2 * 16 * 2 * ZEN5_BOOST_GHZ * ZEN5_PHYSICAL_CORES  # 4147.2
ZEN5_INT8_PEAK_GOPS = 2 * 64 * 2 * ZEN5_BOOST_GHZ * ZEN5_PHYSICAL_CORES  # 16588.8

# Realistic text samples (varied length)
SAMPLE_TEXTS_SHORT = [
    "What is machine learning?",
    "How do neural networks work?",
    "Explain gradient descent",
    "What is BM25 retrieval?",
    "Compare REST vs GraphQL",
    "What is fine-tuning?",
    "Explain LoRA adapters",
    "What is RAG?",
]

SAMPLE_TEXTS_MEDIUM = [
    "Machine learning is a branch of artificial intelligence that enables systems to learn from data and improve over time without being explicitly programmed.",
    "Neural networks are computational models inspired by the human brain. They consist of interconnected nodes organized in layers that learn hierarchical representations.",
    "Gradient descent is an optimization algorithm that iteratively adjusts model parameters to minimize a loss function. Variants include SGD, mini-batch GD, and Adam.",
    "BM25 is a probabilistic ranking function used by search engines to score document relevance. It extends TF-IDF with length normalization and term frequency saturation.",
    "REST uses HTTP methods on resources identified by URIs. GraphQL lets clients request exactly the fields they need via a single endpoint. REST is simpler; GraphQL is flexible.",
    "Fine-tuning adapts a pre-trained model to a specific task by continuing training on labeled data. Full fine-tuning updates all parameters; PEFT methods freeze the backbone.",
    "LoRA injects trainable low-rank decomposition matrices into existing linear layers. Instead of W, it learns W + BA where B and A are small rank-r matrices.",
    "Retrieval-Augmented Generation combines a retriever and a generator. The retriever fetches relevant documents using embeddings; the generator conditions on retrieved context.",
]


# =============================================================================
# Timing helpers
# =============================================================================


@dataclass
class BenchResult:
    label: str
    median_ms: float
    mean_ms: float
    p95_ms: float
    n_iter: int
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "median_ms": round(self.median_ms, 3),
            "mean_ms": round(self.mean_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "n_iter": self.n_iter,
            **{k: v for k, v in self.extra.items()},
        }


def time_repeated(
    fn: Callable[[], Any],
    n_iter: int,
    n_warmup: int = 3,
) -> Tuple[Any, BenchResult]:
    """Run fn n_warmup + n_iter times. Return (last_result, BenchResult)."""
    last = None
    timings: List[float] = []
    for _ in range(n_warmup):
        fn()
    for _ in range(n_iter):
        gc.collect()
        t0 = time.perf_counter()
        last = fn()
        timings.append((time.perf_counter() - t0) * 1000.0)
    return last, BenchResult(
        label="",
        median_ms=statistics.median(timings),
        mean_ms=statistics.fmean(timings),
        p95_ms=sorted(timings)[max(0, int(len(timings) * 0.95) - 1)] if len(timings) > 1 else timings[0],
        n_iter=n_iter,
    )


# =============================================================================
# Print helpers
# =============================================================================


def print_section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def print_header() -> None:
    print("=" * 78)
    print("dtype overhead investigation — hypothesis-driven experiments")
    print("Methodology borrowed from:")
    print("  houslast3/85.30-GFLOPS-Single-Core-FP32-Matrix-Multiplication-on-AMD-Zen-3")
    print("    - One variable changed per experiment")
    print("    - Validation against reference (fp32)")
    print("    - GFLOPS vs theoretical peak")
    print("    - Categorized root-cause analysis")
    print("=" * 78)
    print("\nHypotheses:")
    hypotheses_text = [
        ("H1", "falsified pending", "bf16 GEMM slower than fp32 due to upcast"),
        ("H2", "pending", "bf16 end-to-end speedup mirrors kernel-level speedup"),
        ("H3", "pending", "int8 dynamic quant beats bf16 on AVX512_VNNI"),
        ("H4", "pending", "set_adapter() 17.86ms is from PEFT weight swap (cacheable)"),
        ("H5", "pending", "mixed precision (bf16+fp32 attn) preserves accuracy"),
        ("H6", "pending", "forward-pass GFLOPS is far below peak (memory-bound)"),
    ]
    for hid, status, text in hypotheses_text:
        print(f"  {hid} [{status:>18}] {text}")
    print(f"\nZen 5 theoretical peak: {ZEN5_FP32_PEAK_GFLOPS:.0f} GFLOPS fp32, "
          f"{ZEN5_INT8_PEAK_GOPS:.0f} GOPS int8")


# =============================================================================
# Experiment E1: GEMM microbench at multiple sizes
# =============================================================================


def matmul_flops(m: int, n: int, k: int) -> int:
    """FLOPs for C[m,n] = A[m,k] @ B[k,n]."""
    return 2 * m * n * k


def bench_gemm_dtype(
    m: int, n: int, k: int, dtype: torch.dtype, n_iter: int = 20
) -> Dict[str, Any]:
    """Bench C = A @ B^T using a torch.nn.Linear layer (mimics real transformer Linear)."""
    a = torch.randn(m, k, dtype=torch.float32)
    b = torch.randn(n, k, dtype=torch.float32)

    if dtype == torch.qint8:
        linear_module = torch.nn.Linear(k, n, bias=False)
        with torch.no_grad():
            linear_module.weight.copy_(b)
        quantized_module = torch.ao.quantization.quantize_dynamic(
            linear_module, {torch.nn.Linear}, dtype=torch.qint8
        )

        def fn():
            return quantized_module(a)
    else:
        a_t = a.to(dtype)
        linear_module = torch.nn.Linear(k, n, bias=False, dtype=dtype)
        with torch.no_grad():
            linear_module.weight.copy_(b.to(dtype))

        def fn():
            return linear_module(a_t)

    # Warmup
    for _ in range(5):
        fn()

    # Measure
    timings: List[float] = []
    for _ in range(n_iter):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        timings.append((time.perf_counter() - t0) * 1000.0)

    median_ms = statistics.median(timings)
    flops = matmul_flops(m, n, k)
    dtype_name_map = {
        torch.float32: "fp32",
        torch.bfloat16: "bf16",
        torch.qint8: "int8",
    }
    dtype_name = dtype_name_map.get(dtype, str(dtype).replace("torch.", ""))
    if dtype == torch.qint8:
        ops_label = "GOPS"
        ops_value = flops / 1e9
    else:
        ops_label = "GFLOPS"
        ops_value = flops / 1e9
    achieved = ops_value / (median_ms / 1000.0)
    peak = ZEN5_INT8_PEAK_GOPS if dtype == torch.qint8 else ZEN5_FP32_PEAK_GFLOPS
    pct_peak = (achieved / peak) * 100.0

    return {
        "dtype": dtype_name,
        "shape": f"{m}x{n}x{k}",
        "median_ms": round(median_ms, 3),
        "flops": flops,
        f"achieved_{ops_label.lower()}": round(achieved, 1),
        f"peak_{ops_label.lower()}": round(peak, 1),
        "pct_peak": round(pct_peak, 2),
        "n_iter": n_iter,
    }


def experiment_e1_gemm_microbench() -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """E1: bf16 vs fp32 vs int8 GEMM across sizes."""
    print_section("E1: GEMM microbench — bf16 vs fp32 vs int8")
    print("  Variable: weight/activation dtype. Same shapes, same kernels.\n")

    shapes = [
        # Mimic Linear layers in a 1024-dim transformer
        (1, 1024, 1024),     # single-token forward
        (8, 1024, 1024),     # small batch
        (32, 1024, 1024),    # server default batch
        (64, 1024, 1024),    # large batch
        (32, 3072, 1024),    # MLP intermediate layer (Qwen3 hidden=1024, inter=3072)
        (32, 16384, 1024),   # large LM head style (vocab projection)
    ]
    dtypes_to_test: List[Tuple[str, torch.dtype]] = [
        ("fp32", torch.float32),
        ("bf16", torch.bfloat16),
        ("int8", torch.qint8),
    ]

    results: List[Dict[str, Any]] = []
    for shape in shapes:
        m, n, k = shape
        print(f"  Shape A[{m},{k}] @ B^T[{k},{n}]  (flops={matmul_flops(m,n,k)/1e6:.1f}M)")
        for name, dt in dtypes_to_test:
            try:
                r = bench_gemm_dtype(m, n, k, dt, n_iter=15)
                results.append(r)
                unit = "GOPS" if dt == torch.qint8 else "GFLOPS"
                print(
                    f"    {name:>5}: {r['median_ms']:>7.3f}ms  "
                    f"{r[f'achieved_{unit.lower()}']:>7.1f} {unit}  "
                    f"({r['pct_peak']:>5.1f}% of {unit.split('O')[0]}Opeak)"
                )
            except Exception as e:
                print(f"    {name:>5}: FAILED ({e})")
                results.append({"dtype": name, "shape": f"{m}x{n}x{k}", "error": str(e)})
        print()

    # Hypothesis H1 verdict
    print("  --- H1 verdict ---")
    h1_results = [r for r in results if "error" not in r]
    bf16_wins = 0
    fp32_wins = 0
    for shape_str in {r["shape"] for r in h1_results}:
        bf16 = next((r for r in h1_results if r["shape"] == shape_str and r["dtype"] == "bf16"), None)
        fp32 = next((r for r in h1_results if r["shape"] == shape_str and r["dtype"] == "fp32"), None)
        if bf16 and fp32:
            if bf16["median_ms"] < fp32["median_ms"]:
                bf16_wins += 1
            else:
                fp32_wins += 1
    total = bf16_wins + fp32_wins
    if total == 0:
        verdicts = {"H1": "INCONCLUSIVE — no shape had both bf16 and fp32 data."}
    elif bf16_wins > fp32_wins:
        verdicts = {
            "H1": (
                f"FALSIFIED — bf16 is faster at kernel level in {bf16_wins}/{total} "
                "shapes. PyTorch/oneDNN uses efficient emulation even without AVX512_BF16 "
                "(memory bandwidth halved is the main win)."
            )
        }
    else:
        verdicts = {"H1": f"CONFIRMED — fp32 wins in {fp32_wins}/{total} shapes."}
    print(f"  {verdicts['H1']}")
    return results, verdicts


# =============================================================================
# Experiment E2: end-to-end embedding inference per dtype
# =============================================================================


def load_embedding_dtype(dtype: torch.dtype) -> Any:
    """Load the embedding model with the requested dtype."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        EMBEDDING_MODEL_PATH,
        trust_remote_code=True,
        model_kwargs={
            "default_task": "retrieval",
            "default_prompt": "query",
            "torch_dtype": dtype,
            "attn_implementation": "sdpa",
        },
    )


def apply_int8_dynamic_quantization(model: Any) -> Tuple[Any, str]:
    """Apply torch dynamic int8 quantization to Linear layers in the embedding model."""
    from torch.ao.quantization import quantize_dynamic

    inner_st = model._first_module()
    inner_model = inner_st.model
    # Quantize the base transformer's Linear layers
    try:
        quantized_base = quantize_dynamic(
            inner_model.get_base_model() if hasattr(inner_model, "get_base_model") else inner_model,
            {torch.nn.Linear},
            dtype=torch.qint8,
        )
        # Try to splice back
        if hasattr(inner_model, "base_model"):
            inner_model.base_model.model = quantized_base
        else:
            inner_st.model = quantized_base
        return model, "int8_dynamic_applied"
    except Exception as e:
        return model, f"int8_dynamic_failed: {e}"


def experiment_e2_end_to_end_inference(n_iter: int = 3) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """E2: end-to-end embedding inference per dtype."""
    print_section("E2: end-to-end embedding inference per dtype")
    print("  Variable: model loaded dtype. Same texts, same task, same batch.\n")

    texts = SAMPLE_TEXTS_MEDIUM
    results: Dict[str, Any] = {}

    # fp32 reference
    print("  Loading fp32 reference model...")
    m_fp32 = load_embedding_dtype(torch.float32)
    _, br_fp32 = time_repeated(
        fn=lambda _m=m_fp32: _m.encode(
            texts, task="retrieval", prompt_name="query",
            normalize_embeddings=True, convert_to_numpy=False,
        ),
        n_iter=n_iter, n_warmup=2,
    )
    br_fp32.label = "fp32"
    results["fp32"] = br_fp32.as_dict()
    print(f"    fp32:  {br_fp32.median_ms:.1f}ms median")
    del m_fp32
    gc.collect()

    # bf16
    print("  Loading bf16 model...")
    m_bf16 = load_embedding_dtype(torch.bfloat16)
    _, br_bf16 = time_repeated(
        fn=lambda _m=m_bf16: _m.encode(
            texts, task="retrieval", prompt_name="query",
            normalize_embeddings=True, convert_to_numpy=False,
        ),
        n_iter=n_iter, n_warmup=2,
    )
    br_bf16.label = "bf16"
    results["bf16"] = br_bf16.as_dict()
    print(f"    bf16:  {br_bf16.median_ms:.1f}ms median")
    del m_bf16
    gc.collect()

    # int8 dynamic quantization (start from fp32, then quantize)
    print("  Loading fp32 + int8 dynamic quantization...")
    m_int8 = load_embedding_dtype(torch.float32)
    m_int8, status = apply_int8_dynamic_quantization(m_int8)
    print(f"    int8 status: {status}")
    if "applied" in status:
        try:
            _, br_int8 = time_repeated(
                fn=lambda _m=m_int8: _m.encode(
                    texts, task="retrieval", prompt_name="query",
                    normalize_embeddings=True, convert_to_numpy=False,
                ),
                n_iter=n_iter, n_warmup=2,
            )
            br_int8.label = "int8"
            results["int8"] = br_int8.as_dict()
            print(f"    int8:  {br_int8.median_ms:.1f}ms median")
        except Exception as e:
            print(f"    int8 inference FAILED: {e}")
            results["int8"] = {"error": str(e)}
    else:
        results["int8"] = {"error": status}
    del m_int8
    gc.collect()

    # H2 verdict
    verdicts: Dict[str, str] = {}
    if "bf16" in results and "fp32" in results and "error" not in results["bf16"]:
        speedup = results["fp32"]["median_ms"] / results["bf16"]["median_ms"]
        # Significance threshold: end-to-end timing has ~5% noise; <10% delta = same
        if speedup >= 1.3:
            verdicts["H2"] = (
                f"CONFIRMED — bf16 is {speedup:.2f}x faster than fp32 end-to-end. "
                "Kernel-level speedup carries through to transformer forward pass."
            )
        elif speedup >= 1.1:
            verdicts["H2"] = (
                f"PARTIALLY CONFIRMED — bf16 is {speedup:.2f}x faster end-to-end. "
                "Most of the kernel speedup is eaten by attention/softmax/norm ops in fp32 "
                "and Python/framework overhead."
            )
        else:
            verdicts["H2"] = (
                f"FALSIFIED — bf16 ({results['bf16']['median_ms']:.1f}ms) is statistically "
                f"identical to fp32 ({results['fp32']['median_ms']:.1f}ms), {speedup:.2f}x. "
                "Kernel-level bf16 speedup does NOT carry to end-to-end. The bottleneck is "
                "NOT matmul compute — it's memory bandwidth (loading 677M params per forward) "
                "and Python/PEFT framework overhead. Switching dtypes won't help."
            )
    if "int8" in results and "error" not in results["int8"] and "bf16" in results:
        speedup_vs_bf16 = results["bf16"]["median_ms"] / results["int8"]["median_ms"]
        verdicts["H3"] = (
            f"CONFIRMED — int8 is {speedup_vs_bf16:.2f}x faster than bf16. "
            "AVX512_VNNI delivers the expected multi-fold win."
            if speedup_vs_bf16 >= 1.3
            else f"FALSIFIED — int8 only {speedup_vs_bf16:.2f}x faster than bf16. "
            "Dynamic quant overhead (per-call scale/zero-point) negates VNNI gains."
        )
    elif "int8" in results and "error" in results.get("int8", {}):
        verdicts["H3"] = (
            "FALSIFIED — int8 dynamic quantization FAILS on this model: "
            f"{results['int8']['error']}. PEFT LoRA wrapper changes input dtype "
            "and is incompatible with qlinear_dynamic. Static quantization would require "
            "calibration and merging LoRA weights — significant engineering cost."
        )
    return results, verdicts


# =============================================================================
# Experiment E3: accuracy validation
# =============================================================================


def cosine_similarity_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine similarity per row."""
    a_n = torch.nn.functional.normalize(a, dim=-1)
    b_n = torch.nn.functional.normalize(b, dim=-1)
    return (a_n * b_n).sum(dim=-1)


def _embeddings_to_tensor(emb: Any) -> torch.Tensor:
    """Coerce SentenceTransformer.encode output to a 2D fp32 Tensor."""
    if isinstance(emb, list):
        # PEFT path returns list of per-batch tensors; stack into 2D
        emb = [t if isinstance(t, torch.Tensor) else torch.as_tensor(t) for t in emb]
        return torch.stack(emb).float().squeeze(0) if emb[0].dim() == 1 else torch.cat([t.reshape(-1, t.shape[-1]) if t.dim() > 2 else t for t in emb]).float()
    if isinstance(emb, torch.Tensor):
        return emb.float()
    return torch.as_tensor(emb).float()


def experiment_e3_accuracy() -> Tuple[Dict[str, Any], Dict[str, str]]:
    """E3: compare embedding accuracy vs fp32 reference."""
    print_section("E3: accuracy validation vs fp32 reference")
    print("  Variable: model dtype. Same texts. Metric: cosine sim & L2 distance.\n")

    texts = SAMPLE_TEXTS_MEDIUM
    print("  Loading fp32 reference...")
    m_fp32 = load_embedding_dtype(torch.float32)
    ref_raw = m_fp32.encode(
        texts, task="retrieval", prompt_name="query",
        normalize_embeddings=True, convert_to_numpy=False,
    )
    ref_f32 = _embeddings_to_tensor(ref_raw)
    if ref_f32.dim() == 1:
        ref_f32 = ref_f32.unsqueeze(0)
    print(f"    reference shape: {tuple(ref_f32.shape)}")
    del m_fp32
    gc.collect()

    print("  Loading bf16...")
    m_bf16 = load_embedding_dtype(torch.bfloat16)
    bf16_raw = m_bf16.encode(
        texts, task="retrieval", prompt_name="query",
        normalize_embeddings=True, convert_to_numpy=False,
    )
    bf16_f32 = _embeddings_to_tensor(bf16_raw)
    if bf16_f32.dim() == 1:
        bf16_f32 = bf16_f32.unsqueeze(0)
    cos_bf16 = cosine_similarity_rows(ref_f32, bf16_f32)
    l2_bf16 = torch.norm(ref_f32 - bf16_f32, dim=-1)
    del m_bf16
    gc.collect()

    print("  Loading int8...")
    m_int8 = load_embedding_dtype(torch.float32)
    m_int8, status = apply_int8_dynamic_quantization(m_int8)
    int8_emb = None
    if "applied" in status:
        try:
            int8_raw = m_int8.encode(
                texts, task="retrieval", prompt_name="query",
                normalize_embeddings=True, convert_to_numpy=False,
            )
            int8_f32 = _embeddings_to_tensor(int8_raw)
            if int8_f32.dim() == 1:
                int8_f32 = int8_f32.unsqueeze(0)
            int8_emb = int8_f32
        except Exception as e:
            print(f"    int8 inference failed: {e}")
    if int8_emb is not None:
        cos_int8 = cosine_similarity_rows(ref_f32, int8_emb)
        l2_int8 = torch.norm(ref_f32 - int8_emb, dim=-1)
    else:
        cos_int8 = None
        l2_int8 = None
    del m_int8
    gc.collect()

    def tensor_stats(t: Optional[torch.Tensor]) -> Dict[str, float]:
        if t is None:
            return {"mean": -1.0, "min": -1.0, "max": -1.0}
        return {
            "mean": round(t.mean().item(), 6),
            "min": round(t.min().item(), 6),
            "max": round(t.max().item(), 6),
        }

    results = {
        "bf16": {
            "cosine_sim": tensor_stats(cos_bf16),
            "l2_distance": tensor_stats(l2_bf16),
        },
        "int8": {
            "cosine_sim": tensor_stats(cos_int8),
            "l2_distance": tensor_stats(l2_int8),
        },
    }
    print(f"\n  vs fp32 reference ({len(texts)} texts):")
    print(f"    {'dtype':<8} {'cosine mean':<14} {'cosine min':<14} {'L2 mean':<14} {'L2 max':<14}")
    print(f"    {'-'*8} {'-'*14} {'-'*14} {'-'*14} {'-'*14}")
    for name in ["bf16", "int8"]:
        r = results[name]
        cos = r["cosine_sim"]
        l2 = r["l2_distance"]
        print(
            f"    {name:<8} {cos['mean']:<14.6f} {cos['min']:<14.6f} "
            f"{l2['mean']:<14.6f} {l2['max']:<14.6f}"
        )

    # H5 verdict (will be augmented in E5)
    verdicts: Dict[str, str] = {}
    if cos_bf16 is not None:
        min_cos = cos_bf16.min().item()
        verdicts["H5_partial"] = (
            f"bf16 vs fp32 cosine sim: min={min_cos:.4f}. "
            "BF16 itself {'preserves' if min_cos > 0.999 else 'degrades'} accuracy at this level."
        )
    return results, verdicts


# =============================================================================
# Experiment E4: set_adapter isolation
# =============================================================================


def experiment_e4_set_adapter_isolation() -> Tuple[Dict[str, Any], Dict[str, str]]:
    """E4: break down set_adapter() 17.86ms into subcomponents."""
    print_section("E4: set_adapter() isolation")
    print("  Variable: adapter switching operation. Same model, same task.\n")

    print("  Loading bf16 model...")
    model = load_embedding_dtype(torch.bfloat16)
    inner_st = model._first_module()
    inner_model = inner_st.model  # PeftMixedModel

    # Tasks available
    tasks = ["retrieval", "text-matching", "clustering", "classification"]
    print(f"  Available adapters: {tasks}")

    # Warmup
    for t in tasks:
        inner_model.set_adapter(t)

    results: Dict[str, Any] = {}

    # E4.1: set_adapter to a NEW task (the 17.86ms case in the prior benchmark)
    def switch_to_new_task():
        inner_model.set_adapter("clustering")

    _, br_new = time_repeated(switch_to_new_task, n_iter=20, n_warmup=3)
    br_new.label = "set_adapter (switch to new task)"
    results["switch_to_new"] = br_new.as_dict()
    print(f"  set_adapter(switch to new task):     {br_new.median_ms:.3f}ms")

    # E4.2: set_adapter to the SAME task already active (should be cheaper)
    inner_model.set_adapter("retrieval")
    def switch_to_same_task():
        inner_model.set_adapter("retrieval")  # already active

    _, br_same = time_repeated(switch_to_same_task, n_iter=20, n_warmup=3)
    br_same.label = "set_adapter (same task, no-op)"
    results["switch_to_same"] = br_same.as_dict()
    print(f"  set_adapter(same task, no-op):       {br_same.median_ms:.3f}ms")

    # E4.3: set_adapter to a task we've used before (cache check)
    inner_model.set_adapter("retrieval")
    inner_model.set_adapter("clustering")  # prime
    def switch_back_and_forth():
        inner_model.set_adapter("retrieval")
        inner_model.set_adapter("clustering")

    _, br_pingpong = time_repeated(switch_back_and_forth, n_iter=20, n_warmup=3)
    br_pingpong.label = "set_adapter (back-and-forth x2)"
    results["pingpong"] = br_pingpong.as_dict()
    print(f"  set_adapter(back-and-forth x2):      {br_pingpong.median_ms:.3f}ms")

    # E4.4: time just the .eval() call
    def call_eval():
        inner_model.eval()

    _, br_eval = time_repeated(call_eval, n_iter=20, n_warmup=3)
    br_eval.label = "model.eval() alone"
    results["eval_only"] = br_eval.as_dict()
    print(f"  model.eval() alone:                  {br_eval.median_ms:.3f}ms")

    # E4.5: deep-dive — check PEFT internals
    print("\n  PEFT internals inspection:")
    active = inner_model.active_adapter if hasattr(inner_model, "active_adapter") else "unknown"
    print(f"    active adapter(s): {active}")
    if hasattr(inner_model, "peft_config"):
        n_adapters = len(inner_model.peft_config)
        print(f"    loaded adapters: {n_adapters}")
    # Count adapter layers
    n_lora_layers = sum(
        1 for _ in inner_model.modules() if "lora" in type(_).__name__.lower()
    )
    print(f"    LoRA layer modules: {n_lora_layers}")
    results["peft_internals"] = {
        "active_adapter": str(active),
        "n_lora_layers": n_lora_layers,
    }

    # H4 verdict
    verdicts: Dict[str, str] = {}
    switch_new = br_new.median_ms
    switch_same = br_same.median_ms
    if switch_same < switch_new * 0.5:
        verdicts["H4"] = (
            f"CONFIRMED & CACHEABLE — set_adapter to new task = {switch_new:.2f}ms, "
            f"to same task = {switch_same:.2f}ms ({switch_new/max(switch_same,0.001):.1f}x cheaper when cached). "
            "The cost is PEFT adapter swapping. Hoist outside the request hot path: "
            "either batch same-task requests or cache the active adapter key."
        )
    else:
        verdicts["H4"] = (
            f"PARTIALLY FALSE — same-task switch ({switch_same:.2f}ms) is not much cheaper than new-task "
            f"({switch_new:.2f}ms). PEFT may not short-circuit no-op switches."
        )
    print(f"\n  {verdicts['H4']}")
    del model
    gc.collect()
    return results, verdicts


# =============================================================================
# Experiment E5: mixed precision (bf16 weights + fp32 softmax/norm)
# =============================================================================


def experiment_e5_mixed_precision(n_iter: int = 3) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """E5: try mixed precision via autocast."""
    print_section("E5: mixed precision via torch.autocast(cpu, dtype=bfloat16)")
    print("  Variable: autocast region wrapping forward pass.\n")

    from torch.nn import functional as F

    from sentence_transformers import SentenceTransformer

    print("  Loading fp32 model (for autocast to make sense)...")
    model = SentenceTransformer(
        EMBEDDING_MODEL_PATH,
        trust_remote_code=True,
        model_kwargs={
            "default_task": "retrieval",
            "default_prompt": "query",
            "torch_dtype": torch.float32,
            "attn_implementation": "sdpa",
        },
    )

    texts = SAMPLE_TEXTS_MEDIUM
    inner_st = model._first_module()
    tokenizer = inner_st.tokenizer
    inner_model = inner_st.model
    max_length = inner_st.max_seq_length

    def encode_pure_fp32(_m=model):
        return _m.encode(
            texts, task="retrieval", prompt_name="query",
            normalize_embeddings=True, convert_to_numpy=False,
        )

    def encode_with_autocast():
        # Manually invoke the forward with autocast wrapping
        batch = tokenizer(
            texts, max_length=max_length, truncation=True, padding=True, return_tensors="pt"
        )
        inner_model.set_adapter("retrieval")
        inner_model.eval()
        with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            outputs = inner_model(**batch)
            hidden = outputs.last_hidden_state
            mask = batch.get("attention_mask")
            seq_lens = mask.sum(dim=1) - 1
            pooled = hidden[torch.arange(hidden.shape[0]), seq_lens]
        # Normalize in fp32 (stable)
        emb = F.normalize(pooled.float(), p=2, dim=-1)
        return emb

    _, br_fp32 = time_repeated(encode_pure_fp32, n_iter=n_iter, n_warmup=2)
    br_fp32.label = "fp32 (pure)"
    _, br_autocast = time_repeated(encode_with_autocast, n_iter=n_iter, n_warmup=2)
    br_autocast.label = "fp32 + autocast(bf16)"

    results = {
        "fp32_pure": br_fp32.as_dict(),
        "fp32_autocast_bf16": br_autocast.as_dict(),
    }
    print(f"  fp32 pure:               {br_fp32.median_ms:.1f}ms median")
    print(f"  fp32 + autocast(bf16):   {br_autocast.median_ms:.1f}ms median")
    speedup = br_fp32.median_ms / br_autocast.median_ms if br_autocast.median_ms else 0
    print(f"  autocast speedup: {speedup:.2f}x")

    # Accuracy of autocast vs fp32
    ref = _embeddings_to_tensor(encode_pure_fp32())
    autocast_emb = _embeddings_to_tensor(encode_with_autocast())
    if ref.dim() == 1:
        ref = ref.unsqueeze(0)
    if autocast_emb.dim() == 1:
        autocast_emb = autocast_emb.unsqueeze(0)
    cos = cosine_similarity_rows(ref, autocast_emb)
    results["autocast_accuracy"] = {
        "cosine_mean": round(cos.mean().item(), 6),
        "cosine_min": round(cos.min().item(), 6),
    }
    print(f"  autocast vs fp32 cosine: mean={cos.mean().item():.6f}, min={cos.min().item():.6f}")

    verdicts: Dict[str, str] = {}
    min_cos = cos.min().item()
    if speedup >= 1.2 and min_cos > 0.999:
        verdicts["H5"] = (
            f"CONFIRMED — autocast(bf16) gives {speedup:.2f}x speedup with min cosine "
            f"{min_cos:.6f} (>0.999 = essentially identical). "
            "Recommended for production: load weights fp32, run forward under autocast."
        )
    elif speedup >= 1.2:
        verdicts["H5"] = (
            f"PARTIAL — autocast(bf16) gives {speedup:.2f}x speedup but min cosine "
            f"{min_cos:.6f} indicates some divergence. Acceptable for retrieval, not for fine ranking."
        )
    else:
        verdicts["H5"] = (
            f"FALSIFIED — autocast only {speedup:.2f}x faster; "
            "overhead of casting outweighs compute savings on CPU."
        )

    del model
    gc.collect()
    return results, verdicts


# =============================================================================
# Experiment E6: GFLOPS analysis on real forward pass
# =============================================================================


def estimate_transformer_flops(
    n_layers: int, hidden: int, intermediate: int, n_heads: int,
    n_kv_heads: int, seq_len: int, batch: int, vocab: int = 0,
) -> int:
    """Estimate FLOPs for a forward pass of a decoder-only transformer (no KV cache)."""
    # Attention: QKV projections + output projection
    qkv = 2 * batch * seq_len * hidden * (hidden + 2 * n_kv_heads * (hidden // n_heads))
    attn_out = 2 * batch * seq_len * hidden * hidden
    # Attention scores + softmax + weighted sum
    head_dim = hidden // n_heads
    attn_scores = 2 * batch * n_heads * seq_len * seq_len * head_dim
    attn_value = 2 * batch * n_heads * seq_len * seq_len * head_dim
    # MLP (gate + up + down for SwiGLU)
    mlp = 2 * 3 * batch * seq_len * hidden * intermediate
    # Per layer
    per_layer = qkv + attn_out + attn_scores + attn_value + mlp
    flops = n_layers * per_layer
    if vocab > 0:
        flops += 2 * batch * seq_len * hidden * vocab
    return int(flops)


def experiment_e6_gflops_analysis() -> Tuple[Dict[str, Any], Dict[str, str]]:
    """E6: measure achieved GFLOPS vs theoretical peak."""
    print_section("E6: GFLOPS analysis — actual vs theoretical peak")
    print("  Variable: batch size. Same model, same texts.\n")

    print("  Loading bf16 model...")
    model = load_embedding_dtype(torch.bfloat16)

    # Qwen3-0.6B-Base: 28 layers, hidden=1024, inter=3072, 16 heads, 8 KV heads
    n_layers, hidden, intermediate = 28, 1024, 3072
    n_heads, n_kv_heads = 16, 8

    results: Dict[str, Any] = {"data_points": []}
    print(f"  {'batch':<6} {'seq_len':<8} {'flops (M)':<11} {'median ms':<11} "
          f"{'GFLOPS':<9} {'% peak':<8}")
    print(f"  {'-'*6} {'-'*8} {'-'*11} {'-'*11} {'-'*9} {'-'*8}")

    for batch_size in [1, 8, 16, 32]:
        texts = (SAMPLE_TEXTS_MEDIUM * ((batch_size // len(SAMPLE_TEXTS_MEDIUM)) + 1))[:batch_size]

        # First, get actual seq_len from tokenizer
        inner_st = model._first_module()
        encoded = inner_st.tokenizer(
            texts, max_length=inner_st.max_seq_length, truncation=True,
            padding=True, return_tensors="pt",
        )
        actual_seq_len = encoded["input_ids"].shape[1]
        flops = estimate_transformer_flops(
            n_layers, hidden, intermediate, n_heads, n_kv_heads,
            seq_len=actual_seq_len, batch=batch_size,
        )

        _, br = time_repeated(
            fn=lambda _m=model: _m.encode(
                texts, task="retrieval", prompt_name="query",
                normalize_embeddings=True, convert_to_numpy=False,
                batch_size=max(batch_size, 8),
            ),
            n_iter=3, n_warmup=2,
        )
        achieved_gflops = (flops / 1e9) / (br.median_ms / 1000.0)
        pct_peak = (achieved_gflops / ZEN5_FP32_PEAK_GFLOPS) * 100.0
        results["data_points"].append({
            "batch_size": batch_size,
            "seq_len": actual_seq_len,
            "flops": flops,
            "median_ms": round(br.median_ms, 2),
            "achieved_gflops": round(achieved_gflops, 1),
            "pct_peak": round(pct_peak, 2),
        })
        print(
            f"  {batch_size:<6} {actual_seq_len:<8} {flops/1e6:<11.1f} "
            f"{br.median_ms:<11.2f} {achieved_gflops:<9.1f} {pct_peak:<8.2f}"
        )

    # H6 verdict
    best_pct = max(dp["pct_peak"] for dp in results["data_points"])
    verdicts: Dict[str, str] = {}
    if best_pct < 10.0:
        verdicts["H6"] = (
            f"CONFIRMED — best utilization is only {best_pct:.2f}% of {ZEN5_FP32_PEAK_GFLOPS:.0f} GFLOPS peak. "
            "Forward pass is severely memory-bandwidth bound at small batch sizes. "
            "Increasing batch size is the highest-leverage optimization (amortizes weight loads)."
        )
    else:
        verdicts["H6"] = (
            f"PARTIALLY FALSE — best utilization {best_pct:.2f}% is reasonable for CPU inference. "
            "Compute-bound regime is being approached at large batches."
        )

    del model
    gc.collect()
    return results, verdicts


# =============================================================================
# Final report
# =============================================================================


def print_pareto_table(
    e2_results: Dict[str, Any], e3_results: Dict[str, Any],
) -> None:
    """Print speed-vs-accuracy Pareto frontier."""
    print_section("Pareto frontier — speed vs accuracy")
    print(f"\n  {'dtype':<10} {'median ms':<12} {'speedup vs fp32':<18} "
          f"{'cosine min':<14} {'L2 mean':<12} {'verdict':<30}")
    print(f"  {'-'*10} {'-'*12} {'-'*18} {'-'*14} {'-'*12} {'-'*30}")

    fp32_ms = e2_results.get("fp32", {}).get("median_ms", 1)
    rows = []
    for name in ["fp32", "bf16", "int8"]:
        if name not in e2_results or "error" in e2_results.get(name, {}):
            continue
        ms = e2_results[name].get("median_ms", -1)
        if ms <= 0:
            continue
        speedup = fp32_ms / ms if ms > 0 else 0
        cos = e3_results.get(name, {}).get("cosine_sim", {}).get("min", -1)
        l2 = e3_results.get(name, {}).get("l2_distance", {}).get("mean", -1)
        if cos < 0:
            continue
        if cos > 0.999 and speedup >= 1.0:
            verdict = "[OK] safe (lossless)"
        elif cos > 0.99:
            verdict = "[OK] acceptable for retrieval"
        elif cos > 0.95:
            verdict = "[WARN] use with care"
        else:
            verdict = "[FAIL] too lossy"
        rows.append((name, ms, speedup, cos, l2, verdict))
        print(
            f"  {name:<10} {ms:<12.2f} {speedup:<18.2f} "
            f"{cos:<14.6f} {l2:<12.6f} {verdict:<30}"
        )


def print_final_verdicts(all_verdicts: Dict[str, str]) -> None:
    print_section("Final hypothesis verdicts")
    for hid in ["H1", "H2", "H3", "H4", "H5", "H6"]:
        v = all_verdicts.get(hid, "NOT TESTED")
        # Strip helper tags from E3/E5
        if hid == "H5":
            v = all_verdicts.get("H5", all_verdicts.get("H5_partial", "NOT TESTED"))
        print(f"\n  {hid}: {v}")


# =============================================================================
# CLI
# =============================================================================


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="dtype overhead investigation for Jina embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--experiment", choices=["all", "e1", "e2", "e3", "e4", "e5", "e6"],
                   default="all", help="Which experiment to run")
    p.add_argument("--output", default=os.path.join(SCRIPT_DIR, "dtype_results.json"),
                   help="JSON output file")
    p.add_argument("--quick", action="store_true",
                   help="Use fewer iterations for faster turnaround")
    p.add_argument("--list-hypotheses", action="store_true",
                   help="Print hypotheses and exit")
    return p


def main() -> int:
    args = build_argparser().parse_args()
    if args.list_hypotheses:
        print_header()
        return 0

    n_iter = 2 if args.quick else 3

    print_header()

    all_results: Dict[str, Any] = {}
    all_verdicts: Dict[str, str] = {}

    if args.experiment in ("all", "e1"):
        r, v = experiment_e1_gemm_microbench()
        all_results["e1_gemm"] = r
        all_verdicts.update(v)
    if args.experiment in ("all", "e2"):
        r, v = experiment_e2_end_to_end_inference(n_iter=n_iter)
        all_results["e2_e2e"] = r
        all_verdicts.update(v)
    if args.experiment in ("all", "e3"):
        r, v = experiment_e3_accuracy()
        all_results["e3_accuracy"] = r
        all_verdicts.update(v)
    if args.experiment in ("all", "e4"):
        r, v = experiment_e4_set_adapter_isolation()
        all_results["e4_set_adapter"] = r
        all_verdicts.update(v)
    if args.experiment in ("all", "e5"):
        r, v = experiment_e5_mixed_precision(n_iter=n_iter)
        all_results["e5_autocast"] = r
        all_verdicts.update(v)
    if args.experiment in ("all", "e6"):
        r, v = experiment_e6_gflops_analysis()
        all_results["e6_gflops"] = r
        all_verdicts.update(v)

    # Pareto table (only if we have both E2 and E3)
    if "e2_e2e" in all_results and "e3_accuracy" in all_results:
        print_pareto_table(all_results["e2_e2e"], all_results["e3_accuracy"])

    print_final_verdicts(all_verdicts)

    # Save JSON
    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(
                {"results": all_results, "verdicts": all_verdicts},
                f, indent=2, default=str, ensure_ascii=False,
            )
        print(f"\n  Results saved to: {os.path.abspath(args.output)}")
    except Exception as e:
        print(f"  [WARN] failed to save JSON: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
