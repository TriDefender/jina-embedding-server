#!/usr/bin/env python3
"""
Standalone PyTorch inference profiler for the Jina embedding & reranker models.

Mirrors jina_server.py lifespan loading EXACTLY (same kwargs, same dtype,
same attention impl, same Triton/CUDA-Graph options), then exercises the two
inference paths under three lenses:

  sweep    latency/throughput vs (batch_size, seq_len) for embeddings,
           vs (n_docs, doc_len, _block_size) for reranker.
  profile  torch.profiler op breakdown (top GPU kernels + top CPU ops),
           plus an automatic bottleneck classifier.
  trace    dumps a Chrome .json trace for manual chrome://tracing inspection.

No server needed — loads models directly. Run from repo root:

  uv run benchmark/profile_inference.py --model both --mode profile
  uv run benchmark/profile_inference.py --model embedding --mode sweep
  uv run benchmark/profile_inference.py --model rerank --mode trace --out trace.json

Environment flags (same as the server):
  COMPILE_ON_GPU=1   torch.compile() the model (GPU only; adds JIT time)
  CUDA_GRAPH=1       capture CUDA Graphs for reranker backbone (GPU only)
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import statistics
import sys
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple

# =============================================================================
# Thread tuning — MUST be set before torch import (mirrors jina_server.py:24-38)
# =============================================================================
MAX_THREADS = int(os.environ.get("PROFILE_MAX_THREADS", "12"))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, str(MAX_THREADS))
os.environ.setdefault("TORCH_INTEROP_THREADS", "8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.profiler import ProfilerActivity, profile, schedule

# Make repo root importable so `from optimized_kernels import ...` resolves
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# =============================================================================
# Configuration
# =============================================================================
MODELS_DIR = os.path.join(_REPO_ROOT, "jinaai")
EMBEDDING_MODEL_PATH = os.path.join(MODELS_DIR, "jina-embeddings-v5-text-small")
RERANKER_MODEL_PATH = os.path.join(MODELS_DIR, "jina-reranker-v3")

CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA_AVAILABLE else "cpu")

# Diverse word pool for synthetic workload generation (avoids tokenizer
# over-compression from repeated single tokens).
_WORD_POOL = (
    "algorithm quantum nebula velocity photon catalyst lattice isotope momentum "
    "harvest polymer glacier cipher sodium turbine marble cortex spindle "
    "orchid fraction beacon velvet granite prairie theorem horizon willow "
    "compass ember lattice cinder mariner delta fjord basil quartz ripple "
    "junction solstice fern asteroid cobalt meadow theorem lantern sapphire "
    "vector theorem phosphor willow cascade interval granite helium anchor "
    "matrix ripple cobalt reservoir theorem parish lantern granite theorem"
).split()


# =============================================================================
# Model loading — mirrors jina_server.py lifespan exactly
# =============================================================================
def _detect_attn() -> str:
    if not CUDA_AVAILABLE:
        return "eager"
    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        return "sdpa"
    return "eager"


def _print_env(tag: str) -> None:
    print(f"\n{'=' * 64}\n{tag}\n{'=' * 64}")
    print(f"  device:           {DEVICE}")
    if CUDA_AVAILABLE:
        print(f"  gpu:              {torch.cuda.get_device_name(0)}")
        cap = torch.cuda.get_device_capability()
        print(f"  compute cap:      {cap[0]}.{cap[1]}")
        print(f"  cuda:             {torch.version.cuda}")
        print(f"  vram total:       {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    print(f"  torch:            {torch.__version__}")
    print(f"  threads:          {torch.get_num_threads()} (interop {torch.get_num_interop_threads()})")
    print(f"  attn impl:        {_detect_attn()}")


def load_embedding_model(compile_model: bool = False):
    """Load jina-embeddings-v5-text-small exactly as lifespan does."""
    from sentence_transformers import SentenceTransformer

    attn = _detect_attn()
    model = SentenceTransformer(
        EMBEDDING_MODEL_PATH,
        trust_remote_code=True,
        device=DEVICE,
        default_prompt_name="query",
        model_kwargs={
            "default_task": "retrieval",
            "dtype": torch.bfloat16,
            "attn_implementation": attn,
        },
    )
    if compile_model and DEVICE.type == "cuda":
        print("  [compile] torch.compile(embedding, dynamic=True, max-autotune)...")
        model = torch.compile(model, dynamic=True, mode="max-autotune")
    return model, attn


def load_reranker_model(compile_model: bool = False, cudagraph: bool = False,
                        patch_triton: bool = True):
    """Load jina-reranker-v3 exactly as lifespan does (AutoModel, not ST)."""
    from transformers import AutoModel

    attn = _detect_attn()
    model = AutoModel.from_pretrained(
        RERANKER_MODEL_PATH,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation=attn,
    )
    model.eval()
    model.to(DEVICE)

    if CUDA_AVAILABLE and patch_triton:
        try:
            from optimized_kernels import patch_reranker_attention

            patch_reranker_attention(model)
        except Exception as e:  # noqa: BLE001
            print(f"  [WARN] triton patch failed (falling back to {attn}): {e}")

    if cudagraph and CUDA_AVAILABLE:
        # Reuse the exact CUDA Graph wrapper from the server module
        from jina_server import _enable_cudagraph_reranker  # type: ignore

        state = _enable_cudagraph_reranker(model)
        state.warmup()
    elif compile_model and DEVICE.type == "cuda":
        print("  [compile] torch.compile(reranker, dynamic=True, max-autotune)...")
        model = torch.compile(model, dynamic=True, mode="max-autotune")

    return model, attn


def warmup_embedding(model) -> None:
    """Lifespan warmup call (triggers JIT / graph capture)."""
    _ = model.encode(
        ["dummy_text_here_put_your_string_lol1234567890 for warmup purposes"],
        task="retrieval",
        prompt_name="query",
        convert_to_numpy=False,
        normalize_embeddings=True,
    )


def warmup_reranker(model) -> None:
    try:
        _ = model.rerank("warmup query", ["dummy_text_here_put_your_string_lol"], top_n=1)
    except Exception:  # noqa: BLE001
        pass


# =============================================================================
# Workload generation — token-controlled synthetic text
# =============================================================================
def make_text(target_tokens: int, tokenizer, seed: Optional[int] = None) -> str:
    """Build a string that tokenizes to ~target_tokens tokens."""
    rng = sorted(_WORD_POOL, key=lambda w: hash((w, seed)) if seed is not None else 0)
    raw = " ".join((rng * (target_tokens // len(rng) + 2))[: max(target_tokens, 1) * 2])
    enc = tokenizer(raw, add_special_tokens=False)
    ids = enc["input_ids"][:target_tokens]
    if len(ids) < target_tokens:  # pool too small; loop-expand
        ids = (ids * (target_tokens // len(ids) + 1))[:target_tokens]
    return tokenizer.decode(ids)


def make_texts(n: int, target_tokens: int, tokenizer) -> List[str]:
    return [make_text(target_tokens, tokenizer, seed=i) for i in range(n)]


# =============================================================================
# Timing harness — cuda-synced, warmup + timed iterations
# =============================================================================
@contextmanager
def cuda_sync():
    if CUDA_AVAILABLE:
        torch.cuda.synchronize()
        try:
            yield
        finally:
            torch.cuda.synchronize()
    else:
        yield


def time_call(fn: Callable[[], Any], warmup: int, iters: int) -> Dict[str, float]:
    """Time a callable; returns latency stats in ms."""
    for _ in range(warmup):
        fn()
    if CUDA_AVAILABLE:
        torch.cuda.synchronize()

    samples: List[float] = []
    with cuda_sync():
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            if CUDA_AVAILABLE:
                torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1000.0)  # → ms

    samples.sort()
    half = len(samples) // 2
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "p95_ms": samples[max(0, int(len(samples) * 0.95) - 1)],
        "min_ms": samples[0],
        "max_ms": samples[-1],
        "n_iter": len(samples),
    }


def gpu_mem() -> Dict[str, float]:
    if not CUDA_AVAILABLE:
        return {}
    return {
        "allocated_gb": round(torch.cuda.memory_allocated(0) / (1024**3), 3),
        "reserved_gb": round(torch.cuda.memory_reserved(0) / (1024**3), 3),
        "max_alloc_gb": round(torch.cuda.max_memory_allocated(0) / (1024**3), 3),
    }


# =============================================================================
# Sweep mode
# =============================================================================
def sweep_embeddings(model, tokenizer, out: Dict[str, Any]) -> None:
    """Latency/throughput vs (batch_size, seq_len)."""
    batch_sizes = [1, 8, 32, 64]
    seq_lens = [32, 128, 512, 1024]
    rows: List[Dict[str, Any]] = []

    print(f"\n{'='*78}\n[SWEEP] embeddings — task=retrieval/query, normalize=True\n{'='*78}")
    header = f"{'batch':>6} {'seq_len':>8} {'median':>9} {'p95':>9} {'texts/s':>9} {'tok/s':>10}"
    print(header)
    print("-" * len(header))

    for bs in batch_sizes:
        for sl in seq_lens:
            texts = make_texts(bs, sl, tokenizer)
            # count real tokens for throughput reporting
            n_tok = len(tokenizer(texts[0], add_special_tokens=True)["input_ids"])

            def _run(t=texts):
                return model.encode(
                    t,
                    task="retrieval",
                    prompt_name="query",
                    normalize_embeddings=True,
                    batch_size=min(32, len(t)),
                    convert_to_numpy=False,
                )

            stats = time_call(_run, warmup=2, iters=8)
            stats["batch_size"] = bs
            stats["seq_len"] = sl
            stats["texts_per_s"] = round(bs / (stats["median_ms"] / 1000.0), 1)
            stats["tokens_per_s"] = round(bs * n_tok / (stats["median_ms"] / 1000.0), 0)
            rows.append(stats)
            print(
                f"{bs:>6} {sl:>8} {stats['median_ms']:>8.2f}m "
                f"{stats['p95_ms']:>8.2f}m {stats['texts_per_s']:>9.1f} "
                f"{stats['tokens_per_s']:>10.0f}"
            )
            if CUDA_AVAILABLE:
                torch.cuda.reset_peak_memory_stats(0)

    out["embeddings_sweep"] = rows


def sweep_reranker(model, tokenizer, out: Dict[str, Any]) -> None:
    """Latency vs (n_docs, doc_len, _block_size)."""
    n_docs_grid = [4, 16, 64]
    doc_lens = [64, 256]
    block_sizes = [125, 32]  # default vs small
    rows: List[Dict[str, Any]] = []
    query = "What is the primary bottleneck of attention mechanisms?"

    print(f"\n{'='*78}\n[SWEEP] reranker — query fixed short\n{'='*78}")
    header = (
        f"{'n_docs':>6} {'doc_len':>8} {'block':>6} {'median':>9} "
        f"{'p95':>9} {'docs/s':>8}"
    )
    print(header)
    print("-" * len(header))

    default_block = getattr(model, "_block_size", 125)
    for n_docs in n_docs_grid:
        for dl in doc_lens:
            docs = make_texts(n_docs, dl, tokenizer)
            for blk in block_sizes:
                model._block_size = blk

                def _run(q=query, d=docs):
                    return model.rerank(query=q, documents=d, top_n=min(10, len(d)))

                stats = time_call(_run, warmup=1, iters=4)
                stats["n_docs"] = n_docs
                stats["doc_len"] = dl
                stats["block_size"] = blk
                stats["docs_per_s"] = round(n_docs / (stats["median_ms"] / 1000.0), 1)
                rows.append(stats)
                print(
                    f"{n_docs:>6} {dl:>8} {blk:>6} {stats['median_ms']:>8.2f}m "
                    f"{stats['p95_ms']:>8.2f}m {stats['docs_per_s']:>8.1f}"
                )
                if CUDA_AVAILABLE:
                    torch.cuda.reset_peak_memory_stats(0)
    model._block_size = default_block  # restore

    out["reranker_sweep"] = rows


# =============================================================================
# Profile mode — torch.profiler breakdown + bottleneck classifier
# =============================================================================
# Op-name → category mapping for the bottleneck verdict.
_CATEGORY_RULES: List[Tuple[str, str]] = [
    ("matmul", "GEMM"),
    ("_gemm", "GEMM"),
    ("addmm", "GEMM"),
    ("aten::mm", "GEMM"),
    ("aten::bmm", "GEMM"),
    ("aten::linear", "GEMM"),
    ("ampere_", "GEMM"),
    ("sm90_", "GEMM"),
    ("scaled_dot_product", "attention"),
    ("_efficient_attention", "attention"),
    ("flash", "attention"),
    ("fmha", "attention"),
    ("attention", "attention"),
    ("aten::embedding", "embedding-lookup"),
    ("layer_norm", "norm"),
    ("rms_norm", "norm"),
    ("rmsnorm", "norm"),
    ("aten::mul", "elementwise"),
    ("aten::add", "elementwise"),
    ("aten::sub", "elementwise"),
    ("aten::div", "elementwise"),
    ("aten::pow", "elementwise"),
    ("aten::rsqrt", "elementwise"),
    ("aten::sqrt", "elementwise"),
    ("aten::neg", "elementwise"),
    ("aten::exp", "elementwise"),
    ("aten::erf", "elementwise"),
    ("aten::tanh", "elementwise"),
    ("elementwise_kernel", "elementwise"),
    ("gelu", "activation"),
    ("silu", "activation"),
    ("relu", "activation"),
    ("aten::sigmoid", "activation"),
    ("softmax", "softmax"),
    ("copy_", "data-movement"),
    ("to_copy", "data-movement"),
    ("clone", "data-movement"),
    ("contiguous", "data-movement"),
    ("aten::_cat", "data-movement"),
    ("aten::cat", "data-movement"),
    ("aten::transpose", "data-movement"),
    ("aten::view", "data-movement"),
    ("aten::reshape", "data-movement"),
    ("aten::sum", "reduction"),
    ("aten::mean", "reduction"),
    ("reduction_kernel", "reduction"),
    ("cudaDeviceSynchronize", "host-sync"),
    ("cudaStreamSynchronize", "host-sync"),
    ("cudaLaunchKernel", "host-launch"),
    ("Command Buffer", "host-cmdbuf"),
]


def _categorize(op_name: str) -> str:
    low = op_name.lower()
    # ProfilerStep* is the per-iteration aggregate marker, not a kernel: its
    # self-time is the unattributed gap between ops (Python dispatch overhead).
    if op_name.startswith("ProfilerStep") or op_name == "profilerstep*":
        return "dispatch-overhead"
    for needle, cat in _CATEGORY_RULES:
        if needle in low:
            return cat
    return "other"


def _profile_workload(label: str, fn: Callable[[], Any], warmup: int,
                      active: int, out_path: Optional[str]) -> Dict[str, Any]:
    """Run torch.profiler over fn(); return top kernels + category breakdown."""
    activities = [ProfilerActivity.CPU]
    if CUDA_AVAILABLE:
        activities.append(ProfilerActivity.CUDA)

    prof = profile(
        activities=activities,
        schedule=schedule(wait=1, warmup=warmup, active=active, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    )

    print(f"\n[PROFILE] {label}: wait=1, warmup={warmup}, active={active} iters")
    prof.start()
    steps = 1 + warmup + active
    for _ in range(steps):
        fn()
        if CUDA_AVAILABLE:
            torch.cuda.synchronize()
        prof.step()
    prof.stop()

    ka = prof.key_averages()
    # Aggregate by op key + category
    by_cat: Dict[str, Dict[str, float]] = {}
    op_rows = []
    for evt in ka:
        cat = _categorize(evt.key)
        c = by_cat.setdefault(cat, {"cpu_us": 0.0, "cuda_us": 0.0, "calls": 0})
        c["cpu_us"] += evt.self_cpu_time_total
        c["cuda_us"] += evt.self_device_time_total  # cuda self time
        c["calls"] += evt.count
        op_rows.append(
            {
                "op": evt.key,
                "category": cat,
                "cpu_us": round(evt.self_cpu_time_total, 1),
                "cuda_us": round(evt.self_device_time_total, 1),
                "calls": evt.count,
            }
        )

    total_cuda = sum(c["cuda_us"] for c in by_cat.values()) or 1.0
    total_cpu = sum(c["cpu_us"] for c in by_cat.values()) or 1.0

    cat_summary = sorted(
        (
            {
                "category": cat,
                "cuda_ms": round(v["cuda_us"] / 1000.0, 3),
                "cuda_pct": round(100.0 * v["cuda_us"] / total_cuda, 2),
                "cpu_ms": round(v["cpu_us"] / 1000.0, 3),
                "calls": v["calls"],
            }
            for cat, v in by_cat.items()
        ),
        key=lambda r: r["cuda_ms"],
        reverse=True,
    )

    # Top GPU ops
    top_gpu = sorted(op_rows, key=lambda r: r["cuda_us"], reverse=True)[:12]
    top_cpu = sorted(op_rows, key=lambda r: r["cpu_us"], reverse=True)[:12]

    verdict = _classify(label, cat_summary, total_cpu, total_cuda)

    # ---- print ----
    print(f"\n--- {label}: category breakdown (by GPU self-time) ---")
    print(f"{'category':>18} {'cuda_ms':>10} {'cuda_%':>8} {'cpu_ms':>10} {'calls':>10}")
    for r in cat_summary:
        print(
            f"{r['category']:>18} {r['cuda_ms']:>10.3f} {r['cuda_pct']:>7.2f}% "
            f"{r['cpu_ms']:>10.3f} {r['calls']:>10}"
        )
    print(f"\n  total GPU self-time: {total_cuda/1000:.2f} ms   "
          f"total CPU self-time: {total_cpu/1000:.2f} ms   "
          f"overlap ratio (GPU/CPU): {total_cuda/total_cpu:.2f}x")

    print(f"\n--- {label}: top 12 GPU ops (self-time) ---")
    print(f"{'op':>44} {'cuda_us':>10} {'calls':>8}")
    for r in top_gpu:
        print(f"{r['op'][:44]:>44} {r['cuda_us']:>10.1f} {r['calls']:>8}")
    print(f"\n--- {label}: top 12 CPU ops (self-time) ---")
    print(f"{'op':>44} {'cpu_us':>10} {'calls':>8}")
    for r in top_cpu:
        print(f"{r['op'][:44]:>44} {r['cpu_us']:>10.1f} {r['calls']:>8}")

    print(f"\n>>> VERDICT ({label}): {verdict}")

    if out_path:
        prof.export_chrome_trace(out_path)
        print(f"  [trace] chrome trace → {out_path}")

    return {
        "category_breakdown": cat_summary,
        "top_gpu_ops": top_gpu,
        "top_cpu_ops": top_cpu,
        "total_cuda_ms": round(total_cuda / 1000.0, 3),
        "total_cpu_ms": round(total_cpu / 1000.0, 3),
        "overlap_ratio": round(total_cuda / total_cpu, 2),
        "verdict": verdict,
    }


def _classify(label: str, cat_summary: List[Dict], total_cpu_us: float,
              total_cuda_us: float) -> str:
    """Heuristic bottleneck verdict from the profile."""
    by_cat = {r["category"]: r for r in cat_summary}
    overlap = total_cuda_us / total_cpu_us if total_cpu_us else 0.0

    host_overhead_cats = {
        "dispatch-overhead", "host-sync", "host-launch", "host-cmdbuf", "other",
    }
    host_ms = sum(
        by_cat[c]["cuda_ms"] for c in host_overhead_cats if c in by_cat
    )
    host_pct = 100.0 * (host_ms * 1000.0) / total_cuda_us if total_cuda_us else 0.0

    dom = max(cat_summary, key=lambda r: r["cuda_ms"])
    dom_pct = dom["cuda_pct"]

    if overlap < 0.5 and total_cpu_us > 0:
        cpu_heavy = sorted(cat_summary, key=lambda r: r["cpu_ms"], reverse=True)[0]
        return (
            f"CPU-BOUND (GPU starved): overlap {overlap:.2f}x — host spends "
            f"{total_cpu_us/1000:.1f}ms but GPU only {total_cuda_us/1000:.1f}ms. "
            f"Dominant CPU cost: '{cpu_heavy['category']}' "
            f"({cpu_heavy['cpu_ms']:.1f}ms). Likely tokenization / Python / "
            f"data-prep serializing inference."
        )

    # Host+gap overhead as a share of the GPU timeline: high ⇒ many tiny
    # kernels starve the GPU (kernel-launch/dispatch-bound regime).
    real_compute_cats = {"GEMM", "attention", "embedding-lookup"}
    plus_elem = host_pct + sum(
        by_cat[c]["cuda_pct"] for c in ("elementwise", "data-movement") if c in by_cat
    )
    compute_pct = sum(by_cat[c]["cuda_pct"] for c in real_compute_cats if c in by_cat)

    is_dispatch_top = dom["category"] == "dispatch-overhead" and dom_pct > 25
    if host_pct > 40 or is_dispatch_top or (plus_elem > 50 and compute_pct < 35):
        sub = []
        for c in ("dispatch-overhead", "host-cmdbuf", "host-sync", "host-launch"):
            if c in by_cat and by_cat[c]["cuda_ms"] > 0.01:
                sub.append(f"{c}={by_cat[c]['cuda_ms']:.1f}ms")
        detail = ", ".join(sub) if sub else f"~{host_ms:.1f}ms"
        elem_note = ""
        if "elementwise" in by_cat and by_cat["elementwise"]["cuda_pct"] > 15:
            elem_note = (
                f" Unfused pointwise ({by_cat['elementwise']['cuda_pct']:.0f}%) "
                f"adds bandwidth waste."
            )
        return (
            f"KERNEL-LAUNCH/DISPATCH-BOUND: overhead+unfused = {plus_elem:.0f}% "
            f"vs real compute (GEMM+attn) {compute_pct:.0f}% — GPU is "
            f"underutilized ({detail}).{elem_note} Levers: torch.compile "
            f"(fusion + reduce kernel count), CUDA Graphs (reranker), larger "
            f"batch. Windows WDDM 'Command Buffer Full' flushes add sync — "
            f"consider Linux/WSL2 TCC mode."
        )

    if dom["category"] == "attention" and dom_pct > 30:
        return (
            f"ATTENTION-BOUND: attention = {dom['cuda_ms']:.1f}ms "
            f"({dom_pct:.1f}%). Long-context attention dominates; levers: "
            f"flash/triton kernels, shorter seqs, GQA, fused kernels."
        )
    if dom["category"] == "GEMM" and dom_pct > 35:
        return (
            f"COMPUTE-BOUND (GEMM): linear layers = {dom['cuda_ms']:.1f}ms "
            f"({dom_pct:.1f}%). Levers: larger batch (occupancy), fp8/int8 "
            f"quant, TF32 already on."
        )
    if dom["category"] == "elementwise" and dom_pct > 25:
        return (
            f"MEMORY-BANDWIDTH-BOUND (elementwise): pointwise ops = "
            f"{dom['cuda_ms']:.1f}ms ({dom_pct:.1f}%). Many small mul/add/pow "
            f"kernels (norm/softmax/residual pieces) waste bandwidth; levers: "
            f"torch.compile to fuse pointwise chains, fused norm+act kernels."
        )
    if dom["category"] == "embedding-lookup" and dom_pct > 25:
        return (
            f"EMBEDDING-LOOKUP-BOUND: {dom['cuda_ms']:.1f}ms ({dom_pct:.1f}%). "
            f"Levers: larger batch, reduce vocab, cache."
        )
    if dom["category"] == "data-movement" and dom_pct > 25:
        return (
            f"MEMORY-BANDWIDTH-BOUND (data movement): copy/clone/contiguous = "
            f"{dom['cuda_ms']:.1f}ms ({dom_pct:.1f}%). Levers: drop unnecessary "
            f".contiguous()/.clone(), fuse kernels, keep tensors on-device."
        )
    if dom["category"] == "norm" and dom_pct > 20:
        return (
            f"NORM-BOUND: {dom['cuda_ms']:.1f}ms ({dom_pct:.1f}%). Levers: "
            f"fused norm+act kernels."
        )

    return (
        f"MIXED: dominant '{dom['category']}' = {dom['cuda_ms']:.1f}ms "
        f"({dom_pct:.1f}%); host overhead {host_pct:.0f}%. No single category "
        f"dominates — inspect top_gpu_ops for the long tail."
    )


# =============================================================================
# Main
# =============================================================================
def _get_embedding_tokenizer(model):
    # sentence-transformers exposes the HF tokenizer
    try:
        return model.tokenizer
    except AttributeError:
        return model[0].tokenizer


def _get_reranker_tokenizer(model):
    for attr in ("_tokenizer", "tokenizer"):
        t = getattr(model, attr, None)
        if t is not None:
            return t
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(RERANKER_MODEL_PATH)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["embedding", "reranker", "both"], default="both")
    ap.add_argument("--mode", choices=["sweep", "profile", "trace", "all"], default="profile")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile() the model (GPU). Sets COMPILE_ON_GPU=1 semantics.")
    ap.add_argument("--cudagraph", action="store_true",
                    help="Capture CUDA Graphs for reranker backbone (GPU).")
    ap.add_argument("--no-triton", action="store_true",
                    help="Skip the Triton attention patch on the reranker.")
    ap.add_argument("--out", default=os.path.join(_REPO_ROOT, "benchmark", "profile_results.json"),
                    help="Output JSON path (sweep/profile results).")
    ap.add_argument("--trace", default=None,
                    help="Chrome trace output path (trace mode).")
    ap.add_argument("--warmup-iters", type=int, default=2)
    ap.add_argument("--active-iters", type=int, default=5)
    args = ap.parse_args()

    _print_env(f"PyTorch inference profiler — model={args.model} mode={args.mode}")
    results: Dict[str, Any] = {
        "env": {
            "device": str(DEVICE),
            "torch": torch.__version__,
            "cuda_available": CUDA_AVAILABLE,
            "gpu": torch.cuda.get_device_name(0) if CUDA_AVAILABLE else None,
            "compile": args.compile,
            "cudagraph": args.cudagraph,
            "triton_patched": not args.no_triton,
        },
    }

    do_emb = args.model in ("embedding", "both")
    do_rer = args.model in ("reranker", "both")

    # ---------- EMBEDDING ----------
    if do_emb:
        print("\n[1] loading embedding model...")
        t0 = time.perf_counter()
        emb_model, _ = load_embedding_model(compile_model=args.compile)
        print(f"    loaded in {time.perf_counter()-t0:.1f}s — warming up")
        warmup_embedding(emb_model)
        emb_tok = _get_embedding_tokenizer(emb_model)
        results["env"]["embedding_dim"] = (
            emb_model.get_sentence_embedding_dimension()
        )
        results["env"]["after_embedding_load"] = gpu_mem()

        if args.mode in ("sweep", "all"):
            sweep_embeddings(emb_model, emb_tok, results)
        if args.mode in ("profile", "all"):
            texts = make_texts(32, 512, emb_tok)  # representative batch

            def _emb_work(t=texts):
                return emb_model.encode(
                    t, task="retrieval", prompt_name="query",
                    normalize_embeddings=True, batch_size=32, convert_to_numpy=False,
                )

            results["embedding_profile"] = _profile_workload(
                "embedding (batch=32, seq~512, task=retrieval/query)",
                _emb_work, warmup=args.warmup_iters, active=args.active_iters,
                out_path=args.trace if (args.mode == "trace" and do_emb and not do_rer) else None,
            )
        if args.mode == "trace" and do_rer is False:
            pass  # handled above
        # free embedding before loading reranker (VRAM)
        if do_rer:
            del emb_model
            gc.collect()
            if CUDA_AVAILABLE:
                torch.cuda.empty_cache()

    # ---------- RERANKER ----------
    if do_rer:
        print("\n[2] loading reranker model...")
        t0 = time.perf_counter()
        rer_model, _ = load_reranker_model(
            compile_model=args.compile, cudagraph=args.cudagraph,
            patch_triton=not args.no_triton,
        )
        print(f"    loaded in {time.perf_counter()-t0:.1f}s — warming up")
        warmup_reranker(rer_model)
        rer_tok = _get_reranker_tokenizer(rer_model)
        results["env"]["after_reranker_load"] = gpu_mem()

        if args.mode in ("sweep", "all"):
            sweep_reranker(rer_model, rer_tok, results)
        if args.mode in ("profile", "all"):
            docs = make_texts(16, 256, rer_tok)

            def _rer_work(d=docs):
                return rer_model.rerank(
                    query="What is the primary bottleneck of transformer inference?",
                    documents=d, top_n=10,
                )

            results["reranker_profile"] = _profile_workload(
                "reranker (query + 16 docs @ ~256 tok, block_size=125)",
                _rer_work, warmup=args.warmup_iters, active=args.active_iters,
                out_path=args.trace if (args.mode == "trace" and do_rer) else None,
            )

    # ---------- save ----------
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] results → {args.out}")
    print(f"[MEM] peak gpu: {gpu_mem()}")

    # ---------- summary ----------
    print(f"\n{'='*64}\nSUMMARY\n{'='*64}")
    for prof_key in ("embedding_profile", "reranker_profile"):
        if prof_key in results:
            r = results[prof_key]
            print(f"\n{prof_key}:")
            print(f"  GPU self-time: {r['total_cuda_ms']} ms | "
                  f"CPU self-time: {r['total_cpu_ms']} ms | "
                  f"overlap: {r['overlap_ratio']}x")
            for cat in r["category_breakdown"][:3]:
                print(f"    {cat['category']:>16}: {cat['cuda_ms']} ms ({cat['cuda_pct']}%)")
            print(f"  VERDICT: {r['verdict']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
