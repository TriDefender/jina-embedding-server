"""Isolate the cause of the VRAM spike during batch embedding.

Compares peak/reserved VRAM across four conditions on the SAME workload:
  A) eager (no compile)
  B) eager + torch.cuda.empty_cache() after each batch
  C) torch.compile(mode="max-autotune", dynamic=True)
  D) compile + empty_cache after each batch

Run:  uv run benchmark/memory_probe.py
"""
from __future__ import annotations
import os, sys, time

MAX_THREADS = 12
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, str(MAX_THREADS))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from sentence_transformers import SentenceTransformer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "jinaai", "jina-embeddings-v5-text-small")
DEVICE = torch.device("cuda")


def mem():
    return {
        "alloc_gb": torch.cuda.memory_allocated(0) / 1024**3,
        "reserved_gb": torch.cuda.memory_reserved(0) / 1024**3,
        "peak_gb": torch.cuda.max_memory_allocated(0) / 1024**3,
    }


def fmt(m):
    return f"alloc={m['alloc_gb']:6.3f}GB  reserved={m['reserved_gb']:6.3f}GB  peak={m['peak_gb']:6.3f}GB"


def make_texts(n, tokens):
    words = ("algorithm quantum nebula photon catalyst lattice momentum harvest "
             "polymer cipher turbine marble cortex theorem horizon willow beacon "
             "velvet granite prairie vector cascade interval granite helium").split()
    raw = " ".join(words * (tokens // len(words) + 2))
    import transformers
    tok = transformers.AutoTokenizer.from_pretrained(PATH, trust_remote_code=True)
    ids = tok(raw, add_special_tokens=False)["input_ids"][:tokens]
    base = tok.decode(ids)
    return [base] * n


def run(model, batches, label, empty_between):
    torch.cuda.reset_peak_memory_stats(0)
    for bs, sl in batches:
        texts = make_texts(bs, sl)
        t0 = time.perf_counter()
        _ = model.encode(texts, task="retrieval", prompt_name="query",
                         normalize_embeddings=True, batch_size=32, convert_to_numpy=False)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        m = mem()
        print(f"  [{label}] bs={bs:<3} sl={sl:<5} {dt:7.0f}ms   {fmt(m)}")
        if empty_between:
            torch.cuda.empty_cache()
    print(f"  [{label}] FINAL after run: {fmt(mem())}")


def main():
    batches = [(8, 512), (32, 512), (64, 1024), (32, 512), (8, 128)]

    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB\n")

    # ---- A) eager ----
    print("=== A) eager (no compile) ===")
    m = SentenceTransformer(PATH, trust_remote_code=True, device=DEVICE,
                            default_prompt_name="query",
                            model_kwargs={"default_task": "retrieval", "dtype": torch.bfloat16,
                                          "attn_implementation": "sdpa"})
    _ = m.encode(["warmup text"], task="retrieval", prompt_name="query",
                 normalize_embeddings=True, convert_to_numpy=False)
    print(f"  after load+warmup: {fmt(mem())}")
    run(m, batches, "A-eager", empty_between=False)
    del m; torch.cuda.empty_cache(); torch.cuda.synchronize()

    # ---- B) eager + empty_cache between batches ----
    print("\n=== B) eager + empty_cache() between batches ===")
    m = SentenceTransformer(PATH, trust_remote_code=True, device=DEVICE,
                            default_prompt_name="query",
                            model_kwargs={"default_task": "retrieval", "dtype": torch.bfloat16,
                                          "attn_implementation": "sdpa"})
    _ = m.encode(["warmup text"], task="retrieval", prompt_name="query",
                 normalize_embeddings=True, convert_to_numpy=False)
    print(f"  after load+warmup: {fmt(mem())}")
    run(m, batches, "B-eager+empty", empty_between=True)
    del m; torch.cuda.empty_cache(); torch.cuda.synchronize()

    # ---- C) compile max-autotune ----
    print("\n=== C) torch.compile(mode='max-autotune', dynamic=True) ===")
    m = SentenceTransformer(PATH, trust_remote_code=True, device=DEVICE,
                            default_prompt_name="query",
                            model_kwargs={"default_task": "retrieval", "dtype": torch.bfloat16,
                                          "attn_implementation": "sdpa"})
    m = torch.compile(m, dynamic=True, mode="max-autotune")
    # warmup with the LARGEST shape so the graph is captured before measuring
    big = make_texts(64, 1024)
    print("  [C] capturing/warming compile with bs=64 sl=1024 ...")
    _ = m.encode(big, task="retrieval", prompt_name="query",
                normalize_embeddings=True, batch_size=32, convert_to_numpy=False)
    torch.cuda.synchronize()
    print(f"  after compile warmup: {fmt(mem())}")
    run(m, batches, "C-compile", empty_between=False)
    del m; torch.cuda.empty_cache(); torch.cuda.synchronize()

    # ---- D) compile + empty_cache ----
    print("\n=== D) compile + empty_cache() between batches ===")
    m = SentenceTransformer(PATH, trust_remote_code=True, device=DEVICE,
                            default_prompt_name="query",
                            model_kwargs={"default_task": "retrieval", "dtype": torch.bfloat16,
                                          "attn_implementation": "sdpa"})
    m = torch.compile(m, dynamic=True, mode="max-autotune")
    _ = m.encode(big, task="retrieval", prompt_name="query",
                normalize_embeddings=True, batch_size=32, convert_to_numpy=False)
    torch.cuda.synchronize()
    print(f"  after compile warmup: {fmt(mem())}")
    run(m, batches, "D-compile+empty", empty_between=True)
    del m; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
