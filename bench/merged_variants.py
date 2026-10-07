"""Post-merge optimization probes: compile ceiling, threads, flush_denormal, attention impl."""
from __future__ import annotations

import copy
import os
import statistics
import sys
import time

MAX_THREADS = max(1, int(os.environ.get("JINA_TORCH_THREADS", "12")))
os.environ["OMP_NUM_THREADS"] = str(MAX_THREADS)
os.environ["MKL_NUM_THREADS"] = str(MAX_THREADS)

import torch

torch.set_num_threads(MAX_THREADS)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from microbench_variants import MEDIUM_DOCS, load_embedding, get_inner, t_ms  # noqa: E402


def main():
    st = load_embedding()
    transformer, peft, tokenizer = get_inner(st)
    merged = copy.deepcopy(peft).merge_and_unload(adapter_names=["retrieval"])
    del peft
    gc = __import__("gc")
    gc.collect()

    b1 = tokenizer(["Query: What is machine learning?"], return_tensors="pt")
    b10 = tokenizer([f"Query: {t}" for t in MEDIUM_DOCS], return_tensors="pt",
                    padding=True, truncation=True, max_length=128)
    b32 = tokenizer([f"Query: {t}" for t in (MEDIUM_DOCS * 4)[:32]], return_tensors="pt",
                    padding=True, truncation=True, max_length=128)

    def fwd(b):
        with torch.no_grad():
            return merged(**b)

    # warm
    for b in (b1, b10, b32):
        fwd(b)

    print("== merged eager baseline ==")
    for name, b in (("b1", b1), ("b10", b10), ("b32", b32)):
        med, _ = t_ms(lambda b=b: fwd(b))
        print(f"  {name}: {med:.2f} ms")

    print("\n== flush_denormal ==")
    torch.set_flush_denormal(True)
    for name, b in (("b1", b1), ("b10", b10), ("b32", b32)):
        med, _ = t_ms(lambda b=b: fwd(b))
        print(f"  {name}: {med:.2f} ms")
    torch.set_flush_denormal(False)

    print("\n== threads on merged (1/10/32 texts) ==")
    for nt in [4, 6, 8, 10, 12]:
        torch.set_num_threads(nt)
        fwd(b1); fwd(b32)
        m1, _ = t_ms(lambda: fwd(b1), n_iter=3)
        m10, _ = t_ms(lambda: fwd(b10), n_iter=3)
        m32, _ = t_ms(lambda: fwd(b32), n_iter=3)
        print(f"  threads={nt:>2}: b1={m1:7.2f}  b10={m10:7.2f}  b32={m32:7.2f} ms")
    torch.set_num_threads(MAX_THREADS)

    print("\n== torch.compile ceiling (fixed shape b10) ==")
    t0 = time.perf_counter()
    compiled = torch.compile(merged, dynamic=False)
    fwd_c = lambda b: compiled(**b)
    with torch.no_grad():
        compiled(**b10)
    print(f"  compile time 1 shape: {time.perf_counter() - t0:.1f} s")
    med_c10, _ = t_ms(lambda: fwd_c(b10), n_iter=3, n_warmup=1)
    m10_eager, _ = t_ms(lambda: fwd(b10), n_iter=3, n_warmup=0)
    print(f"  b10 compiled: {med_c10:.2f} ms vs eager {m10_eager:.2f} ms ({m10_eager / med_c10:.3f}x)")

    # new shape with compiled dynamic=False → recompile penalty
    t0 = time.perf_counter()
    with torch.no_grad():
        compiled(**b1)
    print(f"  recompile for new shape b1: {time.perf_counter() - t0:.1f} s")

    print("\n== torch.compile dynamic=True (b10 then b1 then b32) ==")
    del compiled
    gc.collect()
    t0 = time.perf_counter()
    compiled_dyn = torch.compile(merged, dynamic=True)
    with torch.no_grad():
        compiled_dyn(**b10)
    print(f"  first compile: {time.perf_counter() - t0:.1f} s")
    for name, b in (("b1", b1), ("b32", b32)):
        t0 = time.perf_counter()
        with torch.no_grad():
            compiled_dyn(**b)
        print(f"  {name} first: {time.perf_counter() - t0:.1f} s")
    med_d10, _ = t_ms(lambda: compiled_dyn(**b10), n_iter=3, n_warmup=1)
    med_d1, _ = t_ms(lambda: compiled_dyn(**b1), n_iter=3, n_warmup=1)
    med_d32, _ = t_ms(lambda: compiled_dyn(**b32), n_iter=3, n_warmup=1)
    print(f"  dynamic compiled: b1={med_d1:.2f} b10={med_d10:.2f} b32={med_d32:.2f} ms")


if __name__ == "__main__":
    main()
