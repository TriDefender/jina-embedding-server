"""torch.compile ceiling test on the real merged embedding model."""
from __future__ import annotations

import copy
import os
import statistics
import sys
import time

# --- compiler env (must precede torch import) ---
os.environ["PYTHONUTF8"] = "0"
os.environ["VSLANG"] = "1033"
os.environ["TORCHINDUCTOR_VEC_ISA_OK"] = "1"
_VC = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.41.34120"
_SDK = r"C:\Program Files (x86)\Windows Kits\10"
_V = "10.0.22621.0"
os.environ["CXX"] = _VC + r"\bin\Hostx64\x64\cl.exe"
os.environ.setdefault(
    "INCLUDE",
    ";".join([
        _VC.replace("\\", "/") + "/include",
        _SDK.replace("\\", "/") + f"/Include/{_V}/ucrt",
        _SDK.replace("\\", "/") + f"/Include/{_V}/um",
        _SDK.replace("\\", "/") + f"/Include/{_V}/shared",
        _SDK.replace("\\", "/") + f"/Include/{_V}/winrt",
    ]),
)
os.environ.setdefault(
    "LIB",
    ";".join([
        _VC.replace("\\", "/") + "/lib/x64",
        _SDK.replace("\\", "/") + f"/Lib/{_V}/ucrt/x64",
        _SDK.replace("\\", "/") + f"/Lib/{_V}/um/x64",
    ]),
)

MAX_THREADS = 12
os.environ["OMP_NUM_THREADS"] = str(MAX_THREADS)
os.environ["MKL_NUM_THREADS"] = str(MAX_THREADS)

import torch  # noqa: E402

torch.set_num_threads(MAX_THREADS)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from microbench_variants import MEDIUM_DOCS, load_embedding, get_inner, t_ms  # noqa: E402


def main():
    st = load_embedding()
    transformer, peft, tokenizer = get_inner(st)
    merged = copy.deepcopy(peft).merge_and_unload(adapter_names=["retrieval"])
    del peft
    import gc
    gc.collect()

    b1 = tokenizer(["Query: What is machine learning?"], return_tensors="pt")
    b10 = tokenizer([f"Query: {t}" for t in MEDIUM_DOCS], return_tensors="pt",
                    padding=True, truncation=True, max_length=128)
    b32 = tokenizer([f"Query: {t}" for t in (MEDIUM_DOCS * 4)[:32]], return_tensors="pt",
                    padding=True, truncation=True, max_length=128)

    def fwd(m, b):
        with torch.no_grad():
            return m(**b)

    for b in (b1, b10, b32):
        fwd(merged, b)

    print("== merged eager baseline ==")
    base = {}
    for name, b in (("b1", b1), ("b10", b10), ("b32", b32)):
        med, _ = t_ms(lambda b=b: fwd(merged, b))
        base[name] = med
        print(f"  {name}: {med:.2f} ms")

    for mode in ("default", "max-autotune"):
        print(f"\n== torch.compile mode={mode}, dynamic=True ==")
        t0 = time.perf_counter()
        compiled = torch.compile(merged, dynamic=True, mode=mode)
        fwd(compiled, b10)
        print(f"  first compile (b10): {time.perf_counter() - t0:.1f} s")
        for name, b in (("b1", b1), ("b32", b32)):
            t0 = time.perf_counter()
            fwd(compiled, b)
            print(f"  first call {name}: {time.perf_counter() - t0:.1f} s (guard specialize)")
        for name, b in (("b1", b1), ("b10", b10), ("b32", b32)):
            med, _ = t_ms(lambda b=b: fwd(compiled, b), n_iter=3, n_warmup=1)
            print(f"  {name}: {med:.2f} ms  ({base[name] / med:.3f}x vs eager)")
        del compiled
        gc.collect()

    # correctness spot check
    compiled = torch.compile(merged, dynamic=True)
    fwd(compiled, b10)
    import torch.nn.functional as F
    e1 = fwd(merged, b10).last_hidden_state
    e2 = fwd(compiled, b10).last_hidden_state
    print("\n  cosine(eager, compiled) last_hidden:",
          F.cosine_similarity(e1.flatten().float(), e2.flatten().float(), dim=0).item())


if __name__ == "__main__":
    main()
