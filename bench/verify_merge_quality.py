"""Verify numerical quality of LoRA merge variants vs unmerged PEFT.

Compares final normalized embeddings (cosine similarity) for:
  - PEFT unmerged (ground truth)
  - merge in native bf16 (peft default path)
  - merge computed in fp32, cast back to bf16
"""
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
import torch.nn.functional as F

torch.set_num_threads(MAX_THREADS)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB_PATH = os.path.join(PROJECT_ROOT, "jinaai", "jina-embeddings-v5-text-small")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from microbench_variants import MEDIUM_DOCS, load_embedding, get_inner  # noqa: E402

EXTRA_TEXTS = [
    "The stock market rallied after the Federal Reserve signaled rate cuts.",
    "Photosynthesis converts light energy into chemical energy in plants.",
    "Python's GIL prevents true parallelism for CPU-bound threads.",
    "The mitochondria is the powerhouse of the cell.",
    "Quantum entanglement correlates particle states across distances.",
    "A monad is a monoid in the category of endofunctors.",
    "gravity bends spacetime around massive objects like stars",
    "the chef reduced the sauce until it coated the back of a spoon",
]
TEXTS = MEDIUM_DOCS + EXTRA_TEXTS

TASKS = ["retrieval", "text-matching", "clustering", "classification"]
PREFIXES = {
    "retrieval": ("Query: ", "Document: "),
    "text-matching": ("", ""),
    "clustering": ("", ""),
    "classification": ("", ""),
}


def encode_model(model, texts, task, prefix):
    tok = MODEL_TOKENIZER
    inputs = [f"{prefix}{t}" for t in texts]
    batch = tok(inputs, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        out = model(**batch)
        hidden = out.last_hidden_state
        mask = batch["attention_mask"]
        seq_lens = mask.sum(dim=1) - 1
        pooled = hidden[torch.arange(hidden.shape[0]), seq_lens]
        return F.normalize(pooled, p=2, dim=-1)


def merge_fp32(peft_model, task):
    """Merge in fp32 for numerical accuracy, then cast back to bf16."""
    m = copy.deepcopy(peft_model).to(torch.float32)
    merged = m.merge_and_unload(adapter_names=[task])
    return merged.to(torch.bfloat16)


def main():
    global MODEL_TOKENIZER
    st = load_embedding()
    transformer, peft, MODEL_TOKENIZER = get_inner(st)

    cos_results = {"bf16-merge": [], "fp32-merge": []}
    for task in TASKS:
        prefix = PREFIXES[task][0]
        peft.set_adapter(task)
        ref = encode_model(peft, TEXTS, task, prefix).float()

        merged_bf16 = copy.deepcopy(peft).merge_and_unload(adapter_names=[task])
        emb_bf = encode_model(merged_bf16, TEXTS, task, prefix).float()
        cos_bf = F.cosine_similarity(ref, emb_bf, dim=-1)
        cos_results["bf16-merge"].extend(cos_bf.tolist())
        del merged_bf16

        merged_fp32 = merge_fp32(peft, task)
        emb_fp = encode_model(merged_fp32, TEXTS, task, prefix).float()
        cos_fp = F.cosine_similarity(ref, emb_fp, dim=-1)
        cos_results["fp32-merge"].extend(cos_fp.tolist())
        del merged_fp32

        print(f"  task={task:<16} bf16 cos: min={cos_bf.min():.6f} mean={cos_bf.mean():.6f} | "
              f"fp32 cos: min={cos_fp.min():.6f} mean={cos_fp.mean():.6f}")

    for k, v in cos_results.items():
        print(f"  {k}: min={min(v):.6f}  mean={statistics.mean(v):.6f}  n={len(v)}")

    # Ranking invariance sanity: within-task ranking of pairwise sims should be preserved
    print("\n  Pairwise-similarity ranking check (retrieval):")
    peft.set_adapter("retrieval")
    ref = encode_model(peft, TEXTS, "retrieval", "Query: ").float()
    m32 = merge_fp32(peft, "retrieval")
    emb = encode_model(m32, TEXTS, "retrieval", "Query: ").float()
    import numpy as np
    ref_sim = (ref @ ref.T).numpy()
    emb_sim = (emb @ emb.T).numpy()
    iu = np.triu_indices(len(TEXTS), k=1)
    a, b = ref_sim[iu], emb_sim[iu]
    # simple spearman without scipy
    def rankdata(x):
        order = np.argsort(x)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(x))
        return ranks
    rho = np.corrcoef(rankdata(a), rankdata(b))[0, 1]
    print(f"    spearman(pairwise sims, ref vs fp32-merged): {rho:.6f}  (n={len(a)} pairs)")


if __name__ == "__main__":
    main()
