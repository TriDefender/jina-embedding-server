"""
HTTP-level benchmark for the Jina server.

Measures against a running http://127.0.0.1:8000 :
  1. Startup + first-request latency (from process spawn if --spawn given)
  2. Serial embedding latency (1 text) across all tasks
  3. Batch embedding throughput (8/32 texts)
  4. Concurrent request throughput (8/16 parallel clients)
  5. Rerank latency (8/32 docs)
  6. First-request latency per task after only-default warmup

Usage:
  python bench/http_bench.py --spawn           # starts server itself, measures cold start
  python bench/http_bench.py                   # server already running
Output: JSON to bench/results/http_bench_<tag>.json + console tables
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

BASE = "http://127.0.0.1:8000"

QUERIES = [
    "What is machine learning?",
    "How do neural networks work?",
    "Best practices for Python error handling",
    "Explain gradient descent",
    "What is RAG?",
    "Compare REST vs GraphQL",
    "How does attention work in transformers?",
    "What is fine-tuning?",
]
DOCS = [
    "Machine learning is a branch of artificial intelligence that enables systems to learn from data and improve over time without being explicitly programmed.",
    "Neural networks are computational models inspired by the human brain, consisting of interconnected nodes organized in layers that learn hierarchical representations.",
    "Gradient descent is an optimization algorithm that iteratively adjusts parameters to minimize a loss function, with variants like SGD and Adam.",
    "Retrieval-Augmented Generation combines a retriever and a generator; the retriever fetches documents using embeddings and the generator conditions on them.",
    "REST is an architectural style using HTTP methods on resources identified by URIs, while GraphQL lets clients request exactly the fields they need.",
    "The attention mechanism computes weighted averages of value vectors where weights come from query-key similarity across multiple heads.",
    "LoRA injects low-rank decomposition matrices into linear layers, learning W + BA instead of updating W, drastically reducing trainable parameters.",
    "BM25 is a probabilistic ranking function scoring document relevance with term frequency saturation and length normalization.",
]


def stats(ts):
    return {
        "n": len(ts),
        "median_ms": round(statistics.median(ts), 1),
        "p95_ms": round(sorted(ts)[max(0, int(len(ts) * 0.95) - 1)], 1),
        "mean_ms": round(statistics.fmean(ts), 1),
        "min_ms": round(min(ts), 1),
    }


def post(path, payload, timeout=300):
    t0 = time.perf_counter()
    r = requests.post(BASE + path, json=payload, timeout=timeout)
    dt = (time.perf_counter() - t0) * 1000
    if r.status_code != 200:
        raise RuntimeError(f"{path} -> {r.status_code}: {r.text[:200]}")
    return dt, r.json()


def bench_serial_embedding(n_iter=30):
    out = {}
    for task, prompt in [
        ("retrieval", "query"),
        ("retrieval", "document"),
        ("text-matching", None),
        ("clustering", None),
        ("classification", None),
    ]:
        payload = {"input": QUERIES[0], "task": task}
        if prompt:
            payload["prompt_name"] = prompt
        # warm 3
        for _ in range(3):
            post("/v1/embeddings", payload)
        ts = []
        for i in range(n_iter):
            payload_i = dict(payload, input=QUERIES[i % len(QUERIES)])
            dt, _ = post("/v1/embeddings", payload_i)
            ts.append(dt)
        key = f"{task}/{prompt or '-'}"
        out[key] = stats(ts)
        print(f"  serial {key:<28} median={out[key]['median_ms']:>7.1f} ms  p95={out[key]['p95_ms']:>7.1f} ms")
    return out


def bench_batch_embedding():
    out = {}
    for n in (8, 32):
        texts = (DOCS * 8)[:n]
        payload = {"input": texts, "task": "retrieval", "prompt_name": "document"}
        for _ in range(2):
            post("/v1/embeddings", payload)
        ts = []
        for _ in range(8):
            dt, _ = post("/v1/embeddings", payload)
            ts.append(dt)
        s = stats(ts)
        s["texts_per_sec"] = round(n / (s["median_ms"] / 1000), 1)
        out[f"batch{n}"] = s
        print(f"  batch {n:>3} texts: median={s['median_ms']:>8.1f} ms  ({s['texts_per_sec']} texts/s)")
    return out


def bench_concurrent(clients=8, per_client=10):
    payload = {"input": QUERIES[0], "task": "retrieval", "prompt_name": "query"}

    def worker(_):
        ts = []
        for i in range(per_client):
            payload_i = dict(payload, input=QUERIES[i % len(QUERIES)])
            dt, _ = post("/v1/embeddings", payload_i)
            ts.append(dt)
        return ts

    for _ in range(4):
        post("/v1/embeddings", payload)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=clients) as ex:
        all_ts = list(ex.map(worker, range(clients)))
    wall = time.perf_counter() - t0
    flat = [t for ts in all_ts for t in ts]
    s = stats(flat)
    s["clients"] = clients
    s["per_client"] = per_client
    s["wall_s"] = round(wall, 2)
    s["req_per_s"] = round(len(flat) / wall, 1)
    print(f"  concurrent {clients}x{per_client}: median={s['median_ms']} ms  rps={s['req_per_s']}  wall={s['wall_s']}s")
    return s


def bench_rerank():
    out = {}
    for n in (8, 32):
        docs = (DOCS * 8)[:n]
        payload = {"query": "How does retrieval-augmented generation work?", "documents": docs, "top_n": 5}
        for _ in range(2):
            post("/v1/rerank", payload)
        ts = []
        for _ in range(6):
            dt, _ = post("/v1/rerank", payload)
            ts.append(dt)
        s = stats(ts)
        s["docs_per_sec"] = round(n / (s["median_ms"] / 1000), 1)
        out[f"docs{n}"] = s
        print(f"  rerank {n:>3} docs: median={s['median_ms']:>8.1f} ms  ({s['docs_per_sec']} docs/s)")
    return out


def bench_first_request_all_tasks():
    """Latency of the FIRST call for each less-common task (server was warmed on retrieval only)."""
    out = {}
    for task, prompt in [
        ("retrieval", "query"),
        ("retrieval", "document"),
        ("text-matching", None),
        ("clustering", None),
        ("classification", None),
    ]:
        payload = {"input": QUERIES[1], "task": task}
        if prompt:
            payload["prompt_name"] = prompt
        dt, _ = post("/v1/embeddings", payload)  # single cold call
        out[f"{task}/{prompt or '-'}"] = round(dt, 1)
        print(f"  first-call {task}/{prompt or '-'}: {dt:.1f} ms")
    return out


def wait_ready(timeout_s=300.0):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        try:
            r = requests.get(BASE + "/", timeout=2)
            if r.status_code == 200:
                return time.perf_counter() - t0
        except requests.RequestException:
            pass
        time.sleep(0.25)
    raise RuntimeError("server did not become ready")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None, help="label for the results file")
    ap.add_argument("--spawn", action="store_true", help="launch server subprocess and measure cold start")
    ap.add_argument("--settle", type=float, default=0, help="seconds to wait after ready before benchmarking (background LoRA merges)")
    ap.add_argument("--skip-concurrent", action="store_true")
    args = ap.parse_args()

    proc = None
    cold = None
    if args.spawn:
        t0 = time.perf_counter()
        env = dict(os.environ)
        log_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), f"server_{args.tag or 'bench'}.log"
        )
        log_f = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, "jina_server.py"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
        )
        ready_after = wait_ready()
        # wait for models actually loaded (root returns models loaded status)
        while True:
            try:
                j = requests.get(BASE + "/", timeout=2).json()
                if j.get("models", {}).get("embedding") and j.get("models", {}).get("reranker"):
                    break
            except Exception:
                pass
            if proc.poll() is not None:
                with open(log_path, encoding="utf-8", errors="replace") as f:
                    print(f.read()[-3000:])
                raise RuntimeError("server died during startup")
            time.sleep(0.25)
        cold = {
            "spawn_to_health_s": round(ready_after, 2),
            "spawn_to_models_ready_s": round(time.perf_counter() - t0, 2),
        }
        print(f"cold start: health after {cold['spawn_to_health_s']}s, models ready after {cold['spawn_to_models_ready_s']}s")

        # very first request after ready
        dt, _ = post("/v1/embeddings", {"input": QUERIES[0], "task": "retrieval", "prompt_name": "query"})
        cold["first_request_ms"] = round(dt, 1)
        print(f"first request after ready: {dt:.1f} ms")
        if args.settle > 0:
            print(f"settling {args.settle}s (background warmup/merges)...")
            time.sleep(args.settle)

    results = {"tag": args.tag or time.strftime("%Y%m%d_%H%M%S")}

    print("\n-- serial embedding latency --")
    results["serial"] = bench_serial_embedding()
    print("\n-- batch embedding --")
    results["batch"] = bench_batch_embedding()
    if not args.skip_concurrent:
        print("\n-- concurrent (8 clients) --")
        results["concurrent8"] = bench_concurrent(8, 10)
    print("\n-- rerank --")
    results["rerank"] = bench_rerank()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    if cold:
        results["cold_start"] = cold
    out_path = os.path.join(out_dir, f"http_bench_{results['tag']}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nsaved: {out_path}")

    if proc is not None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
