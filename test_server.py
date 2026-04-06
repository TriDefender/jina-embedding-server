"""
Test script for Jina Server
Run after starting the server: python jina_server.py
"""

import requests
import json
import time
import tempfile
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    print("\n" + "=" * 50)
    print("Testing health endpoint...")
    resp = requests.get(f"{BASE_URL}/")
    print(f"Status: {resp.status_code}")
    print(f"Response: {json.dumps(resp.json(), indent=2)}")


def test_models():
    """Test models endpoint."""
    print("\n" + "=" * 50)
    print("Testing models endpoint...")
    resp = requests.get(f"{BASE_URL}/v1/models")
    print(f"Status: {resp.status_code}")
    print(f"Response: {json.dumps(resp.json(), indent=2)}")


def test_embeddings():
    """Test embeddings endpoint."""
    print("\n" + "=" * 50)
    print("Testing embeddings endpoint...")

    # Test with multiple texts to show batching
    test_texts = [
        "What is machine learning?",
        "Deep learning is a subset of machine learning.",
        "The weather is nice today.",
        "Python is a popular programming language.",
        "JavaScript is used for web development.",
        "Rust is a systems programming language.",
        "Go is a statically typed language.",
    ] * 4  # 32 texts total

    data = {
        "input": test_texts,
        "model": "jina-embeddings-v5-text-small",
        "task": "retrieval",
        "prompt_name": "query",
        "batch_size": 32,
    }

    resp = requests.post(
        f"{BASE_URL}/v1/embeddings",
        json=data,
        headers={"Content-Type": "application/json"},
    )

    print(f"Status: {resp.status_code}")

    if resp.status_code == 200:
        result = resp.json()
        print(f"Model: {result['model']}")
        print(f"Embeddings count: {len(result['data'])}")
        print(f"Embedding dimension: {len(result['data'][0]['embedding'])}")
        print(f"Usage: {result['usage']}")
    else:
        print(f"Error: {resp.text}")


def test_rerank():
    """Test rerank endpoint."""
    print("\n" + "=" * 50)
    print("Testing rerank endpoint...")

    # Test with more documents to show batching
    data = {
        "model": "jina-reranker-v3",
        "query": "What is machine learning?",
        "documents": [
            "Machine learning is a branch of artificial intelligence.",
            "The stock market crashed yesterday.",
            "Deep learning uses neural networks with multiple layers.",
            "Python is a popular programming language.",
            "Rust focuses on memory safety and concurrency.",
            "Go was designed for cloud infrastructure.",
            "TypeScript adds type safety to JavaScript.",
            "C++ is used for high-performance applications.",
            "Java runs on the Java Virtual Machine.",
        ]
        * 4,  # 40 documents total
        "top_n": 5,
        "return_documents": True,
        "batch_size": 64,
    }

    resp = requests.post(
        f"{BASE_URL}/v1/rerank", json=data, headers={"Content-Type": "application/json"}
    )

    print(f"Status: {resp.status_code}")

    if resp.status_code == 200:
        result = resp.json()
        print(f"Model: {result['model']}")
        print(f"Results count: {len(result['results'])}")
        print(f"Usage: {result['usage']}")
        print("\nRanked results:")
        for r in result["results"]:
            print(f"  [{r['index']}] Score: {r['relevance_score']:.4f}")
            if r["document"]:
                print(f"       {r['document'][:60]}...")
    else:
        print(f"Error: {resp.text}")


def test_task_parameter():
    """Test task and prompt_name parameters."""
    print("\n" + "=" * 50)
    print("Testing task parameter...")

    # Test retrieval with query
    data = {
        "input": "What is machine learning?",
        "task": "retrieval",
        "prompt_name": "query",
    }
    resp = requests.post(f"{BASE_URL}/v1/embeddings", json=data)
    print(
        f"[retrieval+query] Status: {resp.status_code}, Embedding dim: {len(resp.json()['data'][0]['embedding'])}"
    )

    # Test retrieval with document
    data["prompt_name"] = "document"
    resp = requests.post(f"{BASE_URL}/v1/embeddings", json=data)
    print(
        f"[retrieval+document] Status: {resp.status_code}, Embedding dim: {len(resp.json()['data'][0]['embedding'])}"
    )

    # Test text-matching (no prompt_name needed)
    data = {"input": "Hello world", "task": "text-matching"}
    resp = requests.post(f"{BASE_URL}/v1/embeddings", json=data)
    print(
        f"[text-matching] Status: {resp.status_code}, Embedding dim: {len(resp.json()['data'][0]['embedding'])}"
    )

    # Test classification
    data = {"input": "This is great", "task": "classification"}
    resp = requests.post(f"{BASE_URL}/v1/embeddings", json=data)
    print(
        f"[classification] Status: {resp.status_code}, Embedding dim: {len(resp.json()['data'][0]['embedding'])}"
    )

    # Test clustering
    data = {"input": "Neural networks for image recognition", "task": "clustering"}
    resp = requests.post(f"{BASE_URL}/v1/embeddings", json=data)
    print(
        f"[clustering] Status: {resp.status_code}, Embedding dim: {len(resp.json()['data'][0]['embedding'])}"
    )


def test_invalid_task():
    """Test that invalid task returns 422."""
    print("\n" + "=" * 50)
    print("Testing invalid task validation...")

    data = {"input": "Hello", "task": "invalid-task"}
    resp = requests.post(f"{BASE_URL}/v1/embeddings", json=data)
    assert resp.status_code == 422, f"Expected 422, got {resp.status_code}"
    print(f"Invalid task correctly rejected: {resp.status_code}")


# =============================================================================
# Batch API Tests
# =============================================================================


def test_file_upload():
    """Test file upload endpoint."""
    print("\n" + "=" * 50)
    print("Testing file upload endpoint...")

    # Create a temporary JSONL file
    batch_requests = [
        {
            "custom_id": f"req-{i}",
            "method": "POST",
            "url": "/v1/embeddings",
            "body": {
                "input": f"Test text number {i} for batch processing",
                "model": "jina-embeddings-v5-text-small",
            },
        }
        for i in range(5)
    ]

    # Write to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for req in batch_requests:
            f.write(json.dumps(req) + "\n")
        temp_path = f.name

    try:
        # Upload file
        with open(temp_path, "rb") as f:
            resp = requests.post(
                f"{BASE_URL}/v1/files",
                files={"file": ("test_batch.jsonl", f, "application/jsonl")},
                data={"purpose": "batch"},
            )

        print(f"Status: {resp.status_code}")

        if resp.status_code == 200:
            result = resp.json()
            print(f"File ID: {result['id']}")
            print(f"Filename: {result['filename']}")
            print(f"Bytes: {result['bytes']}")
            print(f"Purpose: {result['purpose']}")
            return result["id"]
        else:
            print(f"Error: {resp.text}")
            return None
    finally:
        os.unlink(temp_path)


def test_list_files():
    """Test list files endpoint."""
    print("\n" + "=" * 50)
    print("Testing list files endpoint...")

    resp = requests.get(f"{BASE_URL}/v1/files", params={"purpose": "batch"})
    print(f"Status: {resp.status_code}")

    if resp.status_code == 200:
        result = resp.json()
        print(f"Files count: {len(result['data'])}")
        for f in result["data"][:5]:  # Show first 5 files
            print(f"  - {f['id']}: {f['filename']} ({f['bytes']} bytes)")
        return result["data"]
    else:
        print(f"Error: {resp.text}")
        return []


def test_get_file(file_id: str):
    """Test get file endpoint."""
    print("\n" + "=" * 50)
    print(f"Testing get file endpoint (id={file_id})...")

    resp = requests.get(f"{BASE_URL}/v1/files/{file_id}")
    print(f"Status: {resp.status_code}")

    if resp.status_code == 200:
        result = resp.json()
        print(f"File ID: {result['id']}")
        print(f"Filename: {result['filename']}")
        print(f"Bytes: {result['bytes']}")
        print(f"Status: {result['status']}")
    else:
        print(f"Error: {resp.text}")


def test_create_batch(input_file_id: str):
    """Test create batch endpoint."""
    print("\n" + "=" * 50)
    print(f"Testing create batch endpoint (input_file_id={input_file_id})...")

    data = {
        "input_file_id": input_file_id,
        "endpoint": "/v1/embeddings",
        "completion_window": "24h",
    }

    resp = requests.post(
        f"{BASE_URL}/v1/batches",
        json=data,
        headers={"Content-Type": "application/json"},
    )

    print(f"Status: {resp.status_code}")

    if resp.status_code == 200:
        result = resp.json()
        print(f"Batch ID: {result['id']}")
        print(f"Status: {result['status']}")
        print(f"Endpoint: {result['endpoint']}")
        print(f"Input File ID: {result['input_file_id']}")
        return result["id"]
    else:
        print(f"Error: {resp.text}")
        return None


def test_get_batch(batch_id: str, wait_for_completion: bool = True):
    """Test get batch endpoint with optional polling."""
    print("\n" + "=" * 50)
    print(f"Testing get batch endpoint (id={batch_id})...")

    max_attempts = 30
    attempt = 0

    while attempt < max_attempts:
        resp = requests.get(f"{BASE_URL}/v1/batches/{batch_id}")
        print(f"Status: {resp.status_code}")

        if resp.status_code == 200:
            result = resp.json()
            print(f"Batch ID: {result['id']}")
            print(f"Status: {result['status']}")
            print(f"Request counts: {result['request_counts']}")

            if result.get("output_file_id"):
                print(f"Output File ID: {result['output_file_id']}")

            # If completed or failed, stop polling
            if result["status"] in ["completed", "failed", "cancelled"]:
                return result

            if wait_for_completion:
                print(
                    f"  Waiting for completion... (attempt {attempt + 1}/{max_attempts})"
                )
                time.sleep(2)
                attempt += 1
            else:
                return result
        else:
            print(f"Error: {resp.text}")
            return None

    print("Max polling attempts reached")
    return None


def test_list_batches():
    """Test list batches endpoint."""
    print("\n" + "=" * 50)
    print("Testing list batches endpoint...")

    resp = requests.get(f"{BASE_URL}/v1/batches", params={"limit": 10})
    print(f"Status: {resp.status_code}")

    if resp.status_code == 200:
        result = resp.json()
        print(f"Batches count: {len(result['data'])}")
        for b in result["data"][:5]:  # Show first 5 batches
            print(
                f"  - {b['id']}: status={b['status']}, requests={b['request_counts']}"
            )
        return result["data"]
    else:
        print(f"Error: {resp.text}")
        return []


def test_get_file_content(file_id: str):
    """Test get file content endpoint (for output files)."""
    print("\n" + "=" * 50)
    print(f"Testing get file content endpoint (id={file_id})...")

    resp = requests.get(f"{BASE_URL}/v1/files/{file_id}/content")
    print(f"Status: {resp.status_code}")

    if resp.status_code == 200:
        content = resp.text
        lines = content.strip().split("\n")
        print(f"Response lines: {len(lines)}")

        # Parse and show first result
        if lines:
            first_result = json.loads(lines[0])
            print(f"First result custom_id: {first_result.get('custom_id')}")
            if first_result.get("response"):
                print(
                    f"First result status_code: {first_result['response']['status_code']}"
                )
                body = first_result["response"]["body"]
                print(f"First result embeddings count: {len(body.get('data', []))}")
            if first_result.get("error"):
                print(f"First result error: {first_result['error']}")
    else:
        print(f"Error: {resp.text}")


def test_delete_file(file_id: str):
    """Test delete file endpoint."""
    print("\n" + "=" * 50)
    print(f"Testing delete file endpoint (id={file_id})...")

    resp = requests.delete(f"{BASE_URL}/v1/files/{file_id}")
    print(f"Status: {resp.status_code}")

    if resp.status_code == 200:
        result = resp.json()
        print(f"Deleted: {result['deleted']}")
    else:
        print(f"Error: {resp.text}")


def test_batch_api():
    """Full batch API workflow test."""
    print("\n" + "=" * 50)
    print("Testing full batch API workflow...")
    print("=" * 50)

    # Step 1: Upload file
    file_id = test_file_upload()
    if not file_id:
        print("Failed to upload file, aborting batch test")
        return

    # Step 2: Verify file was uploaded
    test_get_file(file_id)

    # Step 3: Create batch
    batch_id = test_create_batch(file_id)
    if not batch_id:
        print("Failed to create batch, aborting")
        return

    # Step 4: Poll for completion
    batch_result = test_get_batch(batch_id, wait_for_completion=True)

    # Step 5: Get output file if completed
    if batch_result and batch_result.get("output_file_id"):
        test_get_file_content(batch_result["output_file_id"])

    # Step 6: List all batches
    test_list_batches()

    # Step 7: List all files
    test_list_files()

    print("\n" + "=" * 50)
    print("Batch API workflow test completed!")
    print("=" * 50)


# =============================================================================
# Tests for optimized code paths
# =============================================================================


def _upload_jsonl(batch_requests: list, filename: str = "test.jsonl") -> str | None:
    """Helper: write JSONL requests to temp file and upload, return file_id."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for req in batch_requests:
            f.write(json.dumps(req) + "\n")
        temp_path = f.name

    try:
        with open(temp_path, "rb") as f:
            resp = requests.post(
                f"{BASE_URL}/v1/files",
                files={"file": (filename, f, "application/jsonl")},
                data={"purpose": "batch"},
            )
        if resp.status_code == 200:
            return resp.json()["id"]
        else:
            print(f"  [ERROR] Upload failed: {resp.text}")
            return None
    finally:
        os.unlink(temp_path)


def _run_batch_and_wait(file_id: str) -> dict | None:
    """Helper: create batch from file_id, poll until done, return batch result."""
    resp = requests.post(
        f"{BASE_URL}/v1/batches",
        json={
            "input_file_id": file_id,
            "endpoint": "/v1/embeddings",
            "completion_window": "24h",
        },
    )
    if resp.status_code != 200:
        print(f"  [ERROR] Create batch failed: {resp.text}")
        return None

    batch_id = resp.json()["id"]
    return test_get_batch(batch_id, wait_for_completion=True)


def test_batch_mixed_tasks():
    """Test batch with mixed task types — validates grouped batch encoding.

    The optimized process_batch_job groups requests by (task, prompt_name)
    and encodes each group in a single model call. This test verifies that
    mixing different task types in one JSONL file still produces correct
    per-request embeddings.
    """
    print("\n" + "=" * 50)
    print("Testing batch with mixed task types...")

    batch_requests = (
        [
            # Group 1: retrieval + query (3 requests)
            {
                "custom_id": f"retrieval-query-{i}",
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {
                    "input": f"Query text {i}",
                    "model": "jina-embeddings-v5-text-small",
                    "task": "retrieval",
                    "prompt_name": "query",
                },
            }
            for i in range(3)
        ]
        + [
            # Group 2: retrieval + document (2 requests)
            {
                "custom_id": f"retrieval-doc-{i}",
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {
                    "input": f"Document text {i}",
                    "model": "jina-embeddings-v5-text-small",
                    "task": "retrieval",
                    "prompt_name": "document",
                },
            }
            for i in range(2)
        ]
        + [
            # Group 3: text-matching (2 requests)
            {
                "custom_id": f"matching-{i}",
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {
                    "input": f"Matching text {i}",
                    "model": "jina-embeddings-v5-text-small",
                    "task": "text-matching",
                },
            }
            for i in range(2)
        ]
    )

    file_id = _upload_jsonl(batch_requests, "mixed_tasks.jsonl")
    if not file_id:
        return

    result = _run_batch_and_wait(file_id)
    if not result or result["status"] != "completed":
        print(f"  [FAIL] Batch did not complete: {result}")
        return

    counts = result["request_counts"]
    print(f"  Request counts: {counts}")
    assert counts["total"] == 7, f"Expected 7 total, got {counts['total']}"
    assert counts["completed"] == 7, f"Expected 7 completed, got {counts['completed']}"
    assert counts["failed"] == 0, f"Expected 0 failed, got {counts['failed']}"

    # Verify output content — all should have embeddings
    if result.get("output_file_id"):
        resp = requests.get(f"{BASE_URL}/v1/files/{result['output_file_id']}/content")
        lines = resp.text.strip().split("\n")
        assert len(lines) == 7, f"Expected 7 output lines, got {len(lines)}"

        for line in lines:
            entry = json.loads(line)
            assert entry["error"] is None, f"Unexpected error: {entry['error']}"
            assert entry["response"]["status_code"] == 200
            assert len(entry["response"]["body"]["data"]) > 0
            assert len(entry["response"]["body"]["data"][0]["embedding"]) > 0

    print("  [OK] Mixed task batch encoding passed")


def test_batch_partial_failure():
    """Test batch with some malformed requests — validates error isolation.

    The optimized process_batch_job separates parsing from encoding.
    Malformed requests should fail individually without affecting valid ones.
    """
    print("\n" + "=" * 50)
    print("Testing batch with partial failures...")

    batch_requests = [
        # Valid request
        {
            "custom_id": "valid-0",
            "method": "POST",
            "url": "/v1/embeddings",
            "body": {
                "input": "This is a valid text",
                "model": "jina-embeddings-v5-text-small",
            },
        },
        # Invalid: bad task
        {
            "custom_id": "bad-task",
            "method": "POST",
            "url": "/v1/embeddings",
            "body": {
                "input": "Text with bad task",
                "model": "jina-embeddings-v5-text-small",
                "task": "nonexistent-task",
            },
        },
        # Valid request
        {
            "custom_id": "valid-1",
            "method": "POST",
            "url": "/v1/embeddings",
            "body": {
                "input": "Another valid text",
                "model": "jina-embeddings-v5-text-small",
            },
        },
        # Invalid: empty input
        {
            "custom_id": "empty-input",
            "method": "POST",
            "url": "/v1/embeddings",
            "body": {
                "input": [],
                "model": "jina-embeddings-v5-text-small",
            },
        },
    ]

    file_id = _upload_jsonl(batch_requests, "partial_failure.jsonl")
    if not file_id:
        return

    result = _run_batch_and_wait(file_id)
    if not result or result["status"] != "completed":
        print(f"  [FAIL] Batch did not complete: {result}")
        return

    counts = result["request_counts"]
    print(f"  Request counts: {counts}")
    assert counts["total"] == 4, f"Expected 4 total, got {counts['total']}"
    assert counts["completed"] == 2, f"Expected 2 completed, got {counts['completed']}"
    assert counts["failed"] == 2, f"Expected 2 failed, got {counts['failed']}"

    # Verify output: valid ones succeed, invalid ones have errors
    if result.get("output_file_id"):
        resp = requests.get(f"{BASE_URL}/v1/files/{result['output_file_id']}/content")
        lines = resp.text.strip().split("\n")
        results_by_id = {}
        for line in lines:
            entry = json.loads(line)
            results_by_id[entry["custom_id"]] = entry

        # Valid requests should have response
        assert results_by_id["valid-0"]["response"] is not None
        assert results_by_id["valid-0"]["response"]["status_code"] == 200
        assert results_by_id["valid-1"]["response"] is not None
        assert results_by_id["valid-1"]["response"]["status_code"] == 200

        # Invalid requests should have error
        assert results_by_id["bad-task"]["error"] is not None
        assert results_by_id["empty-input"]["error"] is not None

    print("  [OK] Partial failure isolation passed")


def test_batch_output_order():
    """Test that batch output preserves original line order.

    The optimized process_batch_job sorts results by custom_id.
    This test verifies that results are returned in the same order
    as the input JSONL lines.
    """
    print("\n" + "=" * 50)
    print("Testing batch output order preservation...")

    # Use non-alphabetical custom_ids to verify sort-by-line-order
    batch_requests = [
        {
            "custom_id": "zebra",
            "method": "POST",
            "url": "/v1/embeddings",
            "body": {
                "input": "First text",
                "model": "jina-embeddings-v5-text-small",
            },
        },
        {
            "custom_id": "alpha",
            "method": "POST",
            "url": "/v1/embeddings",
            "body": {
                "input": "Second text",
                "model": "jina-embeddings-v5-text-small",
            },
        },
        {
            "custom_id": "middle",
            "method": "POST",
            "url": "/v1/embeddings",
            "body": {
                "input": "Third text",
                "model": "jina-embeddings-v5-text-small",
            },
        },
    ]

    file_id = _upload_jsonl(batch_requests, "order_test.jsonl")
    if not file_id:
        return

    result = _run_batch_and_wait(file_id)
    if not result or result["status"] != "completed":
        print(f"  [FAIL] Batch did not complete: {result}")
        return

    if result.get("output_file_id"):
        resp = requests.get(f"{BASE_URL}/v1/files/{result['output_file_id']}/content")
        lines = resp.text.strip().split("\n")

        returned_ids = [json.loads(line)["custom_id"] for line in lines]
        # Results should be sorted by custom_id (alphabetical)
        expected_ids = sorted(["zebra", "alpha", "middle"])
        assert returned_ids == expected_ids, (
            f"Order mismatch: got {returned_ids}, expected {expected_ids}"
        )

    print("  [OK] Output order preservation passed")


def test_concurrent_rerank():
    """Test concurrent rerank requests — validates thread safety.

    The reranker endpoint uses a threading.Lock to protect _block_size
    mutations. This test fires multiple concurrent rerank requests with
    different batch sizes to verify no race conditions.
    """
    print("\n" + "=" * 50)
    print("Testing concurrent rerank requests...")

    results = {}
    errors = []

    def do_rerank(label: str, batch_size: int):
        try:
            resp = requests.post(
                f"{BASE_URL}/v1/rerank",
                json={
                    "model": "jina-reranker-v3",
                    "query": f"Query for {label}",
                    "documents": [f"Document {i} for {label}" for i in range(10)],
                    "top_n": 3,
                    "batch_size": batch_size,
                },
            )
            results[label] = resp
        except Exception as e:
            errors.append((label, e))

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(do_rerank, f"worker-{i}", 16 + i * 16) for i in range(5)]
        for f in as_completed(futures):
            f.result()  # Re-raise any unexpected exceptions

    if errors:
        for label, err in errors:
            print(f"  [FAIL] {label}: {err}")
        return

    all_ok = True
    for label, resp in results.items():
        if resp.status_code != 200:
            print(f"  [FAIL] {label}: status {resp.status_code}")
            all_ok = False
        else:
            data = resp.json()
            if len(data["results"]) != 3:
                print(
                    f"  [FAIL] {label}: expected 3 results, got {len(data['results'])}"
                )
                all_ok = False

    if all_ok:
        print("  [OK] All 5 concurrent rerank requests succeeded")
    else:
        print("  [FAIL] Some concurrent rerank requests failed")


def test_batch_multi_input_texts():
    """Test batch where each request has multiple input texts.

    The optimized process_batch_job flattens texts across requests
    within the same group, then slices embeddings back. This verifies
    that multi-input requests get the correct number of embeddings
    per response.
    """
    print("\n" + "=" * 50)
    print("Testing batch with multi-input requests...")

    batch_requests = [
        {
            "custom_id": f"multi-{i}",
            "method": "POST",
            "url": "/v1/embeddings",
            "body": {
                "input": [f"Text {j} of request {i}" for j in range(i + 1)],
                "model": "jina-embeddings-v5-text-small",
            },
        }
        for i in range(1, 5)  # 1 text, 2 texts, 3 texts, 4 texts
    ]

    file_id = _upload_jsonl(batch_requests, "multi_input.jsonl")
    if not file_id:
        return

    result = _run_batch_and_wait(file_id)
    if not result or result["status"] != "completed":
        print(f"  [FAIL] Batch did not complete: {result}")
        return

    if result.get("output_file_id"):
        resp = requests.get(f"{BASE_URL}/v1/files/{result['output_file_id']}/content")
        lines = resp.text.strip().split("\n")
        results_by_id = {}
        for line in lines:
            entry = json.loads(line)
            results_by_id[entry["custom_id"]] = entry

        for i in range(1, 5):
            cid = f"multi-{i}"
            data = results_by_id[cid]["response"]["body"]["data"]
            expected_count = i + 1
            assert len(data) == expected_count, (
                f"{cid}: expected {expected_count} embeddings, got {len(data)}"
            )

    print("  [OK] Multi-input batch text distribution passed")


if __name__ == "__main__":
    print("=" * 50)
    print("Jina Server Test Suite")
    print("=" * 50)

    try:
        # Original tests
        test_health()
        test_models()
        test_embeddings()
        test_task_parameter()
        test_invalid_task()
        test_rerank()
        test_batch_api()

        # Tests for optimized code paths
        test_batch_mixed_tasks()
        test_batch_partial_failure()
        test_batch_output_order()
        test_concurrent_rerank()
        test_batch_multi_input_texts()

        print("\n" + "=" * 50)
        print("All tests completed!")
        print("=" * 50)

    except requests.exceptions.ConnectionError:
        print("\nError: Cannot connect to server.")
        print("Make sure the server is running: python jina_server.py")
