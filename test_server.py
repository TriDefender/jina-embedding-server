"""
Test script for Jina Server
Run after starting the server: python jina_server.py
"""

import requests
import json
import time
import tempfile
import os

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
        "model": "jina-embeddings-v5-text-small-retrieval",
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
                "model": "jina-embeddings-v5-text-small-retrieval",
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


if __name__ == "__main__":
    print("=" * 50)
    print("Jina Server Test Suite")
    print("=" * 50)

    try:
        test_health()
        test_models()
        test_embeddings()
        test_rerank()
        test_batch_api()  # New batch API test

        print("\n" + "=" * 50)
        print("All tests completed!")
        print("=" * 50)

    except requests.exceptions.ConnectionError:
        print("\nError: Cannot connect to server.")
        print("Make sure the server is running: python jina_server.py")
