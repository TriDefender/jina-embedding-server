"""
Test script for Jina Server
Run after starting the server: python jina_server.py
"""

import requests
import json

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


if __name__ == "__main__":
    print("=" * 50)
    print("Jina Server Test Suite")
    print("=" * 50)

    try:
        test_health()
        test_models()
        test_embeddings()
        test_rerank()

        print("\n" + "=" * 50)
        print("All tests completed!")
        print("=" * 50)

    except requests.exceptions.ConnectionError:
        print("\nError: Cannot connect to server.")
        print("Make sure the server is running: python jina_server.py")
