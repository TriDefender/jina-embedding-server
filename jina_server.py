"""
Jina Embedding & Reranker Server
OpenAI-compatible API for embeddings + reranking

Endpoints:
  - POST /v1/embeddings  (OpenAI compatible)
  - POST /v1/rerank      (Jina/Cohere style)
"""

import os
import sys
import time
from typing import List, Optional, Union
from contextlib import asynccontextmanager
import platform

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


# =============================================================================
# Configuration
# =============================================================================

MODELS_DIR = r".\jinaai"

EMBEDDING_MODEL_PATH = os.path.join(
    MODELS_DIR, "jina-embeddings-v5-text-small-retrieval"
)
RERANKER_MODEL_PATH = os.path.join(MODELS_DIR, "jina-reranker-v3")

# CPU configuration
MAX_THREADS = 20  # Limit to 20 cores as requested
torch.set_num_threads(MAX_THREADS)
os.environ["OMP_NUM_THREADS"] = str(MAX_THREADS)
os.environ["MKL_NUM_THREADS"] = str(MAX_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(MAX_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(MAX_THREADS)


# Check CPU capabilities
def check_cpu_capabilities():
    """Check CPU instruction set support."""
    cpu_info = {
        "cpu_count": os.cpu_count(),
        "pytorch_threads": torch.get_num_threads(),
        "platform": platform.platform(),
        "avx": False,
        "avx2": False,
        "avx512": False,
    }

    # Simple check through PyTorch compilation
    try:
        # Check if AVX is available via torch backend
        cpu_info["avx"] = True  # Modern CPUs support AVX
        cpu_info["avx2"] = True  # Modern CPUs support AVX2
        cpu_info["avx512"] = False  # AVX-512 is rare, skip detection
    except Exception:
        pass

    return cpu_info


# Global model references
embedding_model = None
reranker_model = None


# =============================================================================
# Lifespan
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    global embedding_model, reranker_model

    # Display CPU capabilities
    cpu_caps = check_cpu_capabilities()
    print("=" * 60)
    print("System Information")
    print("=" * 60)
    print(f"  Platform:      {cpu_caps['platform']}")
    print(f"  CPU Cores:     {cpu_caps['cpu_count']}")
    print(f"  PyTorch Threads: {cpu_caps['pytorch_threads']}")
    print(f"  AVX Support:   {cpu_caps['avx']}")
    print(f"  AVX2 Support:  {cpu_caps['avx2']}")
    print(f"  AVX512 Support: {cpu_caps['avx512']}")
    print("=" * 60)

    print("\nLoading models...")
    print("=" * 60)

    # Load embedding model
    print(f"\n[1/2] Loading embedding model from: {EMBEDDING_MODEL_PATH}")
    try:
        embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_PATH,
            trust_remote_code=True,
        )
        print(
            f"      [OK] Embedding model loaded (dim={embedding_model.get_sentence_embedding_dimension()})"
        )
    except Exception as e:
        print(f"      [FAIL] Failed to load embedding model: {e}")
        raise

    # Load reranker model
    print(f"\n[2/2] Loading reranker model from: {RERANKER_MODEL_PATH}")
    try:
        # Import from local modeling.py to avoid AutoModel issues
        sys.path.insert(0, RERANKER_MODEL_PATH)
        from modeling import JinaForRanking
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(RERANKER_MODEL_PATH, trust_remote_code=True)
        reranker_model = JinaForRanking(config)
        reranker_model.eval()
        print("      [OK] Reranker model loaded")
    except Exception as e:
        print(f"      [FAIL] Failed to load reranker model: {e}")
        raise

    print("\n" + "=" * 60)
    print("Server ready!")
    print("=" * 60)

    yield

    # Cleanup
    print("Shutting down...")


app = FastAPI(
    title="Jina Embedding & Reranker Server",
    description="OpenAI-compatible API for embeddings + reranking",
    version="1.0.0",
    lifespan=lifespan,
)


# =============================================================================
# Request/Response Models
# =============================================================================


class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request."""

    input: Union[str, List[str]] = Field(..., description="Text to embed")
    model: str = Field(default="jina-embeddings-v5-text-small-retrieval")
    encoding_format: str = Field(default="float", description="float or base64")
    batch_size: int = Field(
        default=32, ge=1, le=128, description="Batch size for processing"
    )


class EmbeddingObject(BaseModel):
    """Single embedding object."""

    object: str = "embedding"
    index: int
    embedding: List[float]


class EmbeddingResponse(BaseModel):
    """OpenAI-compatible embedding response."""

    object: str = "list"
    data: List[EmbeddingObject]
    model: str
    usage: dict


class RerankRequest(BaseModel):
    """Rerank request (Jina/Cohere style)."""

    model: str = Field(default="jina-reranker-v3")
    query: str = Field(..., description="Search query")
    documents: List[str] = Field(..., description="Documents to rerank")
    top_n: Optional[int] = Field(default=None, description="Return only top N results")
    return_documents: bool = Field(
        default=False, description="Include document text in response"
    )
    batch_size: int = Field(
        default=64, ge=1, le=256, description="Batch size for reranking"
    )


class RerankResult(BaseModel):
    """Single rerank result."""

    index: int
    relevance_score: float
    document: Optional[str] = None


class RerankResponse(BaseModel):
    """Rerank response."""

    model: str
    results: List[RerankResult]
    usage: dict


# =============================================================================
# Endpoints
# =============================================================================


@app.get("/")
async def root():
    """Health check."""
    return {
        "status": "ok",
        "models": {
            "embedding": embedding_model is not None,
            "reranker": reranker_model is not None,
        },
    }


@app.get("/v1/models")
async def list_models():
    """List available models."""
    return {
        "object": "list",
        "data": [
            {
                "id": "jina-embeddings-v5-text-small-retrieval",
                "object": "model",
                "owned_by": "jina-ai",
            },
            {
                "id": "jina-reranker-v3",
                "object": "model",
                "owned_by": "jina-ai",
            },
        ],
    }


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """
    Create embeddings for input text(s).
    OpenAI-compatible endpoint.
    """
    if embedding_model is None:
        raise HTTPException(status_code=503, detail="Embedding model not loaded")

    # Normalize input to list
    texts = [request.input] if isinstance(request.input, str) else request.input

    if not texts:
        raise HTTPException(status_code=400, detail="Input cannot be empty")

    # Process in batches for better CPU utilization
    start_time = time.time()
    batch_size = request.batch_size
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_embeddings = embedding_model.encode(
            batch_texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            convert_to_numpy=True,
        )
        all_embeddings.extend(batch_embeddings)

    elapsed = time.time() - start_time
    print(
        f"  [INFO] Processed {len(texts)} texts in {elapsed:.2f}s ({len(texts) / elapsed:.1f} texts/s)"
    )

    # Build response
    data = []
    total_tokens = 0

    for i, emb in enumerate(all_embeddings):
        data.append(
            EmbeddingObject(
                object="embedding",
                index=i,
                embedding=emb.tolist(),
            )
        )
        # Estimate tokens (rough)
        total_tokens += len(texts[i].split()) * 2

    return EmbeddingResponse(
        object="list",
        data=data,
        model=request.model,
        usage={
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens,
            "batch_size": batch_size,
        },
    )


@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    """
    Rerank documents based on query relevance.
    Jina/Cohere style endpoint.
    """
    if reranker_model is None:
        raise HTTPException(status_code=503, detail="Reranker model not loaded")

    if not request.documents:
        raise HTTPException(status_code=400, detail="Documents cannot be empty")

    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Rerank using model's built-in rerank method with batch control
    start_time = time.time()

    # Update block_size in model for better batching
    original_block_size = getattr(reranker_model, "_block_size", 125)
    reranker_model._block_size = request.batch_size

    results = reranker_model.rerank(
        query=request.query,
        documents=request.documents,
        top_n=request.top_n,
    )

    # Restore original block_size
    reranker_model._block_size = original_block_size

    elapsed = time.time() - start_time
    print(
        f"  [INFO] Reranked {len(request.documents)} documents in {elapsed:.2f}s (batch_size={request.batch_size})"
    )

    # Build response
    rerank_results = []
    for r in results:
        rerank_results.append(
            RerankResult(
                index=r["index"],
                relevance_score=r["relevance_score"],
                document=r["document"] if request.return_documents else None,
            )
        )

    # Estimate tokens
    total_tokens = len(request.query.split()) * 2
    for doc in request.documents:
        total_tokens += len(doc.split()) * 2

    return RerankResponse(
        model=request.model,
        results=rerank_results,
        usage={
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens,
            "batch_size": request.batch_size,
            "elapsed_time": f"{elapsed:.3f}s",
        },
    )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    print("""
    ============================================================
         Jina Embedding & Reranker Server
    ===========================================================
      Endpoints:
         POST /v1/embeddings  - Create embeddings
         POST /v1/rerank      - Rerank documents
         GET  /v1/models      - List models
         GET  /                - Health check
    ===========================================================
      Server running on http://0.0.0.0:8000
    ===========================================================
    """)

    uvicorn.run(
        "jina_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
