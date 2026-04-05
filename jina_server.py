"""
Jina Embedding & Reranker Server
OpenAI-compatible API for embeddings + reranking

Endpoints:
  - POST /v1/embeddings  (OpenAI compatible)
  - POST /v1/rerank      (Jina/Cohere style)
  - POST /v1/files       (OpenAI Files API)
  - POST /v1/batches     (OpenAI Batch API)
"""

import os
import sys
import time
import uuid
import json
import threading
from typing import List, Optional, Union, Dict, Any
from contextlib import asynccontextmanager
import platform

import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import Response
from pydantic import BaseModel, Field, field_validator, model_validator
from sentence_transformers import SentenceTransformer


# =============================================================================
# Configuration
# =============================================================================

MODELS_DIR = r".\jinaai"

EMBEDDING_MODEL_PATH = os.path.join(MODELS_DIR, "jina-embeddings-v5-text-small")
RERANKER_MODEL_PATH = os.path.join(MODELS_DIR, "jina-reranker-v3")

# Valid task types for jina-embeddings-v5-text-small (LoRA task adapters)
# See: https://huggingface.co/jinaai/jina-embeddings-v5-text-small
VALID_EMBEDDING_TASKS = ("retrieval", "text-matching", "clustering", "classification")
# For retrieval task, prompt_name selects query vs document prefix
VALID_PROMPT_NAMES = ("query", "document")
# Jina cloud API sends dot-notation tasks; map them to (task, prompt_name)
TASK_ALIAS_MAP = {
    "retrieval.query": ("retrieval", "query"),
    "retrieval.passage": ("retrieval", "document"),
    "text-matching": ("text-matching", None),
    "classification": ("classification", None),
    "clustering": ("clustering", None),
    "separation": ("text-matching", None),
}

# CPU configuration
MAX_THREADS = 20  # Limit to 20 cores as requested
torch.set_num_threads(MAX_THREADS)
os.environ["OMP_NUM_THREADS"] = str(MAX_THREADS)
os.environ["MKL_NUM_THREADS"] = str(MAX_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(MAX_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(MAX_THREADS)


# Check CPU capabilities
def check_cpu_capabilities():
    """Check CPU instruction set support using py-cpuinfo."""
    import cpuinfo

    cpu_info = {
        "cpu_count": os.cpu_count(),
        "pytorch_threads": torch.get_num_threads(),
        "platform": platform.platform(),
        "cpu_brand": cpuinfo.get_cpu_info().get("brand_raw", "Unknown"),
        "avx": False,
        "avx2": False,
        "avx512": False,
        "avx512f": False,
        "avx512_vnni": False,
    }

    # Get all CPU flags
    info = cpuinfo.get_cpu_info()
    flags = info.get("flags", [])

    # Check instruction set support
    cpu_info["avx"] = "avx" in flags
    cpu_info["avx2"] = "avx2" in flags
    cpu_info["avx512f"] = "avx512f" in flags
    cpu_info["avx512_vnni"] = "avx512vnni" in flags

    # AVX512 considered supported if F (foundation) is present
    cpu_info["avx512"] = cpu_info["avx512f"]

    # Log detailed AVX512 subsets if available
    if cpu_info["avx512"]:
        avx512_subsets = [f for f in flags if f.startswith("avx512")]
        cpu_info["avx512_subsets"] = avx512_subsets

    return cpu_info


# Global model references
embedding_model = None
reranker_model = None
reranker_lock = threading.Lock()  # Protects reranker_model._block_size mutations

# In-memory storage for Files and Batches
files_storage: Dict[str, Dict[str, Any]] = {}  # file_id -> file metadata + content
batches_storage: Dict[str, Dict[str, Any]] = {}  # batch_id -> batch metadata


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
    print(f"  Platform:        {cpu_caps['platform']}")
    print(f"  CPU:             {cpu_caps['cpu_brand']}")
    print(f"  CPU Cores:       {cpu_caps['cpu_count']}")
    print(f"  PyTorch Threads: {cpu_caps['pytorch_threads']}")
    print(f"  AVX:             {cpu_caps['avx']}")
    print(f"  AVX2:            {cpu_caps['avx2']}")
    print(f"  AVX-512:         {cpu_caps['avx512']}")
    if cpu_caps.get("avx512"):
        print(f"  AVX-512 Subsets: {', '.join(cpu_caps.get('avx512_subsets', []))}")
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
    model: str = Field(default="jina-embeddings-v5-text-small")
    encoding_format: str = Field(default="float", description="float or base64")
    batch_size: int = Field(
        default=32, ge=1, le=128, description="Batch size for processing"
    )
    task: str = Field(
        default="retrieval",
        description="Task adapter: retrieval, text-matching, clustering, classification",
    )
    prompt_name: Optional[str] = Field(
        default="query",
        description="For retrieval task only: 'query' or 'document'. Required when task='retrieval'.",
    )

    @field_validator("task")
    @classmethod
    def validate_task(cls, v: str) -> str:
        # Accept both plain tasks ("retrieval") and Jina cloud aliases ("retrieval.query")
        # Aliases are resolved after validation — see model_validator below
        if v not in VALID_EMBEDDING_TASKS and v not in TASK_ALIAS_MAP:
            raise ValueError(
                f"Invalid task '{v}'. Must be one of: {VALID_EMBEDDING_TASKS} or aliases: {list(TASK_ALIAS_MAP.keys())}"
            )
        return v

    @field_validator("prompt_name")
    @classmethod
    def validate_prompt_name(cls, v: Optional[str], info) -> Optional[str]:
        if v is not None and v not in VALID_PROMPT_NAMES:
            raise ValueError(
                f"Invalid prompt_name '{v}'. Must be one of: {VALID_PROMPT_NAMES}"
            )
        return v

    @model_validator(mode="after")
    def resolve_task_alias(self) -> "EmbeddingRequest":
        """Expand dot-notation task aliases (e.g. 'retrieval.query') into task + prompt_name."""
        if self.task in TASK_ALIAS_MAP:
            resolved_task, resolved_prompt = TASK_ALIAS_MAP[self.task]
            object.__setattr__(self, "task", resolved_task)
            if self.prompt_name is None:
                object.__setattr__(self, "prompt_name", resolved_prompt)
        # Retrieval task requires prompt_name
        if self.task == "retrieval" and self.prompt_name is None:
            raise ValueError(
                "prompt_name is required when task='retrieval'. Use 'query' or 'document'."
            )
        return self


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
# File & Batch Models (OpenAI Batch API)
# =============================================================================


class FileObject(BaseModel):
    """OpenAI File object."""

    id: str
    object: str = "file"
    bytes: int
    created_at: int
    filename: str
    purpose: str
    status: str = "uploaded"


class FileListResponse(BaseModel):
    """List of files."""

    object: str = "list"
    data: List[FileObject]


class BatchRequest(BaseModel):
    """Batch creation request."""

    input_file_id: str
    endpoint: str = "/v1/embeddings"
    completion_window: str = "24h"
    metadata: Optional[Dict[str, str]] = None


class BatchObject(BaseModel):
    """OpenAI Batch object."""

    id: str
    object: str = "batch"
    endpoint: str
    errors: Optional[Dict[str, Any]] = None
    input_file_id: str
    completion_window: str
    status: str
    output_file_id: Optional[str] = None
    error_file_id: Optional[str] = None
    created_at: int
    in_progress_at: Optional[int] = None
    expires_at: Optional[int] = None
    finalizing_at: Optional[int] = None
    completed_at: Optional[int] = None
    failed_at: Optional[int] = None
    expired_at: Optional[int] = None
    cancelling_at: Optional[int] = None
    cancelled_at: Optional[int] = None
    request_counts: Dict[str, int] = Field(
        default_factory=lambda: {"total": 0, "completed": 0, "failed": 0}
    )
    metadata: Optional[Dict[str, str]] = None


class BatchListResponse(BaseModel):
    """List of batches."""

    object: str = "list"
    data: List[BatchObject]


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
                "id": "jina-embeddings-v5-text-small",
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

    # SentenceTransformer.encode() handles internal batching via batch_size param
    start_time = time.time()
    batch_size = request.batch_size

    # Build encode kwargs based on task and prompt_name
    encode_kwargs = {
        "task": request.task,
        "normalize_embeddings": True,
        "batch_size": batch_size,
        "convert_to_numpy": True,
    }
    if request.prompt_name is not None:
        encode_kwargs["prompt_name"] = request.prompt_name

    all_embeddings = embedding_model.encode(texts, **encode_kwargs)

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

    # Thread-safe: protect _block_size mutation from concurrent requests
    with reranker_lock:
        original_block_size = getattr(reranker_model, "_block_size", 125)
        reranker_model._block_size = request.batch_size

        results = reranker_model.rerank(
            query=request.query,
            documents=request.documents,
            top_n=request.top_n,
        )

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
# File Endpoints (OpenAI Files API)
# =============================================================================


@app.post("/v1/files", response_model=FileObject)
async def upload_file(file: UploadFile = File(...), purpose: str = "batch"):
    """
    Upload a file for batch processing.
    Expects JSONL format with one request per line.
    """
    if purpose != "batch":
        raise HTTPException(status_code=400, detail="Only 'batch' purpose is supported")

    # Read file content
    content = await file.read()

    # Generate file ID
    file_id = f"file-{uuid.uuid4().hex[:24]}"
    created_at = int(time.time())

    # Store file
    files_storage[file_id] = {
        "id": file_id,
        "bytes": len(content),
        "created_at": created_at,
        "filename": file.filename or "batch.jsonl",
        "purpose": purpose,
        "status": "uploaded",
        "content": content,
    }

    print(f"  [INFO] Uploaded file {file_id}: {file.filename} ({len(content)} bytes)")

    return FileObject(
        id=file_id,
        bytes=len(content),
        created_at=created_at,
        filename=file.filename or "batch.jsonl",
        purpose=purpose,
        status="uploaded",
    )


@app.get("/v1/files", response_model=FileListResponse)
async def list_files(purpose: str = "batch"):
    """List all uploaded files."""
    files = []
    for file_id, file_data in files_storage.items():
        if purpose and file_data.get("purpose") != purpose:
            continue
        files.append(
            FileObject(
                id=file_data["id"],
                bytes=file_data["bytes"],
                created_at=file_data["created_at"],
                filename=file_data["filename"],
                purpose=file_data["purpose"],
                status=file_data.get("status", "uploaded"),
            )
        )
    return FileListResponse(object="list", data=files)


@app.get("/v1/files/{file_id}", response_model=FileObject)
async def get_file(file_id: str):
    """Get file metadata."""
    if file_id not in files_storage:
        raise HTTPException(status_code=404, detail="File not found")

    file_data = files_storage[file_id]
    return FileObject(
        id=file_data["id"],
        bytes=file_data["bytes"],
        created_at=file_data["created_at"],
        filename=file_data["filename"],
        purpose=file_data["purpose"],
        status=file_data.get("status", "uploaded"),
    )


@app.delete("/v1/files/{file_id}")
async def delete_file(file_id: str):
    """Delete a file."""
    if file_id not in files_storage:
        raise HTTPException(status_code=404, detail="File not found")

    del files_storage[file_id]
    print(f"  [INFO] Deleted file {file_id}")
    return {"id": file_id, "object": "file", "deleted": True}


@app.get("/v1/files/{file_id}/content")
async def get_file_content(file_id: str):
    """Get file content (for output files)."""
    if file_id not in files_storage:
        raise HTTPException(status_code=404, detail="File not found")

    file_data = files_storage[file_id]
    content = file_data.get("content", b"")
    return Response(
        content=content,
        media_type="application/jsonl",
        headers={
            "Content-Disposition": f'attachment; filename="{file_data["filename"]}"'
        },
    )


# =============================================================================
# Batch Processing Logic
# =============================================================================


def _build_batch_object(batch: Dict[str, Any]) -> BatchObject:
    """Construct a BatchObject from batch storage dict."""
    return BatchObject(
        id=batch["id"],
        endpoint=batch["endpoint"],
        input_file_id=batch["input_file_id"],
        completion_window=batch["completion_window"],
        status=batch["status"],
        created_at=batch["created_at"],
        in_progress_at=batch.get("in_progress_at"),
        completed_at=batch.get("completed_at"),
        failed_at=batch.get("failed_at"),
        output_file_id=batch.get("output_file_id"),
        error_file_id=batch.get("error_file_id"),
        errors=batch.get("errors"),
        request_counts=batch.get(
            "request_counts", {"total": 0, "completed": 0, "failed": 0}
        ),
        metadata=batch.get("metadata"),
    )


async def process_batch_job(batch_id: str):
    """Background task to process a batch job.

    Optimization: collect all texts from same task/prompt_name group,
    batch-encode them in a single model call, then distribute results.
    """
    global embedding_model

    if batch_id not in batches_storage:
        return

    batch = batches_storage[batch_id]
    input_file_id = batch["input_file_id"]

    if input_file_id not in files_storage:
        batch["status"] = "failed"
        batch["failed_at"] = int(time.time())
        batch["errors"] = {"message": "Input file not found"}
        return

    # Update status to in_progress
    batch["status"] = "in_progress"
    batch["in_progress_at"] = int(time.time())
    print(f"  [BATCH] Starting batch {batch_id}")

    try:
        # Read input file
        input_content = files_storage[input_file_id]["content"].decode("utf-8")
        lines = [
            line.strip() for line in input_content.strip().split("\n") if line.strip()
        ]

        total_requests = len(lines)
        default_batch_size = 32

        # ---- Phase 1: Parse all requests, collect valid embedding tasks ----
        # Group: (task, prompt_name) -> [(line_index, text_list)]
        task_groups: Dict[tuple, List[tuple]] = {}
        # Per-line metadata for reconstruction
        line_meta: List[Dict[str, Any]] = []
        parse_errors: List[tuple] = []  # (index, custom_id, error_msg)

        for i, line in enumerate(lines):
            custom_id = f"request-{i}"
            try:
                request_data = json.loads(line)
                custom_id = request_data.get("custom_id", f"request-{i}")
                body = request_data.get("body", {})

                endpoint = request_data.get("endpoint") or batch.get("endpoint")
                if endpoint != "/v1/embeddings":
                    raise ValueError(f"Unsupported endpoint: {endpoint}")

                # Extract input texts
                input_texts = body.get("input", [])
                if isinstance(input_texts, str):
                    input_texts = [input_texts]
                if not input_texts or embedding_model is None:
                    raise ValueError("No input texts or model not loaded")

                # Resolve task/prompt_name
                task = body.get("task", "retrieval")
                prompt_name = body.get("prompt_name", None)
                if task in TASK_ALIAS_MAP:
                    task, prompt_name = TASK_ALIAS_MAP[task]
                if task not in VALID_EMBEDDING_TASKS:
                    raise ValueError(
                        f"Invalid task '{task}'. Must be one of: {VALID_EMBEDDING_TASKS}"
                    )
                if prompt_name is not None and prompt_name not in VALID_PROMPT_NAMES:
                    raise ValueError(
                        f"Invalid prompt_name '{prompt_name}'. Must be one of: {VALID_PROMPT_NAMES}"
                    )

                group_key = (task, prompt_name)
                if group_key not in task_groups:
                    task_groups[group_key] = []
                task_groups[group_key].append((i, input_texts))

                line_meta.append(
                    {
                        "index": i,
                        "custom_id": custom_id,
                        "group_key": group_key,
                        "text_count": len(input_texts),
                        "model": body.get("model", "jina-embeddings-v5-text-small"),
                        "texts": input_texts,
                    }
                )

            except Exception as e:
                parse_errors.append((i, custom_id, str(e)))

        # ---- Phase 2: Batch-encode per (task, prompt_name) group ----
        # Maps: line_index -> list[np.ndarray] (embeddings for that line)
        embeddings_by_line: Dict[int, list] = {}

        for (task, prompt_name), entries in task_groups.items():
            # Flatten all texts in this group, tracking which line they belong to
            flat_texts: List[str] = []
            line_spans: List[tuple] = []  # (line_index, start, end)
            offset = 0
            for line_idx, texts in entries:
                flat_texts.extend(texts)
                line_spans.append((line_idx, offset, offset + len(texts)))
                offset += len(texts)

            encode_kwargs = {
                "task": task,
                "normalize_embeddings": True,
                "batch_size": default_batch_size,
                "convert_to_numpy": True,
            }
            if prompt_name is not None:
                encode_kwargs["prompt_name"] = prompt_name

            all_embs = embedding_model.encode(flat_texts, **encode_kwargs)

            # Distribute embeddings back to their originating lines
            for line_idx, start, end in line_spans:
                embeddings_by_line[line_idx] = all_embs[start:end]

        # ---- Phase 3: Build results ----
        results: List[Dict[str, Any]] = []
        completed = 0
        failed = len(parse_errors)

        # Successful lines (in original order)
        for meta in line_meta:
            idx = meta["index"]
            embs = embeddings_by_line.get(idx, [])
            response_data = []
            total_tokens = 0
            for emb_idx, emb in enumerate(embs):
                response_data.append(
                    {
                        "object": "embedding",
                        "index": emb_idx,
                        "embedding": emb.tolist(),
                    }
                )
                total_tokens += len(meta["texts"][emb_idx].split()) * 2

            results.append(
                {
                    "id": f"resp-{uuid.uuid4().hex[:24]}",
                    "custom_id": meta["custom_id"],
                    "response": {
                        "status_code": 200,
                        "body": {
                            "object": "list",
                            "data": response_data,
                            "model": meta["model"],
                            "usage": {
                                "prompt_tokens": total_tokens,
                                "total_tokens": total_tokens,
                            },
                        },
                    },
                    "error": None,
                }
            )
            completed += 1

        # Failed lines from parse errors
        for err_idx, custom_id, err_msg in parse_errors:
            results.append(
                {
                    "id": f"resp-{uuid.uuid4().hex[:24]}",
                    "custom_id": custom_id,
                    "response": None,
                    "error": {"message": err_msg, "type": "processing_error"},
                }
            )

        # Sort results by original line order
        results.sort(key=lambda r: r["custom_id"])

        # Update progress
        batch["request_counts"] = {
            "total": total_requests,
            "completed": completed,
            "failed": failed,
        }

        # ---- Phase 4: Create output file ----
        output_content = "\n".join(json.dumps(r) for r in results)
        output_bytes = output_content.encode("utf-8")
        output_file_id = f"file-{uuid.uuid4().hex[:24]}"
        files_storage[output_file_id] = {
            "id": output_file_id,
            "bytes": len(output_bytes),
            "created_at": int(time.time()),
            "filename": f"batch_{batch_id}_output.jsonl",
            "purpose": "batch_output",
            "status": "uploaded",
            "content": output_bytes,
        }

        # Update batch status
        batch["status"] = "completed"
        batch["completed_at"] = int(time.time())
        batch["output_file_id"] = output_file_id
        batch["request_counts"] = {
            "total": total_requests,
            "completed": completed,
            "failed": failed,
        }

        print(
            f"  [BATCH] Completed batch {batch_id}: {completed}/{total_requests} succeeded, {failed} failed"
        )

    except Exception as e:
        batch["status"] = "failed"
        batch["failed_at"] = int(time.time())
        batch["errors"] = {"message": str(e)}
        print(f"  [BATCH] Failed batch {batch_id}: {e}")


# =============================================================================
# Batch Endpoints (OpenAI Batch API)
# =============================================================================


@app.post("/v1/batches", response_model=BatchObject)
async def create_batch(request: BatchRequest, background_tasks: BackgroundTasks):
    """Create a new batch job."""
    if request.input_file_id not in files_storage:
        raise HTTPException(status_code=404, detail="Input file not found")

    if request.endpoint != "/v1/embeddings":
        raise HTTPException(
            status_code=400, detail="Only /v1/embeddings endpoint is supported"
        )

    # Generate batch ID
    batch_id = f"batch_{uuid.uuid4().hex[:24]}"
    created_at = int(time.time())

    # Create batch
    batch = {
        "id": batch_id,
        "endpoint": request.endpoint,
        "input_file_id": request.input_file_id,
        "completion_window": request.completion_window,
        "status": "validating",
        "created_at": created_at,
        "metadata": request.metadata,
        "request_counts": {"total": 0, "completed": 0, "failed": 0},
    }
    batches_storage[batch_id] = batch

    print(f"  [BATCH] Created batch {batch_id} with input file {request.input_file_id}")

    # Start background processing
    background_tasks.add_task(process_batch_job, batch_id)

    return _build_batch_object(batch)


@app.get("/v1/batches", response_model=BatchListResponse)
async def list_batches(limit: int = 20):
    """List all batch jobs."""
    batches = [
        _build_batch_object(batch_data) for batch_data in batches_storage.values()
    ]
    return BatchListResponse(object="list", data=batches[:limit])


@app.get("/v1/batches/{batch_id}", response_model=BatchObject)
async def get_batch(batch_id: str):
    """Get batch job status."""
    if batch_id not in batches_storage:
        raise HTTPException(status_code=404, detail="Batch not found")

    batch = batches_storage[batch_id]
    return _build_batch_object(batch)


@app.post("/v1/batches/{batch_id}/cancel", response_model=BatchObject)
async def cancel_batch(batch_id: str):
    """Cancel a batch job."""
    if batch_id not in batches_storage:
        raise HTTPException(status_code=404, detail="Batch not found")

    batch = batches_storage[batch_id]

    if batch["status"] in ["completed", "failed", "cancelled", "expired"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel batch with status: {batch['status']}",
        )

    batch["status"] = "cancelled"
    batch["cancelled_at"] = int(time.time())

    print(f"  [BATCH] Cancelled batch {batch_id}")

    return _build_batch_object(batch)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    print("""
    ============================================================
         Jina Embedding & Reranker Server
    ============================================================
      Endpoints:
         POST /v1/embeddings  - Create embeddings
         POST /v1/rerank      - Rerank documents
         POST /v1/files       - Upload batch file
         GET  /v1/files       - List files
         GET  /v1/files/{id}  - Get file info
         GET  /v1/files/{id}/content - Download file
         POST /v1/batches     - Create batch job
         GET  /v1/batches     - List batches
         GET  /v1/batches/{id}- Get batch status
         GET  /v1/models      - List models
         GET  /               - Health check
    ============================================================
      Server running on http://0.0.0.0:8000
    ============================================================
    """)

    uvicorn.run(
        "jina_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
