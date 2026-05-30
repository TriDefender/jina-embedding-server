"""
Jina Embedding & Reranker Server
OpenAI-compatible API for embeddings + reranking

Endpoints:
  - POST /v1/embeddings  (OpenAI compatible)
  - POST /v1/rerank      (Jina/Cohere style)
  - POST /v1/files       (OpenAI Files API)
  - POST /v1/batches     (OpenAI Batch API)
"""

import asyncio
import gc
import os
import sys
import time
import uuid
import json
import threading
from typing import List, Optional, Union, Dict, Any
from contextlib import asynccontextmanager
import platform

# =============================================================================
# Thread configuration — MUST be set before torch import
# =============================================================================
# Tuned for AMD R9 9900X (12C/24T, Zen 5)
# Use physical cores only; SMT yields negligible gains for dense matmul
MAX_THREADS = 12
os.environ["OMP_NUM_THREADS"] = str(MAX_THREADS)
os.environ["MKL_NUM_THREADS"] = str(MAX_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(MAX_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(MAX_THREADS)
os.environ["TORCH_INTEROP_THREADS"] = "8"
# CUDA allocator config: must be set BEFORE torch import to take effect.
# expandable_segments reduces VRAM fragmentation but is only supported on Linux.
# On Windows, this setting is silently ignored (with a UserWarning).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

torch.set_num_threads(MAX_THREADS)

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import Response
from pydantic import BaseModel, Field, field_validator, model_validator
from sentence_transformers import SentenceTransformer

# =============================================================================
# Device & CUDA Configuration
# =============================================================================
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
IDLE_TIMEOUT_SECONDS = int(os.environ.get("IDLE_TIMEOUT_SECONDS", "300"))
# torch.compile on GPU: gives ~2.8x speedup via operator fusion and CUDA graphs.
# Adds ~10-30s to first load (JIT compilation) but dramatically reduces inference latency.
# Disable for fastest reload (~1-2s) at the cost of slower inference.
COMPILE_ON_GPU = os.environ.get("COMPILE_ON_GPU", "1") == "1"
# CUDA Graph for reranker backbone: captures the Qwen3 transformer as CUDA graphs
# per seq_len bucket, eliminating kernel launch overhead (~10-25% latency reduction).
# Post-processing (projector + cosine scoring) runs eagerly (variable-shape ops).
# Lazy capture: graphs are recorded on first encounter with a new bucket size.
CUDA_GRAPH = os.environ.get("CUDA_GRAPH", "0") == "1" and CUDA_AVAILABLE


def setup_cuda_optimizations():
    """Enable NVIDIA GPU optimizations when CUDA is available."""
    if not CUDA_AVAILABLE:
        return {}
    cap = torch.cuda.get_device_capability()
    info = {
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_memory_gb": round(
            torch.cuda.get_device_properties(0).total_memory / (1024**3), 1
        ),
        "cuda_version": torch.version.cuda,
        "compute_capability": f"{cap[0]}.{cap[1]}",
    }
    # TF32 for Ampere+ (compute capability >= 8.0) — ~2x matmul throughput
    if cap >= (8, 0):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        info["tf32_enabled"] = True
    else:
        info["tf32_enabled"] = False
    # cuDNN autotuner: finds optimal convolution algorithms for fixed input sizes
    torch.backends.cudnn.benchmark = True
    return info


def detect_attention_implementation():
    """Detect best available attention implementation.

    Priority: sdpa > eager
    SDPA (Scaled Dot Product Attention) is PyTorch's native fused attention,
    available in 2.0+ with CUDA. Benchmarked faster than flash-attn on this
    workload.
    """
    if not CUDA_AVAILABLE:
        return "eager"
    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        return "sdpa"
    return "eager"


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


# =============================================================================
# GPU Model Manager — Idle Offloading & Fast Reload
# =============================================================================


class GPUModelManager:
    """Manages a model's GPU lifecycle: offloading when idle, reloading on demand.

    When CUDA is available:
    - Models are loaded to GPU at startup
    - After IDLE_TIMEOUT_SECONDS of no inference, model is moved to CPU RAM (VRAM freed)
    - On next inference request, model is moved back to GPU
    - Parameters are pinned in CPU memory during offload for faster DMA transfer on reload
    - asyncio.Lock ensures safe device transitions

    When CUDA is not available:
    - All methods are no-ops; model stays on CPU permanently

    torch.compile interaction:
    - On CPU: model may be compiled (OptimizedModule). Since DEVICE="cpu", manager is no-op.
    - On GPU with COMPILE_ON_GPU=0: model is raw. Offload/reload work directly.
    - On GPU with COMPILE_ON_GPU=1: model is compiled. OptimizedModule.to() propagates
      to underlying parameters correctly in PyTorch 2.x.
    """

    def __init__(self, model, name: str):
        self.model = model
        self.name = name
        self._on_gpu = DEVICE.type == "cuda"
        self._last_access = time.monotonic()
        self._lock = asyncio.Lock()

    @property
    def is_on_gpu(self) -> bool:
        return self._on_gpu

    @property
    def last_access(self) -> float:
        return self._last_access

    def _pin_cpu_parameters(self):
        """Pin model parameters in CPU memory for potential DMA transfer speedup.

        Pinned (page-locked) memory enables direct DMA transfer to GPU,
        bypassing the internal staging buffer. Note: the actual speedup for
        synchronous model.to() is modest (~10-30%); real gains come from
        non_blocking=True with CUDA stream overlap, which is not used here
        due to simplicity. Kept as a best-effort optimization.
        """
        if not CUDA_AVAILABLE:
            return
        try:
            for param in self.model.parameters():
                if param.device.type == "cpu" and not param.is_pinned():
                    param.data = param.data.pin_memory()
            for buf in self.model.buffers():
                if buf.device.type == "cpu" and not buf.is_pinned():
                    buf.data = buf.data.pin_memory()
        except Exception as e:
            print(f"  [WARN] pin_memory failed for '{self.name}': {e}")

    async def ensure_on_device(self):
        """Ensure model is on the target device (GPU or CPU).
        Called before every inference. Reloads to GPU if offloaded."""
        if DEVICE.type != "cuda":
            self._last_access = time.monotonic()
            return
        async with self._lock:
            if not self._on_gpu:
                t0 = time.monotonic()
                self.model.to(DEVICE)
                # model.to() with default non_blocking=False already
                # calls cudaStreamSynchronize internally per parameter.
                # Release orphaned CPU pinned tensors left after transfer.
                gc.collect()
                self._on_gpu = True
                elapsed = time.monotonic() - t0
                print(f"  [GPU] Reloaded '{self.name}' to GPU in {elapsed:.2f}s")
            self._last_access = time.monotonic()

    async def offload_to_cpu(self):
        """Move model from GPU to CPU RAM to free VRAM.

        After moving to CPU, parameters are pinned in page-locked memory
        to enable faster DMA transfer when reloading to GPU later.
        """
        if DEVICE.type != "cuda":
            return
        async with self._lock:
            if not self._on_gpu:
                return
            self.model.to("cpu")
            # Pin parameters for faster reload via DMA
            self._pin_cpu_parameters()
            gc.collect()
            torch.cuda.empty_cache()
            self._on_gpu = False
            print(f"  [GPU] Offloaded '{self.name}' to CPU (VRAM freed, params pinned)")

    def get_status(self) -> dict:
        """Return model status for health endpoint."""
        loaded = self.model is not None
        return {
            "loaded": loaded,
            "on_gpu": self._on_gpu if loaded else False,
        }


# ---------------------------------------------------------------------------
# CUDA Graph Acceleration for Reranker Backbone
# ---------------------------------------------------------------------------


class _CUDAGraphReranker:
    """CUDA Graph wrapper for JinaForRanking's transformer backbone.

    Strategy: capture CUDA Graphs for the Qwen3 backbone (heavy compute,
    fixed output shapes) and run the projector/scoring in eager mode
    (lightweight, variable shapes due to boolean indexing).

    Uses lazy capture: graphs are recorded on first encounter with a new
    sequence length bucket.  Falls back to eager for oversized inputs.
    """

    # Bucket sizes for typical reranker prompt lengths.
    # Each bucket captures a separate graph; memory ≈ seq_len × hidden_dim × 2 (BF16).
    SEQ_LEN_BUCKETS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]

    def __init__(self, model):
        self.model = model
        # Set lm_head to Identity once (normally done per forward call)
        self.model.lm_head = torch.nn.Identity()

        self._pool = torch.cuda.graph_pool_handle()
        self._graphs: Dict[int, dict] = {}
        self._eager_fallback = False

        # Import helpers from the dynamically-loaded model module
        model_module = sys.modules[model.__class__.__module__]
        self._output_cls = model_module.CausalLMOutputWithScores
        self._format_fn = model_module.format_docs_prompts_func

    def _find_bucket(self, seq_len: int):
        """Return the smallest bucket >= *seq_len*, or None if too large."""
        for b in self.SEQ_LEN_BUCKETS:
            if b >= seq_len:
                return b
        return None

    def _capture(self, seq_len: int):
        """Capture a CUDA Graph for the Qwen3 backbone at *seq_len*."""
        g = torch.cuda.CUDAGraph()
        static_ids = torch.zeros(1, seq_len, dtype=torch.long, device=DEVICE)
        static_mask = torch.ones(1, seq_len, dtype=torch.long, device=DEVICE)

        with torch.no_grad(), torch.cuda.graph(g, pool=self._pool):
            backbone_out = self.model.model(
                input_ids=static_ids,
                attention_mask=static_mask,
                use_cache=False,
                output_hidden_states=True,
            )
            static_hidden = backbone_out.hidden_states[-1]

        self._graphs[seq_len] = {
            "graph": g,
            "input_ids": static_ids,
            "attention_mask": static_mask,
            "hidden_states": static_hidden,
        }
        print(f"      [CUDA Graph] Captured reranker backbone: seq_len={seq_len}")

    @torch.no_grad()
    def compute_single_batch(self, query, docs, instruction=None):
        """CUDA Graph-accelerated replacement for _compute_single_batch."""
        if self._eager_fallback:
            return self.model._compute_single_batch_eager(query, docs, instruction)

        self.model._ensure_tokenizer()
        device = next(self.model.parameters()).device

        prompt = self._format_fn(
            query,
            docs,
            instruction=instruction,
            special_tokens=self.model.special_tokens,
            no_thinking=True,
        )

        batch = self.model._tokenizer(
            text=[prompt],
            padding=True,
            padding_side="left",
            return_tensors="pt",
        ).to(device)

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        bs, seq_len = input_ids.shape

        # Find bucket — fall back to eager if input exceeds largest bucket
        bucket = self._find_bucket(seq_len)
        if bucket is None:
            return self._compute_eager_from_ids(input_ids, attention_mask)

        # Capture graph on first encounter
        if bucket not in self._graphs:
            try:
                self._capture(bucket)
            except Exception as e:
                print(f"      [WARN] CUDA Graph capture failed (seq_len={bucket}): {e}")
                print("      [WARN] Falling back to eager mode for reranker")
                self._eager_fallback = True
                return self._compute_eager_from_ids(input_ids, attention_mask)

        entry = self._graphs[bucket]

        # Copy inputs into static buffers (preserves memory addresses)
        entry["input_ids"][:, :seq_len].copy_(input_ids)
        if seq_len < bucket:
            entry["input_ids"][:, seq_len:].zero_()
        entry["attention_mask"][:, :seq_len].copy_(attention_mask)
        if seq_len < bucket:
            entry["attention_mask"][:, seq_len:].zero_()

        # ── Replay backbone CUDA Graph ──
        entry["graph"].replay()

        # Extract hidden states for actual sequence length
        hidden_states = entry["hidden_states"][:, :seq_len, :]
        dim = hidden_states.shape[-1]

        # ── Eager post-processing (variable-shape ops: boolean indexing) ──
        query_idx = torch.eq(input_ids, self.model.query_embed_token_id)
        doc_idx = torch.eq(input_ids, self.model.doc_embed_token_id)

        doc_embeds = hidden_states[doc_idx].view(bs, -1, dim)
        query_embeds = hidden_states[query_idx].unsqueeze(1)

        doc_embeds = self.model.projector(doc_embeds)
        query_embeds = self.model.projector(query_embeds)

        query_expanded = query_embeds.expand_as(doc_embeds)
        scores = torch.nn.functional.cosine_similarity(
            doc_embeds, query_expanded, dim=-1
        ).squeeze(-1)

        return self._output_cls(
            loss=None,
            logits=None,
            scores=scores,
            query_embeds=query_embeds,
            doc_embeds=doc_embeds,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    def _compute_eager_from_ids(self, input_ids, attention_mask):
        """Eager fallback via the original forward method."""
        return self.model.forward(input_ids=input_ids, attention_mask=attention_mask)

    def warmup(self):
        """Pre-capture a medium-sized graph during startup."""
        if self._eager_fallback:
            return
        try:
            if 512 not in self._graphs:
                self._capture(512)
        except Exception as e:
            print(f"      [WARN] CUDA Graph warmup failed: {e}")
            self._eager_fallback = True


def _enable_cudagraph_reranker(model):
    """Enable CUDA Graph acceleration on a JinaForRanking model.

    Monkey-patches ``_compute_single_batch`` with a version that captures
    the Qwen3 backbone for fixed-shape buckets and runs projector/scoring
    eagerly.
    """
    state = _CUDAGraphReranker(model)
    # Save original for eager fallback
    model._compute_single_batch_eager = model._compute_single_batch
    # Replace with CUDA Graph version
    model._compute_single_batch = (
        lambda query, docs, instruction=None: state.compute_single_batch(
            query, docs, instruction
        )
    )
    return state


# Check CPU capabilities
def check_cpu_capabilities():
    """Check CPU instruction set support using py-cpuinfo."""
    import cpuinfo

    info = cpuinfo.get_cpu_info()
    flags = info.get("flags", [])

    cpu_info = {
        "cpu_count": os.cpu_count(),
        "pytorch_threads": torch.get_num_threads(),
        "platform": platform.platform(),
        "cpu_brand": info.get("brand_raw", "Unknown"),
        "avx": "avx" in flags,
        "avx2": "avx2" in flags,
        "avx512f": "avx512f" in flags,
        "avx512_vnni": "avx512vnni" in flags,
    }
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

# GPU model managers (initialized in lifespan)
embedding_manager: Optional[GPUModelManager] = None
reranker_manager: Optional[GPUModelManager] = None

# In-memory storage for Files and Batches
files_storage: Dict[str, Dict[str, Any]] = {}  # file_id -> file metadata + content
batches_storage: Dict[str, Dict[str, Any]] = {}  # batch_id -> batch metadata

# ---------------------------------------------------------------------------
# Dynamic batching infrastructure for /v1/embeddings
# ---------------------------------------------------------------------------
BATCH_WINDOW_MS = 50  # How long to wait before firing a batch (ms)
BATCH_MAX_SIZE = 64  # Max requests to accumulate before firing early


class _PendingEmbedRequest:
    """Holds a single request awaiting batch encoding."""

    __slots__ = ("texts", "task", "prompt_name", "batch_size", "future")

    def __init__(self, texts, task, prompt_name, batch_size, future):
        self.texts = texts
        self.task = task
        self.prompt_name = prompt_name
        self.batch_size = batch_size
        self.future = future


_pending_embeddings: list[_PendingEmbedRequest] = []
_batch_flush_lock: asyncio.Lock | None = None
_batch_timer_handle: asyncio.TimerHandle | None = None


async def _flush_embedding_batch():
    """Drain pending requests, group by (task, prompt_name), batch-encode."""
    global _pending_embeddings
    if not _pending_embeddings:
        return
    # Ensure model is on GPU before encoding
    if embedding_manager:
        await embedding_manager.ensure_on_device()
    # Grab everything in the queue
    batch = _pending_embeddings[:]
    _pending_embeddings = []

    # Group by (task, prompt_name)
    groups: Dict[tuple, List[_PendingEmbedRequest]] = {}
    for req in batch:
        key = (req.task, req.prompt_name)
        groups.setdefault(key, []).append(req)

    for (task, prompt_name), reqs in groups.items():
        try:
            # Flatten texts, track which request they belong to
            flat_texts: List[str] = []
            spans: List[tuple] = []  # (req_index, start, end)
            offset = 0
            for idx, r in enumerate(reqs):
                flat_texts.extend(r.texts)
                spans.append((idx, offset, offset + len(r.texts)))
                offset += len(r.texts)

            max_bs = max(r.batch_size for r in reqs)
            encode_kwargs = _build_encode_kwargs(task, prompt_name, max_bs)

            all_embs = embedding_model.encode(flat_texts, **encode_kwargs)

            # Distribute results back
            for req_idx, start, end in spans:
                reqs[req_idx].future.set_result(all_embs[start:end])
        except Exception as e:
            # If encoding fails, propagate to all requests in group
            for r in reqs:
                if not r.future.done():
                    r.future.set_exception(e)


def _schedule_batch_flush():
    """Schedule a flush after BATCH_WINDOW_MS unless one is already pending."""
    global _batch_timer_handle
    loop = asyncio.get_running_loop()
    if _batch_timer_handle is not None and not _batch_timer_handle.cancelled():
        return  # Already scheduled
    _batch_timer_handle = loop.call_later(
        BATCH_WINDOW_MS / 1000.0,
        lambda: loop.create_task(_safe_flush()),
    )


async def _safe_flush():
    """Flush with lock to prevent concurrent flushes."""
    global _batch_timer_handle
    if _batch_flush_lock is None:
        return
    async with _batch_flush_lock:
        _batch_timer_handle = None  # Allow next request to schedule new timer
        await _flush_embedding_batch()


def _build_encode_kwargs(
    task: str, prompt_name: Optional[str], batch_size: int
) -> Dict[str, Any]:
    """Build keyword arguments for SentenceTransformer.encode()."""
    kwargs: Dict[str, Any] = {
        "task": task,
        "normalize_embeddings": True,
        "batch_size": batch_size,
        "convert_to_numpy": False,
    }
    if prompt_name is not None:
        kwargs["prompt_name"] = prompt_name
    return kwargs


# =============================================================================
# Lifespan
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup with CUDA support and idle offloading."""
    global embedding_model, reranker_model, _batch_flush_lock
    global embedding_manager, reranker_manager

    # Setup CUDA optimizations
    gpu_info = setup_cuda_optimizations()
    attn_impl = detect_attention_implementation()

    # Display system information
    cpu_caps = check_cpu_capabilities()
    print("=" * 60)
    print("System Information")
    print("=" * 60)
    if CUDA_AVAILABLE:
        print("  CUDA:            Available")
        print(f"  GPU:             {gpu_info.get('gpu_name', 'Unknown')}")
        print(f"  GPU Memory:      {gpu_info.get('gpu_memory_gb', 0)} GB")
        print(f"  CUDA Version:    {gpu_info.get('cuda_version', 'N/A')}")
        print(f"  Compute Cap:     {gpu_info.get('compute_capability', 'N/A')}")
        print(f"  TF32:            {gpu_info.get('tf32_enabled', False)}")
        print(f"  Attention:       {attn_impl}")
    else:
        print("  CUDA:            Not available (CPU mode)")
    print(f"  Device:          {DEVICE}")
    print(f"  Platform:        {cpu_caps['platform']}")
    print(f"  CPU:             {cpu_caps['cpu_brand']}")
    print(f"  CPU Cores:       {cpu_caps['cpu_count']}")
    print(f"  PyTorch Threads: {cpu_caps['pytorch_threads']}")
    print(f"  AVX2:            {cpu_caps['avx2']}")
    print(f"  AVX-512:         {cpu_caps['avx512']}")
    print(f"  Idle Timeout:    {IDLE_TIMEOUT_SECONDS}s")
    if CUDA_AVAILABLE:
        print(f"  Compile on GPU:  {COMPILE_ON_GPU}")
        print(f"  CUDA Graph:      {CUDA_GRAPH}")
    print("=" * 60)

    print("\nLoading models...")
    print("=" * 60)

    # Load embedding model
    print(f"\n[1/2] Loading embedding model from: {EMBEDDING_MODEL_PATH}")
    print(f"      Attention implementation: {attn_impl}")
    try:
        embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_PATH,
            trust_remote_code=True,
            device=DEVICE,
            default_prompt_name="query",
            model_kwargs={
                "default_task": "retrieval",
                "dtype": torch.bfloat16,
                "attn_implementation": attn_impl,
            },
        )
        print(
            f"      [OK] Embedding model loaded on {DEVICE} (dim={embedding_model.get_sentence_embedding_dimension()})"
        )
    except Exception as e:
        print(f"      [FAIL] Failed to load embedding model: {e}")
        raise

    # torch.compile optimization
    # - CPU: always compile (essential for performance)
    # - GPU: optional (COMPILE_ON_GPU=1). Adds ~30-50% throughput but makes
    #   first reload ~10-30s slower due to JIT recompilation after model.to().
    #   Default: off for fastest reload (~1-2s, just parameter transfer).
    should_compile = DEVICE.type == "cpu" or COMPILE_ON_GPU
    if DEVICE.type == "cuda":
        print(f"      compile_on_gpu: {COMPILE_ON_GPU}")
    if should_compile:
        print(f"\n[OPTIMIZE] torch.compile() on embedding model ({DEVICE})...")
        try:
            # Refer to 'https://docs.pytorch.org/docs/stable/generated/torch.compile.html' for more information on torch.compile.
            embedding_model = torch.compile(embedding_model, dynamic=True, mode="max-autotune")
            print("      [OK] torch.compile() applied")
        except Exception as e:
            print(f"      [WARN] torch.compile() failed (falling back to eager): {e}")
    else:
        print("\n[OPTIMIZE] Skipping torch.compile() on GPU (fastest reload)")

    # Load reranker model
    _cudagraph_reranker_state = None
    print(f"\n[2/2] Loading reranker model from: {RERANKER_MODEL_PATH}")
    print(f"      Attention implementation: {attn_impl}")
    try:
        from transformers import AutoModel

        reranker_model = AutoModel.from_pretrained(
            RERANKER_MODEL_PATH,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        )
        reranker_model.eval()
        reranker_model.to(DEVICE)

        # Apply Triton flash attention kernels (fused online-softmax, bf16 tensor cores)
        if CUDA_AVAILABLE:
            try:
                from optimized_kernels import patch_reranker_attention
                patch_reranker_attention(reranker_model)
            except Exception as e:
                print(f"      [WARN] Triton kernel patch failed (falling back to {attn_impl}): {e}")

        if CUDA_GRAPH:
            _cudagraph_reranker_state = _enable_cudagraph_reranker(reranker_model)
            print(f"      [OK] Reranker loaded on {DEVICE} + CUDA Graph backbone")
        elif should_compile:
            reranker_model = torch.compile(reranker_model, dynamic=True, mode="max-autotune")
            print(f"      [OK] Reranker model loaded and compiled ({DEVICE})")
        else:
            print(f"      [OK] Reranker model loaded on {DEVICE}")
    except Exception as e:
        print(f"      [FAIL] Failed to load reranker model: {e}")
        raise

    # Initialize GPU model managers
    embedding_manager = GPUModelManager(embedding_model, "embedding")
    reranker_manager = GPUModelManager(reranker_model, "reranker")

    # Initialize dynamic batching lock (needs running event loop)
    _batch_flush_lock = asyncio.Lock()

    print("\n" + "=" * 60)
    print(f"Server ready! (device={DEVICE})")
    print("=" * 60)

    # Warmup: pre-allocate memory, trigger JIT compilation, and capture CUDA Graphs
    print("\n[WARMUP] Pre-warming models...")
    _ = embedding_model.encode(
        [
            "dummy_text_here_put_your_string_lol1234567890 vgyfsFewwg4rgeghrafW	EDDDDD₫fvvv俄国v恶个过程各方位"
        ],
        task="retrieval",
        prompt_name="query",
        convert_to_numpy=False,
        normalize_embeddings=True,
    )
    # Pre-capture a CUDA Graph bucket for the reranker (if enabled)
    if CUDA_GRAPH and _cudagraph_reranker_state:
        _cudagraph_reranker_state.warmup()
    try:
        _ = reranker_model.rerank(
            "warmup query", ["dummy_text_here_put_your_string_lol"], top_n=1
        )
    except Exception:
        pass
    if CUDA_AVAILABLE:
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        print("      [OK] Models pre-warmed")
        print(
            f"      GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved"
        )
    else:
        print("      [OK] Models pre-warmed")

    # Start idle offload watcher (GPU only)
    async def _idle_offload_watcher():
        """Background task: offload models to CPU after idle timeout."""
        if DEVICE.type != "cuda":
            return
        while True:
            await asyncio.sleep(30)  # Check every 30 seconds
            now = time.monotonic()
            for mgr in [embedding_manager, reranker_manager]:
                if mgr and mgr.is_on_gpu:
                    idle_seconds = now - mgr.last_access
                    if idle_seconds > IDLE_TIMEOUT_SECONDS:
                        await mgr.offload_to_cpu()

    watcher_task = asyncio.create_task(_idle_offload_watcher())

    yield

    # Cleanup
    watcher_task.cancel()
    print("Shutting down...")


app = FastAPI(
    title="Jina Embedding & Reranker Server",
    description="OpenAI-compatible API for embeddings + reranking",
    version="1.5.0",
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
        default=32, ge=1, le=256, description="Batch size for reranking"
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
    """Health check with GPU status."""
    status = {
        "status": "ok",
        "device": str(DEVICE),
        "cuda_available": CUDA_AVAILABLE,
        "cuda_graph_reranker": CUDA_GRAPH,
        "models": {},
    }
    if embedding_manager:
        status["models"]["embedding"] = embedding_manager.get_status()
    else:
        status["models"]["embedding"] = {
            "loaded": embedding_model is not None,
            "on_gpu": False,
        }
    if reranker_manager:
        status["models"]["reranker"] = reranker_manager.get_status()
    else:
        status["models"]["reranker"] = {
            "loaded": reranker_model is not None,
            "on_gpu": False,
        }
    return status


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
    OpenAI-compatible endpoint with dynamic batching.
    """
    if embedding_model is None:
        raise HTTPException(status_code=503, detail="Embedding model not loaded")

    # Normalize input to list
    texts = [request.input] if isinstance(request.input, str) else request.input

    if not texts:
        raise HTTPException(status_code=400, detail="Input cannot be empty")

    start_time = time.time()

    # Submit to dynamic batch queue when available and request is small enough
    if _batch_flush_lock is not None and len(texts) <= BATCH_MAX_SIZE:
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        _pending_embeddings.append(
            _PendingEmbedRequest(
                texts=texts,
                task=request.task,
                prompt_name=request.prompt_name,
                batch_size=request.batch_size,
                future=future,
            )
        )

        # Flush immediately if we hit max batch size, otherwise schedule timer
        if len(_pending_embeddings) >= BATCH_MAX_SIZE:
            loop.create_task(_safe_flush())
        else:
            _schedule_batch_flush()

        all_embeddings = await future
    else:
        # Fallback: encode directly (large single requests or before lifespan init)
        if embedding_manager:
            await embedding_manager.ensure_on_device()
        encode_kwargs = _build_encode_kwargs(
            request.task, request.prompt_name, request.batch_size
        )
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
            "batch_size": request.batch_size,
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

    # Ensure reranker is on GPU before inference
    if reranker_manager:
        await reranker_manager.ensure_on_device()

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
    rerank_results = [
        RerankResult(
            index=r["index"],
            relevance_score=r["relevance_score"],
            document=r["document"] if request.return_documents else None,
        )
        for r in results
    ]

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


def _file_object_from_storage(file_data: Dict[str, Any]) -> FileObject:
    """Construct a FileObject from files_storage dict."""
    return FileObject(
        id=file_data["id"],
        bytes=file_data["bytes"],
        created_at=file_data["created_at"],
        filename=file_data["filename"],
        purpose=file_data["purpose"],
        status=file_data.get("status", "uploaded"),
    )


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
    filename = file.filename or "batch.jsonl"

    # Generate file ID
    file_id = f"file-{uuid.uuid4().hex[:24]}"
    created_at = int(time.time())

    # Store file
    files_storage[file_id] = {
        "id": file_id,
        "bytes": len(content),
        "created_at": created_at,
        "filename": filename,
        "purpose": purpose,
        "status": "uploaded",
        "content": content,
    }

    print(f"  [INFO] Uploaded file {file_id}: {filename} ({len(content)} bytes)")

    return FileObject(
        id=file_id,
        bytes=len(content),
        created_at=created_at,
        filename=filename,
        purpose=purpose,
        status="uploaded",
    )


@app.get("/v1/files", response_model=FileListResponse)
async def list_files(purpose: str = "batch"):
    """List all uploaded files."""
    files = [
        _file_object_from_storage(file_data)
        for file_data in files_storage.values()
        if not purpose or file_data.get("purpose") == purpose
    ]
    return FileListResponse(object="list", data=files)


@app.get("/v1/files/{file_id}", response_model=FileObject)
async def get_file(file_id: str):
    """Get file metadata."""
    if file_id not in files_storage:
        raise HTTPException(status_code=404, detail="File not found")

    file_data = files_storage[file_id]
    return _file_object_from_storage(file_data)


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

    # Ensure model is on GPU before batch encoding
    if embedding_manager:
        await embedding_manager.ensure_on_device()

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
        # Maps: line_index -> list[torch.Tensor] (embeddings for that line)
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

            encode_kwargs = _build_encode_kwargs(task, prompt_name, default_batch_size)

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
