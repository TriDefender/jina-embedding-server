# Jina Embedding & Reranker Server

OpenAI-compatible API server for Jina embeddings and reranking models.

## Features

- **Embedding**: jina-embeddings-v5-text-small (with LoRA task adapters)
- **Reranking**: jina-reranker-v3
- **OpenAI-compatible API**: Works with existing OpenAI client libraries
- **Task-adaptive embeddings**: Switch LoRA adapter per request (`retrieval`, `text-matching`, `clustering`, `classification`)

## Installation

```bash
uv sync

Fetch_Models.bat  #This is for windows, it contains two 'hf download' commands, adapt for linux accordingly
```

### ONNX Backend (Recommended)

The server uses **ONNX Runtime** for embedding inference by default, with PyTorch as fallback. ONNX provides faster inference on CPU by leveraging optimized operators and (on supported hardware) INT8 quantization via AVX-512 VNNI.

**Download ONNX models** (4 task-specific variants with LoRA adapters merged):

```bash
# Already included in Fetch_Models.bat — or run individually:
hf download jinaai/jina-embeddings-v5-text-small-retrieval --local-dir ".\jinaai\jina-embeddings-v5-text-small-retrieval"
hf download jinaai/jina-embeddings-v5-text-small-text-matching --local-dir ".\jinaai\jina-embeddings-v5-text-small-text-matching"
hf download jinaai/jina-embeddings-v5-text-small-classification --local-dir ".\jinaai\jina-embeddings-v5-text-small-classification"
hf download jinaai/jina-embeddings-v5-text-small-clustering --local-dir ".\jinaai\jina-embeddings-v5-text-small-clustering"
```

If ONNX models are not found, the server automatically falls back to PyTorch (SentenceTransformer).

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `USE_ONNX_EMBEDDINGS` | `true` | Set to `false` to force PyTorch backend |
| `OMP_NUM_THREADS` | `12` | Shared thread pool for both backends |
| `ORT_NUM_THREADS` | `6` | ONNX Runtime intra-op threads |
| `MKL_NUM_THREADS` | `12` | MKL thread pool |

Default thread configuration is tuned for **AMD R9 9900X (12C/24T, Zen 5)**: 6 threads ONNX, 6 threads PyTorch (reranker stays on PyTorch).

## Usage

Start the server:

```bash
uv run jina_server.py
```

Or activate the virtual environment:

```bash
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python jina_server.py
```

## API Endpoints

### `POST /v1/embeddings`

Create embeddings for input text. Supports task-adaptive LoRA adapters.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | `string \| string[]` | *(required)* | Text(s) to embed |
| `task` | `string` | `"retrieval"` | Task adapter: `retrieval`, `text-matching`, `clustering`, `classification` |
| `prompt_name` | `string` | `null` | Required when `task="retrieval"`: `"query"` or `"document"` |
| `model` | `string` | `"jina-embeddings-v5-text-small"` | Model name |
| `batch_size` | `int` | `32` | Processing batch size (1–128) |

#### Examples

Retrieval (query side):

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "What is machine learning?", "task": "retrieval", "prompt_name": "query"}'
```

Retrieval (document side):

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Machine learning is a field of AI...", "task": "retrieval", "prompt_name": "document"}'
```

Text matching:

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["Hello", "Hi"], "task": "text-matching"}'
```

Classification:

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "This product is great!", "task": "classification"}'
```

Clustering:

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["neural networks", "monetary policy"], "task": "clustering"}'
```

### `POST /v1/rerank`

Rerank documents based on query relevance.

```bash
curl http://localhost:8000/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "documents": ["Doc 1", "Doc 2"],
    "top_n": 3
  }'
```

### `GET /v1/models`

List available models.

### `GET /`

Health check endpoint.

## Testing

Run the test suite:

```bash
uv run test_server.py
```
## Architecture

- **Embeddings**: ONNX Runtime (task-specific models, last-token pooling, L2 normalization)
- **Reranker**: PyTorch (`jina-reranker-v3` — CausalLM-based, incompatible with ONNX)
- **Thread pools**: Split 6+6 on 12-core CPUs to avoid contention between ONNX and PyTorch

## License

MIT
