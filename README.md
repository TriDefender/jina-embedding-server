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

#For windows, use a flash attention prebuilt, ignore if your system is running linux:

uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.6/flash_attn-2.8.3+cu130torch2.11-cp312-cp312-win_amd64.whl

Fetch_Models.bat  #This is for windows, it contains two 'hf download' commands, adapt for linux accordingly
```

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
## Flags

You can set these flags as environment variables:

| Flag | Default | Description |
|------|---------|-------------|
| `IDLE_TIMEOUT_SECONDS` | `300` | Seconds of inactivity before offloading models from VRAM to CPU RAM |
| `COMPILE_ON_GPU` | `0` | Set `1` to enable `torch.compile()` on GPU. Adds ~30-50% throughput but makes first reload after offload ~10-30s (vs ~1-2s without) |
| `CUDA_GRAPH` | `0` | Set `1` to capture CUDA Graphs for the reranker backbone. Eliminates kernel launch overhead (~10-25% latency reduction). GPU only |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | CUDA memory allocator config. Set before torch import. `expandable_segments` reduces VRAM fragmentation (Linux only; silently ignored on Windows) |

## License

MIT
