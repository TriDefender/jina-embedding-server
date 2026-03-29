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
## WIP:
I'm telling my opencode to implement avx512 for my 9900x rig, brb

## License

MIT
