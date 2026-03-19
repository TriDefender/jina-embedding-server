# Jina Embedding & Reranker Server

OpenAI-compatible API server for Jina embeddings and reranking models.

## Features

- **Embedding**: jina-embeddings-v5-text-small-retrieval
- **Reranking**: jina-reranker-v3
- **OpenAI-compatible API**: Works with existing OpenAI client libraries

## Installation

```bash
uv sync
```

## Usage

Start the server:

```bash
uv run python jina_server.py
```

Or activate the virtual environment:

```bash
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python jina_server.py
```

## API Endpoints

### `POST /v1/embeddings`

Create embeddings for input text.

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, world!"}'
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
uv run python test_server.py
```

## License

MIT
