REM === PyTorch models (required) ===
hf download jinaai/jina-embeddings-v5-text-small --local-dir ".\jinaai\jina-embeddings-v5-text-small" --max-workers 12
hf download jinaai/jina-reranker-v3 --local-dir ".\jinaai\jina-reranker-v3" --max-workers 12

REM === ONNX embedding models (task-specific, LoRA adapters merged) ===
hf download jinaai/jina-embeddings-v5-text-small-retrieval --local-dir ".\jinaai\jina-embeddings-v5-text-small-retrieval" --max-workers 12 --exclude *.gguf *.safetensors
hf download jinaai/jina-embeddings-v5-text-small-text-matching --local-dir ".\jinaai\jina-embeddings-v5-text-small-text-matching" --max-workers 12 --exclude *.gguf *.safetensors
hf download jinaai/jina-embeddings-v5-text-small-classification --local-dir ".\jinaai\jina-embeddings-v5-text-small-classification" --max-workers 12 --exclude *.gguf *.safetensors
hf download jinaai/jina-embeddings-v5-text-small-clustering --local-dir ".\jinaai\jina-embeddings-v5-text-small-clustering" --max-workers 12 --exclude *.gguf *.safetensors
