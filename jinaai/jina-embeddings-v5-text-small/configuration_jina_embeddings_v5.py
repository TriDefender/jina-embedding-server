from transformers import Qwen3Config


class JinaEmbeddingsV5Config(Qwen3Config):
    model_type = "jina_embeddings_v5"
