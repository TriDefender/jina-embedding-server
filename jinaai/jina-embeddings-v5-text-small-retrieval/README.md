---
pipeline_tag: sentence-similarity
tags:
- gguf
- embedding
- qwen3
- llama-cpp
- jina-embeddings-v5
- feature-extraction
- mteb
- vllm
- sentence-transformers
language:
- multilingual
base_model: jinaai/jina-embeddings-v5-text-small
base_model_relation: quantized
inference: false
license: cc-by-nc-4.0
library_name: llama.cpp
---
<br><br>

<p align="center">
<img src="https://huggingface.co/datasets/jinaai/documentation-images/resolve/main/logo.webp" alt="Jina AI: Your Search Foundation, Supercharged!" width="150px">
</p>

### **jina-embeddings-v5-text-small-retrieval**: Retrieval-Targeted Embedding Distillation

[Elastic Inference Service](https://www.elastic.co/docs/explore-analyze/elastic-inference/eis) | [ArXiv](https://arxiv.org/abs/2602.15547) | [Release Note](https://jina.ai/news/jina-embeddings-v5-text-distilling-4b-quality-into-sub-1b-multilingual-embeddings) | [Blog](https://www.elastic.co/search-labs/blog/jina-embeddings-v5-text)

### Model Overview

<p align="center">
<img src="https://jina-ai-gmbh.ghost.io/content/images/2026/02/v5_architecture_1771470917.png" alt="jina-embeddings-v5-text Architecture" width="600px">
</p>
`jina-embeddings-v5-text-small-retrieval` is a compact, high-performance text embedding model designed for information retrieval.

It is part of the **jina-embeddings-v5-text** model family, which also includes [jina-embeddings-v5-text-nano](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano), a smaller model for more resource-constrained use cases.

Trained using a novel approach that combines distillation with task-specific contrastive losses, `jina-embeddings-v5-text-small-retrieval` outperforms existing state-of-the-art models of similar size across diverse embedding benchmarks.
| Feature | Value |
| --- | --- |
| Parameters | 677M |
| Supported Tasks | `retrieval`|
| Max Sequence Length | 32768 |
| Embedding Dimension | 1024 |
| Matryoshka Dimensions | 32, 64, 128, 256, 512, 768, 1024 |
| Pooling Strategy | Last-token pooling |
| Base Model | jinaai/jina-embeddings-v5-text-small |


![v5_benchmarks_combined](https://cdn-uploads.huggingface.co/production/uploads/6476ff2699a5ce743ccea3fc/7WjMQChM6XAOI9LhREChg.png)


### Training and Evaluation

For training details and evaluation results, see our [technical report](https://arxiv.org/abs/2602.15547).

### Usage

<details>
  <summary>Requirements</a></summary>
  
The following Python packages are required:

- `transformers>=5.1.0`
- `torch>=2.8.0`
- `peft>=0.15.2`
- `vllm>=0.15.1`
  
### Optional / Recommended
- **flash-attention**: Installing [flash-attention](https://github.com/Dao-AILab/flash-attention) is recommended for improved inference speed and efficiency, but not mandatory.
- **sentence-transformers**: If you want to use the model via the `sentence-transformers` interface, install this package as well.

</details>

<details open>
  <summary>via <a href="https://www.elastic.co/docs/explore-analyze/elastic-inference/eis">Elastic Inference Service</a></summary>

The fastest way to use v5-text in production. Elastic Inference Service (EIS) provides managed embedding inference with built-in scaling, so you can generate embeddings directly within your Elastic deployment.

```bash
PUT _inference/text_embedding/jina-v5
{
  "service": "elastic",
  "service_settings": {
    "model_id": "jina-embeddings-v5-text-small"
  }
}
```

See the [Elastic Inference Service documentation](https://www.elastic.co/docs/explore-analyze/elastic-inference/eis) for setup details.

</details>


<details>
  <summary>via <a href="https://sbert.net/">sentence-transformers</a></summary>

```python
from sentence_transformers import SentenceTransformer
import torch

model = SentenceTransformer(
    "jinaai/jina-embeddings-v5-text-small-retrieval",
    model_kwargs={"dtype": torch.bfloat16},  # Recommended for GPUs
    config_kwargs={"_attn_implementation": "flash_attention_2"},  # Recommended but optional
)
# Optional: set truncate_dim in encode() to control embedding size

query = "Which planet is known as the Red Planet?"
documents = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
]

# Encode query and documents
query_embeddings = model.encode(sentences=query, prompt_name="query")
document_embeddings = model.encode(sentences=documents, prompt_name="document")
print(query_embeddings.shape, document_embeddings.shape)
# (1024,) (4, 1024)

similarity = model.similarity(query_embeddings, document_embeddings)
print(similarity)
# tensor([[0.4860, 0.7611, 0.5914, 0.6188]])
```
</details>

<details>
  <summary>via <a href="https://github.com/vllm-project/vllm">vLLM</a></summary>

```python
from vllm import LLM
from vllm.config.pooler import PoolerConfig

# Initialize model
name = "jinaai/jina-embeddings-v5-text-small-retrieval"
model = LLM(
    model=name,
    dtype="float16",
    runner="pooling",
    pooler_config=PoolerConfig(seq_pooling_type="LAST", normalize=True),
)

# Create text prompts
query = "Overview of climate change impacts on coastal cities"
query_prompt = f"Query: {query}"

document = "The impacts of climate change on coastal cities are significant.."
document_prompt = f"Document: {document}"

# Encode all prompts
prompts = [query_prompt, document_prompt]
outputs = model.encode(prompts, pooling_task="embed")

```

</details>

<details>
  <summary>via <a href="https://github.com/huggingface/text-embeddings-inference">Text Embeddings Inference</a></summary>

- Via Docker on CPU:
  ```bash
  docker run -p 8080:80 \
    ghcr.io/huggingface/text-embeddings-inference:cpu-1.9 \
    --model-id jinaai/jina-embeddings-v5-text-small-retrieval \
    --dtype float32 --pooling last-token
  ```
- Via Docker on NVIDIA GPU (Turing, Ampere, Ada Lovelace, Hopper or Blackwell):
  ```bash
  docker run --gpus all --shm-size 1g -p 8080:80 \
    ghcr.io/huggingface/text-embeddings-inference:cuda-1.9 \
    --model-id jinaai/jina-embeddings-v5-text-small-retrieval \
    --dtype float16 --pooling last-token
  ```

> Alternatively, you can also run with `cargo`, more information can be found in the [Text Embeddings Inference documentation](https://hf.co/docs/text-embeddings-inference).

Send a request to `/v1/embeddings` to generate embeddings via the [OpenAI Embeddings API](https://platform.openai.com/docs/api-reference/embeddings/create):

```bash
curl -X POST http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jinaai/jina-embeddings-v5-text-small-retrieval",
    "input": [
      "Query: Overview of climate change impacts on coastal cities",
      "Document: The impacts of climate change on coastal cities are significant...",
    ]
  }'
```

Or rather via the [Text Embeddings Inference API specification](https://huggingface.github.io/text-embeddings-inference/) instead, to prevent from manually formatting the inputs:

```bash
curl -X POST http://127.0.0.1:8080/embed \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": "Overview of climate change impacts on coastal cities",
    "prompt_name": "query",
  }'
```

</details>

<details>
  <summary> via <a href="https://github.com/ggml-org/llama.cpp">llama.cpp (GGUF)</a></summary>
After installing <a href="https://github.com/ggml-org/llama.cpp">llama.cpp</a> one can run llama-server to host the embedding model as OpenAI API compatible HTTP server with the respective model version:

```sh
llama-server -hf jinaai/jina-embeddings-v5-text-small-retrieval:F16 --embedding --pooling last -ub 32768
```

Client:

```
curl -X POST "http://127.0.0.1:8080/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "input": [
      "Query: A beautiful sunset over the beach",
      "Query: Un beau coucher de soleil sur la plage",
      "Document: 海滩上美丽的日落",
      "Document: 浜辺に沈む美しい夕日",
      "Document: Golden sunlight melts into the horizon, painting waves in warm amber and rose, while the sky whispers goodnight to the quiet, endless sea."
    ]
  }'
```

</details>

<details>
  <summary> via <a href="https://github.com/ggml-org/llama.cpp">llama.cpp (GGUF)</a></summary>
After installing <a href="https://github.com/ggml-org/llama.cpp">llama.cpp</a> one can run llama-server to host the embedding model as OpenAI API compatible HTTP server with the respective model version:

```sh
llama-server -hf jinaai/jina-embeddings-v5-text-small-retrieval:F16 --embedding --pooling last -ub 32768
```

Client:

```
curl -X POST "http://127.0.0.1:8080/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "input": [
      "Query: A beautiful sunset over the beach",
      "Query: Un beau coucher de soleil sur la plage",
      "Document: 海滩上美丽的日落",
      "Document: 浜辺に沈む美しい夕日",
      "Document: Golden sunlight melts into the horizon, painting waves in warm amber and rose, while the sky whispers goodnight to the quiet, endless sea."
    ]
  }'
```

</details>

<details>
  <summary> via <a href="https://huggingface.co/docs/optimum/index">Optimum (ONNX)</a></summary>

You can run the ONNX-optimized version of the model locally using Hugging Face's `optimum` library. Make sure you have the required dependencies installed (e.g., `pip install optimum[onnxruntime] transformers torch`):

```python
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
import torch

model_id = "jinaai/jina-embeddings-v5-text-small-retrieval"

# 1. Load tokenizer and ONNX model
# We specify the subfolder 'onnx' where the weights are located
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = ORTModelForFeatureExtraction.from_pretrained(
    model_id,
    subfolder="onnx",
    file_name="model.onnx",
    provider="CPUExecutionProvider",  # Or "CUDAExecutionProvider" for GPU
    trust_remote_code=True,
)

# 2. Prepare input
texts = ["Query: How do I use Jina ONNX models?", "Document: Information about semantic matching."]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")


# 4. Inference
with torch.no_grad():
    outputs = model(**inputs)

# 5. Pooling (Crucial for Jina-v5)
# Jina-v5 uses LAST-TOKEN pooling.
# We take the hidden state of the last non-padding token.
last_hidden_state = outputs.last_hidden_state
# Find the indices of the last token (usually the end of the sequence)
sequence_lengths = inputs.attention_mask.sum(dim=1) - 1
embeddings = last_hidden_state[torch.arange(last_hidden_state.size(0)), sequence_lengths]

print('embeddings shape:', embeddings.shape)
print('embeddings:', embeddings)
```

</details>

### License

The model is licensed under CC BY-NC 4.0. For commercial use, please [contact us](sales@jina.ai).

### Citation

If you find `jina-embeddings-v5-text-small-retrieval` useful in your research, please cite the following paper:

```
@misc{akram2026jinaembeddingsv5texttasktargetedembeddingdistillation,
      title={jina-embeddings-v5-text: Task-Targeted Embedding Distillation}, 
      author={Mohammad Kalim Akram and Saba Sturua and Nastia Havriushenko and Quentin Herreros and Michael Günther and Maximilian Werk and Han Xiao},
      year={2026},
      eprint={2602.15547},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.15547}, 
}
```
