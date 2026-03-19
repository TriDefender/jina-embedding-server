---
license: cc-by-nc-4.0
language:
- multilingual
base_model:
- Qwen/Qwen3-0.6B-Base
tags:
- feature-extraction
- mteb
- sentence-transformers
library_name: transformers
---
<br><br>

<p align="center">
<img src="https://huggingface.co/datasets/jinaai/documentation-images/resolve/main/logo.webp" alt="Jina AI: Your Search Foundation, Supercharged!" width="150px">
</p>

# jina-embeddings-v5-text-small

`jina-embeddings-v5-text-small` is the fifth generation of Jina AI's multilingual embedding models, released on February 18, 2026. For a lighter alternative, see [jina-embeddings-v5-text-nano](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano) (239M parameters).

[Elastic Inference Service](https://www.elastic.co/docs/explore-analyze/elastic-inference/eis) | [ArXiv](https://arxiv.org/abs/2602.15547) | [Release Note](https://jina.ai/news/jina-embeddings-v5-text-distilling-4b-quality-into-sub-1b-multilingual-embeddings) | [Blog](https://www.elastic.co/search-labs/blog/jina-embeddings-v5-text)

## Model Overview

<p align="center">
<img src="https://jina-ai-gmbh.ghost.io/content/images/2026/02/v5_architecture_1771470917.png" alt="jina-embeddings-v5-text Architecture" width="600px">
</p>

`jina-embeddings-v5-text-small` scores 71.7 average on MTEB English v2 and 67.7 on MMTEB with 677M parameters, the highest among multilingual embedding models under 1B parameters. Built on Qwen3-0.6B-Base and trained by combining embedding distillation from Qwen3-Embedding-4B with task-specific contrastive losses, it supports 119+ languages with up to 32K tokens and produces embeddings robust under truncation and binary quantization.

It is part of the **jina-embeddings-v5-text** model family, which also includes [jina-embeddings-v5-text-nano](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano), a smaller model for resource-constrained use cases.
| Feature | Value |
| --- | --- |
| Parameters | 677M |
| Supported Tasks | `retrieval`, `text-matching`, `clustering`, `classification` |
| Max Sequence Length | 32768 |
| Embedding Dimension | 1024 |
| Matryoshka Dimensions | 32, 64, 128, 256, 512, 768, 1024 |
| Pooling Strategy | Last-token pooling |
| Base Model | Qwen/Qwen3-0.6B-Base |


![image](https://cdn-uploads.huggingface.co/production/uploads/6476ff2699a5ce743ccea3fc/SJw9j09PkErQ0v9P052S9.png)


## Training and Evaluation

For training details and evaluation results, see our [technical report](https://arxiv.org/abs/2602.15547).

## Usage

<details>
  <summary>Requirements</a></summary>
  
The following Python packages are required:

- `transformers>=4.57.0`
- `torch>=2.8.0`
- `peft>=0.15.2`
  
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
  <summary>via <a href="https://jina.ai/embeddings/">Jina AI Embeddings API</a></summary>

```bash
curl https://api.jina.ai/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $JINA_AI_API_TOKEN" \
  -d @- <<EOFEOF
  {
    "model": "jina-embeddings-v5-text-small",
    "task": "text-matching",
    "input": [
        {
            "text": "غروب جميل على الشاطئ"
        },
        {
            "text": "海滩上美丽的日落"
        },
        {
            "text": "A beautiful sunset over the beach"
        },
        {
            "text": "Un beau coucher de soleil sur la plage"
        },
        {
            "text": "Ein wunderschöner Sonnenuntergang am Strand"
        },
        {
            "text": "Ένα όμορφο ηλιοβασίλεμα πάνω από την παραλία"
        },
        {
            "text": "समुद्र तट पर एक खूबसूरत सूर्यास्त"
        },
        {
            "text": "Un bellissimo tramonto sulla spiaggia"
        },
        {
            "text": "浜辺に沈む美しい夕日"
        },
        {
            "text": "해변 위로 아름다운 일몰"
        },
        {
            "image": "https://i.ibb.co/nQNGqL0/beach1.jpg"
        },
        {
            "image": "https://i.ibb.co/r5w8hG8/beach2.jpg"
        }
    ]
  }
EOFEOF
```

</details>

<details open>
  <summary>via <a href="https://huggingface.co/docs/transformers/en/index">transformers</a></summary>

```python
from transformers import AutoModel
import torch

model = AutoModel.from_pretrained(
    "jinaai/jina-embeddings-v5-text-small",
    trust_remote_code=True,
    _attn_implementation="flash_attention_2",  # Recommended but optional
    dtype=torch.bfloat16,  # Recommended for GPUs
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device=device)

# Optional: set truncate_dim and max_length in encode() to control embedding size and input length

# ========================
# 1. Retrieval Task
# ========================
# Encode query
query_embeddings = model.encode(
    texts=["Overview of climate change impacts on coastal cities"],
    task="retrieval",
    prompt_name="query",
)
# Encode document
document_embeddings = model.encode(
    texts=[
        "Climate change has led to rising sea levels, increased frequency of extreme weather events..."
    ],
    task="retrieval",
    prompt_name="document",
)

# ========================
# 2. Text Matching Task
# ========================
texts = [
    "غروب جميل على الشاطئ",  # Arabic
    "海滩上美丽的日落",  # Chinese
    "Un beau coucher de soleil sur la plage",  # French
    "Ein wunderschöner Sonnenuntergang am Strand",  # German
    "Ένα όμορφο ηλιοβασίλεμα πάνω από την παραλία",  # Greek
    "समुद्र तट पर एक खूबसूरत सूर्यास्त",  # Hindi
    "Un bellissimo tramonto sulla spiaggia",  # Italian
    "浜辺に沈む美しい夕日",  # Japanese
    "해변 위로 아름다운 일몰",  # Korean
]
text_embeddings = model.encode(texts=texts, task="text-matching")

# ========================
# 3. Classification Task
# ========================
texts = [
    "My order hasn't arrived yet and it's been two weeks.",
    "How do I reset my password?",
    "I'd like a refund for my recent purchase.",
    "Your product exceeded my expectations. Great job!",
]
classification_embeddings = model.encode(texts=texts, task="classification")

# ========================
# 4. Clustering Task
# ========================
texts = [
    "We propose a novel neural network architecture for image segmentation.",
    "This paper analyzes the effects of monetary policy on inflation.",
    "Our method achieves state-of-the-art results on object detection benchmarks.",
    "We study the relationship between interest rates and housing prices.",
    "A new attention mechanism is introduced for visual recognition tasks.",
]
clustering_embeddings = model.encode(texts=texts, task="clustering")
```
</details>

<details>
  <summary>via <a href="https://sbert.net/">sentence-transformers</a></summary>
  
```python
from sentence_transformers import SentenceTransformer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SentenceTransformer(
    "jinaai/jina-embeddings-v5-text-small",
    trust_remote_code=True,
    device=device,
    model_kwargs={"dtype": torch.bfloat16},  # Recommended for GPUs
    config_kwargs={"_attn_implementation": "flash_attention_2"},  # Recommended but optional
)

# Optional: set truncate_dim in encode() to control embedding size

# ========================
# 1. Retrieval Task
# ========================
# Encode query
query_embeddings = model.encode(
    sentences=["Overview of climate change impacts on coastal cities"],
    task="retrieval",
    prompt_name="query",
)
# Encode document
document_embeddings = model.encode(
    sentences=[
        "Climate change has led to rising sea levels, increased frequency of extreme weather events..."
    ],
    task="retrieval",
    prompt_name="document",
)

# ========================
# 2. Text Matching Task
# ========================
texts = [
    "غروب جميل على الشاطئ",  # Arabic
    "海滩上美丽的日落",  # Chinese
    "Un beau coucher de soleil sur la plage",  # French
    "Ein wunderschöner Sonnenuntergang am Strand",  # German
    "Ένα όμορφο ηλιοβασίλεμα πάνω από την παραλία",  # Greek
    "समुद्र तट पर एक खूबसूरत सूर्यास्त",  # Hindi
    "Un bellissimo tramonto sulla spiaggia",  # Italian
    "浜辺に沈む美しい夕日",  # Japanese
    "해변 위로 아름다운 일몰",  # Korean
]
text_embeddings = model.encode(sentences=texts, task="text-matching")

# ========================
# 3. Classification Task
# ========================
texts = [
    "My order hasn't arrived yet and it's been two weeks.",
    "How do I reset my password?",
    "I'd like a refund for my recent purchase.",
    "Your product exceeded my expectations. Great job!",
]
classification_embeddings = model.encode(sentences=texts, task="classification")

# ========================
# 4. Clustering Task
# ========================
texts = [
    "We propose a novel neural network architecture for image segmentation.",
    "This paper analyzes the effects of monetary policy on inflation.",
    "Our method achieves state-of-the-art results on object detection benchmarks.",
    "We study the relationship between interest rates and housing prices.",
    "A new attention mechanism is introduced for visual recognition tasks.",
]
clustering_embeddings = model.encode(sentences=texts, task="clustering")
```
</details>

<details>
  <summary>via <a href="https://github.com/vllm-project/vllm">vLLM</a></summary>

We provide separate model versions for each task (`retrieval`, `text-matching`, `classification`, `clustering`).
For each model, the task-specific adapter is merged into the base model weights.  
This modification enables simpler compatibility with vLLM.

Instructions and usage examples for each task are available in their respective model repositories:

- [jina-embeddings-v5-text-small-retrieval](https://huggingface.co/jinaai/jina-embeddings-v5-text-small-retrieval)
- [jina-embeddings-v5-text-small-text-matching](https://huggingface.co/jinaai/jina-embeddings-v5-text-small-text-matching)
- [jina-embeddings-v5-text-small-classification](https://huggingface.co/jinaai/jina-embeddings-v5-text-small-classification)
- [jina-embeddings-v5-text-small-clustering](https://huggingface.co/jinaai/jina-embeddings-v5-text-small-clustering)

</details>

<details>
  <summary>via <a href="https://github.com/huggingface/text-embeddings-inference">Text Embeddings Inference</a></summary>

We provide separate model versions for each task (`retrieval`, `text-matching`, `classification`, `clustering`).
For each model, the task-specific adapter is merged into the base model weights.  
This modification enables simpler compatibility with Text Embeddings Inference.

Instructions and usage examples for each task are available in their respective model repositories:

- [jina-embeddings-v5-text-small-retrieval](https://huggingface.co/jinaai/jina-embeddings-v5-text-small-retrieval)
- [jina-embeddings-v5-text-small-text-matching](https://huggingface.co/jinaai/jina-embeddings-v5-text-small-text-matching)
- [jina-embeddings-v5-text-small-classification](https://huggingface.co/jinaai/jina-embeddings-v5-text-small-classification)
- [jina-embeddings-v5-text-small-clustering](https://huggingface.co/jinaai/jina-embeddings-v5-text-small-clustering)

</details>

<details>
  <summary>via <a href="https://onnx.ai/">ONNX</a> and <a href="https://huggingface.co/docs/optimum/index">Optimum</a></summary>

We provide separate model versions for each task (`retrieval`, `text-matching`, `classification`, `clustering`).
For each model, the task-specific adapter is merged into the base model weights. 
This enables inference using ONNX Runtime and Hugging Face Optimum.

We provide ONNX-formatted weights located within the `onnx` subfolder of each model repository.
Instructions and usage examples for each task are available in their respective model repository:

- [jina-embeddings-v5-text-small-retrieval](https://huggingface.co/jinaai/jina-embeddings-v5-text-small-retrieval)
- [jina-embeddings-v5-text-small-text-matching](https://huggingface.co/jinaai/jina-embeddings-v5-text-small-text-matching)
- [jina-embeddings-v5-text-small-classification](https://huggingface.co/jinaai/jina-embeddings-v5-text-small-classification)
- [jina-embeddings-v5-text-small-clustering](https://huggingface.co/jinaai/jina-embeddings-v5-text-small-clustering)

</details>

## License

jina-embeddings-v5-text-small is licensed under CC BY-NC 4.0. For commercial use, please [contact us](sales@jina.ai).

## Citation

If you find `jina-embeddings-v5-text-small` useful in your research, please cite the following paper:

```bibtex
@article{akram2026jina,
  title={jina-embeddings-v5-text: Task-Targeted Embedding Distillation},
  author={Mohammad Kalim Akram and Saba Sturua and Nastia Havriushenko and Quentin Herreros and Michael G{\"u}nther and Maximilian Werk and Han Xiao},
  journal={arXiv preprint arXiv:2602.15547},
  year={2026}
}
```