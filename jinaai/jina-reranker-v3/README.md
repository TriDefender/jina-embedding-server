---
pipeline_tag: text-ranking
tags:
- transformers
- reranker
- qwen3
language:
- multilingual
base_model:
- Qwen/Qwen3-0.6B
inference: false
license: cc-by-nc-4.0
library_name: transformers
---

<p align="center">
<img src="https://huggingface.co/datasets/jinaai/documentation-images/resolve/main/logo.webp" alt="Jina AI: Your Search Foundation, Supercharged!" width="150px">
</p>


# jina-reranker-v3: Listwise Document Reranker for SOTA Multilingual Retrieval

[Blog](https://jina.ai/news/jina-reranker-v3-0-6b-listwise-reranker-for-sota-multilingual-retrieval) | [API](https://jina.ai/reranker) | [AWS](#) | [Azure](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/jinaai.jina-reranker-v3) | [GCP](https://console.cloud.google.com/marketplace/product/jinaai-public/jina-reranker-v3) | [Arxiv](https://arxiv.org/abs/2509.25085)


> [!TIP]
> [GGUF with quantizations](https://huggingface.co/jinaai/jina-reranker-v3-GGUF) and [MLX](https://huggingface.co/jinaai/jina-reranker-v3-mlx) versions are now available.


`jina-reranker-v3` is a 0.6B parameter multilingual document reranker with a novel *last but not late interaction* architecture. Unlike ColBERT's separate encoding with multi-vector matching, this model performs causal self-attention between query and documents within the same context window, extracting contextual embeddings from the last token of each document.

![jina-reranker-v3 architecture](https://jina-ai-gmbh.ghost.io/content/images/2025/10/Heading--54-.svg)

Built on Qwen3-0.6B with 28 transformer layers and a lightweight MLP projector (1024→512→256), it processes up to 64 documents simultaneously within 131K token context. The model achieves state-of-the-art BEIR performance with 61.94 nDCG@10 while being 10× smaller than generative listwise rerankers.

| Model | Size | BEIR | MIRACL | MKQA | CoIR |
|-------|------|------|--------|------|------|
| **jina-reranker-v3** | 0.6B | **61.94** | 66.83 | 67.92 | 70.64 |
| jina-reranker-v2 | 0.3B | 57.06 | 63.65 | 67.90 | 56.14 |
| jina-reranker-m0 | 2.4B | 58.95 | 66.75 | **68.19** | 63.55 |
| bge-reranker-v2-m3 | 0.6B | 56.51 | **69.32** | 67.88 | 36.28 |
| mxbai-rerank-base-v2 | 0.5B | 58.40 | 55.32 | 64.24 | 65.71 |
| mxbai-rerank-large-v2 | 1.5B | 61.44 | 57.94 | 67.06 | 70.87 |
| Qwen3-Reranker-0.6B | 0.6B | 56.28 | 57.70 | 65.34 | 65.18 |
| Qwen3-Reranker-4B | 4.0B | 61.16 | 67.52 | 67.52 | 73.91 |
| jina-code-embeddings-0.5b | 0.5B | - | - | - | **73.94** |


## Usage

### Local Inference

Use `transformers` for local inference:

**Installation:**

```bash
pip install transformers
```

**Load the model:**

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    'jinaai/jina-reranker-v3',
    dtype="auto",
    trust_remote_code=True,
)
model.eval()
```

**Rank documents:**

```python
query = "What are the health benefits of green tea?"
documents = [
    "Green tea contains antioxidants called catechins that may help reduce inflammation and protect cells from damage.",
    "El precio del café ha aumentado un 20% este año debido a problemas en la cadena de suministro.",
    "Studies show that drinking green tea regularly can improve brain function and boost metabolism.",
    "Basketball is one of the most popular sports in the United States.",
    "绿茶富含儿茶素等抗氧化剂，可以降低心脏病风险，还有助于控制体重。",
    "Le thé vert est riche en antioxydants et peut améliorer la fonction cérébrale.",
]

# Rerank documents
results = model.rerank(query, documents)

# Results are sorted by relevance score (highest first)
for result in results:
    print(f"Score: {result['relevance_score']:.4f}")
    print(f"Document: {result['document'][:100]}...")
    print()

# Output:
# Score: 0.2976
# Document: Green tea contains antioxidants called catechins that may help reduce inflammation and protect ce...
#
# Score: 0.2258
# Document: 绿茶富含儿茶素等抗氧化剂，可以降低心脏病风险，还有助于控制体重。
#
# Score: 0.1911
# Document: Studies show that drinking green tea regularly can improve brain function and boost metabolism.
#
# Score: 0.1640
# Document: Le thé vert est riche en antioxydants et peut améliorer la fonction cérébrale.
```

**API Reference:**

```python
model.rerank(
    query: str,                      # Search query
    documents: List[str],            # Documents to rank
    top_n: Optional[int] = None,     # Return only top N (default: all)
    return_embeddings: bool = False, # Include doc embeddings (default: False)
)
```

**Returns:** List of dicts with keys:
- `document`: Original document text
- `relevance_score`: Float score (higher = more relevant)
- `index`: Position in input documents list
- `embedding`: Document embedding (if `return_embeddings=True`)

**Example with options:**

```python
# Get only top 3 results
top_results = model.rerank(query, documents, top_n=3)

# Get embeddings for further processing
results_with_embeddings = model.rerank(query, documents, return_embeddings=True)
```


### API

Use Jina AI's [Reranker API](https://jina.ai/reranker) for the fastest integration:

```bash
curl -X POST \
  https://api.jina.ai/v1/rerank \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer JINA_API_KEY" \
  -d '{
  "model": "jina-reranker-v3",
  "query": "slm markdown",
  "documents": [
    ...
  ],
  "return_documents": false
}'
```

Response format:

```json
{
  "model":"jina-reranker-v3",
  "usage": {
    "total_tokens":2813
  },
  "results":[
    {
      "index":1,
      "relevance_score":0.9310624287463884
    },
    {
      "index":4,
      "relevance_score":0.8982678574191957
    },
    {
      "index":0,
      "relevance_score":0.890233167219021
    },
    ...
  ]
}
```

## Citation

If you find `jina-reranker-v3` useful in your research, please cite our [technical report](https://arxiv.org/abs/2509.25085):

```bibtex
@misc{wang2025jinarerankerv3lateinteractiondocument,
      title={jina-reranker-v3: Last but Not Late Interaction for Document Reranking}, 
      author={Feng Wang and Yuqing Li and Han Xiao},
      year={2025},
      eprint={2509.25085},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.25085}, 
}
```

## License

`jina-reranker-v3` is listed on AWS & Azure. If you need to use it beyond those platforms or on-premises within your company, note that the model is licensed under CC BY-NC 4.0. For commercial usage inquiries, feel free to [contact us](https://jina.ai/contact-sales/).
