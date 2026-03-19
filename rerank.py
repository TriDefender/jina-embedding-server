#!/usr/bin/env python3
import numpy as np
import subprocess
import tempfile
import os
from typing import Optional, List, Dict
from safetensors import safe_open
import json


class MLPProjector:
    """MLP projector to project hidden states to embedding space."""
    def __init__(self, linear1_weight, linear2_weight):
        self.linear1_weight = linear1_weight
        self.linear2_weight = linear2_weight

    def __call__(self, x):
        # Linear 1
        x = x @ self.linear1_weight.T
        # ReLU
        x = np.maximum(0, x)
        # Linear 2
        x = x @ self.linear2_weight.T
        return x


def load_projector(projector_path: str) -> MLPProjector:
    """Load projector weights from safetensors file."""
    with safe_open(projector_path, framework="numpy") as f:
        w0 = f.get_tensor("projector.0.weight")
        w2 = f.get_tensor("projector.2.weight")

    return MLPProjector(w0, w2)


def sanitize_input(text: str, special_tokens: Dict[str, str]) -> str:
    """Remove special tokens from input text."""
    for token in special_tokens.values():
        text = text.replace(token, "")
    return text


def format_docs_prompts_func(
    query: str,
    docs: list[str],
    instruction: Optional[str] = None,
    special_tokens: Dict[str, str] = {},
) -> str:
    """Format query and documents into a prompt for the model."""
    query = sanitize_input(query, special_tokens)
    docs = [sanitize_input(doc, special_tokens) for doc in docs]

    prefix = (
        "<|im_start|>system\n"
        "You are a search relevance expert who can determine a ranking of the passages based on how relevant they are to the query. "
        "If the query is a question, how relevant a passage is depends on how well it answers the question. "
        "If not, try to analyze the intent of the query and assess how well each passage satisfies the intent. "
        "If an instruction is provided, you should follow the instruction when determining the ranking."
        "<|im_end|>\n<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n"

    doc_emb_token = special_tokens["doc_embed_token"]
    query_emb_token = special_tokens["query_embed_token"]

    prompt = (
        f"I will provide you with {len(docs)} passages, each indicated by a numerical identifier. "
        f"Rank the passages based on their relevance to query: {query}\n"
    )

    if instruction:
        prompt += f'<instruct>\n{instruction}\n</instruct>\n'

    doc_prompts = [f'<passage id="{i}">\n{doc}{doc_emb_token}\n</passage>' for i, doc in enumerate(docs)]
    prompt += "\n".join(doc_prompts) + "\n"
    prompt += f"<query>\n{query}{query_emb_token}\n</query>"

    return prefix + prompt + suffix


class GGUFReranker:
    """GGUF-based implementation of jina-reranker-v3."""

    def __init__(self, model_path: str = "jina-reranker-v3-BF16.gguf", projector_path: str = "projector.safetensors",
                 llama_embedding_path: str = "/tmp/hanxiao-llama.cpp/build/bin/llama-embedding"):
        """Initialize GGUF-based reranker."""
        self.model_path = model_path
        self.llama_embedding_path = llama_embedding_path
        self.projector = load_projector(projector_path)

        # Special tokens
        self.special_tokens = {
            "query_embed_token": "<|rerank_token|>",
            "doc_embed_token": "<|embed_token|>"
        }
        self.doc_embed_token_id = 151670
        self.query_embed_token_id = 151671

    def _get_hidden_states(self, prompt: str) -> np.ndarray:
        """Get per-token hidden states using llama-embedding CLI."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write(prompt)
            prompt_file = f.name

        try:
            result = subprocess.run(
                [
                    self.llama_embedding_path,
                    '-m', self.model_path,
                    '-f', prompt_file,
                    '--pooling', 'none',
                    '--embd-separator', '<#JINA_SEP#>',  # Preserve internal newlines
                    '--embd-normalize', '-1',
                    '--embd-output-format', 'json',
                    '--ubatch-size', '512',
                    '--ctx-size', '8192',
                    '--flash-attn',
                    '-ngl', '99'
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )

            output = json.loads(result.stdout)
            embeddings = [item['embedding'] for item in output['data']]
            return np.array(embeddings)
        finally:
            os.unlink(prompt_file)

    def _tokenize(self, prompt: str) -> List[int]:
        """Tokenize prompt to find special token positions."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write(prompt)
            prompt_file = f.name

        try:
            result = subprocess.run(
                ['llama-tokenize', '-m', self.model_path, '-f', prompt_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=True
            )

            tokens = []
            for line in result.stdout.strip().split('\n'):
                if '->' in line:
                    token_id = int(line.split('->')[0].strip())
                    tokens.append(token_id)
            return tokens
        finally:
            os.unlink(prompt_file)

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        return_embeddings: bool = False,
        instruction: Optional[str] = None
    ) -> List[Dict]:
        """Rerank documents based on relevance to query."""
        # Format prompt
        prompt = format_docs_prompts_func(
            query,
            documents,
            instruction=instruction,
            special_tokens=self.special_tokens
        )

        # Get per-token hidden states using llama-embedding CLI
        embeddings = self._get_hidden_states(prompt)

        # Tokenize to find special token positions
        tokens = self._tokenize(prompt)
        tokens_array = np.array(tokens)

        query_embed_positions_in_tokens = np.where(tokens_array == self.query_embed_token_id)[0]
        doc_embed_positions_in_tokens = np.where(tokens_array == self.doc_embed_token_id)[0]

        if len(query_embed_positions_in_tokens) == 0:
            raise ValueError(f"Query embed token (ID {self.query_embed_token_id}) not found in input")

        if len(doc_embed_positions_in_tokens) == 0:
            raise ValueError(f"Document embed tokens (ID {self.doc_embed_token_id}) not found in input")

        # llama-embedding strips trailing newlines but preserves internal newlines (via --embd-separator)
        # Token positions map directly to embedding indices
        query_pos = query_embed_positions_in_tokens[0]
        doc_positions = doc_embed_positions_in_tokens

        # Extract embeddings at special token positions
        query_hidden = embeddings[query_pos:query_pos+1]  # [1, hidden_size]
        doc_hidden = embeddings[doc_positions]  # [num_docs, hidden_size]

        # Project embeddings
        query_embeds = self.projector(query_hidden)  # [1, 512]
        doc_embeds = self.projector(doc_hidden)  # [num_docs, 512]

        # Compute cosine similarity scores
        # Broadcast query to match doc shape
        query_expanded = np.tile(query_embeds, (len(doc_embeds), 1))  # [num_docs, 512]

        # Cosine similarity
        dot_product = np.sum(doc_embeds * query_expanded, axis=-1)  # [num_docs]
        doc_norm = np.sqrt(np.sum(doc_embeds * doc_embeds, axis=-1))  # [num_docs]
        query_norm = np.sqrt(np.sum(query_expanded * query_expanded, axis=-1))  # [num_docs]
        scores = dot_product / (doc_norm * query_norm)  # [num_docs]

        # Create results
        results = []
        for idx, (doc, score, embed) in enumerate(zip(documents, scores, doc_embeds)):
            result = {
                "index": idx,
                "relevance_score": float(score),
                "document": doc
            }
            if return_embeddings:
                result["embedding"] = embed.tolist()
            results.append(result)

        # Sort by score descending
        results.sort(key=lambda x: x["relevance_score"], reverse=True)

        # Return top_n if specified
        if top_n is not None:
            results = results[:top_n]

        return results


if __name__ == "__main__":
    # Test the reranker
    reranker = GGUFReranker()

    query = "What is the capital of France?"
    documents = [
        "Paris is the capital and largest city of France.",
        "Berlin is the capital of Germany.",
        "The Eiffel Tower is located in Paris."
    ]

    results = reranker.rerank(query, documents)
    for result in results:
        print(f"Doc {result['index']}: {result['relevance_score']:.4f} - {result['document'][:50]}...")
