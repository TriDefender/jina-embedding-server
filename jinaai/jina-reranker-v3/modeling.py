import numpy as np
from dataclasses import dataclass
import torch
from torch import nn
from typing import Optional, List, Dict, Tuple
from transformers.models.qwen3 import modeling_qwen3
from transformers.modeling_outputs import CausalLMOutputWithPast


@dataclass
class CausalLMOutputWithScores(CausalLMOutputWithPast):
    scores: Optional[torch.FloatTensor] = None
    query_embeds: Optional[torch.FloatTensor] = None
    doc_embeds: Optional[torch.FloatTensor] = None


def sanitize_input(text: str, special_tokens: Dict[str, str]) -> str:
    for token in special_tokens.values():
        text = text.replace(token, "")
    return text


def format_docs_prompts_func(
    query: str,
    docs: list[str],
    instruction: Optional[str] = None,
    special_tokens: Dict[str, str] = {},
    no_thinking: bool = True,
) -> str:
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
    if no_thinking:
        suffix += "<think>\n\n</think>\n\n"

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


class JinaForRanking(modeling_qwen3.Qwen3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.padding_side = "left"
        self.projector_dim = 512

        self.lm_head = nn.Identity()

        self.projector = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2, bias=False),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, self.projector_dim, bias=False),
        )

        self.post_init()

        self.special_tokens = {"query_embed_token": "<|rerank_token|>", "doc_embed_token": "<|embed_token|>"}
        self.doc_embed_token_id = 151670
        self.query_embed_token_id = 151671

    def forward(self, *args, **kwargs) -> CausalLMOutputWithScores:
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("use_cache", None)
        assert kwargs.pop("labels", None) is None, "labels should not be passed to forward()"
        input_ids = kwargs.pop("input_ids", None)

        outputs = super().forward(
            *args,
            input_ids=input_ids,
            use_cache=False,
            output_hidden_states=True,
            **kwargs,
        )

        hidden_states = outputs.hidden_states[-1]
        batch_size, _, dim = hidden_states.shape

        query_embed_token_indexes = torch.eq(input_ids, self.query_embed_token_id)
        doc_embed_token_indexes = torch.eq(input_ids, self.doc_embed_token_id)

        doc_embeds = hidden_states[doc_embed_token_indexes].view(batch_size, -1, dim)
        query_embeds = hidden_states[query_embed_token_indexes].unsqueeze(1)

        doc_embeds = self.projector(doc_embeds)
        query_embeds = self.projector(query_embeds)

        query_embeds_expanded = query_embeds.expand_as(doc_embeds)
        scores = torch.nn.functional.cosine_similarity(doc_embeds, query_embeds_expanded, dim=-1).squeeze(-1)

        return CausalLMOutputWithScores(
            loss=None,
            logits=None,
            scores=scores,
            query_embeds=query_embeds,
            doc_embeds=doc_embeds,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _ensure_tokenizer(self):
        if not hasattr(self, "_tokenizer"):
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.name_or_path, trust_remote_code=True)

            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.unk_token
                self._tokenizer.pad_token_id = self._tokenizer.convert_tokens_to_ids(self._tokenizer.pad_token)

            self._tokenizer.padding_side = 'left'

    def _truncate_texts(
        self,
        query: str,
        documents: List[str],
        max_query_length: int = 512,
        max_doc_length: int = 2048,
    ) -> Tuple[str, List[str], List[int], int]:
        self._ensure_tokenizer()

        docs = []
        doc_lengths = []
        for doc in documents:
            doc_tokens = self._tokenizer(doc, truncation=True, max_length=max_doc_length)
            if len(doc_tokens['input_ids']) >= max_doc_length:
                doc = self._tokenizer.decode(doc_tokens['input_ids'])
            doc_lengths.append(len(doc_tokens['input_ids']))
            docs.append(doc)

        query_tokens = self._tokenizer(query, truncation=True, max_length=max_query_length)
        if len(query_tokens['input_ids']) >= max_query_length:
            query = self._tokenizer.decode(query_tokens['input_ids'])

        query_length = len(query_tokens['input_ids'])

        return query, docs, doc_lengths, query_length

    def _compute_single_batch(
        self,
        query: str,
        docs: List[str],
        instruction: Optional[str] = None,
    ) -> CausalLMOutputWithScores:
        self._ensure_tokenizer()
        device = next(self.parameters()).device

        prompt = format_docs_prompts_func(
            query,
            docs,
            instruction=instruction,
            special_tokens=self.special_tokens,
            no_thinking=True,
        )

        batch = self._tokenizer(
            text=[prompt],
            padding=True,
            padding_side="left",
            return_tensors="pt",
        ).to(device)

        return self.forward(**batch)

    def _calculate_cosine_scores(
        self,
        query_embeddings: np.ndarray,
        doc_embeddings: np.ndarray,
    ) -> np.ndarray:
        return np.dot(query_embeddings, doc_embeddings.T) / (
            np.linalg.norm(query_embeddings) * np.linalg.norm(doc_embeddings, axis=1)
        )

    @torch.no_grad()
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        return_embeddings: bool = False,
        max_doc_length: int = 2048,
        max_query_length: int = 512,
    ) -> List[dict]:
        """
        Rerank documents by relevance to a query.

        Args:
            query: Search query string
            documents: List of document strings to rank
            top_n: Return only top N results (default: all)
            return_embeddings: Include embeddings in output (default: False)

        Returns:
            List of dicts with keys:
                - document: Original document text
                - relevance_score: Similarity score (higher = more relevant)
                - index: Position in input documents list
                - embedding: Doc embedding if return_embeddings=True, else None
        """
        self._ensure_tokenizer()

        # Derived from model configuration
        max_length = self._tokenizer.model_max_length

        # Derive block_size from max_length to fit documents efficiently
        # Heuristic: allow ~125 docs per batch for typical doc sizes
        block_size = 125

        query, docs, doc_lengths, query_length = self._truncate_texts(query, documents, max_query_length, max_doc_length)

        length_capacity = max_length - 2 * query_length

        block_docs = []
        doc_embeddings = []
        query_embeddings = []
        block_weights = []

        for length, doc in zip(doc_lengths, docs):
            block_docs.append(doc)
            length_capacity -= length

            if len(block_docs) >= block_size or length_capacity <= max_doc_length:
                outputs = self._compute_single_batch(query, block_docs, instruction=None)

                doc_embeddings.extend(outputs.doc_embeds[0].cpu().float().numpy())
                query_embeddings.append(outputs.query_embeds[0].cpu().float().numpy())
                scores = outputs.scores.view(-1).cpu().float().numpy()
                block_weights.append(((1.0 + scores) / 2.0).max())

                block_docs = []
                length_capacity = max_length - 2 * query_length

        if len(block_docs) > 0:
            outputs = self._compute_single_batch(query, block_docs, instruction=None)

            doc_embeddings.extend(outputs.doc_embeds[0].cpu().float().numpy())
            query_embeddings.append(outputs.query_embeds[0].cpu().float().numpy())
            scores = outputs.scores.view(-1).cpu().float().numpy()
            block_weights.append(((1.0 + scores) / 2.0).max())

        query_embeddings = np.array(query_embeddings)
        doc_embeddings = np.array(doc_embeddings)

        query_embeddings = np.average(query_embeddings, axis=0, weights=block_weights)

        scores = self._calculate_cosine_scores(query_embeddings, doc_embeddings)

        scores_argsort = np.argsort(scores[0])[::-1]

        # Derive top_n: if not specified, return all documents
        if top_n is None:
            top_n = len(documents)
        else:
            top_n = min(top_n, len(documents))

        return [
            {
                'document': documents[scores_argsort[i]],
                'relevance_score': scores[0][scores_argsort[i]],
                'index': scores_argsort[i],
                'embedding': doc_embeddings[scores_argsort[i]] if return_embeddings else None,
            }
            for i in range(top_n)
        ]
