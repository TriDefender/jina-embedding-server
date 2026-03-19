from typing import List, Optional
import os

import torch
import torch.nn.functional as F

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.models.qwen3 import Qwen3Model
from peft import PeftMixedModel, PeftConfig
from .configuration_jina_embeddings_v5 import JinaEmbeddingsV5Config


class JinaEmbeddingsV5Model(PeftMixedModel):
    @classmethod
    def register_for_auto_class(cls, auto_class="AutoModel"):
        return PreTrainedModel.register_for_auto_class.__func__(cls, auto_class)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs):
        if kwargs.get("config", None):
            base_config = kwargs.pop("config")
        else:
            base_config = JinaEmbeddingsV5Config.from_pretrained(
                pretrained_model_name_or_path,
            )
        base_model = Qwen3Model.from_pretrained(
            pretrained_model_name_or_path,
            config=base_config,
#s            dtype=kwargs.pop("dtype", torch.bfloat16),
        )
        
        if os.path.isdir(base_model.name_or_path):
            adapters_dir = os.path.join(base_model.name_or_path, "adapters")
        else:
            adapter_cache_path = snapshot_download(
                repo_id=base_model.name_or_path,
                allow_patterns=["adapters/*"],
            )
            adapters_dir = os.path.join(adapter_cache_path, "adapters")
        adapter_paths = {
            name: os.path.join(adapters_dir, name)
            for name in base_config.task_names
        }

        peft_config = PeftConfig.from_pretrained(adapter_paths["retrieval"], **kwargs)
        model = cls(base_model, peft_config, adapter_name="retrieval")
        model._pretrained_path = pretrained_model_name_or_path
        for adapter_name in base_config.task_names:
            model.load_adapter(
                adapter_paths[adapter_name],
                adapter_name=adapter_name,
                **kwargs,
            )

        model.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=True,
        )
        return model

    def encode(
        self,
        texts: List[str],
        task: str,
        prompt_name: Optional[str] = "document",
        truncate_dim: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> List[torch.Tensor]:
        if task not in self.base_model.config.task_names:
            raise ValueError(f"Unknown task: {task}")

        if prompt_name is None:
            prompt_name = "document"
        if prompt_name not in {"query", "document"}:
            raise ValueError(f"Unknown prompt_name: {prompt_name}")

        prefix = "Query: " if prompt_name == "query" else "Document: "
        inputs = [f"{prefix}{text}" for text in texts]

        if not hasattr(self, "tokenizer") or self.tokenizer is None:
            raise ValueError("Tokenizer not found on model. Load with from_pretrained().")

        max_length = max_length or self.config.max_position_embeddings
        batch = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        device = next(self.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items()}
        self.set_adapter([task])
        self.eval()
        with torch.no_grad():
            outputs = self(**batch)
            hidden = outputs.last_hidden_state
            mask = batch.get("attention_mask")
            if mask is None:
                pooled = hidden[:, -1]
            else:
                sequence_lengths = mask.sum(dim=1) - 1
                pooled = hidden[
                    torch.arange(hidden.shape[0], device=hidden.device),
                    sequence_lengths,
                ]

            if truncate_dim is not None:
                pooled = pooled[:, :truncate_dim]
            embeddings = F.normalize(pooled, p=2, dim=-1)

        return embeddings
