from typing import Any, Dict, List, Literal, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer


class Transformer(nn.Module):

    save_in_root: bool = True

    def __init__(
        self,
        model_name_or_path: str = "jinaai/jina-embeddings-v5-text-small",
        max_seq_length: Optional[int] = None,
        config_args: Optional[Dict[str, Any]] = None,
        model_args: Optional[Dict[str, Any]] = None,
        tokenizer_args: Optional[Dict[str, Any]] = None,
        cache_dir: Optional[str] = None,
        backend: Literal["torch", "onnx", "openvino"] = "torch",
        **kwargs,
    ) -> None:
        super(Transformer, self).__init__()
        if backend != "torch":
            raise ValueError(
                f"Backend '{backend}' is not supported, please use 'torch' instead"
            )
        config_kwargs = config_args or {}
        model_kwargs = model_args or {}

        self.config = AutoConfig.from_pretrained(
            model_name_or_path, cache_dir=cache_dir, **config_kwargs
        )
        self.default_task = model_args.pop("default_task", None)
        if self.default_task and self.default_task not in self.config.task_names:
            raise ValueError(
                f"Invalid task: {self.default_task}. Must be one of {self.config.task_names}."
            )

        self.model = AutoModel.from_pretrained(
            model_name_or_path, config=self.config, cache_dir=cache_dir, **model_kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        self.max_seq_length = max_seq_length or self.config.max_position_embeddings

    def tokenize(
        self, texts: List[str], padding: Union[str, bool] = True
    ) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            texts, max_length=self.max_seq_length, truncation=True, padding=padding, return_tensors="pt"
        )

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        task: Optional[str] = None,
        truncate_dim: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        self.model.eval()
        if task is None:
            if self.default_task is None:
                raise ValueError(
                    "Task must be specified before encoding data. You can set it either during "
                    "loading the model (e.g., model_kwargs={'default_task': 'retrieval'}) or "
                    "pass it as an argument to the encode method (e.g., model.encode(texts, task='retrieval'))."
                )
            task = self.default_task
        else:
            if task not in self.config.task_names:
                raise ValueError(
                    f"Invalid task: {task}. Must be one of {self.config.task_names}."
                )
        self.model.set_adapter(task)

        device = self.model.device

        with torch.no_grad():
            batch = {k: v.to(device) for k, v in features.items() if torch.is_tensor(v)}
            outputs = self.model(
                **batch
            )
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

        features["sentence_embedding"] = embeddings
        return features

    @classmethod
    def load(cls, input_path: str) -> "Transformer":
        return cls(model_name_or_path=input_path)

