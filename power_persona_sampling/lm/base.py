from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class SamplingParams:
    max_new_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0


@dataclass(frozen=True)
class BatchScoreRequest:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_lens: Sequence[int]
    response_lens: Sequence[int]
    suffix_start: Sequence[int] | None
    pool_response_only: bool
    layers: Sequence[int]


@dataclass(frozen=True)
class BatchScoreResult:
    logp_response: list[float]
    logp_suffix: list[float]
    pooled_by_layer: dict[int, torch.Tensor]


class BaseLM(ABC):
    @property
    @abstractmethod
    def device(self) -> torch.device: ...

    @property
    @abstractmethod
    def eos_token_id(self) -> int: ...

    @property
    @abstractmethod
    def pad_token_id(self) -> int: ...

    @abstractmethod
    def encode(self, text: str) -> list[int]: ...

    @abstractmethod
    def decode(self, ids: Sequence[int]) -> str: ...

    @abstractmethod
    def sample_suffix(self, prefix_ids: Sequence[int], params: SamplingParams) -> list[int]:
        """
        Samples a continuation from p0(· | prefix). Returned ids are *new* tokens only.
        """

    def sample_suffix_batch(
        self, prefix_ids_batch: Sequence[Sequence[int]], params: SamplingParams
    ) -> list[list[int]]:
        return [self.sample_suffix(prefix_ids, params=params) for prefix_ids in prefix_ids_batch]

    @abstractmethod
    def score_batch(self, req: BatchScoreRequest) -> BatchScoreResult:
        """
        Computes teacher-forced response log-probs and pooled activations for the requested layers.
        """

    def ensure_ended(self, response_ids: list[int], max_new_tokens: int) -> list[int]:
        if not response_ids:
            return [self.eos_token_id]
        try:
            first_eos = response_ids.index(self.eos_token_id)
            response_ids = response_ids[: first_eos + 1]
        except ValueError:
            pass
        if response_ids[-1] == self.eos_token_id:
            if len(response_ids) <= max_new_tokens:
                return response_ids
            return [*response_ids[: max_new_tokens - 1], self.eos_token_id]
        if len(response_ids) >= max_new_tokens:
            return [*response_ids[: max_new_tokens - 1], self.eos_token_id]
        return [*response_ids, self.eos_token_id]


def log_softmax_select(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(logits, dim=-1)
    return logp.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)


def nucleus_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum = torch.cumsum(probs, dim=-1)
    mask = cum > top_p
    mask[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
    out = torch.full_like(logits, float("-inf"))
    out.scatter_(-1, sorted_idx, sorted_logits)
    return out
