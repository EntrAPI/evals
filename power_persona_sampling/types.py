from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


TokenIds = list[int]


@dataclass(frozen=True)
class Trajectory:
    prompt_ids: TokenIds
    response_ids: TokenIds

    def full_ids(self) -> TokenIds:
        return [*self.prompt_ids, *self.response_ids]

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_ids)

    @property
    def response_len(self) -> int:
        return len(self.response_ids)


@dataclass(frozen=True)
class ScoredTrajectory:
    traj: Trajectory
    logp_response: float
    logp_suffix: float
    persona_scores: dict[str, float]
    pooled_by_layer: dict[int, torch.Tensor]

    def persona_delta(self, other: "ScoredTrajectory", weights: dict[str, float]) -> float:
        total = 0.0
        for name, w in weights.items():
            total += w * (other.persona_scores[name] - self.persona_scores[name])
        return total


def stack_pad(seqs: Sequence[Sequence[int]], pad_id: int) -> torch.Tensor:
    max_len = max(len(s) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out


def stack_pad_left(seqs: Sequence[Sequence[int]], pad_id: int) -> torch.Tensor:
    max_len = max(len(s) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        s = torch.tensor(s, dtype=torch.long)
        out[i, max_len - len(s) :] = s
    return out
