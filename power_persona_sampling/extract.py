from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from tqdm import tqdm

from .config import PersonaSpec
from .io_utils import read_jsonl
from .lm.base import BaseLM
from .lm.hf import make_score_request
from .persona import PersonaVector, PersonaVectorSet


@dataclass(frozen=True)
class PersonaExample:
    persona: str
    label: int
    prompt: str
    response: str


def iter_examples_jsonl(path: str) -> Iterable[PersonaExample]:
    for row in read_jsonl(path):
        yield PersonaExample(
            persona=str(row["persona"]),
            label=int(row["label"]),
            prompt=str(row["prompt"]),
            response=str(row["response"]),
        )


class PersonaVectorExtractor:
    def __init__(self, lm: BaseLM, specs: Sequence[PersonaSpec], batch_size: int = 8):
        self._lm = lm
        self._specs = list(specs)
        self._batch_size = int(batch_size)

    def extract(self, examples: Iterable[PersonaExample], normalize: bool = True) -> PersonaVectorSet:
        layers_by_name = {s.name: s.layer for s in self._specs}

        sum_pos: dict[str, torch.Tensor] = {}
        sum_neg: dict[str, torch.Tensor] = {}
        cnt_pos: dict[str, int] = {}
        cnt_neg: dict[str, int] = {}

        batch: list[PersonaExample] = []
        for ex in tqdm(examples, desc="extract persona vectors"):
            if ex.persona not in layers_by_name:
                continue
            batch.append(ex)
            if len(batch) < self._batch_size:
                continue
            self._accumulate_batch(batch, layers_by_name, sum_pos, sum_neg, cnt_pos, cnt_neg)
            batch = []

        if batch:
            self._accumulate_batch(batch, layers_by_name, sum_pos, sum_neg, cnt_pos, cnt_neg)

        vectors: list[PersonaVector] = []
        for name, layer in layers_by_name.items():
            if cnt_pos.get(name, 0) == 0 or cnt_neg.get(name, 0) == 0:
                raise ValueError(f"need both pos and neg examples for persona={name}")
            mu_pos = sum_pos[name] / float(cnt_pos[name])
            mu_neg = sum_neg[name] / float(cnt_neg[name])
            v = (mu_pos - mu_neg).to(torch.float32)
            if normalize:
                v = v / (v.norm(p=2).clamp(min=1e-12))
            vectors.append(PersonaVector(name=name, layer=layer, vector=v.cpu()))

        return PersonaVectorSet(vectors)

    def _accumulate_batch(
        self,
        batch: Sequence[PersonaExample],
        layers_by_name: dict[str, int],
        sum_pos: dict[str, torch.Tensor],
        sum_neg: dict[str, torch.Tensor],
        cnt_pos: dict[str, int],
        cnt_neg: dict[str, int],
    ) -> None:
        prompt_ids = [self._lm.encode(ex.prompt) for ex in batch]
        response_ids = [self._lm.ensure_ended(self._lm.encode(ex.response), max_new_tokens=10**9) for ex in batch]

        layers = sorted({layers_by_name[ex.persona] for ex in batch})
        req = make_score_request(
            prompt_ids_batch=prompt_ids,
            response_ids_batch=response_ids,
            pad_id=self._lm.pad_token_id,
            layers=layers,
            pool_response_only=True,
        )
        res = self._lm.score_batch(req)

        pooled_by_layer = {k: v.detach().cpu().to(torch.float32) for k, v in res.pooled_by_layer.items()}

        for i, ex in enumerate(batch):
            layer = layers_by_name[ex.persona]
            pooled = pooled_by_layer[layer][i]
            if ex.label == 1:
                sum_pos[ex.persona] = sum_pos.get(ex.persona, torch.zeros_like(pooled)) + pooled
                cnt_pos[ex.persona] = cnt_pos.get(ex.persona, 0) + 1
            else:
                sum_neg[ex.persona] = sum_neg.get(ex.persona, torch.zeros_like(pooled)) + pooled
                cnt_neg[ex.persona] = cnt_neg.get(ex.persona, 0) + 1
