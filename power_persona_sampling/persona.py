from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import torch

from .config import PersonaSpec


@dataclass(frozen=True)
class PersonaVector:
    name: str
    layer: int
    vector: torch.Tensor

    def score_pooled(self, pooled: torch.Tensor) -> torch.Tensor:
        v = self.vector
        if v.device != pooled.device or v.dtype != pooled.dtype:
            v = v.to(device=pooled.device, dtype=pooled.dtype)
        return (pooled * v).sum(dim=-1)


class PersonaVectorSet:
    def __init__(self, vectors: Iterable[PersonaVector]):
        by_name: dict[str, PersonaVector] = {}
        for v in vectors:
            if v.name in by_name:
                raise ValueError(f"duplicate persona vector name: {v.name}")
            by_name[v.name] = v
        self._by_name = by_name

    def names(self) -> list[str]:
        return sorted(self._by_name.keys())

    def get(self, name: str) -> PersonaVector:
        return self._by_name[name]

    def required_layers(self, specs: Iterable[PersonaSpec]) -> list[int]:
        layers = {self.get(s.name).layer for s in specs}
        return sorted(layers)

    def score_from_pooled(
        self,
        pooled_by_layer: Mapping[int, torch.Tensor],
        specs: Iterable[PersonaSpec],
    ) -> list[dict[str, float]]:
        specs = list(specs)
        batch = next(iter(pooled_by_layer.values())).shape[0] if pooled_by_layer else 0
        out: list[dict[str, float]] = [dict() for _ in range(batch)]
        for spec in specs:
            v = self.get(spec.name)
            pooled = pooled_by_layer[v.layer]
            scores = v.score_pooled(pooled)
            for i in range(batch):
                out[i][spec.name] = float(scores[i].item())
        return out

    @staticmethod
    def load(path: str | Path, map_location: str | torch.device = "cpu") -> "PersonaVectorSet":
        obj = torch.load(str(path), map_location=map_location)
        vectors = []
        for item in obj["vectors"]:
            vectors.append(
                PersonaVector(
                    name=str(item["name"]),
                    layer=int(item["layer"]),
                    vector=item["vector"].to(map_location),
                )
            )
        return PersonaVectorSet(vectors)

    def save(self, path: str | Path) -> None:
        payload = {
            "vectors": [
                {"name": v.name, "layer": v.layer, "vector": v.vector.detach().cpu()}
                for v in self._by_name.values()
            ]
        }
        torch.save(payload, str(path))
