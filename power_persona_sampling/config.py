from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PersonaSpec:
    name: str
    layer: int
    beta: float = 1.0
    lam: float = 1.0


@dataclass(frozen=True)
class SamplerConfig:
    alpha: float = 1.0
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    seed: int | None = None
    burn_in: int = 0
    thin: int = 1
