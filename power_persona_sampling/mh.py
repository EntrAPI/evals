from __future__ import annotations

from typing import Mapping


def log_accept_ratio_suffix_resample(
    *,
    alpha: float,
    logp_suffix_x: float,
    logp_suffix_xp: float,
    persona_x: Mapping[str, float],
    persona_xp: Mapping[str, float],
    persona_weights: Mapping[str, float],
) -> float:
    delta_logp = logp_suffix_xp - logp_suffix_x
    delta_persona = 0.0
    for name, w in persona_weights.items():
        delta_persona += float(w) * (float(persona_xp[name]) - float(persona_x[name]))
    return (float(alpha) - 1.0) * float(delta_logp) + float(delta_persona)
