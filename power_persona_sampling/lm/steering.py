from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SteeringVector:
    layer: int
    vector: torch.Tensor
    scale: float
    positions: str = "last"  # "last" | "all"


def _extract_hidden(output):
    if isinstance(output, tuple):
        return output[0], output, True
    return output, output, False


def _replace_hidden(output, hidden, is_tuple: bool):
    if not is_tuple:
        return hidden
    out = list(output)
    out[0] = hidden
    return tuple(out)


def _get_decoder_layers(model) -> Sequence[torch.nn.Module]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return model.model.decoder.layers
    raise ValueError("unable to locate decoder layers for steering")


@contextmanager
def hf_steering_context(
    model,
    steering: Sequence[SteeringVector],
    *,
    enabled: dict[str, bool],
) -> Iterator[None]:
    if not steering:
        yield
        return

    layers = _get_decoder_layers(model)
    handles: list[torch.utils.hooks.RemovableHandle] = []

    by_layer: dict[int, list[SteeringVector]] = {}
    for sv in steering:
        by_layer.setdefault(int(sv.layer), []).append(sv)

    def make_hook(vs: list[SteeringVector]):
        cache: dict[tuple[torch.device, torch.dtype, int], torch.Tensor] = {}

        def hook(_module, _inputs, output):
            if not enabled.get("on", False):
                return output
            hidden, raw, is_tuple = _extract_hidden(output)
            if not isinstance(hidden, torch.Tensor):
                return output
            for sv in vs:
                key = (hidden.device, hidden.dtype, id(sv.vector))
                vec = cache.get(key)
                if vec is None:
                    vec = sv.vector.to(device=hidden.device, dtype=hidden.dtype)
                    cache[key] = vec
                if sv.positions == "all":
                    hidden.add_(vec.view(1, 1, -1) * float(sv.scale))
                else:
                    hidden[:, -1, :].add_(vec.view(1, -1) * float(sv.scale))
            return _replace_hidden(raw, hidden, is_tuple=is_tuple)

        return hook

    try:
        for layer_idx, vs in by_layer.items():
            if layer_idx < 0 or layer_idx >= len(layers):
                raise ValueError(f"layer out of range: {layer_idx} (num_layers={len(layers)})")
            handles.append(layers[layer_idx].register_forward_hook(make_hook(vs)))
        yield
    finally:
        for h in handles:
            h.remove()
