from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..types import stack_pad, stack_pad_left
from .base import (
    BaseLM,
    BatchScoreRequest,
    BatchScoreResult,
    SamplingParams,
    nucleus_filter,
    log_softmax_select,
)
from .steering import SteeringVector, hf_steering_context


@dataclass(frozen=True)
class HFModelConfig:
    model_name_or_path: str
    torch_dtype: str | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class HuggingFaceLM(BaseLM):
    def __init__(self, cfg: HFModelConfig):
        tok = AutoTokenizer.from_pretrained(cfg.model_name_or_path, use_fast=True)
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token

        dtype = None
        if cfg.torch_dtype:
            dtype = getattr(torch, cfg.torch_dtype)

        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model_name_or_path,
                dtype=dtype,
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model_name_or_path,
                torch_dtype=dtype,
            )
        model.to(cfg.device)
        model.eval()

        self._tok = tok
        self._model = model
        self._device = torch.device(cfg.device)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def eos_token_id(self) -> int:
        return int(self._tok.eos_token_id)

    @property
    def pad_token_id(self) -> int:
        return int(self._tok.pad_token_id)

    def encode(self, text: str) -> list[int]:
        return list(self._tok.encode(text, add_special_tokens=False))

    def decode(self, ids: Sequence[int]) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=True)

    @torch.inference_mode()
    def sample_suffix(self, prefix_ids: Sequence[int], params: SamplingParams) -> list[int]:
        input_ids = torch.tensor([list(prefix_ids)], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        out = self._model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=float(params.temperature),
            top_p=float(params.top_p),
            max_new_tokens=int(params.max_new_tokens),
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
        )
        full = out[0].tolist()
        return full[len(prefix_ids) :]

    @torch.inference_mode()
    def sample_suffix_batch(
        self, prefix_ids_batch: Sequence[Sequence[int]], params: SamplingParams
    ) -> list[list[int]]:
        if not prefix_ids_batch:
            return []
        input_ids = stack_pad_left(prefix_ids_batch, pad_id=self.pad_token_id).to(self.device)
        attention_mask = torch.zeros_like(input_ids)
        for i, ids in enumerate(prefix_ids_batch):
            attention_mask[i, -len(ids) :] = 1
        out = self._model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=float(params.temperature),
            top_p=float(params.top_p),
            max_new_tokens=int(params.max_new_tokens),
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
        )
        full = out.tolist()
        suffixes: list[list[int]] = []
        input_len = int(input_ids.shape[1])
        for ids, seq in zip(prefix_ids_batch, full, strict=True):
            suffixes.append(seq[input_len:])
        return suffixes

    @torch.inference_mode()
    def sample_suffix_batch_custom(
        self,
        prefix_ids_batch: Sequence[Sequence[int]],
        params: SamplingParams,
        *,
        steering: Sequence[SteeringVector] = (),
        seed: int | None = None,
        greedy: bool = False,
        steer_only_generated: bool = True,
    ) -> list[list[int]]:
        if not prefix_ids_batch:
            return []

        device = self.device
        input_ids = stack_pad_left(prefix_ids_batch, pad_id=self.pad_token_id).to(device)
        attention_mask = torch.zeros_like(input_ids)
        for i, ids in enumerate(prefix_ids_batch):
            attention_mask[i, -len(ids) :] = 1

        gen = torch.Generator(device=device)
        if seed is not None:
            gen.manual_seed(int(seed))

        enabled = {"on": not steer_only_generated}
        with hf_steering_context(self._model, steering, enabled=enabled):
            out = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
            past = out.past_key_values
            logits = out.logits[:, -1, :]

            if steer_only_generated:
                enabled["on"] = True

            batch = int(input_ids.shape[0])
            eos = torch.full((batch,), int(self.eos_token_id), device=device, dtype=torch.long)
            done = torch.zeros((batch,), device=device, dtype=torch.bool)

            suffixes: list[list[int]] = [[] for _ in range(batch)]
            cur_mask = attention_mask

            for _step in range(int(params.max_new_tokens)):
                if greedy or float(params.temperature) <= 0.0:
                    next_ids = torch.argmax(logits, dim=-1).to(torch.long)
                else:
                    z = logits / float(params.temperature)
                    z = nucleus_filter(z, float(params.top_p))
                    probs = torch.softmax(z, dim=-1)
                    next_ids = torch.multinomial(probs, num_samples=1, generator=gen).squeeze(-1).to(torch.long)
                next_ids = torch.where(done, eos, next_ids)

                for b in range(batch):
                    suffixes[b].append(int(next_ids[b].item()))
                done |= next_ids.eq(eos)
                if bool(done.all().item()):
                    break

                cur_mask = torch.cat(
                    [cur_mask, torch.ones((batch, 1), device=device, dtype=cur_mask.dtype)], dim=1
                )
                out = self._model(
                    input_ids=next_ids.unsqueeze(-1),
                    attention_mask=cur_mask,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
                past = out.past_key_values
                logits = out.logits[:, -1, :]

        return suffixes

    @torch.inference_mode()
    def score_batch(self, req: BatchScoreRequest) -> BatchScoreResult:
        input_ids = req.input_ids.to(self.device)
        attention_mask = req.attention_mask.to(self.device)

        out = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=bool(req.layers),
            return_dict=True,
        )

        logits = out.logits
        hidden_states = out.hidden_states

        batch, seqlen, _ = logits.shape
        assert batch == len(req.prompt_lens)

        logp_response: list[float] = []
        logp_suffix: list[float] = []
        for b in range(batch):
            prompt_len = req.prompt_lens[b]
            response_len = req.response_lens[b]
            full_len = prompt_len + response_len
            if response_len <= 0:
                logp_response.append(0.0)
                logp_suffix.append(0.0)
                continue

            pos_start = max(prompt_len - 1, 0)
            pos_end = full_len - 1
            pred_logits = logits[b, pos_start:pos_end, :]
            target = input_ids[b, pos_start + 1 : pos_end + 1]
            token_logp = log_softmax_select(pred_logits, target)
            token_logp = token_logp * attention_mask[b, pos_start + 1 : pos_end + 1]
            token_logp_resp = token_logp[-response_len:]
            logp_response.append(float(token_logp_resp.sum().item()))

            start = 0
            if req.suffix_start is not None:
                start = int(req.suffix_start[b])
                start = max(0, min(start, response_len))
            logp_suffix.append(float(token_logp_resp[start:].sum().item()))

        pooled_by_layer: dict[int, torch.Tensor] = {}
        if not req.layers:
            return BatchScoreResult(
                logp_response=logp_response, logp_suffix=logp_suffix, pooled_by_layer=pooled_by_layer
            )

        for layer in req.layers:
            hs = hidden_states[layer]
            pooled = []
            for b in range(batch):
                prompt_len = req.prompt_lens[b]
                response_len = req.response_lens[b]
                full_len = prompt_len + response_len
                if response_len <= 0:
                    pooled.append(torch.zeros(hs.shape[-1], device=hs.device, dtype=hs.dtype))
                    continue

                if req.pool_response_only:
                    start = prompt_len
                    end = full_len
                else:
                    start = 0
                    end = full_len
                mask = attention_mask[b, start:end].unsqueeze(-1)
                denom = mask.sum().clamp(min=1.0)
                pooled.append((hs[b, start:end, :] * mask).sum(dim=0) / denom)
            pooled_by_layer[layer] = torch.stack(pooled, dim=0)

        return BatchScoreResult(
            logp_response=logp_response, logp_suffix=logp_suffix, pooled_by_layer=pooled_by_layer
        )


def make_score_request(
    prompt_ids_batch: Sequence[Sequence[int]],
    response_ids_batch: Sequence[Sequence[int]],
    pad_id: int,
    layers: Sequence[int],
    pool_response_only: bool = True,
    suffix_start: Sequence[int] | None = None,
) -> BatchScoreRequest:
    full = [[*p, *r] for p, r in zip(prompt_ids_batch, response_ids_batch, strict=True)]
    input_ids = stack_pad(full, pad_id=pad_id)
    attention_mask = torch.zeros_like(input_ids)
    for i, ids in enumerate(full):
        attention_mask[i, : len(ids)] = 1
    prompt_lens = [len(p) for p in prompt_ids_batch]
    response_lens = [len(r) for r in response_ids_batch]
    if suffix_start is not None:
        if len(suffix_start) != len(full):
            raise ValueError("suffix_start must match batch size")
    return BatchScoreRequest(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lens=prompt_lens,
        response_lens=response_lens,
        suffix_start=suffix_start,
        pool_response_only=pool_response_only,
        layers=tuple(layers),
    )
