from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

from tqdm import tqdm

from .config import PersonaSpec, SamplerConfig
from .lm.base import BaseLM, SamplingParams
from .lm.hf import make_score_request
from .mh import log_accept_ratio_suffix_resample
from .persona import PersonaVectorSet
from .types import ScoredTrajectory, Trajectory


@dataclass(frozen=True)
class BatchStepStats:
    accepted: list[bool]
    log_r: list[float]
    delta_logp: list[float]
    delta_persona: list[float]


@dataclass(frozen=True)
class BatchSamplerResult:
    finals: list[ScoredTrajectory]
    stats: list[BatchStepStats]


class BatchMHSampler:
    def __init__(
        self,
        lm: BaseLM,
        persona_vectors: PersonaVectorSet,
        persona_specs: Sequence[PersonaSpec],
        cfg: SamplerConfig,
    ):
        self._lm = lm
        self._vectors = persona_vectors
        self._specs = list(persona_specs)
        self._cfg = cfg
        self._layers = self._vectors.required_layers(self._specs)
        self._persona_weights = {s.name: float(s.beta) * float(s.lam) for s in self._specs}

        self._seed = cfg.seed if cfg.seed is not None else random.randrange(0, 2**31 - 1)

    def sample(
        self,
        prompts: Sequence[str],
        num_steps: int,
        *,
        init_responses: Sequence[str] | None = None,
        show_progress: bool = True,
    ) -> BatchSamplerResult:
        n = len(prompts)
        if n == 0:
            return BatchSamplerResult(finals=[], stats=[])
        rngs: list[random.Random] = [random.Random(self._seed + i * 10007) for i in range(n)]

        prompt_ids = [self._lm.encode(p) for p in prompts]
        if init_responses is None:
            init_response_ids = self._sample_responses(prompt_ids)
        else:
            if len(init_responses) != len(prompts):
                raise ValueError("init_responses must match prompts length")
            init_response_ids = [
                self._lm.ensure_ended(self._lm.encode(r), max_new_tokens=self._cfg.max_new_tokens)
                for r in init_responses
            ]
        cur = [
            Trajectory(prompt_ids=p, response_ids=r)
            for p, r in zip(prompt_ids, init_response_ids, strict=True)
        ]
        cur_scored = self._score_batch(cur)

        stats: list[BatchStepStats] = []
        it = range(int(num_steps))
        if show_progress:
            it = tqdm(list(it), desc="mh batch sampling")

        for t in it:
            accepted: list[bool] = []
            log_r: list[float] = []
            delta_logp: list[float] = []
            delta_persona: list[float] = []

            props, cuts = self._propose_batch([c.traj for c in cur_scored], rngs)
            pair_trajs: list[Trajectory] = []
            suffix_start: list[int] = []
            for i in range(n):
                pair_trajs.append(cur_scored[i].traj)
                pair_trajs.append(props[i])
                suffix_start.append(cuts[i])
                suffix_start.append(cuts[i])
            pair_scored = self._score_batch(pair_trajs, suffix_start=suffix_start)

            for i in range(n):
                x_idx = 2 * i
                xp_idx = 2 * i + 1
                xp = pair_scored[xp_idx]
                logp_suf_x = float(pair_scored[x_idx].logp_suffix)
                logp_suf_xp = float(pair_scored[xp_idx].logp_suffix)
                dl = float(logp_suf_xp - logp_suf_x)
                lr = log_accept_ratio_suffix_resample(
                    alpha=self._cfg.alpha,
                    logp_suffix_x=logp_suf_x,
                    logp_suffix_xp=logp_suf_xp,
                    persona_x=pair_scored[x_idx].persona_scores,
                    persona_xp=pair_scored[xp_idx].persona_scores,
                    persona_weights=self._persona_weights,
                )
                dp = float(lr - (self._cfg.alpha - 1.0) * dl)
                acc = self._accept(lr, rngs[i])
                accepted.append(acc)
                log_r.append(float(lr))
                delta_logp.append(dl)
                delta_persona.append(dp)
                if acc:
                    cur_scored[i] = xp

            stats.append(
                BatchStepStats(
                    accepted=accepted,
                    log_r=log_r,
                    delta_logp=delta_logp,
                    delta_persona=delta_persona,
                )
            )

        return BatchSamplerResult(finals=cur_scored, stats=stats)

    def decode(self, scored: ScoredTrajectory) -> str:
        return self._lm.decode(scored.traj.response_ids)

    def _accept(self, log_r: float, rng: random.Random) -> bool:
        if log_r >= 0:
            return True
        u = rng.random()
        return math.log(u) < log_r

    def _sample_responses(self, prompt_ids: Sequence[Sequence[int]]) -> list[list[int]]:
        params = SamplingParams(
            max_new_tokens=self._cfg.max_new_tokens,
            temperature=self._cfg.temperature,
            top_p=self._cfg.top_p,
        )
        sampled = self._lm.sample_suffix_batch(prompt_ids, params=params)
        return [self._lm.ensure_ended(s, max_new_tokens=self._cfg.max_new_tokens) for s in sampled]

    def _propose_batch(
        self, trajs: Sequence[Trajectory], rngs: Sequence[random.Random]
    ) -> tuple[list[Trajectory], list[int]]:
        cuts: list[int] = []
        prefixes: list[list[int]] = []
        prefix_full: list[list[int]] = []
        remaining: list[int] = []

        for traj, rng in zip(trajs, rngs, strict=True):
            resp = traj.response_ids
            cut = rng.randrange(0, len(resp)) if resp else 0
            pref = resp[:cut]
            cuts.append(cut)
            prefixes.append(pref)
            prefix_full.append([*traj.prompt_ids, *pref])
            remaining.append(max(1, self._cfg.max_new_tokens - len(pref)))

        bucket = 64
        max_tokens_by_idx: dict[int, int] = {}
        groups: dict[int, list[int]] = {}
        for idx, rem in enumerate(remaining):
            bucketed = int(((rem + bucket - 1) // bucket) * bucket)
            bucketed = min(bucketed, self._cfg.max_new_tokens)
            max_tokens_by_idx[idx] = bucketed
            groups.setdefault(bucketed, []).append(idx)

        suffixes: list[list[int]] = [[] for _ in prefix_full]
        for max_new, idxs in sorted(groups.items()):
            params = SamplingParams(
                max_new_tokens=max_new,
                temperature=self._cfg.temperature,
                top_p=self._cfg.top_p,
            )
            batch_prefix = [prefix_full[i] for i in idxs]
            batch_suffix = self._lm.sample_suffix_batch(batch_prefix, params=params)
            for i, suff in zip(idxs, batch_suffix, strict=True):
                suffixes[i] = suff

        out: list[Trajectory] = []
        for traj, pref, suff, rem in zip(trajs, prefixes, suffixes, remaining, strict=True):
            suff = suff[:rem]
            proposed = self._lm.ensure_ended(pref + suff, max_new_tokens=self._cfg.max_new_tokens)
            out.append(Trajectory(prompt_ids=traj.prompt_ids, response_ids=proposed))
        return out, cuts

    def _score_batch(
        self, trajs: Sequence[Trajectory], *, suffix_start: Sequence[int] | None = None
    ) -> list[ScoredTrajectory]:
        prompt_ids_batch = [t.prompt_ids for t in trajs]
        response_ids_batch = [t.response_ids for t in trajs]
        req = make_score_request(
            prompt_ids_batch=prompt_ids_batch,
            response_ids_batch=response_ids_batch,
            pad_id=self._lm.pad_token_id,
            layers=self._layers,
            pool_response_only=True,
            suffix_start=suffix_start,
        )
        res = self._lm.score_batch(req)
        if self._specs:
            persona_scores = self._vectors.score_from_pooled(res.pooled_by_layer, self._specs)
            pooled_single = [{k: v[i] for k, v in res.pooled_by_layer.items()} for i in range(len(trajs))]
        else:
            persona_scores = [{} for _ in trajs]
            pooled_single = [{} for _ in trajs]
        return [
            ScoredTrajectory(
                traj=trajs[i],
                logp_response=float(res.logp_response[i]),
                logp_suffix=float(res.logp_suffix[i]),
                persona_scores=persona_scores[i],
                pooled_by_layer=pooled_single[i],
            )
            for i in range(len(trajs))
        ]
