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
class StepStats:
    accepted: bool
    log_r: float
    delta_logp: float
    delta_persona: float


@dataclass(frozen=True)
class SamplerResult:
    final: ScoredTrajectory
    kept: list[ScoredTrajectory]
    stats: list[StepStats]


class MHSampler:
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

        seed = cfg.seed if cfg.seed is not None else random.randrange(0, 2**31 - 1)
        self._rng = random.Random(seed)

    def sample(
        self,
        prompt: str,
        num_steps: int,
        *,
        init_response: str | None = None,
        show_progress: bool = True,
    ) -> SamplerResult:
        prompt_ids = self._lm.encode(prompt)

        if init_response is None:
            init = self._sample_response(prompt_ids)
        else:
            init = self._lm.ensure_ended(
                self._lm.encode(init_response), max_new_tokens=self._cfg.max_new_tokens
            )

        cur_traj = Trajectory(prompt_ids=prompt_ids, response_ids=init)
        cur = self._score(cur_traj)

        kept: list[ScoredTrajectory] = []
        stats: list[StepStats] = []

        it = range(int(num_steps))
        if show_progress:
            it = tqdm(list(it), desc="mh sampling")

        for t in it:
            prop_traj, cut = self._propose(cur.traj)
            cur_suf, prop_suf, prop = self._score_pair(cur.traj, prop_traj, cut)

            delta_logp = prop_suf - cur_suf
            log_r = log_accept_ratio_suffix_resample(
                alpha=self._cfg.alpha,
                logp_suffix_x=cur_suf,
                logp_suffix_xp=prop_suf,
                persona_x=cur.persona_scores,
                persona_xp=prop.persona_scores,
                persona_weights=self._persona_weights,
            )
            delta_persona = log_r - (self._cfg.alpha - 1.0) * delta_logp
            accepted = self._accept(log_r)
            stats.append(
                StepStats(
                    accepted=accepted,
                    log_r=float(log_r),
                    delta_logp=float(delta_logp),
                    delta_persona=float(delta_persona),
                )
            )
            if accepted:
                cur = prop

            if t + 1 > self._cfg.burn_in and ((t + 1 - self._cfg.burn_in) % self._cfg.thin == 0):
                kept.append(cur)

        return SamplerResult(final=cur, kept=kept, stats=stats)

    def decode(self, scored: ScoredTrajectory) -> str:
        return self._lm.decode(scored.traj.response_ids)

    def _accept(self, log_r: float) -> bool:
        if log_r >= 0:
            return True
        u = self._rng.random()
        return math.log(u) < log_r

    def _propose(self, traj: Trajectory) -> tuple[Trajectory, int]:
        resp = traj.response_ids
        if len(resp) == 0:
            cut = 0
        else:
            cut = self._rng.randrange(0, len(resp))
        prefix_response = resp[:cut]
        prefix_full = [*traj.prompt_ids, *prefix_response]
        params = SamplingParams(
            max_new_tokens=max(1, self._cfg.max_new_tokens - len(prefix_response)),
            temperature=self._cfg.temperature,
            top_p=self._cfg.top_p,
        )
        suffix = self._lm.sample_suffix(prefix_full, params=params)
        proposed = self._lm.ensure_ended(prefix_response + suffix, max_new_tokens=self._cfg.max_new_tokens)
        return Trajectory(prompt_ids=traj.prompt_ids, response_ids=proposed), cut

    def _sample_response(self, prompt_ids: Sequence[int]) -> list[int]:
        params = SamplingParams(
            max_new_tokens=self._cfg.max_new_tokens,
            temperature=self._cfg.temperature,
            top_p=self._cfg.top_p,
        )
        sampled = self._lm.sample_suffix(list(prompt_ids), params=params)
        return self._lm.ensure_ended(sampled, max_new_tokens=self._cfg.max_new_tokens)

    def _score(self, traj: Trajectory) -> ScoredTrajectory:
        req = make_score_request(
            prompt_ids_batch=[traj.prompt_ids],
            response_ids_batch=[traj.response_ids],
            pad_id=self._lm.pad_token_id,
            layers=self._layers,
            pool_response_only=True,
        )
        res = self._lm.score_batch(req)
        if self._specs:
            persona_scores = self._vectors.score_from_pooled(res.pooled_by_layer, self._specs)[0]
            pooled_single = {k: v[0] for k, v in res.pooled_by_layer.items()}
        else:
            persona_scores = {}
            pooled_single = {}
        return ScoredTrajectory(
            traj=traj,
            logp_response=float(res.logp_response[0]),
            logp_suffix=float(res.logp_suffix[0]),
            persona_scores=persona_scores,
            pooled_by_layer=pooled_single,
        )

    def _score_pair(
        self, x: Trajectory, xp: Trajectory, cut: int
    ) -> tuple[float, float, ScoredTrajectory]:
        req = make_score_request(
            prompt_ids_batch=[x.prompt_ids, xp.prompt_ids],
            response_ids_batch=[x.response_ids, xp.response_ids],
            pad_id=self._lm.pad_token_id,
            layers=self._layers,
            pool_response_only=True,
            suffix_start=[cut, cut],
        )
        res = self._lm.score_batch(req)
        if self._specs:
            scores = self._vectors.score_from_pooled(res.pooled_by_layer, self._specs)
            persona_scores = scores[1]
            pooled_single = {k: v[1] for k, v in res.pooled_by_layer.items()}
        else:
            persona_scores = {}
            pooled_single = {}
        prop = ScoredTrajectory(
            traj=xp,
            logp_response=float(res.logp_response[1]),
            logp_suffix=float(res.logp_suffix[1]),
            persona_scores=persona_scores,
            pooled_by_layer=pooled_single,
        )
        return float(res.logp_suffix[0]), float(res.logp_suffix[1]), prop
