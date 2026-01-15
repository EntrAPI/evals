from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .batch_sampler import BatchMHSampler
from .correctness import extract_gsm8k_answer, is_correct_gsm8k
from .lm.base import BaseLM, SamplingParams
from .persona import PersonaVectorSet
from .config import PersonaSpec, SamplerConfig


@dataclass(frozen=True)
class EvalItem:
    prompt: str
    gold: str


@dataclass(frozen=True)
class EvalRow:
    prompt: str
    gold: str
    base_text: str
    mh_text: str
    base_pred: str | None
    mh_pred: str | None
    base_correct: bool
    mh_correct: bool
    accept_rate: float
    persona_scores: dict[str, float]


@dataclass(frozen=True)
class EvalSummary:
    n: int
    base_acc: float
    mh_acc: float


def run_gsm8k_eval(
    *,
    lm: BaseLM,
    items: Sequence[EvalItem],
    base_params: SamplingParams,
    mh_cfg: SamplerConfig,
    mh_steps: int,
    persona_vectors: PersonaVectorSet | None = None,
    persona_specs: Sequence[PersonaSpec] | None = None,
    batch_size: int = 8,
) -> tuple[EvalSummary, list[EvalRow]]:
    persona_vectors = persona_vectors or PersonaVectorSet([])
    persona_specs = list(persona_specs or [])

    sampler = BatchMHSampler(
        lm=lm,
        persona_vectors=persona_vectors,
        persona_specs=persona_specs,
        cfg=mh_cfg,
    )

    rows: list[EvalRow] = []
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        prompts = [c.prompt for c in chunk]

        prompt_ids = [lm.encode(p) for p in prompts]
        suffixes = lm.sample_suffix_batch(prompt_ids, params=base_params)
        base_texts = [lm.decode(lm.ensure_ended(s, max_new_tokens=base_params.max_new_tokens)) for s in suffixes]

        mh_res = sampler.sample(prompts, num_steps=int(mh_steps), init_responses=base_texts, show_progress=False)
        mh_texts = [sampler.decode(s) for s in mh_res.finals]

        accept_rates = []
        for i in range(len(chunk)):
            accepted = sum(1 for st in mh_res.stats if st.accepted[i])
            accept_rates.append(accepted / max(1, len(mh_res.stats)))

        for i, ex in enumerate(chunk):
            gold = ex.gold
            base_pred = extract_gsm8k_answer(base_texts[i])
            mh_pred = extract_gsm8k_answer(mh_texts[i])
            rows.append(
                EvalRow(
                    prompt=ex.prompt,
                    gold=gold,
                    base_text=base_texts[i],
                    mh_text=mh_texts[i],
                    base_pred=base_pred,
                    mh_pred=mh_pred,
                    base_correct=is_correct_gsm8k(base_texts[i], gold),
                    mh_correct=is_correct_gsm8k(mh_texts[i], gold),
                    accept_rate=float(accept_rates[i]),
                    persona_scores=dict(mh_res.finals[i].persona_scores),
                )
            )

    n = len(rows)
    base_acc = sum(1 for r in rows if r.base_correct) / max(1, n)
    mh_acc = sum(1 for r in rows if r.mh_correct) / max(1, n)
    return EvalSummary(n=n, base_acc=base_acc, mh_acc=mh_acc), rows


def rows_to_jsonl(rows: Sequence[EvalRow]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        out.append(
            {
                "prompt": r.prompt,
                "gold": r.gold,
                "base_pred": r.base_pred,
                "mh_pred": r.mh_pred,
                "base_correct": r.base_correct,
                "mh_correct": r.mh_correct,
                "accept_rate": r.accept_rate,
                "persona_scores": r.persona_scores,
                "base_text": r.base_text,
                "mh_text": r.mh_text,
            }
        )
    return out

