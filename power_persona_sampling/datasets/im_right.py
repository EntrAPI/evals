from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from tqdm import tqdm

from ..correctness import LabeledResponse, is_correct
from ..io_utils import read_jsonl
from ..lm.base import BaseLM, SamplingParams


@dataclass(frozen=True)
class Problem:
    id: str
    prompt: str
    answer: str


def iter_problems_jsonl(path: str) -> Iterable[Problem]:
    for row in read_jsonl(path):
        yield Problem(
            id=str(row.get("id", "")),
            prompt=str(row["prompt"]),
            answer=str(row["answer"]),
        )


class ImRightDatasetBuilder:
    def __init__(
        self,
        lm: BaseLM,
        *,
        persona_name: str = "im_right",
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_new_tokens: int = 256,
        batch_size: int = 8,
        max_rounds: int = 50,
    ):
        self._lm = lm
        self._persona = persona_name
        self._params = SamplingParams(
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
        )
        self._batch_size = int(batch_size)
        self._max_rounds = int(max_rounds)

    def build(
        self,
        problems: Sequence[Problem],
        *,
        target_pos: int = 4,
        target_neg: int = 4,
        show_progress: bool = True,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        it = problems
        if show_progress:
            it = tqdm(list(problems), desc="build im_right dataset")

        for p in it:
            rows.extend(
                self._build_for_problem(
                    p,
                    target_pos=int(target_pos),
                    target_neg=int(target_neg),
                )
            )
        return rows

    def _build_for_problem(self, p: Problem, *, target_pos: int, target_neg: int) -> list[dict[str, Any]]:
        prompt_ids = self._lm.encode(p.prompt)
        pos: list[LabeledResponse] = []
        neg: list[LabeledResponse] = []

        for _ in range(self._max_rounds):
            if len(pos) >= target_pos and len(neg) >= target_neg:
                break

            batch_prompts = [prompt_ids for _ in range(self._batch_size)]
            suffixes = self._lm.sample_suffix_batch(batch_prompts, params=self._params)
            responses = []
            for s in suffixes:
                s = self._lm.ensure_ended(s, max_new_tokens=self._params.max_new_tokens)
                responses.append(self._lm.decode(s))

            for r in responses:
                ok = is_correct(r, p.answer)
                ex = LabeledResponse(prompt=p.prompt, response=r, label=1 if ok else 0, gold=p.answer)
                if ok and len(pos) < target_pos:
                    pos.append(ex)
                if (not ok) and len(neg) < target_neg:
                    neg.append(ex)
                if len(pos) >= target_pos and len(neg) >= target_neg:
                    break

        out: list[dict[str, Any]] = []
        for ex in [*pos, *neg]:
            out.append(
                {
                    "persona": self._persona,
                    "label": int(ex.label),
                    "prompt": ex.prompt,
                    "response": ex.response,
                    "gold": ex.gold,
                    "problem_id": p.id,
                }
            )
        return out


class ImRightGlobalDatasetBuilder:
    def __init__(
        self,
        lm: BaseLM,
        *,
        persona_name: str = "im_right",
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_new_tokens: int = 256,
        batch_size: int = 8,
        max_rounds_per_problem: int = 20,
    ):
        self._lm = lm
        self._persona = persona_name
        self._params = SamplingParams(
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
        )
        self._batch_size = int(batch_size)
        self._max_rounds_per_problem = int(max_rounds_per_problem)

    def build(
        self,
        problems: Sequence[Problem],
        *,
        target_pos: int,
        target_neg: int,
        show_progress: bool = True,
    ) -> list[dict[str, Any]]:
        want_pos = int(target_pos)
        want_neg = int(target_neg)
        if want_pos <= 0 or want_neg <= 0:
            raise ValueError("target_pos and target_neg must be positive")
        if not problems:
            return []

        pos: list[LabeledResponse] = []
        neg: list[LabeledResponse] = []

        it = problems
        if show_progress:
            it = tqdm(list(problems), desc="build im_right dataset (global)")

        for p in it:
            if len(pos) >= want_pos and len(neg) >= want_neg:
                break
            prompt_ids = self._lm.encode(p.prompt)

            for _round in range(self._max_rounds_per_problem):
                if len(pos) >= want_pos and len(neg) >= want_neg:
                    break
                batch_prompts = [prompt_ids for _ in range(self._batch_size)]
                suffixes = self._lm.sample_suffix_batch(batch_prompts, params=self._params)
                for s in suffixes:
                    if len(pos) >= want_pos and len(neg) >= want_neg:
                        break
                    s = self._lm.ensure_ended(s, max_new_tokens=self._params.max_new_tokens)
                    r = self._lm.decode(s)
                    ok = is_correct(r, p.answer)
                    ex = LabeledResponse(prompt=p.prompt, response=r, label=1 if ok else 0, gold=p.answer)
                    if ok and len(pos) < want_pos:
                        pos.append(ex)
                    if (not ok) and len(neg) < want_neg:
                        neg.append(ex)

        out: list[dict[str, Any]] = []
        for ex in [*pos, *neg]:
            out.append(
                {
                    "persona": self._persona,
                    "label": int(ex.label),
                    "prompt": ex.prompt,
                    "response": ex.response,
                    "gold": ex.gold,
                }
            )
        return out
