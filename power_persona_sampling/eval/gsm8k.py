from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class GSM8KExample:
    question: str
    answer: str


def iter_gsm8k(split: str = "test") -> Iterable[GSM8KExample]:
    try:
        from datasets import load_dataset
    except Exception as e:  # pragma: no cover
        raise RuntimeError("missing dependency: install with `pip install -e '.[eval]'`") from e

    ds = load_dataset("gsm8k", "main", split=split)
    for row in ds:
        yield GSM8KExample(question=str(row["question"]), answer=str(row["answer"]))


def gsm8k_prompt(question: str) -> str:
    return gsm8k_prompt_style(question, style="short")


def gsm8k_prompt_style(question: str, *, style: str) -> str:
    q = question.strip()
    if style == "short":
        return (
            "Solve the following grade-school math problem.\n"
            "Give only the final integer answer on the last line as: "
            '"Final answer: X".\n\n'
            f"Problem:\n{q}\n"
        )
    if style == "cot":
        return (
            "Solve the following grade-school math problem. Show your reasoning, then give the final "
            'answer on the last line as: "Final answer: X".\n\n'
            f"Problem:\n{q}\n"
        )
    raise ValueError("style must be one of: short, cot")


def extract_gold_int(gold_answer: str) -> str:
    """
    GSM8K gold uses '#### <int>' at the end.
    """
    import re

    m = re.search(r"####\s*(-?[\d,]+)\s*$", gold_answer.strip())
    if not m:
        raise ValueError("unexpected GSM8K gold format")
    return m.group(1).replace(",", "")
