from __future__ import annotations

import re
from dataclasses import dataclass


_FINAL_RE = re.compile(r"final\s*answer\s*:\s*([^\n\r]+)", flags=re.IGNORECASE)
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_INT_RE = re.compile(r"-?\d+")
_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
_HASH_RE = re.compile(r"^####\s*(.+)$", flags=re.MULTILINE)
_CODE_FENCE_RE = re.compile(r"```.*?```", flags=re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`]*`")


def extract_final_answer(text: str) -> str | None:
    m = _FINAL_RE.search(text)
    if m:
        return m.group(1).strip()
    nums = _NUM_RE.findall(text)
    if not nums:
        return None
    return nums[-1]


def normalize_answer(s: str) -> str:
    s = s.strip()
    s = s.replace(",", "")
    if s.startswith("$"):
        s = s[1:].strip()
    if s.endswith("."):
        s = s[:-1].strip()
    return s


def is_correct(pred_text: str, gold: str) -> bool:
    pred = extract_final_answer(pred_text)
    if pred is None:
        return False
    return normalize_answer(pred) == normalize_answer(gold)


def strip_code(text: str) -> str:
    text = _CODE_FENCE_RE.sub("\n", text)
    text = _INLINE_CODE_RE.sub("", text)
    return text


def extract_gsm8k_answer(text: str) -> str | None:
    """
    GSM8K answers are integers (as strings). This extractor:
    - removes fenced/inline code blocks (to avoid fake python outputs)
    - prefers explicit markers (Final answer:, ####, \\boxed{})
    - otherwise falls back to the last integer in the remaining text
    """
    t = strip_code(text)

    m = _FINAL_RE.search(t)
    if m:
        cand = m.group(1).strip()
        ints = _INT_RE.findall(cand.replace(",", ""))
        return ints[-1] if ints else normalize_answer(cand)

    m = _HASH_RE.search(t)
    if m:
        cand = m.group(1).strip()
        ints = _INT_RE.findall(cand.replace(",", ""))
        return ints[-1] if ints else normalize_answer(cand)

    m = _BOXED_RE.search(t)
    if m:
        cand = m.group(1).strip()
        ints = _INT_RE.findall(cand.replace(",", ""))
        return ints[-1] if ints else normalize_answer(cand)

    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    for ln in reversed(lines[-6:]):
        ints = _INT_RE.findall(ln.replace(",", ""))
        if ints:
            return ints[-1]

    ints = _INT_RE.findall(t.replace(",", ""))
    if not ints:
        return None
    return ints[-1]


def is_correct_gsm8k(pred_text: str, gold_int: str) -> bool:
    pred = extract_gsm8k_answer(pred_text)
    if pred is None:
        return False
    return normalize_answer(pred) == normalize_answer(gold_int)


@dataclass(frozen=True)
class LabeledResponse:
    prompt: str
    response: str
    label: int
    gold: str
