from __future__ import annotations

from pathlib import Path


def read_prompts_txt(path: str | Path) -> list[str]:
    raw = Path(path).read_text(encoding="utf-8")
    parts = []
    buf: list[str] = []
    for line in raw.splitlines():
        if line.strip() == "---":
            s = "\n".join(buf).strip()
            if s:
                parts.append(s)
            buf = []
        else:
            buf.append(line)
    s = "\n".join(buf).strip()
    if s:
        parts.append(s)
    return parts

