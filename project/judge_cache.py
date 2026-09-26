"""One-normalized-completion + one-judge-prompt-version = one stored judgment."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unicodedata
from pathlib import Path
from typing import Callable, Optional

_REPO = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = _REPO / "results" / "judge_cache"


def normalize_completion(text: str) -> str:
    return unicodedata.normalize("NFC", str(text)).strip()


def judgment_key(completion: str, *, prompt: str, judge_model: str) -> str:
    h = hashlib.sha256()
    h.update(normalize_completion(completion).encode("utf-8"))
    h.update(b"\x00")
    h.update(hashlib.sha256(prompt.encode("utf-8")).digest())  # prompt-version fingerprint
    h.update(b"\x00")
    h.update(judge_model.encode("utf-8"))
    return h.hexdigest()


def cached_judgment(
    completion: str,
    *,
    prompt: str,
    judge_model: str,
    call: Callable[[], str],
    parse: Optional[Callable[[str], tuple]] = None,
    cache_dir: Optional[Path] = None,
) -> dict:
    """Return the stored judgment for this (completion, prompt, judge) or call once and store."""
    cdir = Path(cache_dir or os.environ.get("JUDGE_CACHE_DIR", DEFAULT_CACHE_DIR))
    cdir.mkdir(parents=True, exist_ok=True)
    key = judgment_key(completion, prompt=prompt, judge_model=judge_model)
    path = cdir / f"{key}.json"
    if path.exists():
        rec = json.loads(path.read_text(encoding="utf-8"))
        rec["cache_hit"] = True
        return rec

    raw = call()
    label, rationale = parse(raw) if parse else (None, None)
    rec = {"key": key, "judge_model": judge_model, "label": label,
           "rationale": rationale, "raw": raw}
    fd, tmp = tempfile.mkstemp(dir=cdir, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False)
        os.replace(tmp, path)  # atomic; last writer wins with identical content
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    rec["cache_hit"] = False
    return rec
