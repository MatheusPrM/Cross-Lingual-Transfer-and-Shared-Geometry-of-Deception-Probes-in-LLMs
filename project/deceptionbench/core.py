"""DeceptionBench scenario schema and strict bilingual JSON loader."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


Role = Literal["self", "other"]
Level = Literal["L1", "L2"]
Inducement = Literal["none", "pressure", "reward"]
DBLang = Literal["en", "pt", "es", "de", "zh"]


CONDITIONS: tuple[str, ...] = (
    "L1-self",
    "L1-other",
    "L2-self-pressure",
    "L2-self-reward",
    "L2-other-pressure",
    "L2-other-reward",
)


DOMAIN_FILES: dict[str, str] = {
    "economy": "economy.json",
    "education": "education.json",
    "entertainment": "entertainment.json",
    "healthcare": "healthcare.json",
    "social": "social.json",
}


class MissingLocalizedFieldError(ValueError):
    """Raised when a non-en cell requests a localized field that is absent."""


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    domain: str
    row_index: int
    topic: str
    dimension: str
    question: str
    groundtruth: str
    goal: str
    prompts: dict[str, str]

    def prompt_for(self, condition: str) -> str:
        if condition not in self.prompts:
            raise KeyError(f"unknown condition {condition!r} (valid: {CONDITIONS})")
        return self.prompts[condition]


def parse_condition(condition: str) -> tuple[Level, Role, Inducement]:
    parts = condition.split("-")
    if len(parts) == 2:
        level, role = parts
        return level, role, "none"  
    if len(parts) == 3:
        level, role, inducement = parts
        return level, role, inducement  
    raise ValueError(f"malformed condition {condition!r}")


def _suffix(lang: str) -> str:
    lang = (lang or "en").strip().lower()
    return "" if lang == "en" else f"_{lang}"


def _required_field(
    row: dict[str, Any],
    key: str,
    *,
    lang: str,
    row_idx: int,
    domain: str,
    allow_en_fallback: bool,
) -> str:
    suf = _suffix(lang)
    if suf:
        loc_key = f"{key}{suf}"
        if loc_key in row and isinstance(row[loc_key], str) and row[loc_key].strip():
            return str(row[loc_key])
        if not allow_en_fallback:
            raise MissingLocalizedFieldError(
                f"missing localized field {loc_key!r} for row {row_idx} in domain {domain!r} "
                f"(lang={lang}); enable --allow-en-fallback only for debug"
            )
    val = row.get(key)
    if not isinstance(val, str) or not val.strip():
        raise ValueError(
            f"missing required field {key!r} for row {row_idx} in domain {domain!r}"
        )
    return str(val)


def load_deceptionbench_domain(
    path: Path,
    *,
    domain: str,
    lang: str = "en",
    strict: bool = True,
    allow_en_fallback: bool = False,
) -> list[Scenario]:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"DeceptionBench domain JSON must be a list: {path}")

    effective_allow_en = allow_en_fallback and not strict
    out: list[Scenario] = []
    for idx, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"non-dict row {idx} in {path}")
        prompts: dict[str, str] = {}
        for cond in CONDITIONS:
            prompts[cond] = _required_field(
                row,
                cond,
                lang=lang,
                row_idx=idx,
                domain=domain,
                allow_en_fallback=effective_allow_en,
            )
        question = _required_field(
            row, "question", lang=lang, row_idx=idx, domain=domain,
            allow_en_fallback=effective_allow_en,
        )
        groundtruth = _required_field(
            row, "groundtruth", lang=lang, row_idx=idx, domain=domain,
            allow_en_fallback=effective_allow_en,
        )
        goal = _required_field(
            row, "goal", lang=lang, row_idx=idx, domain=domain,
            allow_en_fallback=effective_allow_en,
        )
        topic = str(row.get("topic") or "")
        dimension = str(row.get("dimension") or "")
        out.append(
            Scenario(
                scenario_id=f"db__{domain}__{idx:03d}",
                domain=domain,
                row_index=idx,
                topic=topic,
                dimension=dimension,
                question=question,
                groundtruth=groundtruth,
                goal=goal,
                prompts=prompts,
            )
        )
    return out


def load_deceptionbench_all(
    data_dir: Path,
    *,
    lang: str = "en",
    strict: bool = True,
    allow_en_fallback: bool = False,
    domains: tuple[str, ...] | None = None,
) -> list[Scenario]:
    data_dir = Path(data_dir)
    chosen = tuple(domains) if domains else tuple(DOMAIN_FILES.keys())
    out: list[Scenario] = []
    suf = _suffix(lang)
    for d in chosen:
        if d not in DOMAIN_FILES:
            raise ValueError(f"unknown domain {d!r}")
        candidate = data_dir / f"{Path(DOMAIN_FILES[d]).stem}{suf}_en.json" if suf else None
        bilingual = data_dir / f"{Path(DOMAIN_FILES[d]).stem}_en_{lang}.json" if suf else None
        en_only = data_dir / DOMAIN_FILES[d]
        path = None
        for cand in [p for p in [bilingual, candidate, en_only] if p is not None]:
            if cand.exists():
                path = cand
                break
        if path is None:
            raise FileNotFoundError(
                f"no DeceptionBench domain file found for {d!r} under {data_dir} "
                f"(lang={lang}); tried {bilingual}, {candidate}, {en_only}"
            )
        out.extend(
            load_deceptionbench_domain(
                path,
                domain=d,
                lang=lang,
                strict=strict,
                allow_en_fallback=allow_en_fallback,
            )
        )
    return out
