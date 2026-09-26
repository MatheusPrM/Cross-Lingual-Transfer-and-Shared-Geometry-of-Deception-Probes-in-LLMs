"""
Data loading: read dataset files and return dialogues + labels.
"""

from pathlib import Path
from typing import Literal

from project.types import Dialogue, Label, Message


def _label_from_judgement(judgement: str | int | None) -> Label:
    """
    Map judgement to Label:
    - Integer 1-7 (honesty scale): 1-2 -> HONEST, 3-5 -> AMBIGUOUS, 6-7 -> DECEPTIVE.
    - String labels: "honest"/"0", "deceptive"/"1", "ambiguous", "skip".
    """
    if judgement is None:
        return Label.SKIP
    if isinstance(judgement, int):
        if judgement >= 6:
            return Label.DECEPTIVE
        if judgement <= 2:
            return Label.HONEST
        return Label.AMBIGUOUS
    j = str(judgement).strip().lower()
    if j in ("honest", "0"):
        return Label.HONEST
    if j in ("deceptive", "deceitful", "dishonest", "1"):
        return Label.DECEPTIVE
    if j == "ambiguous":
        return Label.AMBIGUOUS
    if j == "skip":
        return Label.SKIP
    if j.isdigit():
        return _label_from_judgement(int(j))
    return Label.SKIP


def load_repe(
    path: Path,
    *,
    category: str,
    max_rows: int | None = 512,
    variant: str = "you_are_fact_sys",
) -> tuple[list[Dialogue], list[Label]]:
    """Thin wrapper to `project.repe.loading.load_repe` (kept for compatibility)."""
    from project.repe.loading import load_repe as _load

    return _load(path, category=category, max_rows=max_rows, variant=variant)


def _normalize_rollout_items(data: object) -> list[dict]:
    """Parse rollout JSON into a list of item dicts (same rules as load_rollouts)."""
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        if "input_messages" in data and "output_strs" in data:
            return [data]
        items = data.get("rollouts", data.get("data", []))
        if isinstance(items, list):
            return [x for x in items if isinstance(x, dict)]
        return [data]
    return []


def load_rollouts(path: Path) -> tuple[list[Dialogue], list[Label]]:
    import json

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    items = _normalize_rollout_items(data)
    dialogues = []
    labels = []
    for item in items:
        input_messages = item.get("input_messages", [])
        output_strs = item.get("output_strs", [])
        graded = item.get("graded_deception", [])
        if len(graded) < len(output_strs):
            graded = list(graded) + [None] * (len(output_strs) - len(graded))
        for i in range(len(output_strs)):
            dial: Dialogue = []
            for m in input_messages:
                role = m.get("role", "user")
                content = (m.get("content") or "").strip()
                detect = m.get("detect", False)
                dial.append(Message(role=role, content=content, detect=detect))
            out_str = (output_strs[i] or "").rstrip()
            if dial and dial[-1].role == "assistant" and dial[-1].content == "":
                dial[-1] = Message(role="assistant", content=out_str, detect=True)
            else:
                dial.append(Message(role="assistant", content=out_str, detect=True))
            dialogues.append(dial)
            labels.append(_label_from_judgement(graded[i] if i < len(graded) else None))
    return dialogues, labels


def load_rollouts_with_extras(
    path: Path,
) -> tuple[list[Dialogue], list[Label], list[object | None], list[str]]:
    import json

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    items = _normalize_rollout_items(data)
    dialogues: list[Dialogue] = []
    labels: list[Label] = []
    graded_raw: list[object | None] = []
    dialogue_ids: list[str] = []
    for j, item in enumerate(items):
        input_messages = item.get("input_messages", [])
        output_strs = item.get("output_strs", [])
        graded = item.get("graded_deception", [])
        if len(graded) < len(output_strs):
            graded = list(graded) + [None] * (len(output_strs) - len(graded))
        for i in range(len(output_strs)):
            dial: Dialogue = []
            for m in input_messages:
                role = m.get("role", "user")
                content = (m.get("content") or "").strip()
                detect = m.get("detect", False)
                dial.append(Message(role=role, content=content, detect=detect))
            out_str = (output_strs[i] or "").rstrip()
            if dial and dial[-1].role == "assistant" and dial[-1].content == "":
                dial[-1] = Message(role="assistant", content=out_str, detect=True)
            else:
                dial.append(Message(role="assistant", content=out_str, detect=True))
            dialogues.append(dial)
            labels.append(_label_from_judgement(graded[i] if i < len(graded) else None))
            graded_raw.append(graded[i] if i < len(graded) else None)
            dialogue_ids.append(f"item{j}_completion{i}")
    return dialogues, labels, graded_raw, dialogue_ids


def load_dialogues_from_path(
    path: Path,
    format: Literal["repe", "rollouts"] | None = None,
    *,
    repe_category: str | None = None,
) -> tuple[list[Dialogue], list[Label]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if format is None:
        if suffix == ".csv":
            format = "repe"
        elif suffix == ".json":
            format = "rollouts"
        else:
            raise ValueError(f"Cannot infer format from extension '{suffix}'. Pass format= explicitly.")
    if format == "repe" or suffix == ".csv":
        if repe_category is None:
            raise ValueError(
                f"load_dialogues_from_path({path}): REPE CSV requires repe_category=. "
            )
        return load_repe(path, category=repe_category)
    if format == "rollouts" or suffix == ".json":
        return load_rollouts(path)
    raise ValueError(f"Unknown format: {format}")
