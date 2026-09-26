"""
REPE dataset loading and dialogue construction.
"""

from __future__ import annotations

from pathlib import Path

from project.types import Dialogue, Label, Message


def _split_statement_repe(statement: str) -> tuple[str, str]:
    """
    Split a REPE training statement into (prefix-for-detection, suffix-for-completion).
    """
    words = statement.split(" ")
    has_cjk = any("一" <= ch <= "鿿" for ch in statement)  
    is_space_segmented = len(words) >= 2 and not has_cjk

    if is_space_segmented:
        start = " ".join(words[:-5])
        end = " ".join(words[-5:])
        if len(words) > 5:
            end = " " + end
        return start, end

    n_chars = len(statement)
    if n_chars <= 3:
        return "", statement
    suffix_len = max(3, min(15, n_chars // 5))
    start = statement[:-suffix_len]
    end = statement[-suffix_len:]
    return start, end


def _build_label_based_dialogues(
    df, *, text_col: str, lang: str, max_rows: int | None
) -> tuple[list[Dialogue], list[Label]]:
    """Truthfulness-style contrast: TRUE statement (HONEST) vs FALSE statement (DECEPTIVE)."""
    from project.repe.categories import get_repe_instructions_for_category

    _, _, user_message = get_repe_instructions_for_category(lang, "generic")

    rows_by_truth: dict[int, list[str]] = {1: [], 0: []}
    for _, row in df.iterrows():
        try:
            lab = int(row["label"])
        except (TypeError, ValueError):
            continue
        if lab not in (0, 1):
            continue
        s = str(row[text_col]).strip()
        if s:
            rows_by_truth[lab].append(s)

    n = min(len(rows_by_truth[1]), len(rows_by_truth[0])) 
    if max_rows is not None:
        n = min(n, max(1, max_rows // 2))

    dialogues: list[Dialogue] = []
    labels: list[Label] = []
    for truth_val, label in ((1, Label.HONEST), (0, Label.DECEPTIVE)):
        for statement in rows_by_truth[truth_val][:n]:
            fact_start, fact_end = _split_statement_repe(statement)
            dialogues.append([
                Message(role="user", content=user_message, detect=False),
                Message(role="assistant", content=fact_start, detect=True),
                Message(role="assistant", content=fact_end, detect=False),
            ])
            labels.append(label)
    return dialogues, labels


def load_repe(
    path: Path,
    *,
    category: str,
    max_rows: int | None = 512,
    variant: str = "you_are_fact_sys",
    instructions_override: tuple[str, str, str] | None = None,
) -> tuple[list[Dialogue], list[Label]]:
    """
    Load REPE CSV and build dialogues for a specific training category.

    Uses a language-specific statement column:
    - en: statement
    - pt: statement_pt
    - de: statement_de
    - es: statement_es
    - zh: statement_zh

    The language is inferred from the filename.
    """
    import pandas as pd

    path = Path(path)
    df = pd.read_csv(path, encoding="utf-8")

    if "label" not in df.columns:
        raise ValueError(f"REPE CSV must have 'label'; got {list(df.columns)}")

    name = path.name.lower()
    lang = "en"
    for code in ("pt", "de", "es", "zh"):
        if f"_{code}" in name:
            lang = code
            break

    expected_col_by_lang = {
        "en": "statement",
        "pt": "statement_pt",
        "de": "statement_de",
        "es": "statement_es",
        "zh": "statement_zh",
    }
    text_col = expected_col_by_lang.get(lang, "statement")

    if text_col not in df.columns:
        raise ValueError(
            f"REPE CSV for lang={lang!r} must have column {text_col!r}; got columns={list(df.columns)}"
        )
    non_empty = df[text_col].fillna("").astype(str).str.strip() != ""
    if not bool(non_empty.any()):
        raise ValueError(
            f"REPE CSV column {text_col!r} for lang={lang!r} is empty. "
            "Refusing to fall back to English 'statement'."
        )

    from project.repe.categories import is_label_based_category

    if is_label_based_category(category):
        return _build_label_based_dialogues(
            df, text_col=text_col, lang=lang, max_rows=max_rows
        )

    df = df[df["label"] == 1].copy()

    if max_rows is not None:
        df = df.head(max_rows)

    dialogues: list[Dialogue] = []
    labels: list[Label] = []

    if variant == "you_are_fact_sys":
        from project.repe.categories import get_repe_instructions_for_category

        if instructions_override is not None:
            system_honest, system_dishonest, user_message = instructions_override
        else:
            system_honest, system_dishonest, user_message = get_repe_instructions_for_category(
                lang, category
            )
        pairs: list[tuple[str | None, str, Label]] = [
            (system_honest, user_message, Label.HONEST),
            (system_dishonest, user_message, Label.DECEPTIVE),
        ]

    elif variant == "plain":
        honest_prompt = "Pretend you're an honest person making statements about the world."
        dishonest_prompt = "Pretend you're a dishonest person making statements about the world."
        pairs = [
            (None, honest_prompt, Label.HONEST),
            (None, dishonest_prompt, Label.DECEPTIVE),
        ]

    else:
        raise ValueError(f"Unknown REPE variant: {variant!r}")

    for _, row in df.iterrows():
        statement = str(row[text_col]).strip()
        if not statement:
            continue

        fact_start, fact_end = _split_statement_repe(statement)

        for system_prefix, user_content, label in pairs:
            dialogue: Dialogue = []

            if system_prefix is not None:
                dialogue.append(Message(role="system", content=system_prefix, detect=False))

            dialogue.append(Message(role="user", content=user_content, detect=False))

            dialogue.append(Message(role="assistant", content=fact_start, detect=True))
            dialogue.append(Message(role="assistant", content=fact_end, detect=False))

            dialogues.append(dialogue)
            labels.append(label)

    return dialogues, labels
