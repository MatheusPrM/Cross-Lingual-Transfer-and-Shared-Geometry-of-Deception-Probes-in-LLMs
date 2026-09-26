#!/usr/bin/env python3
"""
Direction Specificity analysis. Compares the deception direction against the four auxiliary control categories.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from scripts.analysis.load_probes import DEFAULT_LANGS, load_language_stack

DECEPTION_CATEGORY = "generic"
CONTROL_CATEGORIES = ("formality", "politeness", "truthfulness", "sentiment")

OUT_DIR = _REPO_ROOT / "analysis" / "direction_specificity"


def _require_v2_category_stack(model: str, layer: int, langs, category: str):
    """Load a complete category-path-only v2 stack; never allow a legacy probe fallback."""
    expected = [
        _REPO_ROOT / "probes" / model / lang / category / f"repe_probe_layer{layer}.pt"
        for lang in langs
    ]
    missing = [p for p in expected if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{model}/{category}/layer{layer}: missing required v2 probes: "
            + ", ".join(str(p) for p in missing)
        )
    found, stack = load_language_stack(
        model, layer, langs=langs, category=category, strict=True
    )
    if list(found) != list(langs):
        raise RuntimeError(
            f"{model}/{category}/layer{layer}: loaded languages {found}, expected {list(langs)}"
        )
    return found, stack


def _frozen_layer(model: str) -> int | None:
    p = _REPO_ROOT / "results" / "frozen_layers" / model / "chosen_layer.json"
    if not p.exists():
        return None
    try:
        return int(json.loads(p.read_text(encoding="utf-8"))["chosen_layer"])
    except Exception:
        return None


def _unit_rows(V: np.ndarray) -> np.ndarray:
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)


def _mean_offdiag_cosine(U: np.ndarray) -> float:
    """Mean off-diagonal cosine of unit-row stack U (its own cross-language alignment)."""
    n = U.shape[0]
    if n < 2:
        return float("nan")
    C = U @ U.T
    return float(C[~np.eye(n, dtype=bool)].mean())


def compare_model(model: str, *, langs=DEFAULT_LANGS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (summary_df, per_lang_df) comparing deception vs each available control category."""
    layer = _frozen_layer(model)
    if layer is None:
        print(f"[{model}] no frozen layer; skipping", file=sys.stderr)
        return pd.DataFrame(), pd.DataFrame()

    dec_langs, V_dec = _require_v2_category_stack(
        model, layer, langs, DECEPTION_CATEGORY
    )
    dec_unit = _unit_rows(V_dec)
    dec_by_lang = dict(zip(dec_langs, dec_unit))
    dec_mean_offdiag = _mean_offdiag_cosine(dec_unit)

    summary_rows: list[dict] = []
    per_lang_rows: list[dict] = []
    for cat in CONTROL_CATEGORIES:
        ctrl_langs, V_ctrl = _require_v2_category_stack(model, layer, langs, cat)
        ctrl_by_lang = dict(zip(ctrl_langs, _unit_rows(V_ctrl)))

        common = [l for l in dec_langs if l in ctrl_by_lang]
        if len(common) != len(langs):
            raise RuntimeError(
                f"{model}/{cat}/layer{layer}: incomplete language intersection {common}"
            )

        within = {l: float(np.dot(dec_by_lang[l], ctrl_by_lang[l])) for l in common}
        for l, c in within.items():
            per_lang_rows.append({"model": model, "control_category": cat, "lang": l,
                                  "layer": layer, "cosine_deception_vs_control": c})

        ctrl_mean_offdiag = _mean_offdiag_cosine(np.vstack([ctrl_by_lang[l] for l in common]))
        summary_rows.append({
            "model": model,
            "control_category": cat,
            "layer": layer,
            "n_langs": len(common),
            "mean_within_lang_cosine": float(np.mean(list(within.values()))),
            "max_within_lang_cosine": float(np.max(list(within.values()))),
            "deception_mean_offdiag_cosine": dec_mean_offdiag,
            "control_mean_offdiag_cosine": ctrl_mean_offdiag,
        })

    return pd.DataFrame(summary_rows), pd.DataFrame(per_lang_rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", type=str, required=True,
                    help="Comma-separated model ids (e.g. gemma-2b,llama-8b,qwen3.5-9b,gemma-4-e4b).")
    ap.add_argument("--langs", type=str, default=",".join(DEFAULT_LANGS))
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    langs = tuple(x.strip() for x in args.langs.split(",") if x.strip())
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    all_summary: list[pd.DataFrame] = []
    all_per_lang: list[pd.DataFrame] = []
    for model in models:
        s, pl = compare_model(model, langs=langs)
        if not s.empty:
            all_summary.append(s)
        if not pl.empty:
            all_per_lang.append(pl)

    if not all_summary:
        print("No complete v2 control-concept probe stacks found to compare against deception.",
              file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.concat(all_summary, ignore_index=True)
    per_lang = pd.concat(all_per_lang, ignore_index=True)
    summary.to_csv(args.out_dir / "direction_specificity_summary.csv", index=False)
    per_lang.to_csv(args.out_dir / "direction_specificity_per_lang.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nWrote {args.out_dir / 'direction_specificity_summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
