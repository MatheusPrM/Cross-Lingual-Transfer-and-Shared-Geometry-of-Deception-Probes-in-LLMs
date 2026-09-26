#!/usr/bin/env python3
"""
Frozen-layer selector: reads sweep_layers.py val metrics per layer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_val_rows(sweep_csv: Path) -> list[dict]:
    import pandas as pd
    df = pd.read_csv(sweep_csv)
    if "layer" not in df.columns or "val_auroc" not in df.columns:
        raise ValueError(f"Sweep CSV {sweep_csv} missing val_auroc/layer.")
    df = df.dropna(subset=["val_auroc"]).copy()
    if df.empty:
        raise ValueError(f"No finite val_auroc rows in {sweep_csv}.")
    df = df.sort_values("layer").reset_index(drop=True)
    return df.to_dict("records")


def select_layer(rows: list[dict], *, epsilon: float = 0.005,
                 tie_break: str = "val_margin") -> dict:
    if not rows:
        raise ValueError("Empty rows.")
    aurocs = [float(r["val_auroc"]) for r in rows]
    max_auroc = max(aurocs)
    candidates = [r for r, a in zip(rows, aurocs) if a >= max_auroc - float(epsilon)]
    if len(candidates) == 1:
        chosen = candidates[0]
        tie_breaking_used = False
    else:
        if tie_break != "val_margin":
            raise ValueError(f"Unknown tie_break {tie_break!r}")
        chosen = max(candidates,
                     key=lambda r: float(r.get("val_margin") or float("-inf")))
        tie_breaking_used = True
    return {
        "chosen_layer": int(chosen["layer"]),
        "val_auroc": float(chosen["val_auroc"]),
        "val_margin": (float(chosen["val_margin"])
                       if chosen.get("val_margin") is not None else None),
        "max_val_auroc": float(max_auroc),
        "epsilon": float(epsilon),
        "tie_breaking_used": bool(tie_breaking_used),
        "tie_break_rule": tie_break,
        "n_candidates": int(len(candidates)),
        "candidate_layers": [int(r["layer"]) for r in candidates],
        "all_layer_val_aurocs": [(int(r["layer"]), float(r["val_auroc"]))
                                 for r in rows],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--train-data", default="repe_generic_en")
    ap.add_argument("--sweep-csv", type=Path, default=None)
    ap.add_argument("--epsilon", type=float, default=0.005)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    sweep_csv = args.sweep_csv or (_REPO_ROOT / "results" / "sweeps" /
                                   f"{args.model}_{args.train_data}.csv")
    if not sweep_csv.exists():
        print(f"ERROR: sweep CSV not found: {sweep_csv} — run "
              f"scripts/sweep_layers.py first.", file=sys.stderr)
        return 2

    rows = _load_val_rows(sweep_csv)
    result = select_layer(rows, epsilon=args.epsilon)
    result["model"] = args.model
    result["train_data"] = args.train_data
    result["sweep_csv"] = str(sweep_csv.relative_to(_REPO_ROOT)
                              if str(sweep_csv).startswith(str(_REPO_ROOT))
                              else sweep_csv)

    out = args.out or (_REPO_ROOT / "results" / "frozen_layers" / args.model /
                       "chosen_layer.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    tb = " (tie-break)" if result["tie_breaking_used"] else ""
    print(f"chosen_layer = {result['chosen_layer']}  "
          f"val_auroc={result['val_auroc']:.4f}  "
          f"val_margin={result['val_margin']}  "
          f"candidates={result['candidate_layers']}{tb}")
    print(f"Wrote {out.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
