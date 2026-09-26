#!/usr/bin/env python3
"""
random-direction null for the rank-1 explained energy.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
OUT = _REPO / "analysis" / "rank1_null"
MODELS = ["llama-8b", "aya-expanse-8b", "gemma-2b", "gemma-4-e4b", "qwen3.5-9b"]

FL = {m: json.load(open(_REPO / "results" / "frozen_layers" / m /
                        "chosen_layer.json"))["chosen_layer"] for m in MODELS}


def rank1_energy(mat: np.ndarray) -> float:
    s = np.linalg.svd(mat, compute_uv=False)
    return float(s[0] ** 2 / (s ** 2).sum())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in MODELS:
        d = int(np.load(_REPO / "results" / "geometry" / m /
                        f"layer{FL[m]}" / "v_shared.npy").shape[0])
        rng = np.random.default_rng(42)
        vals = np.empty(2000)
        for i in range(2000):
            v = rng.standard_normal((5, d))
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            vals[i] = rank1_energy(v)
        rows.append({"model": m, "hidden_dim": d,
                     "null_mean": round(float(vals.mean()), 4),
                     "null_q95": round(float(np.percentile(vals, 95)), 4),
                     "null_q99": round(float(np.percentile(vals, 99)), 4),
                     "null_max": round(float(vals.max()), 4),
                     "n_draws": 2000, "seed": 42})
        print(rows[-1])
    pd.DataFrame(rows).to_csv(OUT / "null.csv", index=False)
    (OUT / "meta.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "spec": "5 iid standard-normal d-vectors, unit-normalized, stacked; "
                "uncentered rank-1 energy; 2000 draws; default_rng(42); "
                "descriptive null, no hypothesis test",
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
