#!/usr/bin/env python3
"""
Donor/target decomposition of the transfer grid with matched-language bonus.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
IN = _REPO / "analysis" / "transfer_stats"
OUT = _REPO / "analysis" / "transfer_decomposition"

MODELS = ["llama-8b", "aya-expanse-8b", "gemma-2b", "gemma-4-e4b", "qwen3.5-9b"]
LANGS = ["en", "pt", "es", "de", "zh"]



def design_matrix() -> np.ndarray:
    rows = []
    for si, ti in product(range(5), range(5)):
        row = np.zeros(10)
        row[0] = 1.0
        if si < 4:
            row[1 + si] = 1.0
        else:
            row[1:5] = -1.0
        if ti < 4:
            row[5 + ti] = 1.0
        else:
            row[5:9] = -1.0
        row[9] = 1.0 if si == ti else 0.0
        rows.append(row)
    return np.array(rows)


def effects_from_beta(beta: np.ndarray):
    """Recover full alpha (5), beta_t (5), mu, delta from the coded solution."""
    mu = beta[0]
    a = np.append(beta[1:5], -beta[1:5].sum())
    b = np.append(beta[5:9], -beta[5:9].sum())
    return mu, a, b, beta[9]


def fit_stats(X: np.ndarray, P: np.ndarray, y: np.ndarray):
    beta = P @ y
    resid = y - X @ beta
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - (resid ** 2).sum() / ss_tot if ss_tot > 0 else np.nan
    return beta, r2


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    X = design_matrix()
    P = np.linalg.pinv(X)
    Xa = X[:, :9]
    Pa = np.linalg.pinv(Xa)
    draws = np.load(IN / "bootstrap_draws.npz")
    cells = pd.read_csv(IN / "cells.csv")

    rows = []
    for m in MODELS:
        c = cells[cells.model == m]
        y = np.array([float(c[(c.train_lang == s) & (c.eval_lang == t)]
                            .auroc.iloc[0])
                      for s, t in product(LANGS, LANGS)])
        beta, _r2_full = fit_stats(X, P, y)
        _, r2 = fit_stats(Xa, Pa, y)          
        mu, a, b, delta = effects_from_beta(beta)
        reps = draws[f"{m}_reps"]
        assert reps.shape == (2000, 5, 5), reps.shape
        ES = sorted(LANGS)
        reps = reps[:, :, [ES.index(l) for l in LANGS]]
        for si_c in (0, 4):
            tl_c = LANGS[si_c]
            r_cell = c[(c.train_lang == tl_c) & (c.eval_lang == "pt")].iloc[0]
            v_cell = reps[:, si_c, LANGS.index("pt")]
            lo_c, hi_c = np.nanpercentile(v_cell, [2.5, 97.5])
            assert (abs(lo_c - r_cell.auroc_ci_lo) < 1e-6
                    and abs(hi_c - r_cell.auroc_ci_hi) < 1e-6), (
                f"{m}: replicate axis mapping does not reproduce this run's "
                f"cells.csv CIs")
        Y = reps.reshape(reps.shape[0], 25)  
        finite = np.isfinite(Y).all(axis=1)
        B_eff = int(finite.sum())
        betas = (P @ Y[finite].T).T          
        betas_a = (Pa @ Y[finite].T).T       
        resid_a = Y[finite] - betas_a @ Xa.T
        ss_tot = ((Y[finite] - Y[finite].mean(axis=1, keepdims=True)) ** 2
                  ).sum(axis=1)
        r2_reps = 1.0 - (resid_a ** 2).sum(axis=1) / ss_tot
        mus, als, bts, dls = [], [], [], []
        for bb in betas:
            _mu, _a, _b, _d = effects_from_beta(bb)
            mus.append(_mu); als.append(_a); bts.append(_b); dls.append(_d)
        als, bts, dls = np.array(als), np.array(bts), np.array(dls)

        def ci(v):
            return tuple(np.percentile(v, [2.5, 97.5]))

        base = {"model": m, "B_effective": B_eff,
                "r2_additive_point": round(float(r2), 4),
                "r2_ci_lo": round(float(np.percentile(r2_reps, 2.5)), 4),
                "r2_ci_hi": round(float(np.percentile(r2_reps, 97.5)), 4)}
        d_lo, d_hi = ci(dls)
        rows.append({**base, "term": "delta_matched", "language": "",
                     "estimate": round(float(delta), 4),
                     "ci_lo": round(float(d_lo), 4),
                     "ci_hi": round(float(d_hi), 4)})
        for li, lang in enumerate(LANGS):
            lo, hi = ci(als[:, li])
            rows.append({**base, "term": "alpha_source", "language": lang,
                         "estimate": round(float(a[li]), 4),
                         "ci_lo": round(float(lo), 4),
                         "ci_hi": round(float(hi), 4)})
            lo, hi = ci(bts[:, li])
            rows.append({**base, "term": "beta_target", "language": lang,
                         "estimate": round(float(b[li]), 4),
                         "ci_lo": round(float(lo), 4),
                         "ci_hi": round(float(hi), 4)})
        print(f"{m:16s} delta {delta:+.4f} [{d_lo:+.4f},{d_hi:+.4f}] "
              f"R2 {r2:.3f} (B_eff {B_eff})")

    pd.DataFrame(rows).to_csv(OUT / "decomposition.csv", index=False)
    (OUT / "meta.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "spec": "A = mu + alpha_s + beta_t + delta*1[s=t]; sum-to-zero "
                "contrasts; per-replicate refit on archived shared-frame "
                "draws (transfer_stats/bootstrap_draws.npz); percentile 95% CIs; "
                "no new resampling; r2_additive is the alpha+beta model "
                "WITHOUT the matched indicator; "
                "delta from the full model",
        "cell_order": "itertools.product(LANGS, LANGS) with "
                      "LANGS=[en,pt,es,de,zh]; archived reps axis is "
                      "(B, train-canonical, eval-SORTED), remapped to "
                      "canonical; axis check reproduces this run's "
                      "per-cell CIs to 1e-6 (float32 archive)",
    }, indent=1))
    print(f"saved -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
