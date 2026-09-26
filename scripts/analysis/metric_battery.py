#!/usr/bin/env python3
"""
Per-cell metric battery.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from scripts.analysis.operating_points import _load_probe_geometry  
from scripts.analysis.transfer_stats import auroc_rank

LANGS = ["en", "pt", "es", "de", "zh"]


def cohens_d(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")
    n1, n0 = len(pos), len(neg)
    sp = np.sqrt(((n1 - 1) * pos.var(ddof=1) + (n0 - 1) * neg.var(ddof=1)) / (n1 + n0 - 2))
    return float((pos.mean() - neg.mean()) / sp) if sp > 0 else float("nan")


def brier_and_ece(scores_detector_scale: np.ndarray, labels: np.ndarray,
                  n_bins: int = 10) -> "tuple[float, float, list[dict]]":
    """Brier + ECE with p = sigmoid(TRUE detector logit — caller must include the
    scaler-centering offset). Also returns the per-bin calibration curve."""
    p = 1.0 / (1.0 + np.exp(-np.clip(scores_detector_scale, -60, 60)))
    y = labels.astype(np.float64)
    brier = float(np.mean((p - y) ** 2))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins: list[dict] = []
    for bi, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= 1.0)
        if m.any():
            conf, acc = float(p[m].mean()), float(y[m].mean())
            ece += m.mean() * abs(acc - conf)
            bins.append({"bin": bi, "lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                         "mean_p": conf, "frac_pos": acc})
    return brier, float(ece), bins


def length_stratified_auroc(scores: np.ndarray, labels: np.ndarray,
                            lengths: np.ndarray) -> "tuple[float, float, int]":
    """Mean and min AUROC over length terciles (NaN terciles skipped, counted)."""
    qs = np.quantile(lengths, [1 / 3, 2 / 3])
    strata = np.digitize(lengths, qs)
    vals = [auroc_rank(scores[strata == k], labels[strata == k]) for k in range(3)]
    ok = [v for v in vals if np.isfinite(v)]
    if not ok:
        return float("nan"), float("nan"), 3
    return float(np.mean(ok)), float(np.min(ok)), 3 - len(ok)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    r = spearmanr(a, b).statistic
    return float(r)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--cache-root", type=Path, default=_REPO_ROOT / "results" / "activation_cache")
    ap.add_argument("--out-root", type=Path, default=_REPO_ROOT / "analysis" / "metric_battery")
    ap.add_argument("--category", default="generic")
    args = ap.parse_args()

    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "frozen_layers", _REPO_ROOT / "project" / "frozen_layers.py")
    _fl = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_fl)

    args.out_root.mkdir(parents=True, exist_ok=True)
    meta = {}
    for model in args.models:
        layer = _fl.load_frozen_layer(model)
        dirs, norms, offsets = _load_probe_geometry(model, layer, args.category)
        cell_rows, eval_rows, agree_rows, calib_rows = [], [], [], []
        for el in LANGS:
            npz = args.cache_root / model / "deceptionbench" / f"{el}_layer{layer}.npz"
            with np.load(npz) as z:
                acts = z["acts"].astype(np.float64)
                y = z["labels"].astype(int)
                if "prompt_acts" not in z:
                    raise SystemExit(
                        f"{model}/{el}: cache lacks prompt_acts; regenerate with "
                        f"cache_eval_activations.py")
                pacts = z["prompt_acts"].astype(np.float64)
                lens = z["n_detect_tokens"].astype(np.float64)
                term = np.array([str(t) for t in z["termination"]])
                n_nonadm = int(z["nonadmitted_acts"].shape[0]) if "nonadmitted_acts" in z else 0
                if "nonadmitted_termination" not in z:
                    raise KeyError(
                        f"{npz}: missing nonadmitted_termination; exact all-pool truncation "
                        "cannot be reconstructed"
                    )
                nonadm_term = np.array([str(t) for t in z["nonadmitted_termination"]])
                n_drop_adm = int(z["n_dropped_no_detect"]) if "n_dropped_no_detect" in z else 0
                n_drop_nonadm = (
                    int(z["nonadmitted_n_dropped_no_detect"])
                    if "nonadmitted_n_dropped_no_detect" in z else 0
                )
            if len(nonadm_term) != n_nonadm:
                raise RuntimeError(
                    f"{npz}: non-admitted activation/termination mismatch "
                    f"({n_nonadm} vs {len(nonadm_term)})"
                )
            if n_drop_adm or n_drop_nonadm:
                raise RuntimeError(
                    f"{npz}: detect-token drops prevent exact all-pool truncation accounting "
                    f"(admitted={n_drop_adm}, nonadmitted={n_drop_nonadm})"
                )
            all_term = np.concatenate([term, nonadm_term])
            if len(all_term) != 900:
                raise RuntimeError(
                    f"{npz}: expected 900 accounted DeceptionBench rows, got {len(all_term)}"
                )
            eos = term == "eos"
            n_cap_admitted = int(np.sum(term == "cap"))
            n_cap_all = int(np.sum(all_term == "cap"))
            score_cols = {}
            for tl in LANGS:
                s = acts @ dirs[tl]
                score_cols[tl] = s
                sp = pacts @ dirs[tl]
                brier, ece, bins = brier_and_ece(s * norms[tl] - offsets[tl], y)
                for b in bins:
                    calib_rows.append({"model": model, "train_lang": tl, "eval_lang": el, **b})
                ls_mean, ls_min, ls_skipped = length_stratified_auroc(s, y, lens)
                from sklearn.metrics import average_precision_score
                a = auroc_rank(s, y)
                cell_rows.append({
                    "model": model, "train_lang": tl, "eval_lang": el, "layer": layer,
                    "auroc": a, "auprc": float(average_precision_score(y, s)),
                    "cohens_d": cohens_d(s, y),
                    "cles": a,  
                    "brier": brier, "ece": ece,
                    "response_only_auroc": a,
                    "prompt_only_auroc": auroc_rank(sp, y),
                    "eos_only_auroc": auroc_rank(s[eos], y[eos]),
                    "n_eos_admitted": int(eos.sum()),
                    "cap_rate_admitted": float(np.mean(term == "cap")),
                    "length_strat_auroc_mean": ls_mean, "length_strat_auroc_min": ls_min,
                    "length_strat_skipped": ls_skipped, "n": int(len(y)),
                })
            eval_rows.append({
                "model": model, "eval_lang": el, "layer": layer,
                "length_only_auroc": auroc_rank(lens, y),
                "n_admitted": int(len(y)), "n_nonadmitted_wellformed": n_nonadm,
                "n_total": int(len(all_term)),
                "admitted_rate": float(len(y) / len(all_term)),
                "n_cap_all": n_cap_all,
                "cap_rate_all": float(n_cap_all / len(all_term)),
                "n_cap_admitted": n_cap_admitted,
                "cap_rate_admitted": float(np.mean(term == "cap")),
                "n_deceptive_admitted": int(np.sum(y == 1)),
                "n_honest_admitted": int(np.sum(y == 0)),
                "cap_rate_deceptive_admitted": (
                    float(np.mean(term[y == 1] == "cap")) if (y == 1).any() else float("nan")
                ),
                "cap_rate_honest_admitted": (
                    float(np.mean(term[y == 0] == "cap")) if (y == 0).any() else float("nan")
                ),
            })
            for i, a_ in enumerate(LANGS):
                for b_ in LANGS[i + 1:]:
                    agree_rows.append({"model": model, "eval_lang": el,
                                       "probe_a": a_, "probe_b": b_,
                                       "spearman": spearman(score_cols[a_], score_cols[b_])})
        pd.DataFrame(cell_rows).to_csv(args.out_root / f"{model}_cells.csv", index=False)
        pd.DataFrame(eval_rows).to_csv(args.out_root / f"{model}_evallang.csv", index=False)
        pd.DataFrame(agree_rows).to_csv(args.out_root / f"{model}_score_agreement.csv", index=False)
        pd.DataFrame(calib_rows).to_csv(args.out_root / f"{model}_calibration_bins.csv", index=False)
        meta[model] = {"layer": int(layer), "probe_dir_norms": norms,
                       "probe_logit_offsets": offsets}
        print(f"[{model}] battery: {len(cell_rows)} cells written", flush=True)

    (args.out_root / "meta.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "notes": {
            "cles": "empirical CLES equals AUROC by identity; column kept for table",
            "brier_ece": "p = sigmoid(unit score x direction norm - reconstructed "
                         "offset), including scaler centering and classifier intercept; "
                         "this is the saved detector's true logit; cross-language "
                         "miscalibration is an expected, reported finding",
            "eos_only": "pre-specified sensitivity: rows with termination == 'eos' only",
            "truncation": "cap_rate_all uses all 900 accounted rollouts; admitted-only and "
                          "class-conditional admitted rates are named explicitly",
        }, "models": meta}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
