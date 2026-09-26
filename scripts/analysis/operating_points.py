#!/usr/bin/env python3
"""
Operating points. Runs after db_control_scores.
"""

from __future__ import annotations

import argparse
import hashlib
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

from scripts.analysis.transfer_stats import make_shared_resample  

LANGS = ["en", "pt", "es", "de", "zh"]
QUANTILES = (0.90, 0.95, 0.99)   
SEED_REGISTERED = 42
B_REGISTERED = 2000


def scenario_split(honest_scenarios: "list[str]", seed: int) -> "tuple[set[str], set[str]]":
    """Deterministic scenario-level 50/50 split (calibration, evaluation). Sorted input
    required; calibration gets the floor half."""
    if sorted(honest_scenarios) != list(honest_scenarios):
        raise ValueError("honest scenario list must be sorted for a deterministic split")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(honest_scenarios))
    k = len(honest_scenarios) // 2
    calib = {honest_scenarios[i] for i in perm[:k]}
    ev = set(honest_scenarios) - calib
    return calib, ev


def op_point(scores_honest: np.ndarray, scores_deceptive: np.ndarray,
             threshold: float) -> "tuple[float, float]":
    """(fpr_measured, recall) at score > threshold (deceptive = high score)."""
    fpr = float(np.mean(scores_honest > threshold)) if len(scores_honest) else float("nan")
    rec = float(np.mean(scores_deceptive > threshold)) if len(scores_deceptive) else float("nan")
    return fpr, rec


def clustered_rate_ci(indicator: np.ndarray, scen_idx: np.ndarray,
                      counts: np.ndarray) -> "tuple[float, float]":
    """95% percentile CI for a rate (mean of a 0/1 indicator) under the SHARED scenario
    resample. indicator (n,), scen_idx (n,) into the frame, counts (B, n_frame)."""
    n_frame = counts.shape[1]
    hits = np.bincount(scen_idx, weights=indicator.astype(np.float64), minlength=n_frame)
    sizes = np.bincount(scen_idx, minlength=n_frame).astype(np.float64)
    num = counts @ hits
    den = counts @ sizes
    with np.errstate(invalid="ignore", divide="ignore"):
        rates = np.where(den > 0, num / den, np.nan)
    return (float(np.nanpercentile(rates, 2.5)), float(np.nanpercentile(rates, 97.5)))


def _load_probe_geometry(model: str, layer: int, category: str):
    """
    Unit directions, norms, and offsets needed for the TRUE detector logits.
    """
    from project.detectors import LogisticRegressionDetector
    from scripts.analysis.load_probes import probe_path

    dirs, norms, offsets = {}, {}, {}
    for lang in LANGS:
        pp = probe_path(model, lang, layer, category=category, require_category_path=True)
        if not pp.exists():
            raise FileNotFoundError(f"{model}: missing probe for {lang}: {pp}")
        det = LogisticRegressionDetector.load(pp)
        v = det.effective_direction(layer).float().numpy()
        n = float(np.linalg.norm(v))
        if n <= 0:
            raise ValueError(f"{model}/{lang}: zero-norm probe direction")
        dirs[lang], norms[lang] = v / n, n
        classifier_intercept = 0.0
        if det.clf is not None and getattr(det.clf, "intercept_", None) is not None:
            intercept = np.asarray(det.clf.intercept_, dtype=np.float64).reshape(-1)
            if len(intercept) != 1 or not np.isfinite(intercept[0]):
                raise ValueError(
                    f"{model}/{lang}: invalid binary-classifier intercept {intercept}"
                )
            classifier_intercept = float(intercept[0])
        if det.scaler_mean is not None:
            idx = det.layers.index(layer)
            mu = det.scaler_mean[idx].float().numpy()
            offsets[lang] = float(v @ mu) - classifier_intercept
        else:
            offsets[lang] = -classifier_intercept
    return dirs, norms, offsets


def _load_probe_dirs_and_norms(model: str, layer: int, category: str):
    """Back-compat wrapper (dirs, norms)."""
    dirs, norms, _ = _load_probe_geometry(model, layer, category)
    return dirs, norms


def _load_control_scores(model: str, lang: str, norms: "dict[str, float]",
                         root: Path) -> "dict[str, np.ndarray]":
    p = root / model / f"control_scores_{lang}.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"{model}: control scores missing for {lang} ({p}) — run "
            "scripts/analysis/db_control_scores.py first (needs the Alpaca control corpus).")
    df = pd.read_csv(p)
    out = {}
    for tl in LANGS:
        col = f"score_probe_{tl}"
        if col not in df.columns:
            raise KeyError(f"{p}: missing {col} (stale artifact?)")
        out[tl] = df[col].to_numpy(dtype=np.float64) / norms[tl]  
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--cache-root", type=Path, default=_REPO_ROOT / "results" / "activation_cache")
    ap.add_argument("--control-root", type=Path, default=_REPO_ROOT / "analysis" / "db_control_scores")
    ap.add_argument("--out-root", type=Path, default=_REPO_ROOT / "analysis" / "operating_points")
    ap.add_argument("--category", default="generic")
    ap.add_argument("--b", type=int, default=B_REGISTERED)
    ap.add_argument("--seed", type=int, default=SEED_REGISTERED)
    args = ap.parse_args()

    if (args.b, args.seed) != (B_REGISTERED, SEED_REGISTERED):
        print(f"WARNING: non-default parameters (B={args.b} seed={args.seed}).",
              file=sys.stderr)

    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "frozen_layers", _REPO_ROOT / "project" / "frozen_layers.py")
    _fl = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_fl)

    rows, meta_models = [], {}
    for model in args.models:
        layer = _fl.load_frozen_layer(model)
        dirs, norms = _load_probe_dirs_and_norms(model, layer, args.category)
        print(f"[{model}] layer {layer}", flush=True)
        splits = {}
        for el in LANGS:
            npz = args.cache_root / model / "deceptionbench" / f"{el}_layer{layer}.npz"
            with np.load(npz) as z:
                acts = z["acts"].astype(np.float64)
                y = z["labels"].astype(int)
                scen = np.array([str(s) for s in z["scenario_id"]])
            ctrl = _load_control_scores(model, el, norms, args.control_root)

            honest_scens = sorted(set(scen[y == 0]))
            calib_scens, _ = scenario_split(honest_scens, args.seed)
            in_calib = np.isin(scen, sorted(calib_scens))
            splits[el] = {"n_honest_scens": len(honest_scens),
                          "calib_sha256": hashlib.sha256(
                              "\n".join(sorted(calib_scens)).encode()).hexdigest()}

            frame = sorted(set(scen))
            f_idx = {s: i for i, s in enumerate(frame)}
            counts = make_shared_resample(frame, args.b, args.seed)
            scen_i = np.array([f_idx[s] for s in scen])

            for tl in LANGS:
                s = acts @ dirs[tl]
                pools = {
                    "alpaca": (ctrl[tl], s[y == 0], s[y == 1],
                               (y == 0), (y == 1)),
                    "benchmark": (s[(y == 0) & in_calib],
                                  s[(y == 0) & ~in_calib], s[(y == 1) & ~in_calib],
                                  (y == 0) & ~in_calib, (y == 1) & ~in_calib),
                }
                for source, (cal, sh, sd, mask_h, mask_d) in pools.items():
                    for q in QUANTILES:
                        thr = float(np.quantile(cal, q))
                        fpr, rec = op_point(sh, sd, thr)
                        fpr_ci = clustered_rate_ci((s > thr)[mask_h], scen_i[mask_h], counts)
                        rec_ci = clustered_rate_ci((s > thr)[mask_d], scen_i[mask_d], counts)
                        rows.append({
                            "model": model, "train_lang": tl, "eval_lang": el,
                            "calib_source": source, "quantile": q, "threshold": thr,
                            "registered_headline": (q == 0.99),
                            "fpr_measured": fpr, "recall": rec,
                            "fpr_ci_lo": fpr_ci[0], "fpr_ci_hi": fpr_ci[1],
                            "recall_ci_lo": rec_ci[0], "recall_ci_hi": rec_ci[1],
                            "n_calib": int(len(cal)), "n_honest_eval": int(len(sh)),
                            "n_deceptive_eval": int(len(sd)),
                        })
        meta_models[model] = {"layer": int(layer), "splits": splits,
                              "probe_dir_norms": {k: float(v) for k, v in norms.items()}}

    args.out_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_root / "cells.csv", index=False)
    (args.out_root / "meta.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "B": args.b, "seed": args.seed, "quantiles": list(QUANTILES),
        "registered_headline": "q99, both calibration sources",
        "ci_convention": "scenario-clustered percentile 95%; threshold treated as FIXED "
                         "(calibration uncertainty excluded — measured-FPR convention)",
        "control_rescale": "control CSV scores divided by probe direction norms "
                           "(db_control_scores archived unnormalized-direction scores)",
        "split": "scenario-level 50/50 of the honest pool, seed 42; BOTH measurement "
                 "pools exclude calibration scenarios entirely",
        "models": meta_models,
    }, indent=2))
    print(f"Saved: {args.out_root}/cells.csv ({len(rows)} rows) + meta.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
