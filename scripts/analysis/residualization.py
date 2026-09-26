#!/usr/bin/env python3
"""
Direct construct tests: residualized AUROC + per-item correlations.
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

from scripts.analysis.metric_battery import spearman
from scripts.analysis.transfer_stats import auroc_rank, make_shared_resample

LANGS = ["en", "pt", "es", "de", "zh"]
CONTROL_CATEGORIES = ("formality", "politeness", "truthfulness", "sentiment")
B_REGISTERED = 2000
SEED_REGISTERED = 42
MAX_DEGENERATE_FRACTION = 0.01


def residualize(target: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    """OLS residual of one or more targets on ``covariates`` (+intercept)."""
    X = np.column_stack([np.ones(len(target)), covariates])
    beta, *_ = np.linalg.lstsq(X, target, rcond=None)
    return target - X @ beta


def load_eval_cache(path: Path) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """Load the admitted residualization population and fail closed on row/identity corruption."""
    required = {"acts", "labels", "n_detect_tokens", "scenario_id", "condition"}
    with np.load(path) as z:
        missing = sorted(required - set(z.files))
        if missing:
            raise ValueError(f"{path}: missing residualization cache keys: {missing}")
        acts = z["acts"].astype(np.float64)
        raw_labels = np.asarray(z["labels"])
        lens = z["n_detect_tokens"].astype(np.float64)
        scenario = np.array([str(s) for s in z["scenario_id"]])
        condition = np.array([str(c) for c in z["condition"]])

    if acts.ndim != 2:
        raise ValueError(f"{path}: acts must be 2-D, got {acts.shape}")
    n = acts.shape[0]
    row_counts = {
        "labels": raw_labels.shape[0] if raw_labels.ndim else 0,
        "n_detect_tokens": lens.shape[0] if lens.ndim else 0,
        "scenario_id": scenario.shape[0] if scenario.ndim else 0,
        "condition": condition.shape[0] if condition.ndim else 0,
    }
    if raw_labels.ndim != 1 or lens.ndim != 1 or scenario.ndim != 1 \
            or condition.ndim != 1 or any(rows != n for rows in row_counts.values()):
        raise ValueError(
            f"{path}: cache row-count mismatch; acts={n}, companions={row_counts}"
        )
    if n == 0 or not np.isfinite(acts).all():
        raise ValueError(f"{path}: empty or non-finite activation matrix")
    if not np.isfinite(lens).all() or not (lens > 0).all():
        raise ValueError(f"{path}: n_detect_tokens must be finite and positive")
    if not np.issubdtype(raw_labels.dtype, np.number) \
            or not np.isfinite(raw_labels).all() \
            or set(np.unique(raw_labels)) != {0, 1}:
        raise ValueError(f"{path}: labels must contain both classes and only {{0,1}}")
    y = raw_labels.astype(int)
    invalid_identity = {"", "None", "nan"}
    if any(s in invalid_identity for s in scenario) \
            or any(c in invalid_identity for c in condition):
        raise ValueError(f"{path}: blank scenario_id or condition")
    identities = list(zip(scenario.tolist(), condition.tolist()))
    if len(set(identities)) != n:
        raise ValueError(
            f"{path}: duplicated (scenario_id, condition) identities "
            f"({n - len(set(identities))} duplicates)"
        )
    return acts, y, lens, scenario


def _validate_bootstrap_inputs(
    deception_scores: np.ndarray,
    control_scores: np.ndarray,
    lens: np.ndarray,
    y: np.ndarray,
    scenario_idx: np.ndarray,
    counts: np.ndarray,
) -> None:
    n = len(y)
    if deception_scores.ndim != 2 or deception_scores.shape[0] != n:
        raise ValueError("deception score matrix is not row-aligned with labels")
    if control_scores.ndim != 2 or control_scores.shape[0] != n:
        raise ValueError("control score matrix is not row-aligned with labels")
    if lens.ndim != 1 or len(lens) != n or scenario_idx.ndim != 1 \
            or len(scenario_idx) != n:
        raise ValueError("length/scenario arrays are not row-aligned with labels")
    if counts.ndim != 2 or counts.shape[1] == 0:
        raise ValueError("bootstrap count matrix must be B x nonempty scenario frame")
    if len(scenario_idx) and (
        scenario_idx.min() < 0 or scenario_idx.max() >= counts.shape[1]
    ):
        raise ValueError("scenario index falls outside the shared bootstrap frame")
    if not np.issubdtype(counts.dtype, np.integer) or (counts < 0).any():
        raise ValueError("bootstrap counts must be nonnegative integers")
    if not (counts.sum(axis=1) == counts.shape[1]).all():
        raise ValueError("each bootstrap replicate must draw exactly one scenario frame")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("bootstrap labels must contain both classes and only {0,1}")
    if not all(np.isfinite(x).all() for x in (deception_scores, control_scores, lens)):
        raise ValueError("bootstrap inputs contain non-finite scores or lengths")


def _spearman_matrix(values: np.ndarray) -> np.ndarray:
    """Column-wise Spearman matrix, ranking each resampled variable only once."""
    from scipy.stats import rankdata

    ranks = rankdata(values, axis=0, method="average")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.corrcoef(ranks, rowvar=False)


def bootstrap_eval(
    deception_scores: np.ndarray,
    control_scores: np.ndarray,
    lens: np.ndarray,
    y: np.ndarray,
    scenario_idx: np.ndarray,
    counts: np.ndarray,
) -> "dict[str, np.ndarray]":
    """
    Bootstrap every residualization estimand for one evaluation language.
    """
    deception_scores = np.asarray(deception_scores, dtype=np.float64)
    control_scores = np.asarray(control_scores, dtype=np.float64)
    lens = np.asarray(lens, dtype=np.float64)
    raw_y = np.asarray(y)
    raw_scenario_idx = np.asarray(scenario_idx)
    if raw_y.ndim != 1 or not np.issubdtype(raw_y.dtype, np.number) \
            or not np.isfinite(raw_y).all() or set(np.unique(raw_y)) != {0, 1}:
        raise ValueError("bootstrap labels must contain both classes and only {0,1}")
    if raw_scenario_idx.ndim != 1 \
            or not np.issubdtype(raw_scenario_idx.dtype, np.integer):
        raise ValueError("scenario indices must be a one-dimensional integer array")
    y = raw_y.astype(int)
    scenario_idx = raw_scenario_idx.astype(int)
    counts = np.asarray(counts)
    _validate_bootstrap_inputs(
        deception_scores, control_scores, lens, y, scenario_idx, counts
    )

    b = counts.shape[0]
    n_train = deception_scores.shape[1]
    n_control = control_scores.shape[1]
    out = {
        "auroc_raw": np.full((b, n_train), np.nan),
        "auroc_residualized": np.full((b, n_train), np.nan),
        "auroc_change": np.full((b, n_train), np.nan),
        "spearman_length": np.full((b, n_train), np.nan),
        "spearman_control": np.full((b, n_train, n_control), np.nan),
        "control_auroc_raw": np.full((b, n_control), np.nan),
        "converse_auroc_residualized": np.full((b, n_train, n_control), np.nan),
        "converse_auroc_change": np.full((b, n_train, n_control), np.nan),
    }
    for bi in range(b):
        multiplicity = counts[bi][scenario_idx]
        keep = multiplicity > 0
        rep = np.repeat(np.flatnonzero(keep), multiplicity[keep])
        if not len(rep):
            continue
        y_rep = y[rep]
        s_d = deception_scores[rep]
        s_c = control_scores[rep]
        len_rep = lens[rep]

        covariates = np.column_stack([s_c, len_rep])
        residual_d = residualize(s_d, covariates)  
        for ti in range(n_train):
            raw = auroc_rank(s_d[:, ti], y_rep)
            res = auroc_rank(residual_d[:, ti], y_rep)
            out["auroc_raw"][bi, ti] = raw
            out["auroc_residualized"][bi, ti] = res
            out["auroc_change"][bi, ti] = res - raw

        if n_control:
            for ci in range(n_control):
                out["control_auroc_raw"][bi, ci] = auroc_rank(s_c[:, ci], y_rep)
            for ti in range(n_train):
                converse = residualize(
                    s_c, np.column_stack([s_d[:, ti], len_rep])
                )  
                for ci in range(n_control):
                    res = auroc_rank(converse[:, ci], y_rep)
                    raw = out["control_auroc_raw"][bi, ci]
                    out["converse_auroc_residualized"][bi, ti, ci] = res
                    out["converse_auroc_change"][bi, ti, ci] = res - raw

        correlations = _spearman_matrix(np.column_stack([s_d, len_rep, s_c]))
        length_col = n_train
        for ti in range(n_train):
            out["spearman_length"][bi, ti] = correlations[ti, length_col]
            for ci in range(n_control):
                out["spearman_control"][bi, ti, ci] = correlations[
                    ti, length_col + 1 + ci
                ]
    return out


def _add_ci(row: dict, name: str, reps: np.ndarray, context: str) -> None:
    """Add a 95% percentile CI and invalid-replicate count, refusing unstable output."""
    reps = np.asarray(reps, dtype=np.float64)
    valid = reps[np.isfinite(reps)]
    n_bad = int(reps.size - valid.size)
    if not len(valid):
        raise RuntimeError(f"{context}/{name}: no finite bootstrap replicates")
    if n_bad > MAX_DEGENERATE_FRACTION * reps.size:
        raise RuntimeError(
            f"{context}/{name}: {n_bad}/{reps.size} non-finite replicates exceed "
            f"the {MAX_DEGENERATE_FRACTION:.0%} fail-closed tolerance"
        )
    row[f"{name}_ci_lo"] = float(np.percentile(valid, 2.5))
    row[f"{name}_ci_hi"] = float(np.percentile(valid, 97.5))
    row[f"{name}_n_degenerate_reps"] = n_bad


def _control_dirs(model: str, layer: int) -> "dict[str, dict[str, np.ndarray]]":
    """{category: {lang: unit direction}}; strict category paths, no legacy fallback.
    Missing (category, lang) probes are recorded as absent, not silently substituted."""
    from project.detectors import LogisticRegressionDetector
    from scripts.analysis.load_probes import probe_path

    out: dict[str, dict[str, np.ndarray]] = {}
    for cat in CONTROL_CATEGORIES:
        out[cat] = {}
        for lang in LANGS:
            pp = probe_path(model, lang, layer, category=cat, require_category_path=True)
            if not pp.exists():
                continue
            det = LogisticRegressionDetector.load(pp)
            v = det.effective_direction(layer).float().numpy().astype(np.float64)
            norm = float(np.linalg.norm(v))
            if not np.isfinite(norm) or norm <= 0:
                raise ValueError(
                    f"{model}/{cat}/{lang}/layer{layer}: zero or non-finite direction"
                )
            out[cat][lang] = v / norm
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--cache-root", type=Path, default=_REPO_ROOT / "results" / "activation_cache")
    ap.add_argument("--out-root", type=Path, default=_REPO_ROOT / "analysis" / "residualization")
    ap.add_argument("--b", type=int, default=B_REGISTERED)
    ap.add_argument("--seed", type=int, default=SEED_REGISTERED)
    args = ap.parse_args()

    if args.b <= 0:
        raise ValueError("--b must be positive")
    if (args.b, args.seed) != (B_REGISTERED, SEED_REGISTERED):
        print(
            f"WARNING: non-default residualization bootstrap parameters B={args.b}, seed={args.seed}; "
            f"the paper uses B={B_REGISTERED}, seed={SEED_REGISTERED}.",
            file=sys.stderr,
        )

    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "frozen_layers", _REPO_ROOT / "project" / "frozen_layers.py")
    _fl = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_fl)
    from scripts.analysis.operating_points import _load_probe_dirs_and_norms

    args.out_root.mkdir(parents=True, exist_ok=True)
    meta = {}
    draws: dict[str, np.ndarray] = {}
    for model in args.models:
        layer = _fl.load_frozen_layer(model)
        dec_dirs, _ = _load_probe_dirs_and_norms(model, layer, "generic")
        ctrl = _control_dirs(model, layer)
        missing_registered_controls = [
            f"{category}/{lang}"
            for category in CONTROL_CATEGORIES
            for lang in LANGS
            if lang not in ctrl[category]
        ]
        if missing_registered_controls:
            raise FileNotFoundError(
                f"{model}: four-control grid is incomplete: "
                f"{missing_registered_controls}"
            )
        raw = {}
        for el in LANGS:
            npz = args.cache_root / model / "deceptionbench" / f"{el}_layer{layer}.npz"
            raw[el] = load_eval_cache(npz)

        frame = sorted(set().union(*[set(cell[3]) for cell in raw.values()]))
        if not frame:
            raise ValueError(f"{model}: empty shared scenario frame")
        frame_idx = {scenario: i for i, scenario in enumerate(frame)}
        counts = make_shared_resample(frame, args.b, args.seed)
        if counts.shape != (args.b, len(frame)):
            raise RuntimeError(
                f"{model}: shared bootstrap matrix has shape {counts.shape}, "
                f"expected {(args.b, len(frame))}"
            )
        draws[f"{model}_counts"] = counts.astype(np.int16)
        draws[f"{model}_frame"] = np.array(frame)

        n_lang = len(LANGS)
        n_control = len(CONTROL_CATEGORIES)
        raw_draws = np.full((args.b, n_lang, n_lang), np.nan)
        residual_draws = np.full_like(raw_draws, np.nan)
        change_draws = np.full_like(raw_draws, np.nan)
        length_corr_draws = np.full_like(raw_draws, np.nan)
        control_raw_draws = np.full((args.b, n_lang, n_control), np.nan)
        control_corr_draws = np.full((args.b, n_lang, n_lang, n_control), np.nan)
        converse_draws = np.full_like(control_corr_draws, np.nan)
        converse_change_draws = np.full_like(control_corr_draws, np.nan)
        control_present = np.zeros((n_lang, n_control), dtype=np.int8)

        rows = []
        missing: list[str] = []
        for ei, el in enumerate(LANGS):
            acts, y, lens, scenario = raw[el]
            hidden = acts.shape[1]
            wrong_dec = {lang: tuple(np.asarray(dec_dirs[lang]).shape) for lang in LANGS
                         if np.asarray(dec_dirs[lang]).shape != (hidden,)}
            if wrong_dec:
                raise ValueError(
                    f"{model}/{el}: deception direction/cache dimension mismatch: "
                    f"hidden={hidden}, directions={wrong_dec}"
                )
            cats = [c for c in CONTROL_CATEGORIES if el in ctrl[c]]
            missing.extend(f"{c}/{el}" for c in CONTROL_CATEGORIES if el not in ctrl[c])
            wrong_ctrl = {c: tuple(np.asarray(ctrl[c][el]).shape) for c in cats
                          if np.asarray(ctrl[c][el]).shape != (hidden,)}
            if wrong_ctrl:
                raise ValueError(
                    f"{model}/{el}: control direction/cache dimension mismatch: "
                    f"hidden={hidden}, directions={wrong_ctrl}"
                )
            s_ctrl = {c: acts @ ctrl[c][el] for c in cats}
            control_matrix = np.column_stack([s_ctrl[c] for c in cats]) if cats \
                else np.empty((len(y), 0), dtype=np.float64)
            deception_matrix = np.column_stack([acts @ dec_dirs[tl] for tl in LANGS])
            scenario_idx = np.array([frame_idx[s] for s in scenario], dtype=int)
            boot = bootstrap_eval(
                deception_matrix, control_matrix, lens, y, scenario_idx, counts
            )
            raw_draws[:, :, ei] = boot["auroc_raw"]
            residual_draws[:, :, ei] = boot["auroc_residualized"]
            change_draws[:, :, ei] = boot["auroc_change"]
            length_corr_draws[:, :, ei] = boot["spearman_length"]
            for local_ci, c in enumerate(cats):
                global_ci = CONTROL_CATEGORIES.index(c)
                control_present[ei, global_ci] = 1
                control_raw_draws[:, ei, global_ci] = boot["control_auroc_raw"][:, local_ci]
                control_corr_draws[:, :, ei, global_ci] = (
                    boot["spearman_control"][:, :, local_ci]
                )
                converse_draws[:, :, ei, global_ci] = (
                    boot["converse_auroc_residualized"][:, :, local_ci]
                )
                converse_change_draws[:, :, ei, global_ci] = (
                    boot["converse_auroc_change"][:, :, local_ci]
                )

            cov = np.column_stack([control_matrix, lens])
            for ti, tl in enumerate(LANGS):
                s_d = deception_matrix[:, ti]
                r = residualize(s_d, cov)
                auroc_raw = auroc_rank(s_d, y)
                auroc_residualized = auroc_rank(r, y)
                row = {
                    "model": model, "eval_lang": el, "train_lang": tl, "layer": layer,
                    "auroc_raw": auroc_raw,
                    "auroc_residualized": auroc_residualized,
                    "auroc_change_residualized_minus_raw": (
                        auroc_residualized - auroc_raw
                    ),
                    "controls_used": "+".join(cats) + "+length",
                    "n": int(len(y)),
                    "n_scenarios_observed": int(len(set(scenario))),
                    "n_scenarios_shared_frame": int(len(frame)),
                    "bootstrap_B": int(args.b),
                    "spearman_length": spearman(s_d, lens),
                }
                context = f"{model}/{el}/{tl}"
                _add_ci(row, "auroc_raw", boot["auroc_raw"][:, ti], context)
                _add_ci(
                    row, "auroc_residualized",
                    boot["auroc_residualized"][:, ti], context,
                )
                _add_ci(
                    row, "auroc_change_residualized_minus_raw",
                    boot["auroc_change"][:, ti], context,
                )
                _add_ci(
                    row, "spearman_length",
                    boot["spearman_length"][:, ti], context,
                )
                for c in cats:
                    local_ci = cats.index(c)
                    row[f"spearman_{c}"] = spearman(s_d, s_ctrl[c])
                    conv = residualize(s_ctrl[c], np.column_stack([s_d, lens]))
                    raw_control = auroc_rank(s_ctrl[c], y)
                    residual_converse = auroc_rank(conv, y)
                    row[f"auroc_{c}_raw"] = raw_control
                    row[f"auroc_{c}_residualized_converse"] = residual_converse
                    row[
                        f"auroc_{c}_converse_change_residualized_minus_raw"
                    ] = residual_converse - raw_control
                    _add_ci(
                        row, f"spearman_{c}",
                        boot["spearman_control"][:, ti, local_ci], context,
                    )
                    _add_ci(
                        row, f"auroc_{c}_raw",
                        boot["control_auroc_raw"][:, local_ci], context,
                    )
                    _add_ci(
                        row, f"auroc_{c}_residualized_converse",
                        boot["converse_auroc_residualized"][:, ti, local_ci],
                        context,
                    )
                    _add_ci(
                        row,
                        f"auroc_{c}_converse_change_residualized_minus_raw",
                        boot["converse_auroc_change"][:, ti, local_ci],
                        context,
                    )
                rows.append(row)

        draws[f"{model}_auroc_raw"] = raw_draws.astype(np.float32)
        draws[f"{model}_auroc_residualized"] = residual_draws.astype(np.float32)
        draws[f"{model}_auroc_change"] = change_draws.astype(np.float32)
        draws[f"{model}_spearman_length"] = length_corr_draws.astype(np.float32)
        draws[f"{model}_control_auroc_raw"] = control_raw_draws.astype(np.float32)
        draws[f"{model}_spearman_control"] = control_corr_draws.astype(np.float32)
        draws[f"{model}_converse_auroc_residualized"] = converse_draws.astype(np.float32)
        draws[f"{model}_converse_auroc_change"] = converse_change_draws.astype(np.float32)
        draws[f"{model}_control_present"] = control_present
        pd.DataFrame(rows).to_csv(args.out_root / f"{model}.csv", index=False)
        meta[model] = {
            "layer": int(layer),
            "missing_control_probes": sorted(set(missing)),
            "n_scenarios": len(frame),
            "frame_sha256": hashlib.sha256("\n".join(frame).encode()).hexdigest(),
            "n_items": {lang: int(len(raw[lang][1])) for lang in LANGS},
            "n_scenarios_observed": {
                lang: int(len(set(raw[lang][3]))) for lang in LANGS
            },
        }
        print(f"[{model}] residualization: {len(rows)} rows"
              + (f" (missing controls: {sorted(set(missing))})" if missing else ""), flush=True)

    np.savez_compressed(args.out_root / "bootstrap_draws.npz", **draws)
    (args.out_root / "meta.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "B": args.b,
        "seed": args.seed,
        "registered": (args.b, args.seed) == (B_REGISTERED, SEED_REGISTERED),
        "cluster_unit": "scenario_id; all admitted responses from a drawn scenario repeated "
                        "with that scenario's draw multiplicity",
        "shared_draws": "one deterministic count matrix per model, shared across all eval "
                        "languages, train probes, controls, forward/converse fits, and "
                        "correlations; same construction as transfer_stats",
        "ci": "95% percentile scenario-clustered bootstrap; unadjusted descriptive CIs",
        "bootstrap_refit": "forward and converse OLS residualization refit within every "
                           "bootstrap replicate",
        "design": "residual = OLS(target ~ intercept + language-matched control scores + "
                  "length); converse = control residualized on [deception score, length]",
        "draw_axes": {
            "auroc_raw/auroc_residualized/auroc_change/spearman_length":
                "B x train_lang x eval_lang",
            "control_auroc_raw": "B x eval_lang x control_category",
            "spearman_control/converse_auroc_residualized/converse_auroc_change":
                "B x train_lang x eval_lang x control_category",
            "language_order": LANGS,
            "control_category_order": list(CONTROL_CATEGORIES),
            "auroc_change_sign": "residualized minus raw",
            "converse_change_sign": "residualized converse minus raw control AUROC",
        },
        "models": meta}, indent=2))
    print(
        f"Saved: {args.out_root}/<model>.csv + bootstrap_draws.npz + meta.json",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
