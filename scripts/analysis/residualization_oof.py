#!/usr/bin/env python3
"""
Out-of-fold (scenario-grouped 5-fold) forward residualization.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
IN = _REPO / "analysis" / "residualization"
OUT = _REPO / "analysis" / "residualization_oof"

MODELS = ["llama-8b", "aya-expanse-8b", "gemma-2b", "gemma-4-e4b", "qwen3.5-9b"]
LANGS = ["en", "pt", "es", "de", "zh"]
CONTROL_CATEGORIES = ("formality", "politeness", "truthfulness", "sentiment")
N_FOLDS, FOLD_SEED = 5, 42

FL = {m: json.load(open(_REPO / "results" / "frozen_layers" / m /
                        "chosen_layer.json"))["chosen_layer"] for m in MODELS}


def _stack_imports():
    """Reuse the run-of-record loaders (heavy fit-time deps stubbed)."""
    import importlib.util
    import types
    tr = types.ModuleType("transformers")
    tr.PreTrainedModel = object
    tr.PreTrainedTokenizerBase = object
    sys.modules.setdefault("transformers", tr)
    td = types.ModuleType("project.tokenized_data")
    td.TokenizedDataset = object
    td.stub_tokenized_for_masked_counts = None
    sys.modules.setdefault("project.tokenized_data", td)
    pkg = types.ModuleType("project")
    pkg.__path__ = [str(_REPO / "project")]
    sys.modules.setdefault("project", pkg)
    for name in ("types", "scores", "activations", "detectors"):
        if f"project.{name}" not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                f"project.{name}", _REPO / "project" / f"{name}.py")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[f"project.{name}"] = mod
            spec.loader.exec_module(mod)
    from scripts.analysis.residualization import (load_eval_cache,
                                                   _control_dirs, residualize)
    from scripts.analysis.transfer_stats import auroc_rank
    from scripts.analysis.operating_points import _load_probe_dirs_and_norms
    return load_eval_cache, _control_dirs, residualize, auroc_rank, \
        _load_probe_dirs_and_norms


def main() -> int:
    global MODELS
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", default=MODELS)
    args = ap.parse_args()
    MODELS = args.models
    (load_eval_cache, _control_dirs, residualize, auroc_rank,
     _load_probe_dirs_and_norms) = _stack_imports()
    OUT.mkdir(parents=True, exist_ok=True)
    arch = np.load(IN / "bootstrap_draws.npz")
    arch_cells = {}
    for m in MODELS:
        c = pd.read_csv(IN / f"{m}.csv")
        arch_cells[m] = c.set_index(["train_lang", "eval_lang"])

    draws_out = {}
    all_rows = []
    for m in MODELS:
        layer = FL[m]
        dec_dirs, _ = _load_probe_dirs_and_norms(m, layer, "generic")
        ctrl = _control_dirs(m, layer)
        counts = arch[f"{m}_counts"]
        frame = [str(x) for x in arch[f"{m}_frame"]]
        rng = np.random.default_rng(FOLD_SEED)
        perm = rng.permutation(len(frame))
        fold_of = {frame[i]: int(k % N_FOLDS)
                   for k, i in enumerate(perm)}

        B = counts.shape[0]
        res_draws = np.full((B, 5, 5), np.nan)   
        raw_draws = np.full((B, 5, 5), np.nan)
        rows_model = {}
        for ei, el in enumerate(LANGS):
            acts, y, lens, scenario = load_eval_cache(
                _REPO / "results" / "activation_cache" / m /
                "deceptionbench" / f"{el}_layer{layer}.npz")
            assert set(scenario) <= set(frame), f"{m}/{el}: frame mismatch"
            s_d = np.column_stack([acts @ np.asarray(dec_dirs[tl])
                                   for tl in LANGS])
            s_c = np.column_stack([acts @ np.asarray(ctrl[c][el])
                                   for c in CONTROL_CATEGORIES])
            cov = np.column_stack([s_c, lens])
            folds = np.array([fold_of[s] for s in scenario])
            scen_idx = np.array([frame.index(s) for s in scenario])

            def oof_residual(rows_sel):
                """OOF residuals for the (possibly expanded) row selection."""
                sd, cv, fd = s_d[rows_sel], cov[rows_sel], folds[rows_sel]
                out_res = np.empty_like(sd)
                for f in range(N_FOLDS):
                    tr, te = fd != f, fd == f
                    if not te.any():
                        continue
                    if not tr.any():
                        out_res[te] = np.nan
                        continue
                    X = np.column_stack([np.ones(tr.sum()), cv[tr]])
                    beta, *_ = np.linalg.lstsq(X, sd[tr], rcond=None)
                    Xte = np.column_stack([np.ones(te.sum()), cv[te]])
                    out_res[te] = sd[te] - Xte @ beta
                return out_res

            res_point = oof_residual(np.arange(len(y)))
            for ti, tl in enumerate(LANGS):
                raw_pt = auroc_rank(s_d[:, ti], y)
                a = arch_cells[m].loc[(tl, el)]
                assert abs(raw_pt - a.auroc_raw) < 1e-9, (
                    f"{m} {tl}->{el}: raw AUROC does not reproduce the "
                    f"archived cell ({raw_pt} vs {a.auroc_raw})")
                res_pt = auroc_rank(res_point[:, ti], y)
                rows_model[(tl, el)] = {
                    "model": m, "train_lang": tl, "eval_lang": el,
                    "auroc_raw": raw_pt,
                    "auroc_residualized_oof": round(float(res_pt), 6),
                    "auroc_change_oof": round(float(res_pt - raw_pt), 6),
                    "auroc_residualized_insample_archived":
                        float(a.auroc_residualized),
                }
            for bi in range(B):
                mult = counts[bi][scen_idx]
                keep = mult > 0
                rep = np.repeat(np.flatnonzero(keep),
                                mult[keep].astype(int))
                if not len(rep):
                    continue
                y_rep = y[rep]
                if y_rep.min() == y_rep.max():
                    continue
                res_rep = oof_residual(rep)
                for ti in range(5):
                    raw_draws[bi, ti, ei] = auroc_rank(s_d[rep][:, ti], y_rep)
                    res_draws[bi, ti, ei] = auroc_rank(res_rep[:, ti], y_rep)

        for (tl, el), row in rows_model.items():
            ti, ei = LANGS.index(tl), LANGS.index(el)
            v = res_draws[:, ti, ei]
            ch = res_draws[:, ti, ei] - raw_draws[:, ti, ei]
            row.update({
                "auroc_residualized_oof_ci_lo":
                    round(float(np.nanpercentile(v, 2.5)), 6),
                "auroc_residualized_oof_ci_hi":
                    round(float(np.nanpercentile(v, 97.5)), 6),
                "auroc_change_oof_ci_lo":
                    round(float(np.nanpercentile(ch, 2.5)), 6),
                "auroc_change_oof_ci_hi":
                    round(float(np.nanpercentile(ch, 97.5)), 6),
                "n_degenerate_reps": int(np.isnan(v).sum()),
            })
            all_rows.append(row)
        draws_out[f"{m}_auroc_raw"] = raw_draws.astype(np.float32)
        draws_out[f"{m}_auroc_residualized_oof"] = res_draws.astype(np.float32)
        mr = np.nanmean(res_draws.reshape(B, -1), axis=1)
        mraw = np.nanmean(raw_draws.reshape(B, -1), axis=1)
        mch = np.nanmean((res_draws - raw_draws).reshape(B, -1), axis=1)
        pt_res = np.mean([rows_model[(t, e)]["auroc_residualized_oof"]
                          for t in LANGS for e in LANGS])
        pt_raw = np.mean([rows_model[(t, e)]["auroc_raw"]
                          for t in LANGS for e in LANGS])
        print(f"{m:16s} raw {pt_raw:.3f}  OOF-resid {pt_res:.3f} "
              f"[{np.nanpercentile(mr, 2.5):.3f},"
              f"{np.nanpercentile(mr, 97.5):.3f}]  paired change "
              f"{pt_res - pt_raw:+.3f} [{np.nanpercentile(mch, 2.5):+.3f},"
              f"{np.nanpercentile(mch, 97.5):+.3f}]")

    pd.DataFrame(all_rows).to_csv(OUT / "cells.csv", index=False)
    np.savez_compressed(OUT / "bootstrap_draws.npz", **draws_out)
    (OUT / "meta.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "folds": {"n": N_FOLDS, "seed": FOLD_SEED,
                  "rule": "scenario-grouped, one fold per scenario shared "
                          "across evaluation languages (round-robin over a "
                          "seeded permutation of the archived frame)"},
        "residualizer": "per model and evaluation language: OLS of each "
                        "training-language deception score on [intercept, "
                        "four evaluation-language control scores, response "
                        "length]; fitted on four folds, applied held-out; "
                        "complete procedure repeated inside every ARCHIVED "
                        "bootstrap replicate (no new resampling)",
        "gates": "frames from archived npz; point raw AUROCs reproduce "
                 "archived cells to 1e-9 (asserted per cell)",
        "converse": "NOT recomputed; archived in-sample converse remains "
                    "appendix-tier with explicit labeling",
    }, indent=1))
    print(f"saved -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
