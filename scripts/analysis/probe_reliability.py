#!/usr/bin/env python3
"""
Split-half probe reliability + attenuation-corrected cross-language cosines.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

_TRAIN_TEMPLATE = {
    "sentiment": "datasets/training/sentiment/sentiment_{lang}.csv",
    "sentiment_sst2": "datasets/training/sentiment/sentiment_sst2_{lang}.csv",
    "truthfulness": "datasets/training/repe/true_false_facts_{lang}.csv",
    "generic": "datasets/training/repe/true_false_facts_{lang}.csv",  
}


class _ActsView:
    """Minimal duck-type for LogisticRegressionDetector.fit (carries dialogue counts so
    the detector can pool per dialogue and group by fact)."""

    def __init__(self, t, counts=None):
        self._t = t
        self.detect_token_count_per_dialogue = counts
        self.acts_are_masked_only = True

    def get_masked_activations(self):
        return self._t


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True)
    p.add_argument("--category", default="sentiment", choices=sorted(_TRAIN_TEMPLATE))
    p.add_argument("--langs", nargs="+", default=["en", "pt", "es", "de", "zh"])
    p.add_argument("--n-splits", type=int, default=10,
                   help="seeded split-half repeats per language")
    p.add_argument("--split-seed-base", type=int, default=1000)
    p.add_argument("--reg-coeff", type=float, default=10.0, help="match sweep_layers default")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-root", type=Path, default=_REPO_ROOT / "analysis" / "probe_reliability")
    args = p.parse_args()

    import torch

    from project.activations import extract_masked_activations
    from project.detectors import LogisticRegressionDetector
    from project.loaders import load_dialogues_from_path
    from project.model_loader import load_model_and_tokenizer
    from project.reproducibility import set_seed
    from project.tokenized_data import tokenize_dialogues
    from project.types import Label
    from scripts.analysis.load_probes import probe_path

    set_seed(args.seed)
    from project.frozen_layers import load_frozen_layer  
    layer = load_frozen_layer(args.model)
    out_dir = args.out_root / args.model / args.category
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{args.model}/{args.category}] loading model (layer {layer})...", flush=True)
    hf_model, tok = load_model_and_tokenizer(args.model)
    tok.padding_side = "left"
    if tok.pad_token_id is None and tok.bos_token_id is not None:
        tok.pad_token_id = tok.bos_token_id

    def _unit(v: "torch.Tensor") -> "torch.Tensor":
        return v / v.norm()

    def _fit_direction(tok_rows_h, tok_rows_d, counts_h, counts_d) -> "torch.Tensor":
        det = LogisticRegressionDetector(layers=[layer], reg_coeff=args.reg_coeff, scale=True)
        det.fit(_ActsView(tok_rows_h, counts_h), _ActsView(tok_rows_d, counts_d),
                unit="dialogue") 
        return _unit(det.effective_direction(layer).float())

    rel_rows, rel_by_lang, dm_dirs = [], {}, {}
    for lang in args.langs:
        csv_rel = _TRAIN_TEMPLATE[args.category].format(lang=lang)
        if not (_REPO_ROOT / csv_rel).exists():
            print(f"  [miss] {csv_rel}", flush=True)
            continue
        dialogues, labels = load_dialogues_from_path(_REPO_ROOT / csv_rel, repe_category=args.category)
        print(f"  extracting {lang} ({len(dialogues)} dialogues)...", flush=True)
        acts = extract_masked_activations(
            hf_model, tokenize_dialogues(dialogues, tok, model_id=args.model),
            layers=[layer], batch_size=args.batch_size)
        counts = acts.detect_token_count_per_dialogue
        bounds = np.concatenate([[0], np.cumsum(counts)])
        y = np.array([1 if l == Label.DECEPTIVE else 0 for l in labels])[: len(counts)]

        def token_rows(dialogue_idx: np.ndarray) -> "torch.Tensor":
            rows = [acts.acts[bounds[i]:bounds[i + 1], :, :] for i in dialogue_idx]
            return torch.cat(rows, dim=0)

        tok_labels = np.repeat(y, counts)
        dm = acts.acts[tok_labels == 1, 0, :].float().mean(dim=0) - acts.acts[tok_labels == 0, 0, :].float().mean(dim=0)
        dm_dirs[lang] = _unit(dm)

        idx_h_all = np.where(y == 0)[0]
        idx_d_all = np.where(y == 1)[0]
        n_facts = min(len(idx_h_all), len(idx_d_all))
        cos_halves = []
        for s in range(args.n_splits):
            rng = np.random.default_rng(args.split_seed_base + s)
            fact_perm = rng.permutation(n_facts)
            facts_a = set(fact_perm[: n_facts // 2].tolist())
            dirs = []
            for in_a in (True, False):
                sel = [k for k in range(n_facts) if (k in facts_a) == in_a]
                dirs.append(_fit_direction(token_rows(idx_h_all[sel]), token_rows(idx_d_all[sel]),
                                           [counts[i] for i in idx_h_all[sel]],
                                           [counts[i] for i in idx_d_all[sel]]))
            cos_halves.append(float(torch.dot(dirs[0], dirs[1]).item()))
        rel = float(np.mean(cos_halves))
        from project.stats_utils import spearman_brown
        sb = spearman_brown(rel)
        rel_by_lang[lang] = sb
        half_sd = float(np.std(cos_halves, ddof=1)) if len(cos_halves) > 1 else 0.0
        ci_hw = 1.96 * half_sd / max(np.sqrt(len(cos_halves)), 1)
  
        tok_det = LogisticRegressionDetector(layers=[layer], reg_coeff=args.reg_coeff, scale=True)
        tok_det.fit(_ActsView(token_rows(idx_h_all[:n_facts]), [counts[i] for i in idx_h_all[:n_facts]]),
                    _ActsView(token_rows(idx_d_all[:n_facts]), [counts[i] for i in idx_d_all[:n_facts]]),
                    unit="token")
        tok_dir = _unit(tok_det.effective_direction(layer).float())
        pooled_det = LogisticRegressionDetector(layers=[layer], reg_coeff=args.reg_coeff, scale=True)
        pooled_det.fit(_ActsView(token_rows(idx_h_all[:n_facts]), [counts[i] for i in idx_h_all[:n_facts]]),
                       _ActsView(token_rows(idx_d_all[:n_facts]), [counts[i] for i in idx_d_all[:n_facts]]),
                       unit="dialogue")
        pooled_dir = _unit(pooled_det.effective_direction(layer).float())
        cos_token_pooled = float(torch.dot(tok_dir, pooled_dir).item())

        tok_det.save(out_dir / f"{lang}_token_detector.pkl")
        pooled_det.save(out_dir / f"{lang}_pooled_detector.pkl")
        rel_rows.append({"model": args.model, "category": args.category, "lang": lang, "layer": layer,
                         "reliability_cos_mean": rel, "reliability_cos_std": half_sd,
                         "reliability_cos_ci_lo": rel - ci_hw, "reliability_cos_ci_hi": rel + ci_hw,
                         "reliability_sb_full": sb,
                         "per_seed_cosines": ";".join(f"{c:.6f}" for c in cos_halves),
                         "cos_token_vs_pooled_direction": cos_token_pooled,
                         "n_splits": args.n_splits, "n_dialogues": int(len(y))})
        print(f"  {lang}: split-half cos = {rel:.3f} ± {half_sd:.3f} -> SB full-test {sb:.3f}", flush=True)
        del acts

    full_dirs = {}
    for lang in rel_by_lang:
        pp = probe_path(args.model, lang, layer, category=args.category, require_category_path=True) 
        if pp.exists():
            det = LogisticRegressionDetector.load(pp)
            full_dirs[lang] = _unit(det.effective_direction(layer).float())

    for r in rel_rows:  
        l = r["lang"]
        if l in full_dirs and l in dm_dirs:
            r["cos_lr_diffmeans"] = float(torch.dot(full_dirs[l], dm_dirs[l]).item())
    pd.DataFrame(rel_rows).to_csv(out_dir / "reliability.csv", index=False)
    cos_rows = []
    for a, b in combinations(sorted(full_dirs), 2):
        raw = float(torch.dot(full_dirs[a], full_dirs[b]).item())

        sb_a, sb_b = rel_by_lang[a], rel_by_lang[b]
        denom = float(np.sqrt(sb_a * sb_b)) if sb_a > 0 and sb_b > 0 else float("nan")
        corrected = raw / denom if np.isfinite(denom) else float("nan")
        cos_rows.append({"model": args.model, "category": args.category, "lang_a": a, "lang_b": b,
                         "cos_raw": raw, "cos_corrected": corrected,
                         "rel_sb_a": sb_a, "rel_sb_b": sb_b})
        print(f"  {a}~{b}: raw {raw:+.3f} -> SB-corrected {corrected:+.3f}", flush=True)
    pd.DataFrame(cos_rows).to_csv(out_dir / "corrected_cosines.csv", index=False)
    print(f"Saved: {out_dir}/reliability.csv + corrected_cosines.csv", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
