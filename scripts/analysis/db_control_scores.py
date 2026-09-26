#!/usr/bin/env python3
"""
Alpaca-control score distributions for DeceptionBench probes + v_shared.
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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True)
    p.add_argument("--langs", nargs="+", default=["en", "pt", "es", "de", "zh"])
    p.add_argument("--max-control", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-root", type=Path, default=_REPO_ROOT / "analysis" / "db_control_scores")
    args = p.parse_args()

    import torch

    from project.activations import extract_masked_activations
    from project.detectors import LogisticRegressionDetector
    from project.loaders import load_dialogues_from_path
    from project.model_loader import load_model_and_tokenizer
    from project.reproducibility import set_seed
    from project.tokenized_data import tokenize_dialogues
    from scripts.analysis.load_probes import probe_path

    set_seed(args.seed)
    from project.frozen_layers import load_frozen_layer 
    layer = load_frozen_layer(args.model)
    vs_path = _REPO_ROOT / "results" / "geometry" / args.model / f"layer{layer}" / "v_shared.npy"
    v_shared = None
    if vs_path.exists():
        v = torch.tensor(np.load(vs_path), dtype=torch.float32)
        v_shared = v / v.norm()

    out_dir = args.out_root / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{args.model}] loading model (layer {layer})...", flush=True)
    hf_model, tok = load_model_and_tokenizer(args.model)
    tok.padding_side = "left"
    if tok.pad_token_id is None and tok.bos_token_id is not None:
        tok.pad_token_id = tok.bos_token_id

    rng = np.random.default_rng(args.seed)
    thr_rows = []
    for lang in args.langs:
        cpath = _REPO_ROOT / "datasets" / "evaluation" / "control" / f"control_{lang}.json"
        if not cpath.exists():
            print(f"  [miss] {cpath}", flush=True)
            continue
        all_dialogues, _ = load_dialogues_from_path(cpath)
        idx = rng.choice(len(all_dialogues), size=min(args.max_control, len(all_dialogues)), replace=False)
        dialogues = [all_dialogues[i] for i in sorted(idx)]
        keep, dropped = [], 0
        for d in dialogues:
            try:
                tokenize_dialogues([d], tok, model_id=args.model)
                keep.append(d)
            except Exception:
                dropped += 1
        if dropped:
            print(f"  [{lang}] [warn] dropped {dropped}/{len(dialogues)} control dialogues (mask construction failed)", flush=True)
        dialogues = keep
        print(f"  [{lang}] extracting {len(dialogues)} control dialogues...", flush=True)
        acts = extract_masked_activations(
            hf_model, tokenize_dialogues(dialogues, tok, model_id=args.model),
            layers=[layer], batch_size=args.batch_size)
        per_dial = torch.split(acts.acts[:, 0, :].float(), acts.detect_token_count_per_dialogue)
        pooled = torch.stack([c.mean(dim=0) for c in per_dial if c.shape[0] > 0])

        cols = {}
        for tl in ["en", "pt", "es", "de", "zh"]:
            pp = probe_path(args.model, tl, layer, category="generic", require_category_path=True) 
            if pp.exists():
                det = LogisticRegressionDetector.load(pp)
                w = det.effective_direction(layer).float()
                cols[f"score_probe_{tl}"] = (pooled @ w).numpy()
        if f"score_probe_{lang}" in cols:
            cols["score_native_probe"] = cols[f"score_probe_{lang}"]
        if v_shared is not None:
            cols["score_vshared"] = (pooled @ v_shared).numpy()
        df = pd.DataFrame(cols)
        df.insert(0, "lang", lang)
        df.to_csv(out_dir / f"control_scores_{lang}.csv", index=False)
        for scorer, vals in cols.items():
            thr_rows.append({"model": args.model, "lang": lang, "layer": layer, "scorer": scorer,
                             "n": int(len(vals)),
                             "threshold_q95": float(np.quantile(vals, 0.95)),
                             "threshold_q99": float(np.quantile(vals, 0.99))})
            print(f"  [{lang}] {scorer}: q99={thr_rows[-1]['threshold_q99']:.4f} (n={len(vals)})", flush=True)
        del acts, pooled

    pd.DataFrame(thr_rows).to_csv(out_dir / "thresholds.csv", index=False)
    print(f"Saved: {out_dir}/control_scores_<lang>.csv + thresholds.csv", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
