#!/usr/bin/env python3
"""Layer sweep for frozen-layer selection.

Trains the English generic RepE probe at every candidate layer with grouped
5-fold validation (honest/deceptive versions of the same factual item stay in
one fold; scaler and classifier are refit inside every fold) and records
val_auroc/val_margin per layer. Selection never uses DeceptionBench
performance. Follow with scripts/select_layer.py.

    python scripts/sweep_layers.py --model gemma-2b --layers 10,12,14,16,18
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from project.activations import Activations, extract_masked_activations
from project.datasets import get_dataset_info
from project.detectors import LogisticRegressionDetector
from project.loaders import load_dialogues_from_path
from project.probe_training import split_dialogues_by_label
from project.reproducibility import set_seed
from project.tokenized_data import tokenize_dialogues


def _slice_one_layer(acts: Activations, layer: int) -> Activations:
    idx = acts.layers.index(layer)
    one = acts.acts[:, idx: idx + 1, :]
    return Activations(
        acts=one,
        layers=[layer],
        tokenized_dataset=acts.tokenized_dataset,
        acts_are_masked_only=True,
        detect_token_count_per_dialogue=acts.detect_token_count_per_dialogue,
        dialogue_index_per_token=acts.dialogue_index_per_token,
        labels=acts.labels,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--train-data", default="repe_generic_en")
    ap.add_argument("--layers", required=True,
                    help="comma-separated candidate layers, e.g. 10,12,14,16,18")
    ap.add_argument("--reg-coeff", type=float, default=10.0)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="candidate-checkpoint dir (default results/sweeps/<model>/checkpoints)")
    ap.add_argument("--output-csv", type=Path, default=None,
                    help="sweep metrics CSV (default results/sweeps/<model>_<train-data>.csv)")
    args = ap.parse_args()

    import pandas as pd
    from project.model_loader import load_model_and_tokenizer

    set_seed(args.seed)
    layers = [int(x) for x in args.layers.split(",")]

    info = get_dataset_info(args.train_data)
    if info is None:
        raise SystemExit(f"unknown dataset id: {args.train_data}")
    train_path = info.resolved_path()
    if not train_path.exists():
        raise SystemExit(f"training data not found: {train_path}")
    repe_category = str(info.category or "generic")
    lang = args.train_data.rsplit("_", 1)[-1]
    out_dir = args.out_dir or (_REPO_ROOT / "results" / "sweeps" /
                               args.model / "checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.output_csv or (_REPO_ROOT / "results" / "sweeps" /
                                  f"{args.model}_{args.train_data}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    dialogues, labels = load_dialogues_from_path(
        train_path, repe_category=repe_category)
    honest, deceptive = split_dialogues_by_label(dialogues, labels)
    if not honest or not deceptive:
        raise SystemExit("need both honest and deceptive training dialogues")

    model, tokenizer = load_model_and_tokenizer(args.model)
    if model is None:
        raise SystemExit("layer sweep requires a loaded model")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.bos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.bos_token_id

    tok_h = tokenize_dialogues(honest, tokenizer, model_id=args.model,
                               exclude_last_n_detect_tokens=0)
    tok_d = tokenize_dialogues(deceptive, tokenizer, model_id=args.model,
                               exclude_last_n_detect_tokens=0)
    acts_h = extract_masked_activations(model, tok_h, layers=layers,
                                        batch_size=args.batch_size)
    acts_d = extract_masked_activations(model, tok_d, layers=layers,
                                        batch_size=args.batch_size)

    rows = []
    for layer in layers:
        h1 = _slice_one_layer(acts_h, layer)
        d1 = _slice_one_layer(acts_d, layer)
        detector = LogisticRegressionDetector(
            layers=[layer], reg_coeff=args.reg_coeff, scale=True,
            train_data=args.train_data)
        detector.fit(h1, d1, cv_folds=args.cv_folds)
        out_file = out_dir / f"repe_probe_layer{layer}.pt"
        detector.save(out_file)
        rows.append({
            "model": args.model, "train_data": args.train_data,
            "layer": layer, "val_auroc": detector.val_auroc,
            "val_margin": detector.val_margin,
            "val_n_total": detector.val_n_total,
            "val_n_honest": detector.val_n_honest,
            "val_n_deceptive": detector.val_n_deceptive,
            "probe_path": str(out_file.relative_to(_REPO_ROOT)),
        })
        print(f"  layer {layer}: val_auroc={detector.val_auroc:.4f} "
              f"val_margin={detector.val_margin:.3f}", flush=True)

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"sweep written: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
