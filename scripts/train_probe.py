#!/usr/bin/env python3
"""
Train a linear (logistic regression) deception probe on REPE-style data.

Follows the original methodology: paired honest/deceptive dialogues,
extract activations at the given layer(s), fit a logistic regression probe,
and save it for later evaluation.
"""

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from project.models import DEFAULT_MODEL_ID
from project.paths import REPO_ROOT
from project.probe_training import (
    get_default_layers,
    train_probe_from_act_caches,
    train_probe_from_mixed_act_cache,
    train_probe_from_path,
)
from project.datasets import get_dataset_info, repe_category_from_train_data
from project.reproducibility import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a deception probe on REPE-style (or other) training data, or from activation caches.",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Dataset id (from configs/datasets.yaml) or path to training file (e.g. CSV for REPE).",
    )
    parser.add_argument(
        "--acts",
        type=Path,
        default=None,
        help="Single activation cache (.pkl) with honest + deceptive labels; splits internally.",
    )
    parser.add_argument(
        "--acts-honest",
        type=Path,
        default=None,
        help="With --acts-deceptive: honest-only cache from extract_activations.py.",
    )
    parser.add_argument(
        "--acts-deceptive",
        type=Path,
        default=None,
        help="With --acts-honest: deceptive-only cache from extract_activations.py.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Model id from project registry (e.g. gemma-2b, llama-8b, gemma-4-e4b).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Path to save the trained probe (e.g. probes/gemma-2b/repe_probe.pt).",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layer indices (e.g. 22 or 20,21,22). Default: middle layer.",
    )
    parser.add_argument(
        "--reg-coeff",
        type=float,
        default=10.0,
        help="Inverse regularization: the probe uses C = 1/reg_coeff "
             "(default 10, i.e. C = 0.1, as in the reported runs).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for activation extraction.",
    )
    parser.add_argument(
        "--no-scale",
        action="store_true",
        help="Disable StandardScaler on activations before fitting.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()
    set_seed(args.seed, deterministic=True)

    has_pair = args.acts_honest is not None or args.acts_deceptive is not None
    if has_pair:
        if args.acts_honest is None or args.acts_deceptive is None:
            print("Both --acts-honest and --acts-deceptive are required together.", file=sys.stderr)
            sys.exit(2)
        if args.acts is not None or args.train_data is not None:
            print("Do not combine --acts-honest/--acts-deceptive with --acts or --train-data.", file=sys.stderr)
            sys.exit(2)
    elif args.acts is not None:
        if args.train_data is not None:
            print("Use either --acts or --train-data, not both.", file=sys.stderr)
            sys.exit(2)
    elif args.train_data is not None:
        pass
    else:
        print(
            "Provide one of: --train-data, --acts (mixed-label cache), or --acts-honest with --acts-deceptive.",
            file=sys.stderr,
        )
        sys.exit(2)

    layers = None
    if args.layers is not None:
        layers = [int(x.strip()) for x in args.layers.split(",")]

    if layers is None and args.acts is None and not (args.acts_honest or args.acts_deceptive):
        layers = get_default_layers(args.model)

    if args.acts is not None:
        acts_path = Path(args.acts)
        if not acts_path.is_absolute():
            acts_path = REPO_ROOT / acts_path
        if not acts_path.exists():
            print(f"Error: Cache not found: {acts_path}", file=sys.stderr)
            sys.exit(1)
        print(
            f"Training probe from mixed cache {acts_path}, "
            f"layers={layers if layers is not None else '<from cache>'}, reg={args.reg_coeff}",
            flush=True,
        )
        detector = train_probe_from_mixed_act_cache(
            acts_path,
            layers=layers,
            reg_coeff=args.reg_coeff,
            scale=not args.no_scale,
        )
    elif args.acts_honest is not None:
        h = Path(args.acts_honest)
        d = Path(args.acts_deceptive)
        if not h.is_absolute():
            h = REPO_ROOT / h
        if not d.is_absolute():
            d = REPO_ROOT / d
        if not h.exists() or not d.exists():
            print(f"Error: Cache not found: {h} / {d}", file=sys.stderr)
            sys.exit(1)
        print(
            f"Training probe from caches honest={h}, deceptive={d}, "
            f"layers={layers if layers is not None else '<from cache>'}, reg={args.reg_coeff}",
            flush=True,
        )
        detector = train_probe_from_act_caches(
            h,
            d,
            layers=layers,
            reg_coeff=args.reg_coeff,
            scale=not args.no_scale,
        )
    else:
        assert args.train_data is not None
        train_path: Path | None = None
        if Path(args.train_data).exists():
            train_path = Path(args.train_data)
        else:
            info = get_dataset_info(args.train_data)
            if info is not None:
                train_path = info.resolved_path()
        if train_path is None or not train_path.exists():
            print(f"Error: Training data not found: {args.train_data}", file=sys.stderr)
            sys.exit(1)
        repe_category = repe_category_from_train_data(args.train_data)
        print(f"Training probe on {train_path} with model {args.model}, "
              f"layers {layers}, reg={args.reg_coeff}, category={repe_category}")
        detector = train_probe_from_path(
            train_path,
            args.model,
            layers=layers,
            reg_coeff=args.reg_coeff,
            train_data_id=args.train_data,
            repe_category=repe_category,
            scale=not args.no_scale,
            batch_size=args.batch_size,
        )

    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    sweeps_root = (REPO_ROOT / "results" / "sweeps").resolve()
    out_dir = out.parent.resolve()
    if not out_dir.is_relative_to(sweeps_root):
        for stale in sorted(out_dir.glob("repe_probe_layer*.pt")):
            if stale.resolve() != out.resolve():
                stale.unlink()
                print(f"Removed stale probe {stale}")
    detector.save(out)
    print(f"Saved probe to {out}")


if __name__ == "__main__":
    main()