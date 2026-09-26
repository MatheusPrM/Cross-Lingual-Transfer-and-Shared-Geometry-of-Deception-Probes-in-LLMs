"""
Central path conventions for the project.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

CONFIG_DIR = REPO_ROOT / "configs"
EXPERIMENTS_CONFIG_DIR = CONFIG_DIR / "experiments"
DATASETS_DIR = REPO_ROOT / "datasets"
TRAINING_DATA_DIR = DATASETS_DIR / "training"
EVAL_DATA_DIR = DATASETS_DIR / "evaluation"
ROLLOUTS_DIR = REPO_ROOT / "rollouts"
RESULTS_DIR = REPO_ROOT / "results"

DATASETS_YAML = CONFIG_DIR / "datasets.yaml"


def get_experiment_config_path(name: str) -> Path:
    """Return path to an experiment config by name (with or without .yaml)."""
    base = name.strip()
    if not base.endswith(".yaml") and not base.endswith(".yml"):
        base = f"{base}.yaml"
    return EXPERIMENTS_CONFIG_DIR / base


def rollout_json_path(
    model_id: str,
    *,
    dataset_id: str | None = None,
    dataset_path: Path | None = None,
    file_stem: str | None = None,
    condition: str | None = None,
) -> Path:
    """
    Default path for generated rollouts: ``rollouts/<model>/<family>/<stem>_<model>.json``.
    """
    from project.datasets import get_rollout_filename_stem, rollout_stem_for_dataset_id

    subdir = get_rollout_filename_stem(dataset_id, dataset_path)
    if file_stem is not None:
        fname = file_stem
    elif dataset_id:
        fname = rollout_stem_for_dataset_id(dataset_id)
    elif dataset_path is not None:
        fname = Path(dataset_path).stem
    else:
        fname = subdir
    suffix = f"_{condition}" if condition else ""
    return ROLLOUTS_DIR / model_id / subdir / f"{fname}_{model_id}{suffix}.json"
