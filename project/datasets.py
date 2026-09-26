"""
Dataset registry.

Loads entries from configs/datasets.yaml.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from project.paths import DATASETS_YAML, REPO_ROOT

Split = Literal["training", "evaluation"]


@dataclass(frozen=True)
class DatasetInfo:
    """Single dataset entry."""

    id: str
    path: str
    split: Split
    language: Optional[str] = None
    notes: Optional[str] = None
    family: Optional[str] = None
    category: Optional[str] = None

    def resolved_path(self) -> Path:
        """Returns the absolute dataset path."""
        return REPO_ROOT / self.path

    def __str__(self) -> str:
        return f"{self.id} ({self.split}, {self.language or 'any'})"


def _load_datasets() -> list[DatasetInfo]:
    """Loads datasets from YAML."""
    if not DATASETS_YAML.exists():
        return []

    import yaml

    with open(DATASETS_YAML) as f:
        data = yaml.safe_load(f)

    items = data.get("datasets", [])
    return [
        DatasetInfo(
            id=item["id"],
            path=item["path"],
            split=item["split"],
            language=item.get("language"),
            notes=item.get("notes"),
            family=item.get("family"),
            category=item.get("category"),
        )
        for item in items
    ]


DATASET_REGISTRY: list[DatasetInfo] = _load_datasets()
_REGISTRY_BY_ID: dict[str, DatasetInfo] = {d.id: d for d in DATASET_REGISTRY}


def get_dataset_info(dataset_id: str) -> Optional[DatasetInfo]:
    """Returns DatasetInfo for an id."""
    return _REGISTRY_BY_ID.get(dataset_id)


def get_all_dataset_ids() -> list[str]:
    """Returns all dataset ids."""
    return [d.id for d in DATASET_REGISTRY]


def get_datasets_for_language(lang_id: str) -> list[DatasetInfo]:
    """Returns datasets for a language."""
    return [d for d in DATASET_REGISTRY if d.language == lang_id]


def get_datasets_for_split(split: Split) -> list[DatasetInfo]:
    """Returns datasets for a split."""
    return [d for d in DATASET_REGISTRY if d.split == split]


def probe_language_subdir_from_train_data(train_data: str) -> str:
    """
    Language code for ``probes/<model>/<language>/`` (REPE training ids/paths).
    """
    s = (train_data or "").strip()
    if not s:
        return "en"
    info = get_dataset_info(s)
    if info is not None and info.language:
        return str(info.language).strip().lower()
    p = Path(s)
    if p.suffix:
        stem = p.stem.lower()
        for lang in ("zh", "pt", "es", "de", "en"):
            if stem.endswith(f"_{lang}"):
                return lang
        return "en"
    s_low = s.lower()
    for lang in ("zh", "pt", "es", "de"):
        if s_low.endswith(f"_{lang}"):
            return lang
    return "en"


def repe_category_from_train_data(train_data: str) -> str:
    """Resolve the RepE training category for a ``train_data`` id."""
    info = get_dataset_info(train_data)
    if info is not None and info.category:
        return str(info.category)
    return "generic"


def probe_dir_for_model_and_train_data(model_id: str, train_data: str) -> Path:
    """Directory for shared REPE probes: ``probes/<model_id>/<language>/<category>/``."""
    lang = probe_language_subdir_from_train_data(train_data)
    category = repe_category_from_train_data(train_data)
    return REPO_ROOT / "probes" / model_id / lang / category


def _repe_legacy_probe_dir(model_id: str, train_data: str) -> Path:
    """Legacy path for shared REPE probes: ``probes/<model_id>/<language>/`` (no category).
    """
    lang = probe_language_subdir_from_train_data(train_data)
    return REPO_ROOT / "probes" / model_id / lang


def repe_probe_layer_path(model_id: str, train_data: str, layer: int) -> Path:
    """Path to a single-layer shared REPE probe."""
    new_path = probe_dir_for_model_and_train_data(model_id, train_data) / f"repe_probe_layer{int(layer)}.pt"
    if new_path.exists():
        return new_path
    legacy_path = _repe_legacy_probe_dir(model_id, train_data) / f"repe_probe_layer{int(layer)}.pt"
    if legacy_path.exists():
        import warnings

        warnings.warn(
            f"Using legacy probe path {legacy_path}; run "
            "scripts/migrate_probes_to_category_dirs.py to move it under the "
            "category subfolder.",
            stacklevel=2,
        )
        return legacy_path
    return new_path


def evilmath_rollout_stem_from_split_lang(split: str, lang: str) -> str:
    """
    Rollout JSON filename stem (without ``_<model>``), aligned with roleplaying: ``evilmath_en``,
    ``evilmath_pt``, ``evilmath_train``, ``evilmath_train_pt``, etc.
    """
    lang = (lang or "en").strip().lower()
    s = (split or "test").strip().lower()
    if s == "train":
        return "evilmath_train" if lang == "en" else f"evilmath_train_{lang}"
    return "evilmath_en" if lang == "en" else f"evilmath_{lang}"


_MASK_LANG_SUFFIXES: tuple[str, ...] = ("_de", "_es", "_pt", "_zh", "_en")


def _mask_archetype_from_dataset_id(dataset_id: str) -> str:
    """Extract the archetype slug from a registered MASK dataset id."""
    arch = dataset_id.strip()
    if arch.lower().startswith("mask_"):
        arch = arch[5:]
    for suf in _MASK_LANG_SUFFIXES:
        if arch.lower().endswith(suf):
            arch = arch[: -len(suf)]
            break
    return arch or "known_facts"


def mask_rollout_stem_from_dataset_id(dataset_id: str) -> str:
    """Each archetype-language pair gets a rollout stem."""
    info = get_dataset_info(dataset_id)
    if info is not None and (info.family or "").strip().lower() == "mask":
        lang = (info.language or "en").strip().lower()
        archetype = _mask_archetype_from_dataset_id(dataset_id)
        return f"mask_{archetype}_{lang}"
    low = dataset_id.lower()
    archetype = _mask_archetype_from_dataset_id(dataset_id)
    for lang in ("zh", "pt", "es", "de"):
        if low.endswith(f"_{lang}"):
            return f"mask_{archetype}_{lang}"
    return f"mask_{archetype}_en"


def mask_rollout_stem_from_mask_path(mask_path: Path) -> str:
    """Resolve ``mask_<archetype>_<lang>`` when generating from ``--mask-data`` without ``--dataset-id``."""
    p = Path(mask_path).resolve()
    for d in DATASET_REGISTRY:
        if (d.family or "").strip().lower() != "mask":
            continue
        try:
            if d.resolved_path().resolve() == p:
                lang = (d.language or "en").strip().lower()
                archetype = _mask_archetype_from_dataset_id(d.id)
                return f"mask_{archetype}_{lang}"
        except OSError:
            continue
    stem = Path(mask_path).stem.lower()
    for lang in ("zh", "pt", "es", "de"):
        if stem.endswith(f"_{lang}"):
            archetype = stem[: -(len(lang) + 1)] or "known_facts"
            return f"mask_{archetype}_{lang}"
    archetype = stem or "known_facts"
    return f"mask_{archetype}_en"


def rollout_stem_for_dataset_id(dataset_id: str) -> str:
    """
    Filename stem for ``rollouts/<model>/<family>/<stem>_<model>.json``.
    """
    info = get_dataset_info(dataset_id)
    fam = (info.family or "").strip().lower() if info else ""
    if fam == "mask" or dataset_id.lower().startswith("mask_"):
        return mask_rollout_stem_from_dataset_id(dataset_id)
    if info is None or fam != "evilmath":
        return dataset_id
    lang = (info.language or "en").strip().lower()
    if dataset_id.startswith("evilmath_train"):
        return evilmath_rollout_stem_from_split_lang("train", lang)
    if dataset_id.startswith("evilmath_test") or dataset_id == "evilmath_test":
        return evilmath_rollout_stem_from_split_lang("test", lang)
    return dataset_id


def get_rollout_filename_stem(dataset_id: Optional[str], dataset_path: Optional[Path] = None) -> str:
    """Returns the rollout category name."""
    if dataset_id:
        info = get_dataset_info(dataset_id)
        if info is not None and info.family:
            fam = info.family.strip().lower()
            if fam in {"mask", "evilmath", "roleplaying", "repe", "control", "wmdp_sandbagging", "deceptionbench"}:
                return "wmdp_sandbagging" if fam == "sandbagging" else fam

        id_lower = dataset_id.lower()
        if "repe" in id_lower:
            return "repe"
        if "roleplaying" in id_lower:
            return "roleplaying"
        if "control" in id_lower:
            return "control"
        if id_lower.startswith("mask_"):
            return dataset_id
        if "mask" in id_lower:
            return "mask"
        if "evilmath" in id_lower:
            return "evilmath"
        if "wmdp" in id_lower:
            return "wmdp_sandbagging"
        if "deceptionbench" in id_lower:
            return "deceptionbench"

    if dataset_path is not None:
        path_str = str(dataset_path).lower()
        if "repe" in path_str:
            return "repe"
        if "roleplaying" in path_str:
            return "roleplaying"
        if "control" in path_str:
            return "control"
        if "mask" in path_str and "raw" in path_str:
            return f"mask_{Path(dataset_path).stem}"
        if "mask" in path_str:
            return "mask"
        if "evilmath" in path_str:
            return "evilmath"
        if "wmdp" in path_str:
            return "wmdp_sandbagging"
        if "deceptionbench" in path_str:
            return "deceptionbench"
        return Path(dataset_path).stem

    return "rollouts"