"""
Probe training helper.
"""

from pathlib import Path
from typing import Optional

from project.activations import Activations, extract_masked_activations
from project.detectors import LogisticRegressionDetector
from project.loaders import load_dialogues_from_path
from project.models import get_model_info
from project.tokenized_data import get_tokenized_dataset_pair, tokenize_dialogues
from project.types import Dialogue, Label


def split_dialogues_by_label(
    dialogues: list[Dialogue],
    labels: list[Label],
) -> tuple[list[Dialogue], list[Dialogue]]:
    """
    Splits mixed training data into honest and deceptive dialogues. Assumes paired data and stable ordering.
    """
    honest: list[Dialogue] = []
    deceptive: list[Dialogue] = []

    for dial, lab in zip(dialogues, labels):
        if lab == Label.HONEST:
            honest.append(dial)
        elif lab == Label.DECEPTIVE:
            deceptive.append(dial)
        else:
            continue

    if len(honest) != len(deceptive):
        raise ValueError(
            "Paired training requires equal numbers of HONEST and DECEPTIVE dialogues. "
            f"Got {len(honest)} honest and {len(deceptive)} deceptive."
        )

    return honest, deceptive


def get_default_layers(model_id: str) -> list[int]:
    """Returns the default layer range."""
    info = get_model_info(model_id)
    if info is None:
        return [0]

    n = info.num_layers
    return list(range(int(0.25 * n), int(0.75 * n)))


def train_probe(
    dialogues: list[Dialogue],
    labels: list[Label],
    model_id: str,
    *,
    layers: Optional[list[int]] = None,
    reg_coeff: float = 10.0,
    scale: bool = True,
    batch_size: int = 8,
    model_cache_dir: Optional[Path] = None,
    train_data_id: Optional[str] = None,
    cv_folds: int = 5,
) -> LogisticRegressionDetector:
    """
    Trains a probe from mixed labelled dialogues.

    Splits the data into honest and deceptive pairs first.
    """
    from project.model_loader import load_model_and_tokenizer

    if layers is None:
        layers = get_default_layers(model_id)

    honest_dial, deceptive_dial = split_dialogues_by_label(dialogues, labels)
    if not honest_dial or not deceptive_dial:
        raise ValueError(
            "Need both honest and deceptive dialogues; "
            f"got {len(honest_dial)} honest and {len(deceptive_dial)} deceptive."
        )

    model, tokenizer = load_model_and_tokenizer(model_id, cache_dir=model_cache_dir)
    if model is None:
        raise RuntimeError("Probe training requires a loaded model.")

    tokenizer.padding_side = "left"  # type: ignore
    if tokenizer.pad_token_id is None and tokenizer.bos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.bos_token_id

    honest_tok = tokenize_dialogues(
        honest_dial, tokenizer, model_id=model_id, exclude_last_n_detect_tokens=0
    )
    deceptive_tok = tokenize_dialogues(
        deceptive_dial, tokenizer, model_id=model_id, exclude_last_n_detect_tokens=0
    )

    honest_acts = extract_masked_activations(model, honest_tok, layers=layers, batch_size=batch_size)
    deceptive_acts = extract_masked_activations(model, deceptive_tok, layers=layers, batch_size=batch_size)

    detector = LogisticRegressionDetector(
        layers=layers,
        reg_coeff=reg_coeff,
        scale=scale,
        train_data=train_data_id,
    )
    detector.fit(honest_acts, deceptive_acts, cv_folds=cv_folds)
    return detector


def train_probe_paired(
    honest_dialogues: list[Dialogue],
    deceptive_dialogues: list[Dialogue],
    model_id: str,
    *,
    layers: Optional[list[int]] = None,
    reg_coeff: float = 10.0,
    scale: bool = True,
    batch_size: int = 8,
    model_cache_dir: Optional[Path] = None,
    enforce_equal_detect_lengths: bool = False,
    train_data_id: Optional[str] = None,
) -> LogisticRegressionDetector:
    """Trains a probe from paired honest and deceptive dialogues."""
    from project.model_loader import load_model_and_tokenizer

    if len(honest_dialogues) != len(deceptive_dialogues):
        raise ValueError(
            "Paired training requires equal numbers of examples: "
            f"honest={len(honest_dialogues)} deceptive={len(deceptive_dialogues)}."
        )

    if layers is None:
        layers = get_default_layers(model_id)

    model, tokenizer = load_model_and_tokenizer(model_id, cache_dir=model_cache_dir)
    if model is None:
        raise RuntimeError("Probe training requires a loaded model.")

    tokenizer.padding_side = "left"  # type: ignore
    if tokenizer.pad_token_id is None and tokenizer.bos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.bos_token_id

    honest_tok, deceptive_tok = get_tokenized_dataset_pair(
        honest_dialogues,
        deceptive_dialogues,
        tokenizer,
        model_id=model_id,
        strict_detection_mask=True,
        enforce_equal_detect_lengths=enforce_equal_detect_lengths,
    )

    honest_acts = Activations.from_model(model, honest_tok, layers=layers, batch_size=batch_size)
    deceptive_acts = Activations.from_model(model, deceptive_tok, layers=layers, batch_size=batch_size)

    detector = LogisticRegressionDetector(
        layers=layers,
        reg_coeff=reg_coeff,
        scale=scale,
        train_data=train_data_id,
    )
    detector.fit(honest_acts, deceptive_acts)
    return detector


def train_probe_from_path(
    train_path: Path,
    model_id: str,
    *,
    layers: Optional[list[int]] = None,
    reg_coeff: float = 10.0,
    scale: bool = True,
    batch_size: int = 8,
    model_cache_dir: Optional[Path] = None,
    train_data_id: Optional[str] = None,
    repe_category: Optional[str] = None,
    cv_folds: int = 5,
) -> LogisticRegressionDetector:
    """Loads training data from a path and trains a probe."""
    if repe_category is None:
        if train_data_id is not None:
            from project.datasets import repe_category_from_train_data
            repe_category = repe_category_from_train_data(train_data_id)
        else:
            # Fallback: ad-hoc training paths without a registered ID get generic.
            repe_category = "generic"

    dialogues, labels = load_dialogues_from_path(train_path, repe_category=repe_category)
    if not dialogues:
        raise ValueError(f"No dialogues loaded from {train_path}")

    if train_data_id is None:
        train_data_id = Path(train_path).stem

    return train_probe(
        dialogues,
        labels,
        model_id,
        layers=layers,
        reg_coeff=reg_coeff,
        scale=scale,
        batch_size=batch_size,
        model_cache_dir=model_cache_dir,
        train_data_id=train_data_id,
        cv_folds=cv_folds,
    )


def train_probe_from_act_caches(
    honest_cache_path: Path,
    deceptive_cache_path: Path,
    *,
    layers: Optional[list[int]] = None,
    reg_coeff: float = 10.0,
    scale: bool = True,
    train_data_id: Optional[str] = None,
) -> LogisticRegressionDetector:
    """Trains from separate honest and deceptive activation caches."""
    from project.activation_cache import load_activation_cache, record_to_activations

    h_rec = load_activation_cache(Path(honest_cache_path))
    d_rec = load_activation_cache(Path(deceptive_cache_path))

    if h_rec.layers != d_rec.layers:
        raise ValueError(
            f"Cache layer mismatch: honest {h_rec.layers} vs deceptive {d_rec.layers}"
        )

    use_layers = list(layers) if layers is not None else list(h_rec.layers)
    if layers is not None and use_layers != h_rec.layers:
        raise ValueError(
            f"Requested layers {use_layers} must match cache layers {h_rec.layers}"
        )

    h_acts = record_to_activations(h_rec)
    d_acts = record_to_activations(d_rec)
    h_acts.labels = None
    d_acts.labels = None

    detector = LogisticRegressionDetector(
        layers=use_layers,
        reg_coeff=reg_coeff,
        scale=scale,
        train_data=train_data_id,
    )
    detector.fit(h_acts, d_acts)
    return detector


def train_probe_from_mixed_act_cache(
    cache_path: Path,
    *,
    layers: Optional[list[int]] = None,
    reg_coeff: float = 10.0,
    scale: bool = True,
    train_data_id: Optional[str] = None,
) -> LogisticRegressionDetector:
    """Trains from one mixed-label activation cache."""
    from project.activation_cache import load_activation_cache, split_record_by_label_for_training

    rec = load_activation_cache(Path(cache_path))
    use_layers = list(layers) if layers is not None else list(rec.layers)

    if layers is not None and use_layers != rec.layers:
        raise ValueError(
            f"Requested layers {use_layers} must match cache layers {rec.layers}"
        )

    h_acts, d_acts = split_record_by_label_for_training(rec)

    detector = LogisticRegressionDetector(
        layers=use_layers,
        reg_coeff=reg_coeff,
        scale=scale,
        train_data=train_data_id,
    )
    detector.fit(h_acts, d_acts)
    return detector