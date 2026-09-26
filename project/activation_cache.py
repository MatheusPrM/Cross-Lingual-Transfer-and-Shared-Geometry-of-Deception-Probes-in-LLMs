"""
Save/load activation caches for probe training and evaluation without rerunning the model.

Default format is masked-only: [n_detect_tokens, n_layers, hidden] plus grouping metadata.
Optional full format (--save-full) stores full sequence tensors and masks.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Literal, Optional

import torch
from torch import Tensor

from project.activations import Activations, extract_activations
from project.tokenized_data import TokenizedDataset, stub_tokenized_for_masked_counts, tokenize_dialogues
from project.types import Dialogue, Label


CACHE_VERSION = 1


def default_activation_cache_path(
    *,
    repo_root: Path,
    model_id: str,
    dataset_id: str,
    layers: List[int],
    cache_format: Literal["masked", "full"] = "masked",
) -> Path:
    """Builds the default cache path."""
    layers_str = ",".join(str(int(x)) for x in layers)
    return (
        Path(repo_root)
        / "caches"
        / model_id
        / f"{dataset_id}_{model_id}__layers={layers_str}__fmt={cache_format}.pkl"
    )


def repe_training_cache_path(
    *,
    repo_root: Path,
    model_id: str,
    category: str,
    language: str,
    layers: List[int],
    cache_format: Literal["masked", "full"] = "masked",
) -> Path:
    """Path for a category-aware RepE *training* activation cache."""
    if not category:
        raise ValueError(
            "repe_training_cache_path requires an explicit category. "
            "Cache keys MUST include category — categories share source CSVs but "
            "produce different activations."
        )
    layers_str = ",".join(str(int(x)) for x in layers)
    return (
        Path(repo_root)
        / "caches"
        / model_id
        / "repe_train"
        / category
        / f"{language}__layers={layers_str}__fmt={cache_format}.pkl"
    )


@dataclass
class ActivationCacheRecord:
    """Saved activation cache."""

    cache_version: int
    format: Literal["masked", "full"]
    masked_activations: Tensor
    dialogue_index_per_token: Tensor
    detect_token_count_per_dialogue: List[int]
    layers: List[int]
    label_values: List[str]
    graded_deception: Optional[List[Any]]
    dialogue_ids: List[str]
    model_id: str
    tokenizer_name_or_path: str
    hf_model_revision: Optional[str]
    full_activations: Optional[Tensor] = None
    attention_mask: Optional[Tensor] = None
    detection_mask: Optional[Tensor] = None


def _labels_from_values(values: List[str]) -> List[Label]:
    """Turns saved label strings back into Label values."""
    out: List[Label] = []
    for v in values:
        try:
            out.append(Label(v))
        except ValueError:
            out.append(Label.SKIP)
    return out


def build_masked_metadata(tokenized: TokenizedDataset) -> tuple[Tensor, List[int]]:
    """Builds token-to-dialogue metadata from the detection mask."""
    mask = tokenized.detection_mask
    if mask is None:
        raise ValueError("tokenized.detection_mask is required")

    counts = [int(c) for c in mask.sum(dim=1).tolist()]
    dialogue_index_per_token = torch.repeat_interleave(
        torch.arange(mask.shape[0], dtype=torch.long),
        torch.tensor(counts, dtype=torch.long),
    )
    return dialogue_index_per_token, counts


def build_cache_record(
    activations: Activations,
    *,
    labels: List[Label],
    graded_deception: Optional[List[Any]],
    dialogue_ids: List[str],
    model_id: str,
    tokenizer_name_or_path: str,
    hf_model_revision: Optional[str],
    save_full: bool,
) -> ActivationCacheRecord:
    """Builds a cache record from activations."""
    masked = activations.get_masked_activations().cpu()
    dial_idx, counts = build_masked_metadata(activations.tokenized_dataset)

    if dial_idx.numel() != masked.shape[0]:
        raise RuntimeError("Detection mask does not match masked activations")

    label_values = [lab.value for lab in labels]

    full_acts = None
    attn = None
    det_m = None
    if save_full:
        full_acts = activations.acts.cpu()
        attn = activations.tokenized_dataset.attention_mask.cpu()
        det_m = activations.tokenized_dataset.detection_mask.cpu()

    return ActivationCacheRecord(
        cache_version=CACHE_VERSION,
        format="full" if save_full else "masked",
        masked_activations=masked,
        dialogue_index_per_token=dial_idx.cpu(),
        detect_token_count_per_dialogue=counts,
        layers=list(activations.layers),
        label_values=label_values,
        graded_deception=graded_deception,
        dialogue_ids=list(dialogue_ids),
        model_id=model_id,
        tokenizer_name_or_path=tokenizer_name_or_path,
        hf_model_revision=hf_model_revision,
        full_activations=full_acts,
        attention_mask=attn,
        detection_mask=det_m,
    )


def truncate_cache_record(
    record: ActivationCacheRecord,
    max_dialogues: int,
) -> ActivationCacheRecord:
    """Keeps only the first max_dialogues dialogues."""
    if max_dialogues <= 0 or len(record.detect_token_count_per_dialogue) <= max_dialogues:
        return record

    n_keep = max_dialogues
    row_sel = record.dialogue_index_per_token < n_keep

    new_masked = record.masked_activations[row_sel]
    new_dial_idx = record.dialogue_index_per_token[row_sel]
    new_counts = list(record.detect_token_count_per_dialogue[:n_keep])
    new_labels = list(record.label_values[:n_keep])
    new_ids = list(record.dialogue_ids[:n_keep])
    new_graded = (
        list(record.graded_deception[:n_keep]) if record.graded_deception is not None else None
    )

    full_a = record.full_activations[:n_keep].clone() if record.full_activations is not None else None
    attn = record.attention_mask[:n_keep].clone() if record.attention_mask is not None else None
    det_m = record.detection_mask[:n_keep].clone() if record.detection_mask is not None else None

    return ActivationCacheRecord(
        cache_version=record.cache_version,
        format=record.format,
        masked_activations=new_masked,
        dialogue_index_per_token=new_dial_idx.clone(),
        detect_token_count_per_dialogue=new_counts,
        layers=list(record.layers),
        label_values=new_labels,
        graded_deception=new_graded,
        dialogue_ids=new_ids,
        model_id=record.model_id,
        tokenizer_name_or_path=record.tokenizer_name_or_path,
        hf_model_revision=record.hf_model_revision,
        full_activations=full_a,
        attention_mask=attn,
        detection_mask=det_m,
    )


def save_activation_cache(path: Path, record: ActivationCacheRecord) -> None:
    """Saves a cache record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(record, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_activation_cache(path: Path) -> ActivationCacheRecord:
    """Loads a cache record."""
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, ActivationCacheRecord):
        return obj

    raise TypeError(f"Expected ActivationCacheRecord in {path}, got {type(obj)}")


def record_to_activations(
    record: ActivationCacheRecord,
    *,
    prefer_full: bool = False,
) -> Activations:
    """Rebuilds Activations from a cache record."""
    if prefer_full and record.full_activations is not None and record.attention_mask is not None:
        if record.detection_mask is None:
            raise ValueError("Full cache is missing detection_mask")

        n = record.attention_mask.shape[0]
        tok = torch.zeros_like(record.attention_mask, dtype=torch.long)
        stub_dialogues: List[Dialogue] = [[] for _ in range(n)]

        tokenized = TokenizedDataset(
            tokens=tok,
            attention_mask=record.attention_mask.long(),
            detection_mask=record.detection_mask.bool(),
            dialogues=stub_dialogues,
            formatted_dialogues=[""] * n,
        )

        labels = _labels_from_values(record.label_values)
        return Activations(
            acts=record.full_activations,
            layers=list(record.layers),
            tokenized_dataset=tokenized,
            acts_are_masked_only=False,
            detect_token_count_per_dialogue=None,
            dialogue_index_per_token=None,
            labels=labels,
        )

    stub = stub_tokenized_for_masked_counts(record.detect_token_count_per_dialogue)
    labels = _labels_from_values(record.label_values)

    return Activations(
        acts=record.masked_activations,
        layers=list(record.layers),
        tokenized_dataset=stub,
        acts_are_masked_only=True,
        detect_token_count_per_dialogue=list(record.detect_token_count_per_dialogue),
        dialogue_index_per_token=record.dialogue_index_per_token.clone(),
        labels=labels,
    )


def subset_activations_by_dialogue_indices(
    record: ActivationCacheRecord,
    dialogue_indices: List[int],
) -> Activations:
    """Keeps only selected dialogues from a cache."""
    if not dialogue_indices:
        raise ValueError("dialogue_indices must be non-empty")

    keep = torch.tensor(dialogue_indices, dtype=torch.long)
    row_sel = torch.isin(record.dialogue_index_per_token.cpu(), keep)

    sub_masked = record.masked_activations[row_sel]
    sub_dial = record.dialogue_index_per_token[row_sel]

    old_to_new = {old: k for k, old in enumerate(dialogue_indices)}
    new_idx = torch.tensor([old_to_new[int(x)] for x in sub_dial.tolist()], dtype=torch.long)
    counts = [int((new_idx == k).sum().item()) for k in range(len(dialogue_indices))]
    stub = stub_tokenized_for_masked_counts(counts)

    return Activations(
        acts=sub_masked,
        layers=list(record.layers),
        tokenized_dataset=stub,
        acts_are_masked_only=True,
        detect_token_count_per_dialogue=counts,
        dialogue_index_per_token=new_idx,
        labels=None,
    )


def split_record_by_label_for_training(record: ActivationCacheRecord) -> tuple[Activations, Activations]:
    """Splits a mixed cache into honest and deceptive activations."""
    labels = _labels_from_values(record.label_values)
    honest_idx = [i for i, lab in enumerate(labels) if lab == Label.HONEST]
    dec_idx = [i for i, lab in enumerate(labels) if lab == Label.DECEPTIVE]

    if not honest_idx or not dec_idx:
        raise ValueError(
            f"Need both HONEST and DECEPTIVE dialogues in cache; got {len(honest_idx)} honest, {len(dec_idx)} deceptive."
        )

    return (
        subset_activations_by_dialogue_indices(record, honest_idx),
        subset_activations_by_dialogue_indices(record, dec_idx),
    )


def extract_and_cache_rollouts(
    dialogues: List[Dialogue],
    labels: List[Label],
    graded_deception: Optional[List[Any]],
    dialogue_ids: List[str],
    model,
    tokenizer,
    model_id: str,
    *,
    layers: List[int],
    batch_size: int,
    save_full: bool,
    hf_model_revision: Optional[str] = None,
) -> ActivationCacheRecord:
    """Tokenizes dialogues, extracts activations, and builds a cache."""
    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None and tokenizer.bos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.bos_token_id

    tokenized = tokenize_dialogues(dialogues, tokenizer, model_id=model_id)
    acts = extract_activations(model, tokenized, layers=layers, batch_size=batch_size)
    tok_name = getattr(tokenizer, "name_or_path", None) or str(model_id)

    return build_cache_record(
        acts,
        labels=labels,
        graded_deception=graded_deception,
        dialogue_ids=dialogue_ids,
        model_id=model_id,
        tokenizer_name_or_path=str(tok_name),
        hf_model_revision=hf_model_revision,
        save_full=save_full,
    )