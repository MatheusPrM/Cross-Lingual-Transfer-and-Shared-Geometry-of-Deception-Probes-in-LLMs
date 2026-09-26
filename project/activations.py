"""Activation extraction and masking over tokenized dialogues."""

from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor
from transformers import PreTrainedModel

from project.tokenized_data import TokenizedDataset, stub_tokenized_for_masked_counts
from project.types import Label


@dataclass
class Activations:
    """
    Stores either full activations:
        [n_dialogues, seq_len, n_layers, hidden]

    or masked-only activations:
        [n_detect_tokens, n_layers, hidden]
    """

    acts: Tensor
    layers: List[int]
    tokenized_dataset: TokenizedDataset
    acts_are_masked_only: bool = False
    detect_token_count_per_dialogue: Optional[List[int]] = None
    dialogue_index_per_token: Optional[Tensor] = None
    labels: Optional[List[Label]] = None

    @classmethod
    def from_model(
        cls,
        model: PreTrainedModel,
        tokenized_dataset: TokenizedDataset,
        *,
        layers: List[int],
        batch_size: int = 8,
    ) -> "Activations":
        """Main entry point for full extraction."""
        return extract_activations(
            model,
            tokenized_dataset,
            layers=list(layers),
            batch_size=batch_size,
        )

    def __post_init__(self) -> None:
        if self.acts_are_masked_only:
            if self.detect_token_count_per_dialogue is None:
                raise ValueError("detect_token_count_per_dialogue is required for masked-only activations")

            expected = sum(self.detect_token_count_per_dialogue)
            if self.acts.shape[0] != expected:
                raise ValueError(
                    f"Masked activations have {self.acts.shape[0]} rows, but expected {expected}"
                )

            if self.dialogue_index_per_token is not None:
                if self.dialogue_index_per_token.shape[0] != self.acts.shape[0]:
                    raise ValueError("dialogue_index_per_token length does not match activation rows")

            if self.labels is not None and self.detect_token_count_per_dialogue is not None:
                if len(self.labels) != len(self.detect_token_count_per_dialogue):
                    raise ValueError(
                        f"labels length {len(self.labels)} != dialogues "
                        f"{len(self.detect_token_count_per_dialogue)} for masked activations"
                    )
        elif self.labels is not None:
            if len(self.labels) != self.tokenized_dataset.n_dialogues:
                raise ValueError(
                    f"labels length {len(self.labels)} != n_dialogues {self.tokenized_dataset.n_dialogues}"
                )

    def get_masked_activations(self) -> Tensor:
        """Returns only the activations at detection positions."""
        if self.acts_are_masked_only:
            return self.acts

        assert self.tokenized_dataset.detection_mask is not None
        return self.acts[self.tokenized_dataset.detection_mask]


def _run_model_for_hidden_states(
    model: PreTrainedModel,
    *,
    tokens: Tensor,
    attention_mask: Tensor,
) -> tuple[Tensor, ...]:
    """Runs the model and returns hidden states; uses the inner decoder to skip the unused full-vocab lm_head when its shape is provable."""
    model_device = next(model.parameters()).device
    use_amp = model_device.type == "cuda"
    autocast_ctx = (
        torch.amp.autocast("cuda", enabled=True) if use_amp else nullcontext()
    )
    base = getattr(model, "model", model) 
    n_layers = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    hidden_states = None
    with autocast_ctx:
        if base is not model and n_layers is not None: 
            try:
                out = base(tokens, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
                hs = out.hidden_states
                if hs is not None and len(hs) == n_layers + 1:
                    hidden_states = hs
            except (TypeError, AttributeError):
                hidden_states = None
        if hidden_states is None:  
            out = model(tokens, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
            hidden_states = out.hidden_states

    if hidden_states is None:
        raise RuntimeError("Model did not return hidden states")

    return hidden_states


def _normalize_hidden_state_shape(h: Tensor, expected_seq_len: Optional[int] = None) -> Tensor:
    """
    Makes sure hidden states are [batch, seq, hidden].
    """
    if h.dim() == 3:
        return h

    if h.dim() == 4:
        b, d1, d2, d3 = h.size()

        if d1 == 1:
            return h.squeeze(1)

        if expected_seq_len is None:
            raise ValueError(
                f"Got 4D hidden state {tuple(h.shape)} but expected_seq_len is missing"
            )

        if d2 == expected_seq_len:
            return h.permute(0, 2, 1, 3).contiguous().view(b, d2, d1 * d3)

        if d1 == expected_seq_len:
            return h.contiguous().view(b, d1, d2 * d3)

        raise ValueError(
            f"Could not normalize hidden state {tuple(h.shape)} with expected_seq_len={expected_seq_len}"
        )

    raise ValueError(
        f"Unexpected hidden state rank {h.dim()} for shape {tuple(h.shape)}"
    )


@torch.no_grad()
def extract_activations(
    model: PreTrainedModel,
    tokenized: TokenizedDataset,
    layers: List[int],
    batch_size: int = 8,
) -> Activations:
    """Extracts full activations for all sequence positions."""
    device = next(model.parameters()).device
    n_dialogues = tokenized.n_dialogues
    seq_len = tokenized.seq_len
    layers = list(layers)
    n_layers = len(layers)
    result: Optional[Tensor] = None
    num_batches = (n_dialogues + batch_size - 1) // batch_size

    for batch_index, start in enumerate(range(0, n_dialogues, batch_size)):
        end = min(start + batch_size, n_dialogues)
        print(
            f"  Extracting activations batch {batch_index + 1}/{num_batches} "
            f"({end - start} dialogues)...",
            flush=True,
        )

        batch_tokens = tokenized.tokens[start:end].to(device)
        batch_attn = tokenized.attention_mask[start:end].to(device)

        hidden_states = _run_model_for_hidden_states(
            model,
            tokens=batch_tokens,
            attention_mask=batch_attn,
        )

        layer_tensors = []
        for layer_idx in layers:
            if layer_idx >= len(hidden_states):
                raise IndexError(
                    f"Layer index {layer_idx} out of range for model with {len(hidden_states)} hidden state tensors"
                )

            h = _normalize_hidden_state_shape(
                hidden_states[layer_idx],
                expected_seq_len=int(batch_tokens.shape[1]),
            )

            if h.shape[1] != batch_tokens.shape[1]:
                raise RuntimeError(
                    f"Normalized hidden states have seq_len={h.shape[1]}, "
                    f"but tokens have seq_len={batch_tokens.shape[1]}"
                )

            layer_tensors.append(h.detach().cpu())

        batch_result = torch.stack(layer_tensors, dim=2)

        if result is None:
            result = torch.zeros(
                n_dialogues,
                seq_len,
                n_layers,
                int(batch_result.shape[3]),
                dtype=batch_result.dtype,
                device=batch_result.device,
            )

        result[start:end] = batch_result

    if result is None:
        hidden_size = int(getattr(model.config, "hidden_size", 0) or 0)
        result = torch.zeros(n_dialogues, seq_len, n_layers, hidden_size)

    return Activations(
        acts=result,
        layers=layers,
        tokenized_dataset=tokenized,
        labels=None,
    )


@torch.no_grad()
def extract_masked_activations(
    model: PreTrainedModel,
    tokenized: TokenizedDataset,
    layers: List[int],
    batch_size: int = 8,
) -> Activations:
    """Extracts only activations at detection positions."""
    device = next(model.parameters()).device
    n_dialogues = tokenized.n_dialogues
    layers = list(layers)
    n_layers = len(layers)
    hidden_size = int(getattr(model.config, "hidden_size", 0) or 0)

    detect_mask = tokenized.detection_mask
    if detect_mask is None:
        raise ValueError("tokenized.detection_mask is required for masked-only extraction")

    detect_token_count_per_dialogue: List[int] = detect_mask.sum(dim=1).tolist()
    total_detect = int(sum(detect_token_count_per_dialogue))

    if total_detect == 0:
        stub = stub_tokenized_for_masked_counts(detect_token_count_per_dialogue)
        return Activations(
            acts=torch.zeros(0, n_layers, hidden_size),
            layers=layers,
            tokenized_dataset=stub,
            acts_are_masked_only=True,
            detect_token_count_per_dialogue=detect_token_count_per_dialogue,
            dialogue_index_per_token=torch.zeros(0, dtype=torch.long),
            labels=None,
        )

    stub = stub_tokenized_for_masked_counts(detect_token_count_per_dialogue)
    dialogue_index_per_token = torch.repeat_interleave(
        torch.arange(n_dialogues, dtype=torch.long),
        torch.tensor(detect_token_count_per_dialogue, dtype=torch.long),
    )

    acts_chunks: list[Tensor] = []
    num_batches = (n_dialogues + batch_size - 1) // batch_size

    for batch_index, start in enumerate(range(0, n_dialogues, batch_size)):
        end = min(start + batch_size, n_dialogues)
        print(
            f"  Extracting masked activations batch {batch_index + 1}/{num_batches} "
            f"({end - start} dialogues)...",
            flush=True,
        )

        batch_tokens = tokenized.tokens[start:end].to(device)
        batch_attn = tokenized.attention_mask[start:end].to(device)
        batch_detect = detect_mask[start:end].to(device)

        hidden_states = _run_model_for_hidden_states(
            model,
            tokens=batch_tokens,
            attention_mask=batch_attn,
        )

        layer_masked: list[Tensor] = []
        for layer_idx in layers:
            if layer_idx >= len(hidden_states):
                raise IndexError(
                    f"Layer index {layer_idx} out of range for model with {len(hidden_states)} hidden state tensors"
                )

            h = _normalize_hidden_state_shape(
                hidden_states[layer_idx],
                expected_seq_len=int(batch_tokens.shape[1]),
            )

            if h.shape[1] != batch_tokens.shape[1]:
                raise RuntimeError(
                    f"Normalized hidden states have seq_len={h.shape[1]}, "
                    f"but tokens have seq_len={batch_tokens.shape[1]}"
                )

            h_masked = h[batch_detect]
            layer_masked.append(h_masked.detach().cpu())

        batch_result = torch.stack(layer_masked, dim=1)
        acts_chunks.append(batch_result)

    acts = torch.cat(acts_chunks, dim=0)
    if acts.shape[0] != total_detect:
        raise RuntimeError(
            f"Masked activation row count mismatch: got {acts.shape[0]}, expected {total_detect}"
        )

    return Activations(
        acts=acts,
        layers=layers,
        tokenized_dataset=stub,
        acts_are_masked_only=True,
        detect_token_count_per_dialogue=detect_token_count_per_dialogue,
        dialogue_index_per_token=dialogue_index_per_token,
        labels=None,
    )


def get_activations_at_layer(activations: Activations, layer_index: int) -> Tensor:
    """Returns masked activations for one layer."""
    idx = activations.layers.index(layer_index)
    masked = activations.get_masked_activations()
    return masked[:, idx, :]


@torch.no_grad()
def extract_masked_activations_multi(
    model: PreTrainedModel,
    tokenized: TokenizedDataset,
    *,
    layers: List[int],
    masks: dict[str, Tensor],
    batch_size: int = 8,
) -> dict[str, Activations]:
    """Single-forward-pass extractor that emits one masked-only Activations per named mask."""
    if not masks:
        raise ValueError("masks dict cannot be empty")
    n_dialogues = tokenized.n_dialogues
    seq_len = tokenized.seq_len
    for name, m in masks.items():
        if m.shape != (n_dialogues, seq_len):
            raise ValueError(
                f"mask {name!r} has shape {tuple(m.shape)}, expected {(n_dialogues, seq_len)}"
            )

    device = next(model.parameters()).device
    layers = list(layers)
    n_layers = len(layers)
    hidden_size = int(getattr(model.config, "hidden_size", 0) or 0)

    counts: dict[str, List[int]] = {}
    totals: dict[str, int] = {}
    for name, m in masks.items():
        c = m.sum(dim=1).tolist()
        counts[name] = c
        totals[name] = int(sum(c))

    chunks: dict[str, list[Tensor]] = {name: [] for name in masks}
    num_batches = (n_dialogues + batch_size - 1) // batch_size

    for batch_index, start in enumerate(range(0, n_dialogues, batch_size)):
        end = min(start + batch_size, n_dialogues)
        print(
            f"  Extracting multi-masked activations batch {batch_index + 1}/{num_batches} "
            f"({end - start} dialogues, masks={list(masks.keys())})...",
            flush=True,
        )

        batch_tokens = tokenized.tokens[start:end].to(device)
        batch_attn = tokenized.attention_mask[start:end].to(device)

        hidden_states = _run_model_for_hidden_states(
            model,
            tokens=batch_tokens,
            attention_mask=batch_attn,
        )

        per_layer_h: list[Tensor] = []
        for layer_idx in layers:
            if layer_idx >= len(hidden_states):
                raise IndexError(
                    f"Layer index {layer_idx} out of range for model with {len(hidden_states)} hidden state tensors"
                )
            h = _normalize_hidden_state_shape(
                hidden_states[layer_idx],
                expected_seq_len=int(batch_tokens.shape[1]),
            )
            if h.shape[1] != batch_tokens.shape[1]:
                raise RuntimeError(
                    f"Normalized hidden states have seq_len={h.shape[1]}, "
                    f"but tokens have seq_len={batch_tokens.shape[1]}"
                )
            per_layer_h.append(h)

        for name, m in masks.items():
            batch_m = m[start:end].to(device)
            layer_masked: list[Tensor] = []
            for h in per_layer_h:
                layer_masked.append(h[batch_m].detach().cpu())
            batch_result = torch.stack(layer_masked, dim=1)
            chunks[name].append(batch_result)

    out: dict[str, Activations] = {}
    for name in masks:
        if totals[name] == 0:
            stub = stub_tokenized_for_masked_counts(counts[name])
            out[name] = Activations(
                acts=torch.zeros(0, n_layers, hidden_size),
                layers=layers,
                tokenized_dataset=stub,
                acts_are_masked_only=True,
                detect_token_count_per_dialogue=counts[name],
                dialogue_index_per_token=torch.zeros(0, dtype=torch.long),
                labels=None,
            )
            continue
        acts = torch.cat(chunks[name], dim=0)
        if acts.shape[0] != totals[name]:
            raise RuntimeError(
                f"mask {name!r}: got {acts.shape[0]} rows, expected {totals[name]}"
            )
        stub = stub_tokenized_for_masked_counts(counts[name])
        dialogue_index_per_token = torch.repeat_interleave(
            torch.arange(n_dialogues, dtype=torch.long),
            torch.tensor(counts[name], dtype=torch.long),
        )
        out[name] = Activations(
            acts=acts,
            layers=layers,
            tokenized_dataset=stub,
            acts_are_masked_only=True,
            detect_token_count_per_dialogue=counts[name],
            dialogue_index_per_token=dialogue_index_per_token,
            labels=None,
        )
    return out


def compute_prompt_mask(tokenized: TokenizedDataset) -> Tensor:
    """Prompt-only mask = attended tokens that are NOT in the detection mask."""
    if tokenized.detection_mask is None:
        raise ValueError("tokenized.detection_mask is required to derive a prompt_mask")
    return tokenized.attention_mask.bool() & ~tokenized.detection_mask