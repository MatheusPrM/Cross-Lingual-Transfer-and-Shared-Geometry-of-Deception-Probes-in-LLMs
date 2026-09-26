"""
Tokenize dialogues and build detection masks.

Builds token ids, attention mask, and detection_mask for probe scoring.
"""

import re
import warnings
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from project.types import Dialogue, Message

DEFAULT_PADDING: dict[str, dict[str, int]] = {
    "gemma": {"left": 0, "right": 0},
    "gemma4": {"left": 0, "right": 0},
    "mistral": {"left": 0, "right": 0},
    "llama": {"left": 0, "right": 0},
    "llama4": {"left": 0, "right": 0},
    "qwen": {"left": 0, "right": 0},
    "cohere": {"left": 0, "right": 0},
}


@dataclass
class TokenizedDataset:
    """
    Tokenized dialogues and masks.

    - tokens: [n_dialogues, seq_len]
    - attention_mask: [n_dialogues, seq_len]
    - detection_mask: [n_dialogues, seq_len]
    """

    tokens: Tensor
    attention_mask: Tensor
    detection_mask: Tensor
    dialogues: list[Dialogue]
    formatted_dialogues: list[str]
    dataset_id: Optional[str] = None

    @property
    def n_dialogues(self) -> int:
        return self.tokens.shape[0]

    @property
    def seq_len(self) -> int:
        return self.tokens.shape[1]

    @staticmethod
    def _prefix_pattern(tokenizer_type: str) -> re.Pattern[str]:
        """Regex for message boundaries in formatted chat text."""
        if tokenizer_type == "gemma":
            begin_of_text = r"(<bos>|<pad>)*"
            end_of_last = r"(<end_of_turn>\n)?"
            start_of_turn = r"<start_of_turn>(user|model)\n"
            return re.compile(rf"({begin_of_text}{start_of_turn})|({end_of_last}{start_of_turn})|(\n\n)?")

        if tokenizer_type == "gemma4":
            end_of_last = r"(<turn\|>\n?)?"
            start_of_turn = r"<\|turn>(system|user|assistant|model)\n?"
            begin_of_text = r"(<bos>)?"
            return re.compile(rf"({begin_of_text}{start_of_turn})|({end_of_last}{start_of_turn})|(\n\n)?")

        if tokenizer_type == "qwen":
            begin_of_text = r"(<\|begin_of_text\|>)?"
            im_start = r"<\|im_start\|>"
            im_end = r"<\|im_end\|>\n?"
            role = r"(system|user|assistant)"
            return re.compile(rf"({begin_of_text}{im_start}{role}\n)?({im_end}{im_start}{role}\n)?|(\n\n)?")

        if tokenizer_type == "mistral":
            message_end = r"(</s>)?"
            message_start = r"((<pad>)*<s>)?"
            system_block = r"(\[SYSTEM_PROMPT\].*?\[/SYSTEM_PROMPT\])?"
            inst_start = r" \[INST\] "
            inst_end = r" \[/INST\] "
            return re.compile(
                rf"{message_end}{message_start}{system_block}{inst_start}|{inst_end}|(\n\n)|{message_end}",
                re.DOTALL,
            )

        if tokenizer_type == "llama":
            begin_of_text = r"((<\|pad\|>)*(<\|begin_of_text\|>))?"
            header = r"<\|start_header_id\|>(system|user|assistant)<\|end_header_id\|>\n\n"
            end_of_turn = r"<\|eot_id\|>"
            date_info = (
                r"(Cutting Knowledge Date: December 2023\nToday Date: \d\d \w+ 20\d\d\n\n)?"
            )
            return re.compile(
                rf"({begin_of_text}{header}{date_info})?({end_of_turn}{header})?|(\n\n)?"
            )

        if tokenizer_type == "llama4":
            begin_of_text = r"(<\|begin_of_text\|>)?"
            header = r"<\|header_start\|>(system|user|assistant)<\|header_end\|>\n\n"
            end_of_turn = r"<\|eot\|>"
            return re.compile(
                rf"({begin_of_text}{header})?({end_of_turn}{header})?|(\n\n)?"
            )

        if tokenizer_type == "phi":
            begin_of_text = r"(<\|endoftext\|>|<s>)?"
            role_open = r"<\|(system|user|assistant)\|>"
            end_of_turn = r"<\|end\|>"
            return re.compile(
                rf"({begin_of_text}{role_open})?({end_of_turn}\n?{role_open})?|(\n\n)?"
            )

        if tokenizer_type == "cohere":
            begin_of_text = r"(<BOS_TOKEN>)?"
            header = r"<\|START_OF_TURN_TOKEN\|><\|(?:SYSTEM|USER|CHATBOT)_TOKEN\|>"
            end_of_turn = r"<\|END_OF_TURN_TOKEN\|>"
            return re.compile(
                rf"({begin_of_text}{header})?({end_of_turn}{header})?|(\n\n)?"
            )

        raise ValueError(f"Unknown tokenizer type: {tokenizer_type!r}")

    @staticmethod
    def _char_to_token(
        offset_mapping: list[list[tuple[int, int]]],
        batch_idx: int,
        char_idx: int,
    ) -> Optional[int]:
        """Return token index for a character index."""
        if batch_idx >= len(offset_mapping):
            return None

        for token_idx, (start, end) in enumerate(offset_mapping[batch_idx]):
            if start <= char_idx < end:
                return token_idx
            if char_idx < start:
                return token_idx if token_idx > 0 else None

        return len(offset_mapping[batch_idx]) - 1 if offset_mapping[batch_idx] else None

    @staticmethod
    def _get_detection_mask(
        tokens: Tensor,
        offset_mapping: list[list[tuple[int, int]]],
        dialogues: list[Dialogue],
        formatted_dialogues: list[str],
        padding: dict[str, int],
        tokenizer_type: str,
    ) -> Tensor:
        """Build detection_mask from offset_mapping."""
        detection_mask = torch.zeros_like(tokens, dtype=torch.bool)
        pad_left = padding.get("left", 0)
        pad_right = padding.get("right", 0)
        pattern = TokenizedDataset._prefix_pattern(tokenizer_type)

        for batch_idx, dialogue in enumerate(dialogues):
            char_idx = 0
            formatted_text = formatted_dialogues[batch_idx]

            for message in dialogue:
                match = re.match(pattern, formatted_text[char_idx:])
                if match is None:
                    break

                start_char_idx = char_idx + match.end()
                if tokenizer_type == "qwen" and message.role == "assistant":
                    start_char_idx = _skip_qwen_assistant_preamble(
                        formatted_text, start_char_idx, message.content
                    )
                end_char_idx = start_char_idx + len(message.content)

                if end_char_idx > len(formatted_text):
                    break
                if formatted_text[start_char_idx:end_char_idx] != message.content:
                    break

                if message.detect:
                    start_tok = TokenizedDataset._char_to_token(
                        offset_mapping, batch_idx, start_char_idx
                    )
                    end_tok = TokenizedDataset._char_to_token(
                        offset_mapping, batch_idx, end_char_idx - 1
                    )

                    if start_tok is not None and end_tok is not None:
                        lo = max(0, start_tok - pad_left)
                        hi = min(tokens.shape[1], end_tok + 1 + pad_right)
                        detection_mask[batch_idx, lo:hi] = True

                char_idx = end_char_idx

        return detection_mask


def _normalize_dialogue_for_mask(dialogue: Dialogue, family: Optional[str]) -> Dialogue:
    """
    Normalize dialogue like chat templating does, but preserve `detect`.
    """
    if not dialogue:
        return []

    fold_system = family in {"gemma", "mistral"}
    out: list[Message] = []
    idx = 0

    if fold_system and dialogue[0].role == "system":
        out.append(Message(role="user", content=dialogue[0].content + "\n\n", detect=dialogue[0].detect))
        idx = 1

    for m in dialogue[idx:]:
        if out and out[-1].role == m.role:
            prev = out[-1]
            out[-1] = Message(
                role=prev.role,
                content=prev.content + m.content,
                detect=prev.detect or m.detect,
            )
        else:
            out.append(m)

    return out


def _skip_qwen_assistant_preamble(
    formatted_text: str,
    start_char_idx: int,
    message_content: str = "",
) -> int:
    """
    Qwen3/QwQ/Qwen2.5 chat template may inject a <think>...</think> block
    """
    if start_char_idx < 0 or start_char_idx >= len(formatted_text):
        return start_char_idx

    if message_content and formatted_text.startswith(message_content, start_char_idx):
        return start_char_idx

    if message_content:
        found = formatted_text.find(message_content, start_char_idx)
        if found != -1:
            return found

    if not formatted_text.startswith("<think>", start_char_idx):
        return start_char_idx
    if message_content.startswith("<think>"):
        return start_char_idx

    close = formatted_text.find("</think>", start_char_idx)
    if close == -1:
        return start_char_idx
    i = close + len("</think>")
    while i < len(formatted_text) and formatted_text[i] in (" ", "\t", "\n", "\r"):
        i += 1
    return i

def _format_dialogue_for_chat(
    tokenizer: PreTrainedTokenizerBase,
    dialogue: Dialogue,
    model_id: Optional[str] = None,
    for_generation: bool = False,
) -> str:
    """Format dialogue with the chat formatter."""
    from project.chat_format import format_dialogue_for_chat as format_chat

    return format_chat(
        dialogue,
        tokenizer,
        model_id=model_id,
        add_generation_prompt=for_generation,
        for_generation=for_generation,
    )


def tokenize_dialogues(
    dialogues: list[Dialogue],
    tokenizer: PreTrainedTokenizerBase,
    max_length: Optional[int] = None,
    padding_side: str = "left",
    dataset_id: Optional[str] = None,
    model_id: Optional[str] = None,
    padding: Optional[dict[str, dict[str, int]]] = None,
    *,
    detect_only_start_of_turn: bool = False,
    detect_only_last_token: bool = False,
    strict_detection_mask: bool = True,
    exclude_last_n_detect_tokens: int = 0,
) -> TokenizedDataset:
    """
    Tokenize dialogues and build detection_mask.

    Detection modes:
    - default: all tokens in detect=True messages
    - detect_only_start_of_turn: first token only
    - detect_only_last_token: last token only
    """
    from project.chat_format import get_template_family

    if detect_only_start_of_turn and detect_only_last_token:
        raise ValueError(
            "Only one detection policy may be active: choose start_of_turn or last_token."
        )

    tokenizer.padding_side = padding_side  

    family = get_template_family(tokenizer, model_id)
    if family not in ("gemma", "gemma4", "mistral", "llama", "llama4", "qwen", "cohere"):
        family = None

    pad_dict = (padding or DEFAULT_PADDING).copy()
    for key in ("gemma", "gemma4", "mistral", "llama", "llama4", "qwen", "cohere"):
        pad_dict.setdefault(key, {"left": 0, "right": 0})

    formatted_dialogues = [
        _format_dialogue_for_chat(tokenizer, dialogue, model_id=model_id, for_generation=False)
        for dialogue in dialogues
    ]
    mask_dialogues: list[Dialogue] = [_normalize_dialogue_for_mask(d, family) for d in dialogues]

    enc = tokenizer(
        formatted_dialogues,
        return_tensors="pt",
        padding=True,
        truncation=max_length is not None,
        max_length=max_length,
        return_attention_mask=True,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )

    tokens = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    batch_size, seq_len = tokens.shape

    use_baseline_mask = family is not None
    failure_context: dict[str, object] | None = None

    if use_baseline_mask:
        if hasattr(enc, "char_to_token"):
            detection_mask = torch.zeros_like(tokens, dtype=torch.bool)
            pad_left = pad_dict[family].get("left", 0)
            pad_right = pad_dict[family].get("right", 0)
            pattern = TokenizedDataset._prefix_pattern(family)

            for batch_idx, dialogue in enumerate(mask_dialogues):
                char_idx = 0
                formatted_text = formatted_dialogues[batch_idx]

                for msg_idx, message in enumerate(dialogue):
                    match = re.match(pattern, formatted_text[char_idx:])
                    if match is None:
                        failure_context = {
                            "reason": "prefix_pattern_no_match",
                            "family": family,
                            "dialogue_index": batch_idx,
                            "message_index": msg_idx,
                            "content_preview": (
                                message.content[:80] + "…"
                                if len(message.content) > 80
                                else message.content
                            ),
                        }
                        use_baseline_mask = False
                        break

                    start_char_idx = char_idx + match.end()
                    raw_start_char_idx = start_char_idx
                    qwen_think_turn = False
                    if family == "qwen" and message.role == "assistant":
                        start_char_idx = _skip_qwen_assistant_preamble(
                            formatted_text, start_char_idx, message.content
                        )
                        if "</think>" in message.content and not formatted_text.startswith(
                            message.content, start_char_idx
                        ):
                            qwen_think_turn = True
                            start_char_idx = raw_start_char_idx

                    if qwen_think_turn:
                        turn_end = formatted_text.find("<|im_end|>", start_char_idx)
                        end_char_idx = turn_end if turn_end != -1 else len(formatted_text)
                    else:
                        end_char_idx = start_char_idx + len(message.content)

                    if end_char_idx > len(formatted_text):
                        failure_context = {
                            "reason": "end_char_out_of_bounds",
                            "family": family,
                            "dialogue_index": batch_idx,
                            "message_index": msg_idx,
                            "content_preview": (
                                message.content[:80] + "…"
                                if len(message.content) > 80
                                else message.content
                            ),
                        }
                        use_baseline_mask = False
                        break

                    if not qwen_think_turn and formatted_text[start_char_idx:end_char_idx] != message.content:
                        failure_context = {
                            "reason": "message_content_mismatch",
                            "family": family,
                            "dialogue_index": batch_idx,
                            "message_index": msg_idx,
                            "content_preview": (
                                message.content[:80] + "…"
                                if len(message.content) > 80
                                else message.content
                            ),
                        }
                        use_baseline_mask = False
                        break

                    if message.detect:
                        start_tok = enc.char_to_token(batch_idx, start_char_idx)
                        end_tok = enc.char_to_token(batch_idx, end_char_idx - 1)

                        if start_tok is None or end_tok is None:
                            failure_context = {
                                "reason": "char_to_token_failed",
                                "family": family,
                                "dialogue_index": batch_idx,
                                "message_index": msg_idx,
                                "content_preview": (
                                    message.content[:80] + "…"
                                    if len(message.content) > 80
                                    else message.content
                                ),
                            }
                            use_baseline_mask = False
                            break

                        lo = max(0, int(start_tok) - pad_left)
                        hi = min(tokens.shape[1], int(end_tok) + 1 + pad_right)
                        detection_mask[batch_idx, lo:hi] = True

                    char_idx = end_char_idx

                if not use_baseline_mask:
                    break

        else:
            offset_mapping = enc.get("offset_mapping")
            if offset_mapping is not None:
                if isinstance(offset_mapping, torch.Tensor):
                    offset_mapping = [
                        [(int(a), int(b)) for a, b in row]
                        for row in offset_mapping.tolist()
                    ]
                else:
                    offset_mapping = [list(map(tuple, row)) for row in offset_mapping]

                detection_mask = TokenizedDataset._get_detection_mask(
                    tokens,
                    offset_mapping,
                    mask_dialogues,
                    formatted_dialogues,
                    pad_dict[family],
                    family,
                )
            else:
                failure_context = {
                    "reason": "no_char_to_token_or_offset_mapping",
                    "family": family,
                    "dialogue_index": None,
                    "message_index": None,
                    "content_preview": None,
                }
                use_baseline_mask = False

    if not use_baseline_mask:
        if strict_detection_mask:
            ctx = failure_context or {
                "reason": "unknown_detection_mask_failure",
                "family": family,
                "dialogue_index": None,
                "message_index": None,
                "content_preview": None,
            }
            raise ValueError(
                "Failed to construct detection_mask via family-specific span matching. "
                f"context={ctx}. If you want a fallback mask for debugging, set strict_detection_mask=False."
            )

        warnings.warn(
            "Falling back to last non-pad token for detection_mask because span matching failed. "
            f"family={family!r}, dataset_id={dataset_id!r}, context={failure_context}",
            RuntimeWarning,
        )

        detection_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        for batch_idx in range(batch_size):
            non_pad = (attention_mask[batch_idx] == 1).sum().item()
            if non_pad > 0:
                detection_mask[batch_idx, non_pad - 1] = True

    if exclude_last_n_detect_tokens > 0:
        for batch_idx in range(batch_size):
            true_indices = detection_mask[batch_idx].nonzero(as_tuple=False).flatten()
            n_true = true_indices.numel()
            if n_true == 0:
                continue
            n_drop = min(exclude_last_n_detect_tokens, n_true)
            detection_mask[batch_idx, true_indices[n_true - n_drop:]] = False

    if detect_only_start_of_turn or detect_only_last_token:
        reduced_mask = torch.zeros_like(detection_mask)

        for batch_idx in range(batch_size):
            row = detection_mask[batch_idx]
            true_indices = row.nonzero(as_tuple=False).flatten()
            if true_indices.numel() == 0:
                continue

            breaks = (true_indices[1:] != true_indices[:-1] + 1).nonzero(as_tuple=False).flatten()
            starts = torch.cat([true_indices[:1], true_indices[breaks + 1]])
            ends = torch.cat([true_indices[breaks], true_indices[-1:]])

            if detect_only_start_of_turn:
                reduced_mask[batch_idx, starts] = True
            else:
                reduced_mask[batch_idx, ends] = True

        detection_mask = reduced_mask

    return TokenizedDataset(
        tokens=tokens,
        attention_mask=attention_mask,
        detection_mask=detection_mask,
        dialogues=dialogues,
        formatted_dialogues=formatted_dialogues,
        dataset_id=dataset_id,
    )


def get_tokenized_dataset_pair(
    honest_dialogues: list[Dialogue],
    deceptive_dialogues: list[Dialogue],
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: Optional[int] = None,
    padding_side: str = "left",
    model_id: Optional[str] = None,
    padding: Optional[dict[str, dict[str, int]]] = None,
    detect_only_start_of_turn: bool = False,
    detect_only_last_token: bool = False,
    strict_detection_mask: bool = True,
    enforce_equal_detect_lengths: bool = False,
    exclude_last_n_detect_tokens: int = 0,  # keep the complete statement in the detect mask
) -> tuple[TokenizedDataset, TokenizedDataset]:
    """
    Tokenize honest and deceptive dialogues with the same settings.

    If enforce_equal_detect_lengths=True, each pair must have the same number
    of detection tokens.
    """
    honest_tok = tokenize_dialogues(
        honest_dialogues,
        tokenizer,
        max_length=max_length,
        padding_side=padding_side,
        dataset_id=None,
        model_id=model_id,
        padding=padding,
        detect_only_start_of_turn=detect_only_start_of_turn,
        detect_only_last_token=detect_only_last_token,
        strict_detection_mask=strict_detection_mask,
        exclude_last_n_detect_tokens=exclude_last_n_detect_tokens,
    )

    deceptive_tok = tokenize_dialogues(
        deceptive_dialogues,
        tokenizer,
        max_length=max_length,
        padding_side=padding_side,
        dataset_id=None,
        model_id=model_id,
        padding=padding,
        detect_only_start_of_turn=detect_only_start_of_turn,
        detect_only_last_token=detect_only_last_token,
        strict_detection_mask=strict_detection_mask,
        exclude_last_n_detect_tokens=exclude_last_n_detect_tokens,
    )

    if enforce_equal_detect_lengths:
        if honest_tok.n_dialogues != deceptive_tok.n_dialogues:
            raise ValueError(
                "Paired tokenization requires equal number of dialogues: "
                f"honest={honest_tok.n_dialogues}, deceptive={deceptive_tok.n_dialogues}."
            )

        honest_counts = honest_tok.detection_mask.sum(dim=1).tolist()
        deceptive_counts = deceptive_tok.detection_mask.sum(dim=1).tolist()
        mismatches = [
            idx
            for idx, (honest_count, deceptive_count) in enumerate(
                zip(honest_counts, deceptive_counts)
            )
            if int(honest_count) != int(deceptive_count)
        ]

        if mismatches:
            first_bad = mismatches[0]
            raise ValueError(
                "Detection-token length mismatch under enforce_equal_detect_lengths=True. "
                f"First mismatch at index {first_bad}: "
                f"honest={int(honest_counts[first_bad])}, "
                f"deceptive={int(deceptive_counts[first_bad])}."
            )

    return honest_tok, deceptive_tok


def stub_tokenized_for_masked_counts(
    detect_token_count_per_dialogue: list[int],
) -> TokenizedDataset:
    """
    Minimal TokenizedDataset for masked-only activations.
    """
    if not detect_token_count_per_dialogue:
        return TokenizedDataset(
            tokens=torch.zeros(0, 1, dtype=torch.long),
            attention_mask=torch.zeros(0, 1, dtype=torch.long),
            detection_mask=torch.zeros(0, 1, dtype=torch.bool),
            dialogues=[],
            formatted_dialogues=[],
        )

    n_dialogues = len(detect_token_count_per_dialogue)
    max_seq = max(detect_token_count_per_dialogue)

    tokens = torch.zeros(n_dialogues, max_seq, dtype=torch.long)
    attention_mask = torch.zeros(n_dialogues, max_seq, dtype=torch.long)
    detection_mask = torch.zeros(n_dialogues, max_seq, dtype=torch.bool)

    for idx, count in enumerate(detect_token_count_per_dialogue):
        if count > 0:
            attention_mask[idx, :count] = 1
            detection_mask[idx, :count] = True

    return TokenizedDataset(
        tokens=tokens,
        attention_mask=attention_mask,
        detection_mask=detection_mask,
        dialogues=[[] for _ in range(n_dialogues)],
        formatted_dialogues=[""] * n_dialogues,
    )