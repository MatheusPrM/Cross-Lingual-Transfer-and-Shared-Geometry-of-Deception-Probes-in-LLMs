"""Build Dialogue objects from message lists (DeceptionBench, single-turn user only); length-based detect mask."""

from __future__ import annotations

from typing import TYPE_CHECKING

from project.types import Dialogue, Message

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
    from project.tokenized_data import TokenizedDataset


def messages_to_dialogue(messages: list[dict[str, str]]) -> Dialogue:
    d: Dialogue = []
    for m in messages:
        d.append(Message(role=str(m["role"]), content=str(m.get("content", "")), detect=False))
    d.append(Message(role="assistant", content="", detect=True))
    return d


def fill_assistant_response(d: Dialogue, text: str) -> Dialogue:
    out = list(d[:-1]) if d and d[-1].role == "assistant" else list(d)
    out.append(Message(role="assistant", content=text, detect=True))
    return out


def tokenize_dialogues_length_masked(
    dialogues: list[Dialogue],
    tokenizer: "PreTrainedTokenizerBase",
    *,
    model_id: str | None = None,
    enable_thinking: bool | None = False,
) -> "TokenizedDataset":
    """
    Tokenize DB dialogues; detection_mask = tokens past the prompt-only render length (robust to chat-template normalization).
    """
    import torch
    from project.chat_format import format_dialogue_for_chat
    from project.tokenized_data import TokenizedDataset

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.bos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.bos_token_id

    full_texts: list[str] = []
    prompt_token_counts: list[int] = []
    for d in dialogues:
        prompt_only = [Message(role=m.role, content=m.content, detect=False) for m in d if m.role != "assistant"]
        prompt_text = format_dialogue_for_chat(
            prompt_only, tokenizer, model_id=model_id,
            add_generation_prompt=True, for_generation=False,
            enable_thinking=enable_thinking,
        )
        full_text = format_dialogue_for_chat(
            d, tokenizer, model_id=model_id,
            add_generation_prompt=False, for_generation=False,
            enable_thinking=enable_thinking,
        )
        full_texts.append(full_text)
        prompt_token_counts.append(len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"]))

    enc = tokenizer(
        full_texts, return_tensors="pt", padding=True,
        return_attention_mask=True, add_special_tokens=False,
    )
    tokens = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    n_batch, seq_len = tokens.shape

    detection_mask = torch.zeros_like(tokens, dtype=torch.bool)
    for i in range(n_batch):
        n_real = int(attention_mask[i].sum())
        n_prompt = prompt_token_counts[i]
        n_response = max(0, n_real - n_prompt)
        if n_response > 0:
            start_pad = seq_len - n_real
            detection_mask[i, start_pad + n_prompt : start_pad + n_real] = True

    return TokenizedDataset(
        tokens=tokens,
        attention_mask=attention_mask,
        detection_mask=detection_mask,
        dialogues=dialogues,
        formatted_dialogues=full_texts,
    )


def tokenize_prompt_only_length_masked(
    dialogues: "list[Dialogue]",
    tokenizer: "PreTrainedTokenizerBase",
    *,
    model_id: str | None = None,
    enable_thinking: bool | None = False,
) -> "TokenizedDataset":
    """PROMPT-side counterpart of tokenize_dialogues_length_masked.
    """
    import torch
    from project.chat_format import format_dialogue_for_chat
    from project.tokenized_data import TokenizedDataset

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.bos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.bos_token_id

    prompt_texts: list[str] = []
    for d in dialogues:
        prompt_only = [Message(role=m.role, content=m.content, detect=False)
                       for m in d if m.role != "assistant"]
        prompt_texts.append(format_dialogue_for_chat(
            prompt_only, tokenizer, model_id=model_id,
            add_generation_prompt=True, for_generation=False,
            enable_thinking=enable_thinking,
        ))

    enc = tokenizer(
        prompt_texts, return_tensors="pt", padding=True,
        return_attention_mask=True, add_special_tokens=False,
    )
    tokens = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    n_batch, seq_len = tokens.shape

    detection_mask = torch.zeros_like(tokens, dtype=torch.bool)
    for i in range(n_batch):
        n_real = int(attention_mask[i].sum())
        if n_real > 0:
            start_pad = seq_len - n_real
            detection_mask[i, start_pad:] = True

    return TokenizedDataset(
        tokens=tokens,
        attention_mask=attention_mask,
        detection_mask=detection_mask,
        dialogues=dialogues,
        formatted_dialogues=prompt_texts,
    )
