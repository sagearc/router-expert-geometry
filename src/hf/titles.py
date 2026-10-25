"""Tokenize the retained 256 public-model titles, with no special tokens."""

import hashlib
import json
from pathlib import Path
from typing import Literal

from transformers import AutoTokenizer

from src.constants import (
    OLMOE_MODEL_ID,
    OLMOE_TOKENIZER_REVISION,
    PUBLIC_MAX_TOKENS,
    PUBLIC_TITLE_COUNT,
    QWEN_MODEL_ID,
    QWEN_REVISION,
)
from src.types import IntArray

TOKENIZERS: dict[str, tuple[str, str]] = {
    'qwen': (QWEN_MODEL_ID, QWEN_REVISION),
    'olmoe': (OLMOE_MODEL_ID, OLMOE_TOKENIZER_REVISION),
}


def load_titles(path: Path) -> list[str]:
    titles = json.loads(path.read_text())
    if not isinstance(titles, list) or len(titles) != PUBLIC_TITLE_COUNT or not all(isinstance(t, str) and t for t in titles):
        raise ValueError('expected exactly 256 nonempty title strings')
    return titles


def tokenize_titles(
    titles: list[str], model: Literal['qwen', 'olmoe'], titles_path: Path,
) -> tuple[IntArray, IntArray, dict[str, object]]:
    """Right-padded token IDs, lengths and provenance metadata for one tokenizer."""
    model_id, revision = TOKENIZERS[model]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    encoded = tokenizer(titles, padding=True, truncation=True, max_length=PUBLIC_MAX_TOKENS,
                        add_special_tokens=False, return_tensors='np')
    metadata: dict[str, object] = {
        'tokenizer_model': model_id, 'revision': revision,
        'max_length': PUBLIC_MAX_TOKENS, 'add_special_tokens': False,
        'titles_sha256': hashlib.sha256(titles_path.read_bytes()).hexdigest(),
    }
    lengths = encoded['attention_mask'].sum(-1)
    if (encoded['input_ids'].ndim != 2 or encoded['input_ids'].shape[0] != PUBLIC_TITLE_COUNT
            or lengths.shape != (PUBLIC_TITLE_COUNT,) or (lengths < 1).any() or (lengths > PUBLIC_MAX_TOKENS).any()):
        raise ValueError('tokenizer must produce 256 nonempty sequences of at most 64 tokens')
    return encoded['input_ids'], lengths, metadata
