"""Fixed evaluation inputs: shard packing, domain batching and retained row selection.

NumPy only. The capture scripts read these inputs and the reducers pair them
through the hashes the scripts record.
"""

from collections.abc import Iterator
from itertools import pairwise
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from src.artifacts import RetainedInputs
from src.constants import (
    CAUSAL_CALIBRATION_SEQUENCES,
    CAUSAL_SEQUENCE_LENGTH,
    CAUSAL_SOURCE_SEQUENCES,
    PAPER_SEQUENCE_LENGTH,
    PAPER_VOCAB_SIZE,
    PUBLIC_TITLE_COUNT,
    SPECIALIST_CODE_DOMAIN,
    SPECIALIST_SEQUENCE_COUNT,
)
from src.types import IntArray


def load_token_shard(path: Path) -> IntArray:
    """Memory-map a v3-small validation shard; the historical files may be raw uint16."""
    with path.open('rb') as stream:
        is_npy = stream.read(6) == b'\x93NUMPY'
    if not is_npy and path.stat().st_size % 2:
        raise ValueError(f'invalid uint16 token file: {path}')
    tokens = np.load(path, mmap_mode='r', allow_pickle=False) if is_npy else np.memmap(path, dtype='<u2', mode='r')
    if not np.issubdtype(tokens.dtype, np.integer):
        raise ValueError(f'{path}: expected integer tokens')
    return cast(IntArray, tokens)


def pack_shard(path: Path) -> tuple[NDArray[np.uint16], int]:
    """Complete 2048-token sequences from one shard, dropping only its incomplete tail."""
    tokens = load_token_shard(path)
    if tokens.ndim != 1:
        raise ValueError(f'{path}: expected one-dimensional integer tokens')
    if (tokens < 0).any() or (tokens >= PAPER_VOCAB_SIZE).any():
        raise ValueError(f'{path}: token outside paper vocabulary')
    n = len(tokens) // PAPER_SEQUENCE_LENGTH
    if not n:
        raise ValueError(f'{path}: no complete sequence')
    packed = np.asarray(tokens[:n * PAPER_SEQUENCE_LENGTH], dtype=np.uint16).reshape(n, PAPER_SEQUENCE_LENGTH)
    return packed, len(tokens) % PAPER_SEQUENCE_LENGTH


def causal_rows(path: Path) -> IntArray:
    """The retained causal source: twenty 513-token rows, the first four for calibration.

    The retained source flattened each shard and split it into 513-token rows.
    """
    tokens = load_token_shard(path)
    if tokens.size < CAUSAL_SOURCE_SEQUENCES * CAUSAL_SEQUENCE_LENGTH:
        raise ValueError('requires at least twenty 513-token integer sequences per domain')
    rows = np.asarray(tokens).reshape(-1)[:CAUSAL_SOURCE_SEQUENCES * CAUSAL_SEQUENCE_LENGTH]
    return cast(IntArray, rows.reshape(CAUSAL_SOURCE_SEQUENCES, CAUSAL_SEQUENCE_LENGTH).copy())


def evaluation_sequences(path: Path) -> IntArray:
    """The rows after the calibration rows. All-controls does not use matching."""
    return causal_rows(path)[CAUSAL_CALIBRATION_SEQUENCES:]


def code_sequences(path: Path) -> IntArray:
    """The first 240 code-domain rows of a packed error-sequence NPZ, in order."""
    with np.load(path, allow_pickle=False) as data:
        domain = data['domain_names'].tolist().index(SPECIALIST_CODE_DOMAIN)
        sequences = data['sequences'][data['domain_idx'] == domain][:SPECIALIST_SEQUENCE_COUNT]
    if sequences.shape != (SPECIALIST_SEQUENCE_COUNT, PAPER_SEQUENCE_LENGTH) or not np.issubdtype(sequences.dtype, np.integer):
        raise ValueError('requires 240 full 2048-token code-domain sequences')
    return cast(IntArray, sequences)


def domain_batches(domains: IntArray, batch_size: int) -> Iterator['slice[int, int, None]']:
    """Keep historical partial batches at contiguous domain boundaries."""
    boundaries = np.r_[0, np.flatnonzero(domains[1:] != domains[:-1]) + 1, len(domains)]
    for start, end in pairwise(boundaries):
        for offset in range(start, end, batch_size):
            yield slice(offset, min(offset + batch_size, end))


def final_token_rows(data: RetainedInputs) -> IntArray:
    """Select the retained final token per title, preserving title order."""
    count = len(data['router_inputs'])
    if 'sample_title_index' not in data and 'sample_token_index' not in data:
        if count != PUBLIC_TITLE_COUNT:
            raise ValueError('legacy capture must contain 256 last-token rows')
        return np.arange(count)
    if 'sample_title_index' not in data or 'sample_token_index' not in data:
        raise ValueError('all-token capture requires both title and token indices')
    titles, tokens = data['sample_title_index'], data['sample_token_index']
    if (titles.shape != (count,) or tokens.shape != (count,)
            or not np.issubdtype(titles.dtype, np.integer)
            or not np.issubdtype(tokens.dtype, np.integer) or (tokens < 0).any()
            or not np.array_equal(np.unique(titles), np.arange(PUBLIC_TITLE_COUNT))):
        raise ValueError('invalid token indices or incomplete 256-title capture')
    rows = []
    for title in range(PUBLIC_TITLE_COUNT):
        candidates = np.flatnonzero(titles == title)
        rows.append(candidates[np.argmax(tokens[candidates])])
    return np.asarray(rows)
