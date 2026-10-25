"""Tokenize the retained 256 public-model titles, with no special tokens."""

import argparse
import json
from pathlib import Path
from typing import Literal

import numpy as np

from src.constants import PUBLIC_TITLES_PATH
from src.hf.titles import load_titles, tokenize_titles


class Args:
    model: Literal['qwen', 'olmoe']
    out: Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['qwen', 'olmoe'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(namespace=Args())
    if args.out.exists():
        parser.error('output already exists')
    input_ids, lengths, metadata = tokenize_titles(load_titles(PUBLIC_TITLES_PATH), args.model, PUBLIC_TITLES_PATH)
    with args.out.open('xb') as stream:
        np.savez_compressed(stream, input_ids=input_ids, lengths=lengths, metadata_json=json.dumps(metadata))
    print(f'{len(input_ids)} titles, {lengths.sum()} non-padding tokens')
