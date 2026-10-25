"""Rank router-axis removal against 32 weight-derived controls at all 16 layers.

Uses the upstream OLMo-core factory without overriding normalization.
"""

import argparse
from pathlib import Path

from src.configs.public_olmoe import OLMOE_CHECKPOINT_REVISIONS
from src.hf.historical_causal import run


class Args:
    checkpoint_step: int
    eval_root: Path
    out: Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-step', type=int, choices=OLMOE_CHECKPOINT_REVISIONS, required=True,
                        help='retained experiment checkpoint; its commit is fixed in configuration')
    parser.add_argument('--eval-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(namespace=Args())
    run(OLMOE_CHECKPOINT_REVISIONS[args.checkpoint_step], args.eval_root, args.out)
