"""Public OLMoE router-axis removal versus all 32 orthogonal controls.

Checkpoint settings through Transformers, without conversion or architecture
overrides. Four fixed interventions at all 16 layers, FP32, 512 predictions,
16 sequences/domain after skipping the four original calibration rows.
"""

import argparse
import json
from pathlib import Path

import torch

from src.configs.public_olmoe import OLMOE_CHECKPOINT_REVISIONS
from src.hf.causal import run


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
    if not torch.cuda.is_available() or args.out.exists():
        parser.error('requires CUDA and a new output file')
    result = run(OLMOE_CHECKPOINT_REVISIONS[args.checkpoint_step], args.eval_root)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
