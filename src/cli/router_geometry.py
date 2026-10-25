"""Read only native checkpoint router tensors and summarize off-diagonal cosines."""

import argparse
import json

from src.capture.checkpoint_geometry import checkpoint_geometry


class Args:
    checkpoint: str



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', help='Native OLMo-core model_and_optim directory')
    args = parser.parse_args(namespace=Args())
    print(json.dumps(checkpoint_geometry(args.checkpoint), indent=2, allow_nan=False))
