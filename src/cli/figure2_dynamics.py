"""Original 1B Figure 2 dynamics and controls on six retained top-10 captures."""

import argparse
import json
from pathlib import Path

from src.analysis.figure2 import capture_paths, dynamics


class Args:
    captures: Path
    out: Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(namespace=Args())
    if args.out.exists():
        parser.error('output already exists')
    try:
        paths = capture_paths(args.captures)
    except ValueError as error:
        parser.error(str(error))
    result = dynamics(paths)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
