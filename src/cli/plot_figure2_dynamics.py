"""Plot original 1B score–response dynamics and final-checkpoint controls."""

import argparse
from pathlib import Path

from src.plots.figure2_dynamics import plot, read_result


class Args:
    result: Path
    out: Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(namespace=Args())
    if args.out.exists() or args.out.suffix.lower() not in ('.png', '.pdf', '.svg'):
        parser.error('choose a new PNG, PDF or SVG output')
    result, values = read_result(args.result)
    plot(result, values, args.out)
