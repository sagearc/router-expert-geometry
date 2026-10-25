"""Plot OLMoE score–gate-response Spearman correlation against layer depth."""

import argparse
from pathlib import Path

from src.plots.olmoe_layers import plot, read_result


class Args:
    checkpoint: list[list[str]]
    out: Path
    show_spread: bool


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', nargs=2, action='append', required=True,
                        metavar=('LABEL', 'JSON'), help='repeat for each checkpoint')
    parser.add_argument('--out', type=Path, required=True, help='new PNG, PDF or SVG path')
    parser.add_argument('--show-spread', action='store_true', help='shade ±1 title SD, not a confidence interval')
    args = parser.parse_args(namespace=Args())
    if args.out.exists() or args.out.suffix.lower() not in ('.png', '.pdf', '.svg'):
        parser.error('choose a new PNG, PDF or SVG output')
    results = [(label, read_result(Path(path))) for label, path in args.checkpoint]
    plot(results, args.out, args.show_spread)
