"""Train either retained local 6.9B recipe, not a balancing-only ablation."""

import argparse

from src.configs.runs import (
    GeometryArgs,
    add_run_arguments,
    build_7b_config,
    parse_run_args,
    print_or_run,
)
from src.constants import BALANCED_ROUTINGS, GEOMETRY_MICROBATCH_SEQUENCES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', choices=BALANCED_ROUTINGS, required=True)
    add_run_arguments(parser, GEOMETRY_MICROBATCH_SEQUENCES)
    args = parse_run_args(parser, GeometryArgs())
    print_or_run(build_7b_config(args), args.print_config)
