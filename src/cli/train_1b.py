"""Train one of the four paper routers with the OLMo-core trainer."""

import argparse

from src.configs.runs import (
    TrainingArgs,
    add_run_arguments,
    build_config,
    parse_run_args,
    print_or_run,
)
from src.constants import PAPER_MICROBATCH_SEQUENCES, PAPER_ROUTINGS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--routing', choices=PAPER_ROUTINGS, required=True)
    add_run_arguments(parser, PAPER_MICROBATCH_SEQUENCES)
    args = parse_run_args(parser, TrainingArgs())
    print_or_run(build_config(args), args.print_config)
