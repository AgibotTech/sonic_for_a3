"""Command-line interface for scalable Isaac evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .runner import (
    BucketedPreparation,
    aggregate_output,
    apply_cli_overrides,
    context_as_json,
    load_user_config,
    prepare_from_config,
    prepare_run_context,
    run_evaluation,
    validate_output,
)


def _add_config_and_paths(parser: argparse.ArgumentParser, *, checkpoint: bool) -> None:
    parser.add_argument("--config", type=Path, default=None)
    if checkpoint:
        parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--prepared-output", type=Path, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--sorting-mode", choices=("off", "on", "auto"), default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gear_sonic.evaluation",
        description="SONIC A3 scalable Isaac evaluation",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="prepare data and run Isaac evaluation")
    _add_config_and_paths(run_parser, checkpoint=True)
    run_parser.add_argument("--dry-run", action="store_true")

    prepare_parser = subparsers.add_parser("prepare", help="prepare a length-aware dataset")
    _add_config_and_paths(prepare_parser, checkpoint=False)

    aggregate_parser = subparsers.add_parser("aggregate", help="aggregate saved batch results")
    aggregate_parser.add_argument("--output", type=Path, required=True)
    aggregate_parser.add_argument("--allow-partial", action="store_true")

    validate_parser = subparsers.add_parser("validate", help="validate final metrics")
    validate_parser.add_argument("--output", type=Path, required=True)
    return parser


def _resolved_config(args: argparse.Namespace):
    config = load_user_config(args.config)
    return apply_cli_overrides(
        config,
        checkpoint=getattr(args, "checkpoint", None),
        dataset=args.dataset,
        output=args.output,
        prepared_output=args.prepared_output,
        num_envs=args.num_envs,
        sorting_mode=args.sorting_mode,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        config = _resolved_config(args)
        if args.dry_run:
            context = prepare_run_context(config, write_manifest=False)
            print(json.dumps(context_as_json(context), ensure_ascii=False, indent=2))
            return 0
        return run_evaluation(config)
    if args.command == "prepare":
        prepared = prepare_from_config(_resolved_config(args))
        if isinstance(prepared, BucketedPreparation):
            print(
                json.dumps(
                    {
                        "motion_dir": str(prepared.prepared.motion_dir),
                        "manifest": str(prepared.prepared.manifest_path),
                        "motion_count": len(prepared.prepared.manifest.entries),
                        "reused": prepared.prepared.reused,
                        "bucket_plan": str(prepared.plan_path),
                        "buckets": [
                            {
                                "ordinal": view.assignment.ordinal,
                                "name": view.assignment.name,
                                "motion_count": view.assignment.motion_count,
                                "num_envs": view.assignment.num_envs,
                                "motion_dir": str(view.motion_dir),
                                "manifest": str(view.manifest_path),
                                "reused": view.reused,
                            }
                            for view in prepared.views
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        print(
            json.dumps(
                {
                    "motion_dir": str(prepared.motion_dir),
                    "manifest": str(prepared.manifest_path),
                    "motion_count": len(prepared.manifest.entries),
                    "reused": prepared.reused,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "aggregate":
        path = aggregate_output(args.output, allow_partial=args.allow_partial)
        print(path)
        return 0
    if args.command == "validate":
        report = validate_output(args.output)
        print(json.dumps(report.__dict__, ensure_ascii=False, indent=2))
        return 0 if report.valid else 2
    raise AssertionError(f"unhandled command: {args.command}")
