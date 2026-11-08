"""Single entrypoint for trajectory construction and UMM execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.umm_cli import (
    add_backend_arguments,
    add_image_arguments,
)

EXECUTIONS = ("online", "cached")
PROTOCOLS = ("uniweaver", "minimal")
DATASETS = ("single", "bench")
PLANNERS = ("api", "bagel")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run every active trajectory and UMM path through one entrypoint."
    )
    parser.add_argument("--trajectory-only", action="store_true")
    parser.add_argument("--execution", choices=EXECUTIONS)
    parser.add_argument("--protocol", choices=PROTOCOLS, default="uniweaver")
    parser.add_argument("--dataset", choices=DATASETS, default="bench")
    parser.add_argument("--planner", choices=PLANNERS, default="bagel")
    add_backend_arguments(parser, required=False, model_required=False)
    parser.set_defaults(backend="bagel")
    parser.add_argument(
        "--allow-unreported-backbone",
        action="store_true",
        help=(
            "run the UniWeaver protocol on a backbone outside bagel, omnigen2, "
            "and tokenflow, which are the only ones reported"
        ),
    )

    parser.add_argument("--task-key")
    parser.add_argument("--benchmark-file", type=Path)
    parser.add_argument("--trajectory")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--parallel", type=int, default=1)

    parser.add_argument("--api", default="deepseek")
    parser.add_argument("--key", dest="api_key_override")
    parser.add_argument("--base-url", dest="api_base_url")
    parser.add_argument("--model")
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--review-retry",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "bounded UniWeaver Step review/regeneration; enabled by default for "
            "every image-executing UniWeaver run and unavailable for Minimal; "
            "use --no-review-retry to disable"
        ),
    )

    add_image_arguments(parser, include_text=True, image_shape=(1024, 1024))
    return parser


def _normalise_args(args):
    args.execution = args.execution or "online"
    args.base_only = bool(args.base_only or not args.sft_checkpoint)
    args.task_key_explicit = args.task_key is not None
    if args.review_retry is None:
        args.review_retry = bool(
            args.protocol == "uniweaver" and not args.trajectory_only
        )
    if args.dataset == "bench" and args.benchmark_file is None:
        from core.tasks import DEFAULT_BENCHMARK_FILE

        args.benchmark_file = DEFAULT_BENCHMARK_FILE
    return args


def _validate_trajectory(args, parser):
    if args.execution != "online":
        parser.error("trajectory-only requires --execution online")
    if args.protocol != "uniweaver":
        parser.error("trajectory-only requires --protocol uniweaver")
    if not args.task_key:
        parser.error("trajectory-only requires --task-key")
    if args.benchmark_file is None or not args.benchmark_file.is_file():
        parser.error("trajectory-only requires an existing --benchmark-file")
    if args.planner == "api" and args.backend != "bagel":
        parser.error("trajectory API scope does not use an image backend")


def _validate_cached(args, parser):
    if not args.trajectory:
        parser.error("cached execution requires --trajectory")
    if not args.output_dir:
        parser.error("cached execution requires --output-dir")
    if args.protocol != "uniweaver":
        parser.error("cached execution requires --protocol uniweaver")
    if args.execution != "cached":
        parser.error("cached execution requires --execution cached")
    if args.dataset != "single":
        parser.error("cached execution uses one existing trajectory at a time")


def _validate_online(args, parser):
    if args.execution != "online":
        parser.error("online execution requires --execution online")
    if args.planner != "bagel":
        parser.error(
            "full execution uses the selected UMM backend as Planner; "
            "use --trajectory-only for API Planner"
        )
    if not args.output_dir:
        parser.error("full execution requires --output-dir")
    if args.dataset == "single" and not args.task_key:
        parser.error("full single-task scope requires --task-key")


def _validate(args, parser):
    if args.protocol == "uniweaver" and not args.trajectory_only:
        from baselines.umm_backend_factory import backend_spec

        if (
            not backend_spec(args.backend).uniweaver_reported
            and not args.allow_unreported_backbone
        ):
            parser.error(
                f"{args.backend} is not one of the backbones reported with "
                "UniWeaver (bagel, omnigen2, tokenflow); add "
                "--allow-unreported-backbone to run it anyway"
            )
    if (
        args.dataset == "bench"
        and getattr(args, "task_key_explicit", False)
        and not args.trajectory_only
    ):
        parser.error(
            "--task-key selects a single task; add --dataset single for a "
            "one-task run, or drop --task-key for the 120-task suite"
        )
    if args.review_retry and args.protocol != "uniweaver":
        parser.error("Reviewer/retry requires --protocol uniweaver")
    if args.review_retry and args.trajectory_only:
        parser.error("Reviewer/retry requires image execution")
    if args.trajectory_only:
        _validate_trajectory(args, parser)
    elif args.execution == "cached":
        _validate_cached(args, parser)
    else:
        _validate_online(args, parser)


def _run_trajectory(args, parser):
    if args.planner == "api":
        from sft_prepare_data import construct_trajectory_api as runner
    else:
        from sft_prepare_data import construct_trajectory_bagel as runner
    return runner.run(args, parser=parser)


def _load_trajectory(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _umm_provenance(args, backend):
    from baselines.umm_benchmark_runner import build_benchmark_provenance

    return build_benchmark_provenance(
        backend,
        args,
        schema_version=None,
        mode="uniweaver",
        seed_policy=(
            "image_seed_for(base_seed, kind, identity); "
            "kinds are condition and step"
        ),
        extra={
            "condition_policy": (
                "retain at most the first four usable Conditions in declared order; "
                "remove unavailable references and record projection plus backend consumption"
            ),
        },
    )


def _run_cached_umm(args):
    from baselines.umm_cli import (
        require_backend_repo,
        require_positive_runtime_timeouts,
    )
    from baselines.umm_runtime_backend import load_supervised_backend
    from baselines.umm_uniweaver_research import run_uniweaver_images

    require_backend_repo(args)
    require_positive_runtime_timeouts(args)
    backend = load_supervised_backend(args)
    trajectory = _load_trajectory(args.trajectory)
    output = Path(args.output_dir)
    if args.resume:
        if not output.is_dir():
            raise SystemExit(f"resume output directory not found: {output}")
    else:
        output.mkdir(parents=True, exist_ok=False)
    provenance = _umm_provenance(args, backend)
    return run_uniweaver_images(
        backend,
        trajectory,
        output,
        provenance,
        condition_shape=(512, 512),
        step_shape=(args.image_height, args.image_width),
        image_steps=args.image_steps,
        resume=args.resume,
        review_retry=args.review_retry,
    )


def _run_online(args):
    if args.dataset == "bench":
        from baselines import umm_runner

        args.run_dir = Path(args.output_dir)
        return umm_runner.run(args)
    if args.protocol == "minimal":
        from baselines import umm_minimal_runner

        return umm_minimal_runner.run(args)
    from baselines import umm_uniweaver_research

    return umm_uniweaver_research.run(args)


def main(argv=None) -> int:
    parser = _parser()
    args = _normalise_args(parser.parse_args(argv))
    _validate(args, parser)
    if args.trajectory_only:
        _run_trajectory(args, parser)
        return 0
    if args.execution == "cached":
        _run_cached_umm(args)
        return 0
    result = _run_online(args)
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
