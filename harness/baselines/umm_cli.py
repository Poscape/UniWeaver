"""Shared argument construction and boundary validation for UMM CLIs."""

from __future__ import annotations

import argparse

from baselines.umm_backend_factory import SUPPORTED_BACKENDS
from baselines.umm_framework import (
    DEFAULT_IMAGE_SHAPE,
    DEFAULT_MAX_STEPS,
    DEFAULT_TEXT_MAX_TOKENS,
    official_repository_identity,
)
from baselines.umm_runtime_backend import (
    DEFAULT_BACKEND_STARTUP_TIMEOUT_SECONDS,
    DEFAULT_MODEL_CALL_TIMEOUT_SECONDS,
)
from baselines.umm_runtime_limits import (
    DEFAULT_MAX_CHECKS_PER_STEP,
    DEFAULT_MAX_RUNTIME_CONDITIONS,
    DEFAULT_MAX_RUNTIME_STEPS,
    DEFAULT_TASK_TIMEOUT_SECONDS,
)
from core.production import DEFAULT_IMAGE_SEED


def add_backend_arguments(
    parser: argparse.ArgumentParser,
    *,
    required: bool = True,
    model_required: bool | None = None,
) -> None:
    if model_required is None:
        model_required = required
    parser.add_argument("--backend", choices=SUPPORTED_BACKENDS, required=required)
    parser.add_argument("--repo-path")
    parser.add_argument("--model-path", required=model_required)
    parser.add_argument("--vq-path")
    parser.add_argument("--llm-path")
    parser.add_argument(
        "--understanding-vision-path",
        help=(
            "vision tokenizer used by a backend's review path when it is not "
            "part of --vq-path (TokenFlow: tokenflow_siglip_32k.pt)"
        ),
    )
    parser.add_argument("--bagel-mode", type=int, default=1)
    parser.add_argument("--offload-folder")
    parser.add_argument("--sft-checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--model-call-timeout-seconds",
        type=float,
        default=DEFAULT_MODEL_CALL_TIMEOUT_SECONDS,
        help="hard wall-clock deadline for each isolated model call",
    )
    parser.add_argument(
        "--backend-startup-timeout-seconds",
        type=float,
        default=DEFAULT_BACKEND_STARTUP_TIMEOUT_SECONDS,
        help="hard wall-clock deadline for loading the isolated backend",
    )
    parser.add_argument(
        "--task-timeout-seconds", type=float,
        default=DEFAULT_TASK_TIMEOUT_SECONDS,
        help="hard wall-clock deadline for one complete task",
    )
    parser.add_argument(
        "--max-runtime-conditions", type=int,
        default=DEFAULT_MAX_RUNTIME_CONDITIONS,
        help=(
            "cap on declared Condition items before execution; the default "
            "mirrors the Plan3 0-2 reference contract"
        ),
    )
    parser.add_argument(
        "--max-runtime-steps", type=int,
        default=DEFAULT_MAX_RUNTIME_STEPS,
    )
    parser.add_argument(
        "--max-checks-per-step", type=int,
        default=DEFAULT_MAX_CHECKS_PER_STEP,
    )


def add_image_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_text: bool,
    max_steps: int = DEFAULT_MAX_STEPS,
    text_max_tokens: int = DEFAULT_TEXT_MAX_TOKENS,
    text_temperature: float = 0.0,
    image_shape: tuple[int, int] = DEFAULT_IMAGE_SHAPE,
    image_steps: int = 50,
    seed: int = DEFAULT_IMAGE_SEED,
) -> None:
    if include_text:
        parser.add_argument("--max-steps", type=int, default=max_steps)
        parser.add_argument("--text-max-tokens", type=int, default=text_max_tokens)
        parser.add_argument("--text-temperature", type=float, default=text_temperature)
        parser.add_argument("--sample", action="store_true")
    parser.add_argument("--image-steps", type=int, default=image_steps)
    parser.add_argument("--image-height", type=int, default=image_shape[0])
    parser.add_argument("--image-width", type=int, default=image_shape[1])
    parser.add_argument("--seed", type=int, default=seed)


def require_backend_repo(args) -> None:
    if args.backend == "bagel":
        args.repo_source_identity = None
        return
    if not args.repo_path:
        raise SystemExit("--repo-path is required for non-BAGEL backends")
    try:
        args.repo_source_identity = official_repository_identity(
            args.repo_path, args.backend)
    except ValueError as error:
        raise SystemExit(str(error)) from error


def require_positive_runtime_timeouts(args) -> None:
    if (args.model_call_timeout_seconds <= 0
            or args.backend_startup_timeout_seconds <= 0
            or args.task_timeout_seconds <= 0):
        raise SystemExit("backend and task runtime timeouts must be positive")
    resource_limits = (
        args.max_runtime_conditions,
        args.max_runtime_steps,
        args.max_checks_per_step,
    )
    if any(value < 1 for value in resource_limits):
        raise SystemExit("runtime resource limits must be positive")


def require_positive_image_steps(args) -> None:
    if args.image_steps < 1:
        raise SystemExit("image settings must be positive")


def require_positive_text_limits(args) -> None:
    if args.max_steps < 1 or args.text_max_tokens < 1:
        raise SystemExit("step and token limits must be positive")


def require_sampling_temperature(args) -> None:
    if args.sample and args.text_temperature <= 0:
        raise SystemExit("sample mode requires a positive text temperature")
