#!/usr/bin/env python3
"""Fail-fast readiness probe for one backend environment; loads no model weights."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BAGEL_MODEL_ARTIFACTS = ("ema.safetensors", "ae.safetensors", "llm_config.json")
SEEDX_EDIT_ARTIFACTS = (
    "seed_x_edit/llm",
    "seed_x_edit/agent/pytorch_model.bin",
    "seed_detokenizer/second_stage/pytorch_model.bin",
)


def _report(reason: str) -> int:
    print(f"NOT_READY {reason}", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--repo-path")
    parser.add_argument("--model-path")
    parser.add_argument("--vq-path")
    parser.add_argument("--llm-path")
    parser.add_argument("--understanding-vision-path")
    parser.add_argument("--sft-checkpoint")
    parser.add_argument(
        "--require-flash-attn",
        action="store_true",
        help="also require an importable flash_attn (BAGEL needs the official one)",
    )
    arguments = parser.parse_args(argv)

    try:
        import torch
    except ImportError as error:
        return _report(f"PyTorch unavailable in this interpreter: {error}")
    if not torch.cuda.is_available():
        return _report("torch.cuda.is_available() is false")

    resolved: list[str] = []
    for label, value in (
        ("repo-path", arguments.repo_path),
        ("model-path", arguments.model_path),
        ("vq-path", arguments.vq_path),
        ("llm-path", arguments.llm_path),
        ("understanding-vision-path", arguments.understanding_vision_path),
        ("sft-checkpoint", arguments.sft_checkpoint),
    ):
        if not value:
            continue
        path = Path(value).expanduser()
        if not path.exists():
            return _report(f"{label} does not exist: {path}")
        resolved.append(f"{label}={path}")

    if arguments.backend == "bagel":
        if not arguments.model_path:
            return _report("bagel requires --model-path")
        root = Path(arguments.model_path).expanduser()
        missing = [
            name for name in BAGEL_MODEL_ARTIFACTS if not (root / name).is_file()
        ]
        if missing:
            return _report(f"BAGEL model artifacts missing from {root}: {missing}")

    if arguments.backend == "seed-x" and arguments.model_path:
        root = Path(arguments.model_path).expanduser()
        missing = [
            name for name in SEEDX_EDIT_ARTIFACTS if not (root / name).exists()
        ]
        if missing:
            return _report(
                f"seed-x editing artifacts missing from {root}: {missing}")

    if arguments.require_flash_attn:
        try:
            import flash_attn
        except ImportError as error:
            return _report(f"flash_attn unavailable in this interpreter: {error}")
        resolved.append(f"flash_attn={flash_attn.__version__}")

    print(
        f"READY backend={arguments.backend} "
        f"torch={torch.__version__} cuda_devices={torch.cuda.device_count()} "
        + " ".join(resolved)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
