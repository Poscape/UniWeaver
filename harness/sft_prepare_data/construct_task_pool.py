"""Construct four-stage trajectories for a generated UniWeaver task pool."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sft_prepare_data.construct_trajectory_api import construct_trajectory
from sft_prepare_data.trajectory_construct_driver import run_task_specs
from sft_prepare_data.trajectory_providers import (
    ApiStageSender,
    create_client,
)


def load_task_pool(path: str | Path) -> list[dict]:
    """Load and validate the task keys and prompts needed by the constructor."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    tasks = payload.get("tasks") if isinstance(payload, dict) else None
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("task pool must contain a nonempty tasks list")
    records = []
    seen = set()
    for item in tasks:
        if not isinstance(item, dict):
            raise TypeError("task pool entries must be objects")
        key = item.get("task_key") or item.get("task_id")
        prompt = item.get("task_prompt") or item.get("user_prompt")
        if not isinstance(key, str) or not key.strip():
            raise ValueError("task pool entry has no task_key/task_id")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"task pool entry {key} has no task prompt")
        if key in seen:
            raise ValueError(f"duplicate task key in task pool: {key}")
        seen.add(key)
        records.append({
            "task_key": key,
            "task_prompt": prompt,
            "task_family": item.get("task_family"),
            "difficulty": item.get("difficulty"),
        })
    return records


def output_path_for(output_dir: Path, task_key: str) -> Path:
    safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_key).strip("._") or "task"
    return output_dir / f"{safe_key}.json"


def construct_task_pool(
    task_pool: str | Path,
    output_dir: str | Path,
    run_one: Callable[[str, str, str], object],
    *,
    parallel: int = 1,
    resume: bool = False,
    limit: int | None = None,
) -> list:
    """Construct one trajectory per task-pool entry."""
    tasks = load_task_pool(task_pool)
    if limit is not None:
        tasks = tasks[:max(0, limit)]
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    specs = []
    for item in tasks:
        path = output_path_for(root, item["task_key"])
        if resume and path.is_file():
            continue
        specs.append((item["task_key"], item["task_prompt"], str(path)))
    if not specs:
        return []
    return run_task_specs(specs, run_one, parallel=parallel)


def run(args) -> int:
    config, _client = create_client(
        args.api, args.api_key_override, args.api_base_url)
    model = args.model or config.get("model", "deepseek-v4-pro")

    def run_one(task_key: str, task_prompt: str, output_path: str):
        sender = ApiStageSender(
            args.api, model, args.api_key_override, args.api_base_url)
        return construct_trajectory(
            task_key, task_prompt, output_path, sender,
            extra_fields={
                "model_provider": args.api,
                "model_name": model,
            },
        )

    results = construct_task_pool(
        args.task_pool, args.output_dir, run_one,
        parallel=args.parallel, resume=args.resume, limit=args.limit,
    )
    print(f"constructed {len(results)} trajectories in {args.output_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-pool", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api", default="deepseek")
    parser.add_argument("--model")
    parser.add_argument("--key", dest="api_key_override")
    parser.add_argument("--base-url", dest="api_base_url")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.parallel < 1:
        parser.error("--parallel must be at least 1")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be nonnegative")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
