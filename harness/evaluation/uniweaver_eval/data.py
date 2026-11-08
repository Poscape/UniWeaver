from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.serialization import atomic_write_json, atomic_write_text
from core.tasks import BENCHMARK_SPECS, validate_uniweaver_bench_tasks


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    family: str
    prompt: str | None
    evaluation_reference: dict[str, Any]
    difficulty: str | None = None


@dataclass(frozen=True)
class Artifact:
    task_id: str
    run_id: str
    prompt: str
    trajectory: dict[str, Any]
    image_paths: list[Path]
    source_path: Path


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    atomic_write_json(path, value)


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    payload = "".join(
        json.dumps(record, ensure_ascii=True) + "\n" for record in records
    )
    atomic_write_text(path, payload)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return records


def _registry_base(repository_root: Path, registry_path: Path, path_base: str) -> Path:
    if path_base == "delivery_root":
        return repository_root
    return registry_path.parent


def load_run_registry(registry_path: Path, repository_root: Path) -> dict[str, dict[str, Any]]:
    registry = read_json(registry_path)
    path_base = registry.get("path_base", "registry")
    base = _registry_base(repository_root, registry_path, path_base)
    runs = registry.get("runs")
    if not isinstance(runs, list):
        raise TypeError("Run registry must contain a runs list.")
    resolved: dict[str, dict[str, Any]] = {}
    for entry in runs:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise TypeError("Each run registry entry must have an id.")
        run_id = entry["id"]
        trajectory_dir = entry.get("trajectory_dir")
        image_root = entry.get("image_root")
        if not isinstance(trajectory_dir, str) or not isinstance(image_root, str):
            raise TypeError(
                f"Run registry entry {run_id} needs trajectory_dir and image_root."
            )
        resolved[run_id] = {
            **entry,
            "trajectory_dir": Path(trajectory_dir)
            if Path(trajectory_dir).is_absolute()
            else base / trajectory_dir,
            "image_root": Path(image_root)
            if Path(image_root).is_absolute()
            else base / image_root,
        }
    return resolved


def load_matrix_tasks(path: Path) -> dict[str, TaskSpec]:
    matrix = read_json(path)
    tasks = matrix.get("tasks")
    if not isinstance(tasks, list):
        raise TypeError(f"Benchmark matrix must contain a tasks list: {path}")
    result: dict[str, TaskSpec] = {}
    for task in tasks:
        if not isinstance(task, dict) or not isinstance(task.get("id"), str):
            raise TypeError(f"Invalid task entry in {path}")
        task_id = task["id"]
        if task_id in result:
            raise ValueError(f"Duplicate task id in {path}: {task_id}")
        family = task.get("task_family") or task.get("family")
        if not isinstance(family, str) or not family:
            raise ValueError(f"Missing task family for {task_id}")
        prompt = task.get("user_prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"Missing user prompt for {task_id}")
        reference = task.get("evaluation_reference")
        if not isinstance(reference, dict):
            raise TypeError(f"Missing evaluation reference for {task_id}")
        result[task_id] = TaskSpec(
            task_id=task_id,
            family=family,
            prompt=prompt,
            evaluation_reference=reference,
            difficulty=task.get("difficulty"),
        )
    declared_count = matrix.get("task_count")
    if declared_count is not None and declared_count != len(result):
        raise ValueError(
            f"Declared task count {declared_count} does not match {len(result)} in {path}"
        )
    return result


def load_benchmark(
    repository_root: Path, benchmark_name: str = "UniWeaver-Bench"
) -> dict[str, TaskSpec]:
    spec = BENCHMARK_SPECS.get(benchmark_name)
    if spec is None:
        raise ValueError(f"Unknown UniWeaver-Bench benchmark: {benchmark_name}")
    tasks = load_matrix_tasks(
        repository_root / "datasets/benchmarks/uniweaver_bench" / spec["filename"]
    )
    if len(tasks) == spec["task_count"]:
        validate_uniweaver_bench_tasks(tasks.values())
    return tasks


def trajectory_directory(
    repository_root: Path,
    run_id: str,
    run_registry: dict[str, dict[str, Any]] | None = None,
    run_dir: Path | None = None,
) -> Path:
    if run_dir is not None:
        return run_dir / "text"
    if run_registry and run_id in run_registry:
        return run_registry[run_id]["trajectory_dir"]
    return repository_root / "output_run" / "runs" / run_id / "text"


def _is_trajectory_file(path: Path) -> bool:
    return path.suffix == ".json" and not path.name.startswith("._") and path.name != "summary.json"


def resolve_step_images(
    repository_root: Path,
    run_id: str,
    task_id: str,
    image_root: Path | None = None,
) -> list[Path]:
    root = (
        image_root
        if image_root is not None
        else repository_root / "output_run" / "runs" / run_id / "images"
    )
    if not root.exists():
        return []
    matches: list[Path] = []
    for task_dir in root.rglob(f"{task_id}__*"):
        steps_dir = task_dir / "steps"
        if not steps_dir.is_dir():
            continue
        matches.extend(
            path
            for path in steps_dir.glob("*.png")
            if not path.name.startswith("._")
        )
    return sorted(set(matches), key=lambda path: str(path).lower())


def load_artifacts(
    repository_root: Path,
    run_id: str,
    run_registry_path: Path | None = None,
    run_dir: Path | None = None,
) -> dict[str, Artifact]:
    run_registry = (
        load_run_registry(run_registry_path, repository_root)
        if run_registry_path
        else None
    )
    run_config = run_registry.get(run_id) if run_registry else None
    artifacts: dict[str, Artifact] = {}
    root = trajectory_directory(repository_root, run_id, run_registry, run_dir)
    if not root.exists():
        raise FileNotFoundError(f"No trajectory directory for {run_id}: {root}")
    image_root = (
        run_dir / "images"
        if run_dir is not None
        else (run_config.get("image_root") if run_config else None)
    )

    for path in sorted(root.glob("*.json")):
        if not _is_trajectory_file(path):
            continue
        try:
            payload = read_json(path)
        except json.JSONDecodeError:
            continue
        task_id = payload.get("task_key") or payload.get("id")
        if not task_id or "plan" not in payload:
            continue
        prompt = payload.get("task_prompt") or payload.get("prompt") or ""
        artifacts[task_id] = Artifact(
            task_id=task_id,
            run_id=run_id,
            prompt=prompt,
            trajectory=payload,
            image_paths=resolve_step_images(
                repository_root, run_id, task_id, image_root
            ),
            source_path=path,
        )
    return artifacts
