"""Leakage-safe training holdout sourced from the official UniWeaver-Bench suite."""

from __future__ import annotations

import json
import os
from pathlib import Path

from core.hashing import sha256_file, sha256_text
from core.text_normalization import normalize_prompt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOLDOUT_PATH = (
    PROJECT_ROOT / "datasets" / "benchmarks" / "uniweaver_bench"
    / "bench.json"
)
HOLDOUT_PATH = Path(
    os.environ.get("BENCHMARK_FILE") or DEFAULT_HOLDOUT_PATH)
HOLDOUT_SET_VERSION = "uniweaver-bench-v1"


def holdout_manifest_sha256() -> str:
    """Hash of the official UniWeaver-Bench manifest."""
    return sha256_file(HOLDOUT_PATH)


def __getattr__(name: str) -> str:
    """Keep the backward-compatible module-level constant importable and lazy."""
    if name == "HOLDOUT_MANIFEST_SHA256":
        return holdout_manifest_sha256()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def load_benchmark_holdout_tasks() -> list[dict]:
    """Return the official UniWeaver-Bench task IDs and prompt hashes for exclusion."""
    payload = json.loads(HOLDOUT_PATH.read_text(encoding="utf-8"))
    tasks = payload.get("tasks") if isinstance(payload, dict) else None
    if not isinstance(tasks, list) or len(tasks) != 120:
        raise ValueError("UniWeaver-Bench holdout must contain exactly 120 tasks")

    records = []
    for item in tasks:
        key = item.get("id")
        prompt = item.get("user_prompt")
        if not isinstance(key, str) or not key.strip():
            raise ValueError("UniWeaver-Bench holdout contains an invalid task id")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"UniWeaver-Bench holdout prompt is unavailable: {key}")
        records.append({
            "key": key,
            "prompt": prompt,
            "prompt_sha256": sha256_text(normalize_prompt(prompt)),
        })
    if len({item["key"] for item in records}) != len(records):
        raise ValueError("UniWeaver-Bench holdout task IDs must be unique")
    return records
