"""Write ``ablation_manifest.json``: what the run was, what code produced it,
and which module attributes the patch layer rebound."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from core.hashing import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Production files the ablation layer patches by name.  Their hashes are
# recorded so a run can be rejected if the release line moved underneath it.
TRACKED_PRODUCTION_FILES = (
    "core/contract.py",
    "core/runner.py",
    "core/validation.py",
    "core/render_inputs.py",
    "core/serialization.py",
    "baselines/umm_uniweaver_research.py",
)


def release_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def worktree_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"],
            capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return True
    return bool([line for line in out.splitlines()
                 if not line.strip().endswith("ablation/")])


def build(variant_id: str, description: dict, applied: list[str],
          argv: list[str]) -> dict:
    return {
        "ablation_variant": variant_id,
        "counted_as_experiment": description["counted_as_experiment"],
        "release_commit": release_commit(),
        "release_dirty": worktree_dirty(),
        "profile": description,
        "applied_patches": applied,
        "ablation_code_sha256": {
            name: sha256_file(PROJECT_ROOT / "ablation" / name)
            for name in ("profiles.py", "prompts.py", "patches.py",
                         "run_variant.py", "manifest.py")
        },
        "production_file_sha256": {
            name: sha256_file(PROJECT_ROOT / name)
            for name in TRACKED_PRODUCTION_FILES
        },
        "runner_argv": argv,
    }


def write(output_dir: Path, payload: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "ablation_manifest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    return path


def pending_path(output_dir: Path) -> Path:
    """Where a pre-run copy goes without creating the output root."""
    return output_dir.parent / f"{output_dir.name}.ablation_manifest.pending.json"


def write_pending(output_dir: Path, payload: dict) -> Path:
    path = pending_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    return path


def clear_pending(output_dir: Path) -> None:
    pending_path(output_dir).unlink(missing_ok=True)


def evaluation_package_record(output_dir: Path) -> dict:
    """Record that this variant is scored with the report evaluation track."""
    return {
        "report_track_usable": True,
        "scored_tracks": ["report"],
    }
