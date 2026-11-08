"""Shared model-artifact identity for UMM backends."""

from __future__ import annotations

import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path

from core.hashing import sha256_file

# Hugging Face snapshot files often resolve to content-addressed blobs whose
# name is already the SHA256 of the exact bytes.
_BLOB_NAME = re.compile(r"[0-9a-f]{64}")


@lru_cache(maxsize=256)
def _file_sha256(path: str, size: int, mtime_ns: int) -> str:
    del size, mtime_ns
    resolved = Path(path)
    if _BLOB_NAME.fullmatch(resolved.name):
        return resolved.name
    return sha256_file(resolved)


def _artifact_signature(path: Path) -> dict:
    resolved = Path(os.path.realpath(os.path.expanduser(str(path))))
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "sha256": _file_sha256(str(resolved), stat.st_size, stat.st_mtime_ns),
    }


def _add_artifacts(artifacts: dict, prefix: str, source: Path) -> None:
    source = Path(os.path.realpath(os.path.expanduser(str(source))))
    if source.is_file():
        artifacts[prefix] = _artifact_signature(source)
        return
    if not source.is_dir():
        raise FileNotFoundError(f"model artifact does not exist: {source}")
    for path in sorted(source.rglob("*")):
        if path.is_file() and ".cache" not in path.parts:
            relative = path.relative_to(source).as_posix()
            artifacts[f"{prefix}/{relative}"] = _artifact_signature(path)


def model_artifact_identity(
    model_root: str | os.PathLike,
    *,
    extra_artifacts: dict[str, str | os.PathLike] | None = None,
) -> dict:
    """Bind a backend to the exact bytes of all checkpoint components."""
    root = Path(os.path.realpath(os.path.expanduser(str(model_root))))
    if not root.is_dir():
        raise FileNotFoundError(f"model root does not exist: {root}")
    artifacts: dict[str, dict] = {}
    _add_artifacts(artifacts, "model", root)
    for name, source in sorted((extra_artifacts or {}).items()):
        _add_artifacts(artifacts, name, Path(source))
    digest_payload = {
        name: {"size": value["size"], "sha256": value["sha256"]}
        for name, value in sorted(artifacts.items())
    }
    digest = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "model_root": str(root),
        "sha256": digest,
        "artifacts": digest_payload,
    }
