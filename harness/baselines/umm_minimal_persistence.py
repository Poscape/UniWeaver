"""Atomic incremental persistence for the sequential minimal protocol."""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path

from PIL import Image

from core.hashing import sha256_file
from core.policy_registry import (
    INTERRUPTION_RECOVERY_EVENT,
    MINIMAL_STEP_OPERATION_EVENT,
    ORCHESTRATION_FAILURE_EVENT,
    PERSISTENCE_FAILURE_EVENT,
)
from core.serialization import atomic_write_json, atomic_write_text

DEFAULT_LEASE_SECONDS = 300.0


class MinimalPersistenceSession:
    """Persist every minimal model operation before advancing the loop."""

    def __init__(self, output_dir: str | Path, initial: dict, resume: bool = False):
        self.output = Path(output_dir)
        if resume and self.output.is_dir() and (self.output / "task_result.json").is_file():
            self.steps_dir = self.output / "steps"
            self.state = json.loads(
                (self.output / "task_result.json").read_text(encoding="utf-8")
            )
            if any(self.state.get(key) != value for key, value in initial.items()):
                raise ValueError("minimal resume identity does not match persisted state")
            if self.state.get("status") == "running":
                self._validate_resume_owner()
            active = self.state.get("active_operation")
            if isinstance(active, int) and active < len(self.state.get("events", [])):
                event = self.state["events"][active]
                event.update({"status": "interrupted", "finished_at": time.time()})
                self.state.setdefault("interruptions", []).append({
                    "policy_id": INTERRUPTION_RECOVERY_EVENT["policy_id"],
                    "status": "interrupted",
                    "step_index": event.get("step_index"),
                    "detected_at": time.time(),
                    "previous_owner": self.state.get("owner", {}),
                })
            self.state["active_operation"] = None
            self.state["status"] = "running"
            self.state["resumed_at"] = time.time()
            self.state["owner"] = self._owner_identity()
            self._refresh_lease(self._configured_operation_timeout())
            self._write()
            return
        if resume and self.output.exists() and any(self.output.iterdir()):
            raise ValueError("minimal resume output has no task_result.json")
        self.output.mkdir(parents=True, exist_ok=False)
        self.steps_dir = self.output / "steps"
        self.steps_dir.mkdir()
        self.state = {
            **initial,
            "status": "running",
            "started_at": time.time(),
            "steps": [],
            "events": [],
            "active_operation": None,
            "owner": self._owner_identity(),
            "heartbeat_at": time.time(),
            "lease_expires_at": time.time() + max(
                DEFAULT_LEASE_SECONDS,
                float(
                    initial.get("runtime_configuration", {})
                    .get("runtime_control", {})
                    .get("model_call_timeout_seconds") or 0
                ),
            ),
        }
        self._write()

    @staticmethod
    def _owner_identity():
        return {"hostname": socket.gethostname(), "pid": os.getpid()}

    @staticmethod
    def _pid_is_alive(pid):
        if not isinstance(pid, int) or pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def _validate_resume_owner(self):
        owner = self.state.get("owner", {})
        same_host = owner.get("hostname") == socket.gethostname()
        if (same_host and owner.get("pid") != os.getpid()
                and self._pid_is_alive(owner.get("pid"))):
            raise RuntimeError(
                f"refusing to resume a live minimal run owned by pid {owner.get('pid')}"
            )
        if (not same_host
                and self.state.get("lease_expires_at", 0) > time.time()):
            raise RuntimeError(
                "refusing to resume a minimal run whose cross-host lease is still active"
            )

    def _refresh_lease(self, deadline_seconds=None):
        now = time.time()
        self.state["heartbeat_at"] = now
        self.state["lease_expires_at"] = now + max(
            DEFAULT_LEASE_SECONDS, float(deadline_seconds or 0)
        )

    def _configured_operation_timeout(self):
        value = (
            self.state.get("runtime_configuration", {})
            .get("runtime_control", {})
            .get("model_call_timeout_seconds")
        )
        return value if isinstance(value, (int, float)) else None

    def completed_steps(self):
        """Load only the contiguous persisted prefix with matching image hashes."""
        from baselines.umm_framework import SequentialStep

        completed = []
        for expected, record in enumerate(self.state.get("steps", []), 1):
            if not isinstance(record, dict) or record.get("step_index") != expected:
                raise ValueError("minimal resume steps are not a contiguous prefix")
            image_path = self.output / str(record.get("image_path", ""))
            if not image_path.is_file() or sha256_file(image_path) != record.get("image_sha256"):
                raise ValueError(f"minimal resume image hash mismatch at step {expected}")
            with Image.open(image_path) as image:
                persisted_image = image.copy()
            completed.append(SequentialStep(
                step_index=expected,
                content=record.get("content", ""),
                raw_text=record.get("raw_text", ""),
                image=persisted_image,
                done=record.get("done") is True,
                image_seed=record.get("image_seed"),
            ))
        return completed

    def _write(self):
        atomic_write_json(self.output / "task_result.json", self.state)

    def step_started(self, step_index: int, raw_input: str, request_config: dict):
        event = {
            "policy_id": MINIMAL_STEP_OPERATION_EVENT["policy_id"],
            "stage": "minimal_step",
            "step_index": step_index,
            "status": "started",
            "started_at": time.time(),
            "finished_at": None,
            "raw_input": raw_input,
            "raw_output": None,
            "request_config": request_config,
            "error": None,
        }
        self.state["events"].append(event)
        self.state["active_operation"] = len(self.state["events"]) - 1
        timeout = request_config.get("model_call_timeout_seconds")
        if not isinstance(timeout, (int, float)):
            timeout = self._configured_operation_timeout()
        self._refresh_lease(timeout if isinstance(timeout, (int, float)) else None)
        self._write()

    def step_done(self, step):
        stem = f"step_{step.step_index:02d}"
        text_path = self.steps_dir / f"{stem}.txt"
        image_path = self.steps_dir / f"{stem}.png"
        event = self.state["events"][self.state["active_operation"]]
        event.update({
            "raw_output": step.raw_text,
            "model_returned_at": time.time(),
        })
        self._write()
        atomic_write_text(text_path, step.content + "\n")
        temporary = image_path.with_suffix(".png.tmp")
        try:
            step.image.save(temporary, format="PNG")
            temporary.replace(image_path)
        finally:
            temporary.unlink(missing_ok=True)
        event.update({
            "status": "completed",
            "finished_at": time.time(),
            "raw_output": step.raw_text,
        })
        self.state["steps"].append({
            "step_index": step.step_index,
            "content": step.content,
            "raw_text": step.raw_text,
            "done": step.done,
            "image_seed": step.image_seed,
            "image_path": str(image_path.relative_to(self.output)),
            "image_sha256": sha256_file(image_path),
        })
        self.state["active_operation"] = None
        self._refresh_lease()
        self._write()

    def step_failed(self, step_index: int, error: BaseException, evidence: dict):
        active = self.state.get("active_operation")
        if isinstance(active, int) and active < len(self.state["events"]):
            event = self.state["events"][active]
            if event.get("step_index") == step_index:
                event["raw_output"] = evidence.get("raw_text")
                event["error"] = {
                    "type": type(error).__name__, "message": str(error),
                }
                self._write()

    def step_terminal(self, step_index: int, raw_text: str, done: bool):
        """Close a step that produced no visible content without failing the task."""
        active = self.state.get("active_operation")
        if isinstance(active, int) and active < len(self.state["events"]):
            event = self.state["events"][active]
            if event.get("step_index") == step_index:
                event.update({
                    "status": "terminal",
                    "finished_at": time.time(),
                    "raw_output": raw_text,
                    "terminal_declaration": bool(done),
                })
        self.state["active_operation"] = None
        self._refresh_lease()
        self._write()

    def fail(self, error: BaseException, *, stage="minimal_step", status="error"):
        """Persist a terminal failure under the same name summary.json uses."""
        active = self.state.get("active_operation")
        if isinstance(active, int) and active < len(self.state["events"]):
            self.state["events"][active].update({
                "status": "failed",
                "finished_at": time.time(),
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            })
        preflight = stage == "preflight"
        self.state.update({
            "status": status,
            "failure_kind": status,
            "finished_at": time.time(),
            "active_operation": None,
            "failure_provenance": {
                "policy_id": (
                    ORCHESTRATION_FAILURE_EVENT["policy_id"]
                    if preflight else MINIMAL_STEP_OPERATION_EVENT["policy_id"]
                ),
                "stage": stage,
                "classification": (
                    "orchestration_failure" if preflight
                    else "model_or_runtime_failure"
                ),
                "exception_type": type(error).__name__,
                "message": str(error),
                "resolved": not preflight,
            },
        })
        self._refresh_lease()
        self._write()

    def finish(self, run: dict) -> dict:
        self.state.update({
            key: value for key, value in run.items() if key != "steps"
        })
        self.state["status"] = run["status"]
        self.state["finished_at"] = time.time()
        self.state["active_operation"] = None
        self._refresh_lease()
        transcript_lines = [
            f"# {run['task_key']}", "", run["task_prompt"], "",
        ]
        for step in self.state["steps"]:
            transcript_lines.extend([
                f"## Step {step['step_index']}", "", step["content"], "",
                f"![Step {step['step_index']}]({step['image_path']})", "",
            ])
        atomic_write_text(
            self.output / "transcript.md", "\n".join(transcript_lines)
        )
        atomic_write_json(self.output / "run.json", self.state)
        self._write()
        return self.state

    def persistence_failure(self, error: BaseException):
        """Write an auxiliary record for persistence diagnostics."""
        record = {
            "policy_id": PERSISTENCE_FAILURE_EVENT["policy_id"],
            "status": "unresolved",
            "exception_type": type(error).__name__,
            "message": str(error),
            "recorded_at": time.time(),
        }
        try:
            atomic_write_json(self.output / "persistence_error.json", record)
        except Exception:  # noqa: BLE001
            return False
        return True
