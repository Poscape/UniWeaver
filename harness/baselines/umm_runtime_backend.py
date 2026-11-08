"""Process-isolated, time-bounded access to one persistent UMM backend."""

from __future__ import annotations

import atexit
import ctypes
import multiprocessing
import os
import signal
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict

from baselines.umm_framework import BackendCapabilities

DEFAULT_MODEL_CALL_TIMEOUT_SECONDS = 3600.0
DEFAULT_BACKEND_STARTUP_TIMEOUT_SECONDS = 3600.0


class BackendCallTimeout(TimeoutError):
    """A backend worker did not finish one model call by its deadline."""

    def __init__(self, operation: str, timeout_seconds: float):
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"backend operation {operation!r} exceeded "
            f"{timeout_seconds:g} seconds"
        )


class BackendWorkerError(RuntimeError):
    """An exception raised inside the isolated backend worker."""


class TaskDeadlineExceeded(BackendCallTimeout):
    """The total task deadline expired during or before a backend call."""

    def __init__(self, operation: str, timeout_seconds: float):
        self.active_operation = operation
        super().__init__("task_deadline", timeout_seconds)


_RPC_METHODS = (
    "begin_minimal_task",
    "generate_image",
    "generate_minimal_step",
    "generate_planner_text",
    "generate_text",
    "generate_uniweaver_image",
    "planner_token_counts",
    "review_image",
)


def _failure_evidence(backend: object | None) -> dict:
    """Collect one adapter's last-call diagnostics for a failed worker call."""
    if backend is None:
        return {}
    evidence = getattr(backend, "last_minimal_step_attempt", None)
    return evidence if isinstance(evidence, dict) else {}


def _worker_error_message(backend: object | None, error: BaseException) -> dict:
    """Build the error envelope for one failed worker call."""
    return {
        "kind": "error",
        "error_type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
        "evidence": _failure_evidence(backend),
    }


def _bind_worker_lifetime_to_parent() -> None:
    """On Linux, prevent a stuck CUDA worker from surviving its supervisor."""
    if not sys.platform.startswith("linux"):
        return
    parent_pid = os.getppid()
    libc = ctypes.CDLL(None)
    if libc.prctl(1, signal.SIGTERM) != 0:  # PR_SET_PDEATHSIG
        raise OSError("could not configure backend parent-death signal")
    if os.getppid() != parent_pid:
        os.kill(os.getpid(), signal.SIGTERM)


def _worker_main(connection, args) -> None:
    backend = None
    try:
        _bind_worker_lifetime_to_parent()
        from baselines.umm_backend_factory import load_backend

        backend = load_backend(args)
        capabilities = getattr(
            backend,
            "capabilities",
            BackendCapabilities(name=str(getattr(backend, "name", "unknown"))),
        )
        connection.send({
            "kind": "ready",
            "name": str(getattr(backend, "name", "unknown")),
            "model_identity": getattr(backend, "model_identity", {}),
            "capabilities": asdict(capabilities),
            "methods": [
                name for name in _RPC_METHODS
                if callable(getattr(backend, name, None))
            ],
        })
        while True:
            request = connection.recv()
            if request.get("kind") == "close":
                return
            operation = request["operation"]
            try:
                result = getattr(backend, operation)(
                    *request.get("args", ()), **request.get("kwargs", {})
                )
                connection.send({"kind": "result", "value": result})
            except BaseException as error:  # noqa: BLE001 - cross-process boundary
                connection.send(_worker_error_message(backend, error))
    except BaseException as error:  # noqa: BLE001 - preserve startup diagnostics
        try:
            connection.send({
                "kind": "startup_error",
                "error_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            })
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class SupervisedBackend:
    """Keep a model in a child process and enforce a deadline on every call."""

    def __init__(
        self,
        args,
        *,
        call_timeout_seconds: float = DEFAULT_MODEL_CALL_TIMEOUT_SECONDS,
        startup_timeout_seconds: float = DEFAULT_BACKEND_STARTUP_TIMEOUT_SECONDS,
        worker_target=_worker_main,
    ):
        if call_timeout_seconds <= 0 or startup_timeout_seconds <= 0:
            raise ValueError("backend timeouts must be positive")
        self._args = args
        self.call_timeout_seconds = float(call_timeout_seconds)
        self.startup_timeout_seconds = float(startup_timeout_seconds)
        self._context = multiprocessing.get_context("spawn")
        self._lock = threading.Lock()
        self._process = None
        self._connection = None
        self._methods: set[str] = set()
        self._last_minimal_step_attempt: dict = {}
        self._task_deadline_at = None
        self._task_timeout_seconds = None
        self._worker_target = worker_target
        self._start()
        atexit.register(self.close)

    def _start(self, timeout_seconds: float | None = None) -> None:
        timeout = min(
            self.startup_timeout_seconds,
            float(timeout_seconds) if timeout_seconds is not None else self.startup_timeout_seconds,
        )
        parent, child = self._context.Pipe()
        process = self._context.Process(
            target=self._worker_target,
            args=(child, self._args),
            daemon=False,
            name="umm-backend-worker",
        )
        process.start()
        child.close()
        self._process = process
        self._connection = parent
        if not parent.poll(timeout):
            self._terminate()
            if (self._task_deadline_at is not None
                    and self._task_deadline_at <= time.monotonic()):
                raise TaskDeadlineExceeded(
                    "backend_startup", self._task_timeout_seconds)
            raise BackendCallTimeout(
                "backend_startup", timeout)
        try:
            message = parent.recv()
        except EOFError as error:
            self._terminate()
            raise BackendWorkerError(
                "backend worker exited during startup") from error
        if message.get("kind") != "ready":
            self._terminate()
            raise BackendWorkerError(
                f"backend startup failed: {message.get('error_type')}: "
                f"{message.get('message')}\n{message.get('traceback', '')}"
            )
        self.name = message["name"]
        self.model_identity = message.get("model_identity", {})
        self.capabilities = BackendCapabilities(**message["capabilities"])
        self._methods = set(message["methods"])

    def _terminate(self) -> None:
        connection, process = self._connection, self._process
        self._connection = None
        self._process = None
        if connection is not None:
            connection.close()
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=10)

    def close(self) -> None:
        with self._lock:
            if (self._connection is not None and self._process is not None
                    and self._process.is_alive()):
                try:
                    self._connection.send({"kind": "close"})
                    self._process.join(timeout=10)
                except (BrokenPipeError, EOFError, OSError):
                    pass
            self._terminate()

    def _call(self, operation: str, *args, **kwargs):
        with self._lock:
            self._last_minimal_step_attempt = {}
            timeout = self.call_timeout_seconds
            if self._task_deadline_at is not None:
                remaining = self._task_deadline_at - time.monotonic()
                if remaining <= 0:
                    raise TaskDeadlineExceeded(
                        operation, self._task_timeout_seconds)
                timeout = min(timeout, remaining)
            if self._process is None or not self._process.is_alive():
                self._terminate()
                startup_budget = None
                if self._task_deadline_at is not None:
                    startup_budget = self._task_deadline_at - time.monotonic()
                    if startup_budget <= 0:
                        raise TaskDeadlineExceeded(
                            operation, self._task_timeout_seconds)
                self._start(startup_budget)
                if self._task_deadline_at is not None:
                    remaining = self._task_deadline_at - time.monotonic()
                    if remaining <= 0:
                        self._terminate()
                        raise TaskDeadlineExceeded(
                            operation, self._task_timeout_seconds)
                    timeout = min(self.call_timeout_seconds, remaining)
            connection = self._connection
            try:
                connection.send({
                    "kind": "call",
                    "operation": operation,
                    "args": args,
                    "kwargs": kwargs,
                })
            except (BrokenPipeError, EOFError, OSError) as error:
                self._terminate()
                raise BackendWorkerError(
                    f"backend worker unavailable before {operation}") from error
            if not connection.poll(timeout):
                self._terminate()
                if (self._task_deadline_at is not None
                        and self._task_deadline_at <= time.monotonic()):
                    raise TaskDeadlineExceeded(
                        operation, self._task_timeout_seconds)
                raise BackendCallTimeout(operation, self.call_timeout_seconds)
            try:
                message = connection.recv()
            except EOFError as error:
                self._terminate()
                raise BackendWorkerError(
                    f"backend worker exited during {operation}") from error
            if message.get("kind") == "error":
                evidence = message.get("evidence")
                self._last_minimal_step_attempt = (
                    evidence if isinstance(evidence, dict) else {}
                )
                raise BackendWorkerError(
                    f"backend {operation} failed: {message.get('error_type')}: "
                    f"{message.get('message')}\n{message.get('traceback', '')}"
                )
            if message.get("kind") != "result":
                raise BackendWorkerError(
                    f"backend {operation} returned an invalid worker response")
            return message.get("value")

    @contextmanager
    def task_scope(self, timeout_seconds: float):
        if timeout_seconds <= 0:
            raise ValueError("task timeout must be positive")
        previous_deadline = self._task_deadline_at
        previous_timeout = self._task_timeout_seconds
        self._task_timeout_seconds = float(timeout_seconds)
        self._task_deadline_at = time.monotonic() + timeout_seconds
        try:
            yield
        finally:
            self._task_deadline_at = previous_deadline
            self._task_timeout_seconds = previous_timeout

    def __getattr__(self, name: str):
        if name in self._methods:
            return lambda *args, **kwargs: self._call(name, *args, **kwargs)
        raise AttributeError(name)

    @property
    def last_minimal_step_attempt(self) -> dict:
        """Diagnostics recorded by the adapter for the most recent failed call."""
        return self._last_minimal_step_attempt


def load_supervised_backend(args) -> SupervisedBackend:
    """Construct the default process-isolated backend used by every UMM CLI."""
    return SupervisedBackend(
        args,
        call_timeout_seconds=args.model_call_timeout_seconds,
        startup_timeout_seconds=args.backend_startup_timeout_seconds,
    )


def is_timeout_error(error: BaseException) -> bool:
    return isinstance(error, BackendCallTimeout)
