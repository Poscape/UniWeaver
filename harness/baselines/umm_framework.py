"""Shared execution contract for all unified multimodal model backends."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image

from baselines.umm_backend_factory import backend_spec
from core.production import (
    CONTRACT_VERSION,
    PLANNER_JSON_ONLY_SUFFIX,
    TRAJECTORY_DESIGN,
    image_seed_for,
)

DONE_MARKER = "<DONE>"
# Wrapper lines some backends wrap every generation in. They carry no task
# content, so a generation made only of these counts as empty.
SCAFFOLDING_LINES = frozenset({"<beginning of the code>", "<end of the code>"})
DEFAULT_MAX_STEPS = 8
DEFAULT_TEXT_MAX_TOKENS = 1024
DEFAULT_IMAGE_SHAPE = (1024, 1024)
STEP_SYSTEM_PROMPT = f"""You are completing a task one step at a time.
Output only the natural-language description of the current step; do not output JSON, a plan, future steps, markdown fences, or analysis.
Use the task and the completed-step history to decide the next action.
When the current step finishes the task, put {DONE_MARKER} alone on the final line.
Do not output {DONE_MARKER} unless the task is complete."""
STEP_REQUEST_PROMPT = f"""Output exactly one next step for the task above.
Do not repeat the full task, write a complete report, list future steps, or describe the image as an observer.
Write only the user-visible natural-language action or result for this step.
A non-final step contains one action sentence and no completion marker.
A final step uses exactly this form:
The task is complete.
{DONE_MARKER}
The marker must be alone on its own final line; never append it to a sentence."""



def pil_lanczos():
    """Return the LANCZOS resampling filter on old and new Pillow."""
    resampling = getattr(Image, "Resampling", None)
    return resampling.LANCZOS if resampling is not None else Image.LANCZOS


def _normalise_remote(remote: str) -> str:
    remote = remote.strip().lower()
    if remote.startswith("git@github.com:"):
        remote = "https://github.com/" + remote.split(":", 1)[1]
    remote = remote.removesuffix("/").removesuffix(".git")
    return remote


def official_repository_identity(repo_path: str, backend_name: str) -> dict:
    spec = backend_spec(backend_name)
    expected_remote = spec.official_remote
    expected_commit = spec.official_commit
    if expected_remote is None or expected_commit is None:
        raise ValueError(f"no official repository is registered for {backend_name}")
    path = Path(repo_path).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"repository path does not exist: {path}")

    def git(*arguments):
        try:
            result = subprocess.run(
                ["git", "-C", str(path), *arguments],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise ValueError(
                f"cannot inspect git repository {path}: {error}") from error
        return result.stdout.strip()

    root = Path(git("rev-parse", "--show-toplevel")).resolve()
    remote = git("remote", "get-url", "origin")
    normalized_remote = _normalise_remote(remote)
    normalized_expected = _normalise_remote(expected_remote)
    if normalized_remote != normalized_expected:
        raise ValueError(
            f"{backend_name} repository origin must be {expected_remote}, "
            f"got {remote}")
    commit = git("rev-parse", "HEAD")
    if commit != expected_commit:
        raise ValueError(
            f"{backend_name} repository HEAD must be {expected_commit}, got {commit}"
        )
    dirty = bool(git("status", "--porcelain", "--untracked-files=all"))
    if dirty:
        raise ValueError(f"{backend_name} repository must have a clean worktree")
    return {
        "path": str(root),
        "remote": remote,
        "expected_remote": expected_remote,
        "expected_commit": expected_commit,
        "commit": commit,
        "dirty": dirty,
    }


def parse_step_text(text: str) -> tuple[str, bool]:
    """Normalize the visible step and detect only an exact final marker line."""
    raw_text = str(text or "").strip()
    lines = [
        line for line in raw_text.splitlines()
        if line.strip().lower() not in SCAFFOLDING_LINES
    ]
    done = bool(lines and lines[-1].strip() == DONE_MARKER)
    if done:
        lines.pop()
    content = "\n".join(lines).strip()
    return content, done


@dataclass(frozen=True)
class BackendCapabilities:
    """Declare the inputs and generation controls supported by a backend."""

    # Backend identifier used in run ids, provenance, and error messages.
    name: str

    # --- Condition axis (UniWeaver only) ---
    # When False, the harness records the Step's condition references as
    # consumed by nobody instead of claiming them.
    condition_images_consumed: bool = True
    # When False, the harness passes at most one condition image per Step; any
    # others stay recorded as ignored rather than claimed.
    multiple_condition_images_consumed: bool = True

    # --- Image-generation axis ---
    # Whether ``generate_image`` uses the previous Step image.
    previous_image_consumed: bool = True
    # Whether the requested shape selects the generation geometry. False means
    # the adapter generates at its native resolution and resamples to the
    # requested delivery size, so persisted Step PNGs are 1024x1024 either way.
    requested_shape_consumed: bool = True
    # Whether ``--image-steps`` applies; token-autoregressive backends use the
    # visual-token budget in ``image_config`` instead.
    requested_image_steps_consumed: bool = True
    # Whether the derived per-image seed is honoured.
    requested_seed_consumed: bool = True

    # --- Review and bookkeeping ---
    # Whether the adapter can answer a Plan4 review prompt; False records
    # ``BACKEND_REVIEW_UNSUPPORTED`` and skips regeneration without failing.
    reviewer_supported: bool = False
    # True only for the BAGEL adapter, which runs this repository's own UniWeaver
    # inference path. Recorded in provenance only; no logic branches on it.
    native_model_backend: bool = False
    # UniWeaver only. Human-readable note on how Planner prompts are delivered.
    # Recorded in provenance only; no logic branches on it.
    planner_transport: str = "adapter-specific"
    # Maximum sequence length declared by the released text model, when known.
    planner_context_limit: int | None = None
    # Fixed generation settings for this backend, merged into run provenance and
    # into the benchmark image metadata.
    image_config: dict | None = None
    # How many adapter calls the harness may issue concurrently.
    max_parallel_calls: int = 1


@dataclass
class UMMRunConfig:
    max_steps: int
    text_max_tokens: int
    text_temperature: float
    image_shape: tuple[int, int]
    image_steps: int
    seed: int
    do_sample: bool = False

@dataclass
class SequentialStep:
    step_index: int
    content: str
    raw_text: str
    image: Image.Image
    done: bool
    image_seed: int | None = None



def _capability_metadata(capabilities: BackendCapabilities) -> dict:
    return {
        "condition_images_consumed": capabilities.condition_images_consumed,
        "previous_image_consumed": capabilities.previous_image_consumed,
        "multiple_condition_images_consumed": capabilities.multiple_condition_images_consumed,
        "requested_shape_consumed": capabilities.requested_shape_consumed,
        "requested_image_steps_consumed": capabilities.requested_image_steps_consumed,
        "requested_seed_consumed": capabilities.requested_seed_consumed,
        "reviewer_supported": capabilities.reviewer_supported,
        "native_model_backend": capabilities.native_model_backend,
        "planner_context_limit": capabilities.planner_context_limit,
        "max_parallel_calls": capabilities.max_parallel_calls,
    }


def backend_provenance(backend: object, mode: str) -> dict:
    capabilities = capabilities_for(backend)
    is_uniweaver = mode == "uniweaver"
    backend_name = capabilities.name
    provenance = {
        "backend": backend_name,
        "native_bagel_uniweaver_inference": backend_name == "bagel" and is_uniweaver,
        "contract_comparability": backend_name == "bagel" and is_uniweaver,
        "protocol_comparability": is_uniweaver,
        "planner_transport": capabilities.planner_transport if is_uniweaver else "not applicable",
        "model_identity": _model_identity_for(backend),
        "backend_capabilities": _capability_metadata(capabilities),
        "backend_image_config": capabilities.image_config,
    }
    try:
        sft_identity = object.__getattribute__(
            backend, "sft_checkpoint_training_identity")
    except AttributeError:
        sft_identity = None
    if isinstance(sft_identity, dict):
        provenance["sft_checkpoint_training_identity"] = sft_identity
    return provenance


def build_umm_provenance(
    backend: object,
    mode: str,
    *,
    schema_version: str | None,
    base_seed: int,
    seed_policy: str,
    extra: dict | None = None,
) -> dict:
    provenance = {
        "framework": "umm-unified-v1",
        "model_family": "umm",
        "adapter": "umm-unified-v1",
        **backend_provenance(backend, mode),
        "trajectory_design": TRAJECTORY_DESIGN if mode == "uniweaver" else None,
        "contract_version": CONTRACT_VERSION if mode == "uniweaver" else None,
        "base_seed": base_seed,
        "seed_policy": seed_policy,
    }
    if extra:
        provenance.update(extra)
    if schema_version is not None:
        provenance["schema_version"] = schema_version
    return provenance


def _model_identity_for(backend: object) -> dict:
    try:
        identity = object.__getattribute__(backend, "model_identity")
    except AttributeError:
        return {}
    return identity if isinstance(identity, dict) else {}


def run_serial_batch(
    tasks: list[dict],
    process_task: Callable[[int, dict], dict],
    on_progress: Callable[[int, dict, dict, list[dict], float], None],
    *,
    start_index: int = 0,
    initial_results: list[dict] | None = None,
) -> list[dict]:
    started = time.time()
    results = list(initial_results or [])
    for index, task in enumerate(tasks[start_index:], start_index + 1):
        record = process_task(index, task)
        results.append(record)
        on_progress(index, task, record, results, started)
    return results


def classify_task_error(error: Exception, *, protocol: str = "uniweaver") -> str:
    """Name the failed stage for one benchmark task."""
    from baselines.umm_runtime_backend import is_timeout_error
    from core.runner import TrajectoryError

    if is_timeout_error(error):
        return "timed_out"
    if protocol == "minimal":
        return "step_error"
    return "planner_error" if isinstance(error, TrajectoryError) else "image_error"


class UMMBackend(Protocol):
    name: str
    capabilities: BackendCapabilities

    def generate_text(
        self,
        prompt: str,
        previous_image: Image.Image | None,
        max_tokens: int,
        temperature: float,
    ) -> str:
        ...

    def generate_image(
        self,
        content: str,
        previous_image: Image.Image | None,
        seed: int,
        image_shape: tuple[int, int],
        image_steps: int,
        previous_images: list[Image.Image] | None = None,
    ) -> Image.Image:
        ...

    def review_image(
        self,
        *,
        image: Image.Image,
        prompt: str,
        condition_images: dict[str, Image.Image],
        max_tokens: int,
    ) -> str:
        ...

    def generate_minimal_step(
        self,
        task_prompt: str,
        history: list[str],
        previous_image: Image.Image | None,
        step_index: int,
        config: UMMRunConfig,
    ) -> tuple[str, str, bool, Image.Image | None, int | None]:
        ...


def _history_text(history: list[str]) -> str:
    if not history:
        return "(no completed steps)"
    return "\n".join(
        f"Step {index}: {content}" for index, content in enumerate(history, 1)
    )


def _step_prompt(task_prompt: str, history: list[str]) -> str:
    return (
        f"{STEP_SYSTEM_PROMPT}\n\n"
        f"Task:\n{task_prompt}\n\n"
        f"Completed-step history:\n{_history_text(history)}\n\n"
        f"{STEP_REQUEST_PROMPT}"
    )


def _image_list(previous_image: Image.Image | None) -> list[Image.Image] | None:
    return [previous_image] if previous_image is not None else None


def _normalise_previous_images(
    previous_image: Image.Image | None,
    previous_images: list[Image.Image] | None,
) -> list[Image.Image]:
    if previous_images is not None:
        return [image for image in previous_images if image is not None]
    return _image_list(previous_image) or []


def seeded_generator(device, seed, modulus: int = 2**63 - 1):
    """Build the noise generator a backend samples with."""
    import torch

    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(int(seed) % modulus)
    return generator


def planner_prompt(*, user_text: str, system_text: str | None) -> str:
    return f"{system_text}\n\n{user_text}" if system_text else user_text


def send_planner_text(
    backend: object,
    *,
    user_text: str,
    system_text: str | None,
    max_tokens: int,
    do_sample: bool,
    temperature: float,
) -> str:
    # Shared closing instruction for every Planner request.
    user_text = f"{user_text}{PLANNER_JSON_ONLY_SUFFIX}"
    sender = getattr(backend, "generate_planner_text", None)
    if sender is not None:
        return sender(
            user_text=user_text,
            system_text=system_text,
            max_tokens=max_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )
    del do_sample
    prompt = planner_prompt(user_text=user_text, system_text=system_text)
    prepare = getattr(backend, "_prepare_planner_prompt", None)
    if prepare is not None:
        prompt = prepare(prompt)
    return backend.generate_text(prompt, None, max_tokens, temperature)


class StatelessMinimalBackend:
    """Implement the shared minimal loop for stateless backend adapters."""

    def generate_planner_text(
        self, *, user_text, system_text, max_tokens, do_sample, temperature
    ):
        del do_sample
        return self.generate_text(
            self._prepare_planner_prompt(
                planner_prompt(user_text=user_text, system_text=system_text)
            ),
            None,
            max_tokens,
            temperature,
        )

    def _prepare_planner_prompt(self, prompt: str) -> str:
        return prompt

    def review_image(self, *, image, prompt, condition_images, max_tokens):
        del condition_images
        return self.generate_text(prompt, image, max_tokens, 0.0)

    def generate_minimal_step(
        self,
        task_prompt: str,
        history: list[str],
        previous_image: Image.Image | None,
        step_index: int,
        config: UMMRunConfig,
    ) -> tuple[str, str, bool, Image.Image | None, int | None]:
        self.last_minimal_step_attempt = {"raw_text": None}
        raw_text = self.generate_text(
            _step_prompt(task_prompt, history),
            previous_image,
            config.text_max_tokens,
            config.text_temperature,
        )
        self.last_minimal_step_attempt["raw_text"] = raw_text
        content, done = parse_step_text(raw_text)
        if not content:
            # Nothing to illustrate or show a reader: stop without a step.
            return raw_text, content, done, None, None
        image_seed = image_seed_for(config.seed, "step", step_index)
        image = self.generate_image(
            content,
            previous_image,
            image_seed,
            config.image_shape,
            config.image_steps,
        )
        if not isinstance(image, Image.Image):
            raise TypeError(f"step {step_index} returned no PIL image")
        return raw_text, content, done, image, image_seed


def run_minimal_task(
    backend: UMMBackend,
    task_prompt: str,
    task_key: str,
    config: UMMRunConfig,
    *,
    on_step_start: Callable[[int, str, dict], None] | None = None,
    on_step_done: Callable[[SequentialStep], None] | None = None,
    on_step_terminal: Callable[[int, str, bool], None] | None = None,
    on_step_error: Callable[[int, BaseException, dict], None] | None = None,
    initial_steps: list[SequentialStep] | None = None,
) -> dict:
    """Run one sequential minimal task through the common backend contract."""
    if not config.max_steps or config.max_steps < 1:
        raise ValueError("max_steps must be positive")
    if not config.text_max_tokens or config.text_max_tokens < 1:
        raise ValueError("text_max_tokens must be positive")

    steps: list[SequentialStep] = list(initial_steps or [])
    history: list[str] = [step.content for step in steps]
    previous_image = steps[-1].image if steps else None
    begin = getattr(backend, "begin_minimal_task", None)
    if begin is not None:
        begin(task_prompt)
    next_step = (
        config.max_steps + 1
        if steps and steps[-1].done else len(steps) + 1
    )
    declared_done = False
    for step_index in range(next_step, config.max_steps + 1):
        raw_input = _step_prompt(task_prompt, history)
        if on_step_start is not None:
            on_step_start(step_index, raw_input, {
                "max_tokens": config.text_max_tokens,
                "temperature": config.text_temperature,
                "do_sample": config.do_sample,
                "image_shape": list(config.image_shape),
                "image_steps": config.image_steps,
            })
        try:
            raw_text, content, done, image, step_image_seed = backend.generate_minimal_step(
                task_prompt,
                history,
                previous_image,
                step_index,
                config,
            )
        except BaseException as error:
            if on_step_error is not None:
                evidence = getattr(backend, "last_minimal_step_attempt", {})
                on_step_error(
                    step_index,
                    error,
                    evidence if isinstance(evidence, dict) else {},
                )
            raise
        if not content:
            # The model produced no visible step: stop in a controlled way and
            # keep every step already produced. A bare marker still counts as an
            # explicit completion declaration.
            declared_done = done
            if on_step_terminal is not None:
                on_step_terminal(step_index, raw_text, done)
            break
        step = SequentialStep(
            step_index=step_index,
            content=content,
            raw_text=raw_text,
            image=image,
            done=done,
            image_seed=step_image_seed,
        )
        steps.append(step)
        if on_step_done is not None:
            on_step_done(step)
        if done:
            break
        history.append(content)
        previous_image = image

    capabilities = capabilities_for(backend)
    from baselines.umm_runtime_limits import minimal_runtime_usage

    return {
        "framework": "umm-unified-minimal-v1",
        "backend": backend.name,
        "backend_capabilities": _capability_metadata(capabilities),
        "task_key": task_key,
        "task_prompt": task_prompt,
        "status": (
            "completed"
            if (steps and steps[-1].done) or declared_done
            else "incomplete"
        ),
        "max_steps": config.max_steps,
        "runtime_usage": minimal_runtime_usage(config.max_steps),
        "text_max_tokens": config.text_max_tokens,
        "do_sample": config.do_sample,
        "text_temperature": config.text_temperature,
        "image_shape": list(config.image_shape),
        "image_steps": config.image_steps,
        "image_seed": config.seed,
        "steps": steps,
    }


def capabilities_for(backend: object) -> BackendCapabilities:
    try:
        capabilities = object.__getattribute__(backend, "capabilities")
    except AttributeError:
        capabilities = None
    if isinstance(capabilities, BackendCapabilities):
        return capabilities
    try:
        name = object.__getattribute__(backend, "name")
    except AttributeError:
        name = "unknown"
    try:
        condition_images_consumed = object.__getattribute__(
            backend, "condition_images_consumed")
    except AttributeError:
        condition_images_consumed = True
    return BackendCapabilities(
        name=str(name),
        condition_images_consumed=bool(condition_images_consumed),
    )
