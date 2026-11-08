from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from core.hashing import sha256_file
from core.policy_registry import (
    REPORT_IMAGE_PAIRING_GATE,
    REPORT_ORDERED_UNITS_GATE,
    REPORT_READER_CONTENT_GATE,
    REPORT_TERMINAL_STATUS_GATE,
)

from .data import Artifact, load_artifacts, read_json

# Statuses the terminal gate accepts, mirrored from the registry so the
# completion-source diagnostic agrees with what the gate actually accepts.
_TERMINAL_STATUSES = frozenset(REPORT_TERMINAL_STATUS_GATE["accepted_statuses"])

# A Minimal step may carry no reader-facing text (code-fence markers, a repeated
# completion sentence, or a truncation), so these checks look only at the text.
_STEP_NUMBER_PREFIX = re.compile(r"^\s*step\s*\d+\s*[:.\-]\s*", re.IGNORECASE)
_SCAFFOLDING_LINE = re.compile(
    r"^\s*</?(?:beginning|end) of(?: the)? code>\s*$", re.IGNORECASE
)
# Only markers this harness or the backends actually emit are stripped, matched
# as prefixes, so legitimate angle-bracket prose survives.
_MARKER_FRAGMENT = re.compile(r"<[^>\n]*>?")
_MARKER_NAMES = (
    "beginning of the code",
    "end of the code",
    "start of description",
    "end of the description",
    "end of description",
    "start_header_id",
    "end_header_id",
    "image description goes here",
    "assistedive text:",
    "initial state:",
    "final step:",
    "instruction:",
    "think",
    "point",
    "response",
    "done",
    "img",
)
# Shortest shared prefix (with a marker name) that still counts as that marker.
# Long enough that no HTML/XML tag name collides with one.
_MARKER_PREFIX_MATCH = 6
# Shortest truncation that still counts as a marker when it is a prefix of one.
# Guards single-letter HTML tags such as <b> or <p> from matching "beginning" or
# "point".
_MARKER_MIN_TRUNCATION = 3
# Text shorter than this, after removing the markers above, is treated as a
# truncated marker stub rather than reader-facing content.
_MIN_READER_TEXT = 12
_COMPLETION_SENTENCE = re.compile(
    r"""
    ^\s*
    (?:
        (?:the\s+)?(?:task|report|guide|plan|story|brief|work|it)\s+
        (?:(?:is|was|are|were)\s+)?(?:now\s+)?(?:complete|completed|done|finished)
      | i\s+(?:have|'ve)\s+(?:now\s+)?(?:completed|finished|done)
        (?:\s+the\s+(?:task|report|guide|plan|story|brief|work))?
      | (?:complete|completed|done|finished)
    )
    \s*[.!]?\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _is_completion_boilerplate(content: Any) -> bool:
    """True when a step only announces completion and carries no task content."""
    text = str(content or "").strip()
    if not text:
        return False
    lines = [
        line.strip()
        for line in _STEP_NUMBER_PREFIX.sub("", text).splitlines()
        if line.strip() and not _SCAFFOLDING_LINE.match(line)
    ]
    return bool(lines) and all(_COMPLETION_SENTENCE.match(line) for line in lines)


def _reader_text(content: Any) -> str:
    """Return what a reader would actually read from one Step's text."""
    lines = [
        line
        for line in str(content or "").splitlines()
        if not _SCAFFOLDING_LINE.match(line)
        and not _COMPLETION_SENTENCE.match(_STEP_NUMBER_PREFIX.sub("", line.strip()))
    ]
    return re.sub(r"\s+", " ", _remove_marker_fragments("\n".join(lines))).strip()


def _remove_marker_fragments(text: str) -> str:
    """Drop angle-bracket runs that name a known marker, keeping anything else."""
    def shared_prefix(left: str, right: str) -> int:
        limit = min(len(left), len(right))
        index = 0
        while index < limit and left[index] == right[index]:
            index += 1
        return index

    def replace(match: re.Match[str]) -> str:
        inner = match.group(0).strip("<>").strip("|/ \t").lower()
        if not inner:
            return " "
        for name in _MARKER_NAMES:
            if inner.startswith(name) or (
                len(inner) >= _MARKER_MIN_TRUNCATION and name.startswith(inner)
            ):
                return " "
            if shared_prefix(inner, name) >= _MARKER_PREFIX_MATCH:
                return " "
        return match.group(0)

    return _MARKER_FRAGMENT.sub(replace, text)


def _has_reader_content(content: Any) -> bool:
    """Whether a Step carries text a reader could read."""
    if _announces_completion(content):
        return False
    return len(_reader_text(content)) >= _MIN_READER_TEXT


def _announces_completion(content: Any) -> bool:
    """Whether a Step declares the run finished."""
    if _is_completion_boilerplate(content):
        return True
    return bool(_COMPLETION_SENTENCE.match(_reader_text(content)))


@dataclass(frozen=True)
class ReportArtifact:
    task_id: str
    run_id: str
    prompt: str
    units: list[dict[str, Any]]
    image_paths: list[Path]
    source_path: Path
    declared_status: str | None
    # How completion was declared: "marker", "prose", or None. Diagnostic only.
    completion_source: str | None = None
    # Harness-side prompts retained in the raw trace. Diagnostic only; no gate
    # reads them.
    harness_instructions: tuple[str, ...] = ()


def _step_index(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.startswith("step_"):
        try:
            return int(value.rsplit("_", 1)[-1])
        except ValueError:
            return None
    return None


def _bagel_image_for_step(image_paths: list[Path], step: Any) -> Path | None:
    index = _step_index(step)
    if index is not None:
        expected = f"step_{index}.png"
        matches = [path for path in image_paths if path.name == expected]
        if len(matches) == 1:
            return matches[0]
    return None


def _instruction_texts(payload: Any) -> tuple[str, ...]:
    """Collect non-empty ``system_text`` and ``raw_input`` strings from a trace."""
    found: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"system_text", "raw_input"} and isinstance(item, str):
                    text = item.strip()
                    if text:
                        found.append(text)
                else:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(payload)
    return tuple(dict.fromkeys(found))


def _finalize_report_units(
    entries: list[tuple[Any, Path | None]],
    declared_status: str | None,
) -> tuple[list[dict[str, Any]], list[Path], str | None, str | None]:
    """Turn ordered ``(content, image)`` pairs into contiguous report units."""
    kept = [
        (content, image)
        for content, image in entries
        if _has_reader_content(content)
    ]
    units = [
        {
            "unit_index": position,
            "content": content,
            "image_path": str(image) if image is not None else None,
        }
        for position, (content, image) in enumerate(kept, 1)
    ]
    images = [image for _, image in kept if image is not None]
    announced = any(_announces_completion(content) for content, _ in entries)
    if declared_status in _TERMINAL_STATUSES:
        completion_source = "marker"
    elif announced:
        declared_status = "completed"
        completion_source = "prose"
    else:
        completion_source = None
    return units, images, declared_status, completion_source


def _from_bagel(artifact: Artifact) -> ReportArtifact:
    generated = artifact.trajectory.get("step_generations") or {}
    ordered: list[tuple[int, str, dict[str, Any]]] = []
    for step_name, value in generated.items():
        if not isinstance(value, dict):
            continue
        parsed = value.get("parsed")
        if not isinstance(parsed, dict):
            continue
        index = _step_index(step_name)
        if index is not None:
            ordered.append((index, step_name, parsed))

    if not ordered:
        plan2 = ((artifact.trajectory.get("plan") or {}).get("plan2") or {}).get("steps") or []
        for step in plan2:
            if not isinstance(step, dict):
                continue
            index = step.get("step_index")
            if isinstance(index, int):
                ordered.append((index, f"step_{index}", step))

    ordered.sort(key=lambda item: item[0])
    units, images, declared_status, completion_source = _finalize_report_units(
        [
            (parsed.get("content"), _bagel_image_for_step(artifact.image_paths, index))
            for index, _step_name, parsed in ordered
        ],
        artifact.trajectory.get("status"),
    )
    return ReportArtifact(
        task_id=artifact.task_id,
        run_id=artifact.run_id,
        prompt=artifact.prompt,
        units=units,
        image_paths=images,
        source_path=artifact.source_path,
        declared_status=declared_status,
        completion_source=completion_source,
        harness_instructions=_instruction_texts(
            artifact.trajectory.get("raw_attempts")
        ),
    )


def _from_sequential(payload: dict[str, Any], source_path: Path, run_id: str) -> ReportArtifact:
    task_id = payload.get("task_key")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError(f"Missing task_key in {source_path}")
    steps = payload.get("steps")
    if not isinstance(steps, list):
        raise TypeError(f"Missing steps list in {source_path}")
    events = payload.get("events")
    instructions = _instruction_texts(events) if isinstance(events, list) else ()

    if any(
        not isinstance(step, dict) or not isinstance(step.get("step_index"), int)
        for step in steps
    ):
        raise TypeError(f"Invalid step in {source_path}")
    ordered_steps = sorted(steps, key=lambda step: step["step_index"])
    units, image_paths, declared_status, completion_source = _finalize_report_units(
        [
            (
                step.get("content"),
                (
                    source_path.parent / step["image_path"]
                    if isinstance(step.get("image_path"), str)
                    else None
                ),
            )
            for step in ordered_steps
        ],
        payload.get("status") if isinstance(payload.get("status"), str) else None,
    )
    return ReportArtifact(
        task_id=task_id,
        run_id=run_id,
        prompt=str(payload.get("task_prompt") or ""),
        units=units,
        image_paths=image_paths,
        source_path=source_path,
        declared_status=declared_status,
        completion_source=completion_source,
        harness_instructions=instructions,
    )


def load_report_artifacts(
    repository_root: Path,
    run_id: str,
    source: str,
    report_root: Path | None = None,
    run_registry_path: Path | None = None,
    run_dir: Path | None = None,
) -> dict[str, ReportArtifact]:
    if source == "bagel":
        return {
            task_id: _from_bagel(artifact)
            for task_id, artifact in load_artifacts(
                repository_root, run_id, run_registry_path, run_dir
            ).items()
        }
    if source == "sequential":
        if report_root is None:
            raise ValueError("--report-root is required for sequential reports")
        artifacts: dict[str, ReportArtifact] = {}
        for source_path in sorted(report_root.glob("*/run.json")):
            payload = read_json(source_path)
            if payload.get("framework") not in {
                "sequential-minimal-v1", "umm-unified-minimal-v1",
            }:
                raise ValueError(f"Unexpected framework in {source_path}")
            artifact = _from_sequential(payload, source_path, run_id)
            if artifact.task_id in artifacts:
                raise ValueError(f"Duplicate task artifact: {artifact.task_id}")
            artifacts[artifact.task_id] = artifact
        return artifacts
    raise ValueError(f"Unknown report source: {source}")


# Long enough that a short answer sharing domain words with a prompt is not
# treated as a restatement; short prompts still qualify when nearly every
# output token matches the prompt.
_ECHO_MIN_RUN = 12
_ECHO_MIN_LEFTOVER = 20


def _is_input_echo(content: Any, prompt: str) -> bool:
    """Report whether a unit is nothing but a restatement of one input prompt."""
    text = re.sub(r"\s+", " ", str(content or "")).strip().lower()
    if not text:
        return False
    source = re.sub(r"\s+", " ", str(prompt or "")).strip().lower()
    for length in range(
        min(len(source), len(text)), _ECHO_MIN_RUN - 1, -5
    ):
        chunk = source[:length]
        if chunk and chunk in text:
            text = text.replace(chunk, " ", 1)
    leftover = re.sub(r"[^a-z0-9]+", " ", text).strip()
    if len(leftover) < _ECHO_MIN_LEFTOVER:
        return True
    source_tokens = re.findall(r"[a-z0-9]+", source)
    output_tokens = re.findall(r"[a-z0-9]+", text)
    if len(output_tokens) < _ECHO_MIN_RUN or len(output_tokens) > len(source_tokens) + 3:
        return False
    return SequenceMatcher(
        None, source_tokens, output_tokens, autojunk=False
    ).ratio() >= 0.85


def input_echo_units(artifact: ReportArtifact) -> list[dict[str, Any]]:
    """List units whose text only restates the task prompt or harness input.

    Recorded so a reader can judge output quality; it never changes a gate.
    """
    sources = (
        (("task_prompt", artifact.prompt),)
        + tuple(
            ("harness_instructions", instruction)
            for instruction in artifact.harness_instructions
        )
    )
    echoes = []
    for unit in artifact.units:
        content = unit.get("content")
        echo_source = next(
            (
                label
                for label, source in sources
                if _is_input_echo(content, source)
            ),
            None,
        )
        if echo_source is not None:
            echoes.append({
                "unit_index": unit.get("unit_index"),
                "echo_source": echo_source,
                "content": str(content or "")[:200],
            })
    return echoes


def prose_completion_tasks(
    artifacts: dict[str, ReportArtifact],
) -> list[str]:
    """Tasks whose completion was declared in prose rather than with the marker."""
    return sorted(
        task_id
        for task_id, artifact in artifacts.items()
        if artifact.completion_source == "prose"
    )


def report_contract(artifact: ReportArtifact) -> dict[str, Any]:
    units = artifact.units
    indexes = [unit.get("unit_index") for unit in units]
    expected = list(range(1, len(units) + 1))
    content_count = sum(bool(str(unit.get("content") or "").strip()) for unit in units)
    unit_image_values = [unit.get("image_path") for unit in units]
    existing = sum(
        isinstance(path, str) and Path(path).is_file() for path in unit_image_values
    )
    image_hashes = [
        sha256_file(Path(path))
        for path in unit_image_values
        if isinstance(path, str) and Path(path).is_file()
    ]
    unique_images = len(set(image_hashes))
    minimum_units = int(REPORT_ORDERED_UNITS_GATE.get("minimum_units", 1))
    hard_minimum = int(REPORT_ORDERED_UNITS_GATE.get("hard_minimum_units", 1))
    max_units = int(REPORT_ORDERED_UNITS_GATE.get("max_units", 0)) or None
    declared = artifact.declared_status in set(
        REPORT_TERMINAL_STATUS_GATE["accepted_statuses"]
    )
    # A run that produced units but never announced completion is treated as
    # having finished at its last step. The fact is kept for the warning; only
    # the gate outcome changes.
    inferred_completion = bool(units) and not declared
    if inferred_completion and REPORT_TERMINAL_STATUS_GATE.get("infer_from_units"):
        declared = True
    warnings = []
    if units and len(units) < minimum_units:
        warnings.append("fewer_than_minimum_units")
    if inferred_completion:
        warnings.append("completion_inferred_from_last_step")
    checks = [
        {
            "criterion": REPORT_ORDERED_UNITS_GATE["criterion"],
            "passed": (
                len(units) >= hard_minimum
                and indexes == expected
                and (max_units is None or len(units) <= max_units)
            ),
            "observation": (
                f"unit_count={len(units)} (minimum {minimum_units}, "
                f"hard minimum {hard_minimum}, "
                f"maximum {max_units if max_units else 'none'}); "
                f"unit_indexes={indexes}"
            ),
        },
        {
            "criterion": REPORT_READER_CONTENT_GATE["criterion"],
            "passed": bool(units) and content_count == len(units),
            "observation": f"content_units={content_count}/{len(units)}",
        },
        {
            "criterion": REPORT_IMAGE_PAIRING_GATE["criterion"],
            "passed": bool(units)
            and len(unit_image_values) == len(units)
            and unique_images == len(units)
            and existing == len(units),
            "observation": (
                f"images={existing}/{len(units)}; unique_hashes={unique_images}"
            ),
        },
    ]
    diagnostics = [
        {
            "criterion": REPORT_TERMINAL_STATUS_GATE["criterion"],
            "passed": declared,
            "observation": (
                f"declared_status={artifact.declared_status!r}"
                + (
                    "; completion inferred from the last step"
                    if inferred_completion else ""
                )
            ),
        },
    ]
    return {
        "status": "PASS" if all(check["passed"] for check in checks) else "FAIL",
        "checks": checks,
        "diagnostics": diagnostics,
        "warnings": warnings,
    }
