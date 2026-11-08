from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from http.client import IncompleteRead, RemoteDisconnected
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from core.hashing import sha256_file, sha256_text
from core.policy_registry import (
    EVALUATION_JUDGE_CALL_EVENT,
    JUDGE_JSON_EXTRACTION_POLICY,
    REPORT_FAILURE_PROVENANCE_GATE,
    REPORT_REQUEST_LAYERS,
    validate_operation_provenance,
)

from .rubric import TIER_TO_ANCHOR, dimensions_for


class JudgeProviderError(RuntimeError):
    pass


class JudgeArtifactInputError(JudgeProviderError):
    """A prepared packet cannot be safely resolved against its transferred run."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


def _sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        default=str,
    )
    return sha256_text(payload)


def _recovery_operation(policy_id, operation, before, after, sequence):
    return {
        "sequence": sequence,
        "policy_id": policy_id,
        "operation": operation,
        "input_sha256": _sha256(before),
        "output_sha256": _sha256(after),
    }


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("provider") != "chat_completions":
        raise ValueError("Only the chat_completions adapter is implemented.")
    base_url = config.get("base_url") or os.environ.get(
        config.get("base_url_env", "UMM_JUDGE_BASE_URL"), "")
    if not base_url:
        raise ValueError("The judge endpoint is not configured")
    config["base_url"] = base_url
    config["api_key_env"] = config.get("api_key_env", "UMM_JUDGE_API_KEY")
    return config


def _data_url(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _resolve_run_relative_file(run_dir: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise JudgeArtifactInputError(
            "Portable judge packet contains an invalid artifact path.",
            code="INVALID_ARTIFACT_PATH",
        )
    relative = Path(value)
    if relative.is_absolute():
        raise JudgeArtifactInputError(
            f"Portable judge packet contains an absolute artifact path: {value}",
            code="ABSOLUTE_ARTIFACT_PATH",
        )
    root = run_dir.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise JudgeArtifactInputError(
            f"Artifact path escapes the transferred run directory: {value}",
            code="ARTIFACT_PATH_ESCAPE",
        ) from exc
    if not resolved.is_file():
        raise JudgeArtifactInputError(
            f"Transferred artifact file is missing: {value}",
            code="ARTIFACT_FILE_MISSING",
        )
    return resolved


def _resolve_image_inputs(
    prepared: dict[str, Any], run_dir: Path | None, max_images: int,
) -> list[Path]:
    raw_paths = prepared.get("image_paths")
    if not isinstance(raw_paths, list) or not raw_paths:
        raise JudgeArtifactInputError(
            "Visual judge packet contains no image paths.",
            code="IMAGE_PATHS_MISSING",
        )
    if len(raw_paths) > max_images:
        raise JudgeArtifactInputError(
            f"Visual judge packet has {len(raw_paths)} images but the configured limit is {max_images}.",
            code="IMAGE_LIMIT_EXCEEDED",
        )

    schema = prepared.get("artifact_path_schema")
    if schema is None:
        # Legacy same-machine packets remain readable, but missing files are no longer
        # omitted from the judge call.
        resolved = [Path(value).resolve() for value in raw_paths if isinstance(value, str)]
        if len(resolved) != len(raw_paths):
            raise JudgeArtifactInputError(
                "Legacy judge packet contains a non-string image path.",
                code="INVALID_ARTIFACT_PATH",
            )
        missing = [str(path) for path in resolved if not path.is_file()]
        if missing:
            raise JudgeArtifactInputError(
                f"Judge image files are missing: {missing}",
                code="ARTIFACT_FILE_MISSING",
            )
        return resolved

    if schema != "umm-run-relative-v1" or prepared.get("artifact_path_base") != "run_dir":
        raise JudgeArtifactInputError(
            f"Unsupported artifact path schema: {schema!r}",
            code="UNSUPPORTED_ARTIFACT_PATH_SCHEMA",
        )
    if run_dir is None:
        raise JudgeArtifactInputError(
            "Portable visual judge packets require --run-dir.",
            code="RUN_DIR_REQUIRED",
        )
    metadata = prepared.get("image_artifacts")
    if not isinstance(metadata, list) or len(metadata) != len(raw_paths):
        raise JudgeArtifactInputError(
            "Portable judge packet image metadata does not match image_paths.",
            code="IMAGE_METADATA_MISMATCH",
        )

    resolved = []
    for path_value, artifact in zip(raw_paths, metadata, strict=True):
        if not isinstance(artifact, dict) or artifact.get("path") != path_value:
            raise JudgeArtifactInputError(
                "Portable judge packet image metadata path mismatch.",
                code="IMAGE_METADATA_MISMATCH",
            )
        expected_hash = artifact.get("sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise JudgeArtifactInputError(
                f"Portable judge packet has an invalid image SHA256 for {path_value!r}.",
                code="INVALID_IMAGE_SHA256",
            )
        image_path = _resolve_run_relative_file(run_dir, path_value)
        actual_hash = sha256_file(image_path)
        if actual_hash != expected_hash:
            raise JudgeArtifactInputError(
                f"Transferred image SHA256 mismatch for {path_value}: expected {expected_hash}, got {actual_hash}",
                code="IMAGE_SHA256_MISMATCH",
            )
        resolved.append(image_path)
    return resolved


def _verify_source_artifact(prepared: dict[str, Any], run_dir: Path | None) -> None:
    """Verify the non-visual source file bound into every portable packet."""
    if prepared.get("artifact_path_schema") is None:
        return
    if (prepared.get("artifact_path_schema") != "umm-run-relative-v1"
            or prepared.get("artifact_path_base") != "run_dir"):
        raise JudgeArtifactInputError(
            "Portable judge packet has an unsupported source artifact schema.",
            code="UNSUPPORTED_ARTIFACT_PATH_SCHEMA",
        )
    if run_dir is None:
        raise JudgeArtifactInputError(
            "Portable judge packets require --run-dir.", code="RUN_DIR_REQUIRED",
        )
    source = prepared.get("source_artifact")
    expected_hash = prepared.get("source_artifact_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise JudgeArtifactInputError(
            "Portable judge packet has an invalid source artifact SHA256.",
            code="INVALID_SOURCE_ARTIFACT_SHA256",
        )
    path = _resolve_run_relative_file(run_dir, source)
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise JudgeArtifactInputError(
            f"Transferred source artifact SHA256 mismatch for {source}: "
            f"expected {expected_hash}, got {actual_hash}",
            code="SOURCE_ARTIFACT_SHA256_MISMATCH",
        )


def _extract_json_text(api_response: dict[str, Any]) -> str:
    try:
        message = api_response["choices"][0]["message"]
        content = message["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise JudgeProviderError("Response does not contain Chat Completions message content.") from exc
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        for field in ("text", "content", "output_text"):
            value = content.get(field)
            if isinstance(value, str):
                return value
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                for field in ("text", "content", "output_text"):
                    value = part.get(field)
                    if isinstance(value, str):
                        parts.append(value)
                        break
        if parts:
            return "".join(parts)
    content_type = type(content).__name__
    reasoning = message.get("reasoning") if isinstance(message, dict) else None
    if isinstance(reasoning, str) and "{" in reasoning and "}" in reasoning:
        return reasoning
    raise JudgeProviderError(f"Unsupported Chat Completions content type: {content_type}.")


def _close_unbalanced(text: str) -> str | None:
    """Append the closing brackets a truncated JSON body is missing."""
    stack: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if not stack:
                return None
            stack.pop()
    if in_string or not stack:
        return None
    closer = {"{": "}", "[": "]"}
    return text + "".join(closer[bracket] for bracket in reversed(stack))


def parse_json_object(
    text: str, *, return_provenance: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], list[dict[str, Any]]]:
    cleaned = text.strip()
    operations = []
    if cleaned.startswith("```json"):
        before = cleaned
        cleaned = cleaned[7:]
        operations.append(_recovery_operation(
            JUDGE_JSON_EXTRACTION_POLICY["policy_id"], "judge_markdown_fence",
            before, cleaned, len(operations) + 1,
        ))
    elif cleaned.startswith("```"):
        before = cleaned
        cleaned = cleaned[3:]
        operations.append(_recovery_operation(
            JUDGE_JSON_EXTRACTION_POLICY["policy_id"], "judge_markdown_fence",
            before, cleaned, len(operations) + 1,
        ))
    before_suffix = cleaned
    cleaned = cleaned.removesuffix("```")
    cleaned = cleaned.strip()
    if cleaned != before_suffix:
        operations.append(_recovery_operation(
            JUDGE_JSON_EXTRACTION_POLICY["policy_id"], "judge_markdown_fence",
            before_suffix, cleaned, len(operations) + 1,
        ))
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as first_error:
        # Repair truncated closers first so a complete body that only lost its
        # final brace survives; the caller still schema-checks the result.
        closed = _close_unbalanced(cleaned)
        closed_value = None
        if closed is not None:
            try:
                closed_value = json.loads(closed)
            except json.JSONDecodeError:
                closed_value = None
        if closed_value is not None:
            value = closed_value
            operations.append(_recovery_operation(
                JUDGE_JSON_EXTRACTION_POLICY["policy_id"],
                "judge_stack_closure", cleaned, closed, len(operations) + 1,
            ))
        else:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start < 0 or end <= start:
                raise JudgeProviderError(
                    "Judge did not return a valid JSON object.") from first_error
            try:
                extracted = cleaned[start : end + 1]
                value = json.loads(extracted)
                operations.append(_recovery_operation(
                    JUDGE_JSON_EXTRACTION_POLICY["policy_id"],
                    "judge_object_extraction", cleaned, extracted,
                    len(operations) + 1,
                ))
            except json.JSONDecodeError as exc:
                raise JudgeProviderError(
                    "Judge did not return a valid JSON object.") from exc
    if not isinstance(value, dict):
        raise JudgeProviderError("Judge JSON response must be an object.")
    if return_provenance:
        return value, operations
    return value


def _validate_verdicts(payload: dict[str, Any], prepared: dict[str, Any]) -> list[dict[str, Any]]:
    verdicts = payload.get("verdicts")
    if not isinstance(verdicts, dict):
        raise JudgeProviderError("Judge JSON is missing verdicts.")
    expected = dimensions_for(
        prepared["layer"],
        prepared.get("criteria"),
    )
    if set(verdicts) != set(expected):
        raise JudgeProviderError(
            f"Judge verdict metrics must be exactly {sorted(expected)}, found {sorted(verdicts)}."
        )
    for metric, expected_dimensions in expected.items():
        values = verdicts.get(metric)
        if not isinstance(values, dict) or set(values) != set(expected_dimensions):
            raise JudgeProviderError(
                f"{metric} verdict criteria must be exactly {list(expected_dimensions)}."
            )
        for criterion, result in values.items():
            if not isinstance(result, dict):
                raise JudgeProviderError(f"{metric}.{criterion} must be an object.")
            if result.get("tier") not in TIER_TO_ANCHOR:
                raise JudgeProviderError(
                    f"{metric}.{criterion}.tier must be MET, PARTIAL, or NOT_MET."
                )
            evidence = result.get("evidence")
            if not isinstance(evidence, str) or len(evidence.strip()) < 8:
                raise JudgeProviderError(
                    f"{metric}.{criterion}.evidence must be a concrete observation."
                )
    return []


class ChatCompletionsJudge:
    def __init__(self, config: dict[str, Any], run_dir: Path | None = None) -> None:
        self.config = config
        self.run_dir = Path(run_dir).resolve() if run_dir is not None else None
        key_name = config.get("api_key_env", "UMM_JUDGE_API_KEY")
        self.api_key = os.environ.get(key_name)
        if not self.api_key:
            raise JudgeProviderError(f"Missing API key environment variable: {key_name}")

    def judge(self, prepared: dict[str, Any]) -> dict[str, Any]:
        layer = prepared.get("layer")
        if layer not in REPORT_REQUEST_LAYER_SET:
            raise JudgeProviderError(f"Unsupported report layer: {layer!r}")
        _verify_source_artifact(prepared, self.run_dir)
        section = self.config[
            "text_judge" if layer in REPORT_TEXT_LAYERS else "vision_judge"
        ]
        user_content: str | list[dict[str, Any]]
        if layer in {"visual_atomic", "report_interleaved_atomic"}:
            max_images = int(section.get("max_images_per_task", 8))
            image_paths = _resolve_image_inputs(prepared, self.run_dir, max_images)
            user_content = [{"type": "text", "text": prepared["user_prompt"]}]
            for image_path in image_paths:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _data_url(image_path),
                            "detail": section.get("image_detail", "high"),
                        },
                    }
                )
        else:
            user_content = prepared["user_prompt"]

        request_body: dict[str, Any] = {
            "model": section["model"],
            "messages": [
                {"role": "system", "content": prepared["system_prompt"]},
                {"role": "user", "content": user_content},
            ],
            "temperature": section.get("temperature", 0),
            "max_tokens": section.get("max_tokens", 2400),
        }
        if isinstance(section.get("reasoning"), dict):
            request_body["reasoning"] = section["reasoning"]
        elif section.get("reasoning_effort"):
            request_body["reasoning_effort"] = section["reasoning_effort"]
        if isinstance(section.get("thinking"), dict):
            request_body["thinking"] = section["thinking"]
        if section.get("json_mode", True):
            request_body["response_format"] = {"type": "json_object"}

        base_url = self.config["base_url"].rstrip("/")
        request_path = self.config.get("request_path", "/chat/completions")
        url = base_url + request_path
        raw_response = None
        content = None
        parser_operations = []
        parsed = None
        try:
            raw_response = self._post(url, request_body)
            content = _extract_json_text(raw_response)
            parsed, parser_operations = parse_json_object(
                content, return_provenance=True,
            )
            schema_operations = _validate_verdicts(parsed, prepared)
        except BaseException as error:
            if not hasattr(error, "raw_response"):
                error.raw_response = raw_response
            if not hasattr(error, "raw_content"):
                error.raw_content = content
            failed_operations = [
                {**operation, "sequence": sequence}
                for sequence, operation in enumerate([
                    *parser_operations,
                    *((parsed or {}).get("_recovery_provenance", {}).get(
                        "operations", []
                    ) if isinstance(parsed, dict) else []),
                ], 1)
            ]
            validate_operation_provenance(failed_operations)
            error.recovery_provenance = {
                "operations": failed_operations,
                "semantic_values_added": False,
            }
            raise
        combined_operations = [
            {**operation, "sequence": sequence}
            for sequence, operation in enumerate(
                [*parser_operations, *schema_operations], 1
            )
        ]
        validate_operation_provenance(combined_operations)
        return {
            "policy_id": EVALUATION_JUDGE_CALL_EVENT["policy_id"],
            "status": "ok",
            "request_id": prepared["request_id"],
            "task_id": prepared["task_id"],
            "run_id": prepared["run_id"],
            "family": prepared["family"],
            "layer": layer,
            "criteria": prepared.get("criteria", []),
            "request_content_sha256": prepared.get("request_content_sha256"),
            "judge_model": section["model"],
            "judge_label": section.get("label", section["model"]),
            "judged_at": datetime.now(timezone.utc).isoformat(),
            "judgment": parsed,
            "raw_input": prepared,
            "raw_response": raw_response,
            "raw_content": content,
            "recovery_provenance": {
                "operations": combined_operations,
                "semantic_values_added": False,
            },
            "usage": raw_response.get("usage", {}),
        }

    def _post(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        encoded = json.dumps(body).encode("utf-8")
        request = Request(
            url,
            data=encoded,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        retries = int(self.config.get("max_retries", 3))
        timeout = int(self.config.get("timeout_seconds", 120))
        for attempt in range(retries + 1):
            try:
                with urlopen(request, timeout=timeout) as response:
                    raw_body = response.read().decode("utf-8")
                    try:
                        return json.loads(raw_body)
                    except json.JSONDecodeError as exc:
                        error = JudgeProviderError(
                            "Judge HTTP response body is not valid JSON."
                        )
                        error.raw_content = raw_body
                        raise error from exc
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                retryable = exc.code in {408, 409, 429, 500, 502, 503, 504}
                error = JudgeProviderError(f"HTTP {exc.code}: {detail}")
            except (URLError, TimeoutError, OSError, IncompleteRead, RemoteDisconnected) as exc:
                retryable = True
                error = JudgeProviderError(f"Network error: {exc}")
            if not retryable or attempt == retries:
                raise error
            # Use 8/16/32 second backoff to wait beyond short rate-limit bursts
            # while keeping transient-error retries bounded.
            time.sleep(min(60, 2 ** (attempt + 2)))
        raise JudgeProviderError("Unreachable retry state.")


REPORT_TEXT_LAYERS = frozenset(
    {"report_text_atomic", "content_quality_atomic"})
REPORT_REQUEST_LAYER_SET = frozenset(REPORT_REQUEST_LAYERS.values())


def usage_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Total provider-reported token usage across judge calls."""
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    token_record_count = 0
    for result in results:
        usage = result.get("usage")
        if not isinstance(usage, dict):
            continue
        saw_tokens = False
        for field in totals:
            value = usage.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[field] += value
                saw_tokens = True
        if saw_tokens:
            token_record_count += 1
    return {
        **totals,
        "token_record_count": token_record_count,
    }


def judge_label_for(layer: str, config: dict[str, Any]) -> str:
    """The judge label a call for ``layer`` will carry under ``config``."""
    if layer not in REPORT_REQUEST_LAYER_SET:
        raise ValueError(f"Unsupported report layer: {layer!r}")
    section = config["text_judge" if layer in REPORT_TEXT_LAYERS else "vision_judge"]
    return section.get("label", section["model"])


def judge_requests(
    prepared_requests: list[dict[str, Any]],
    config: dict[str, Any],
    workers: int,
    repeat: int,
    run_dir: Path | None = None,
    completed_calls: set[tuple[str, str, int]] | None = None,
) -> list[dict[str, Any]]:
    """Judge every (request, repeat) pair not already covered."""
    if workers <= 0 or repeat <= 0:
        raise ValueError("judge workers and repeat must be positive")
    expanded: list[dict[str, Any]] = []
    for prepared in prepared_requests:
        label = judge_label_for(prepared["layer"], config)
        for repeat_index in range(repeat):
            if completed_calls and (
                    prepared["request_id"], label, repeat_index + 1) in completed_calls:
                continue
            copy = dict(prepared)
            copy["request_id"] = f"{prepared['request_id']}:r{repeat_index + 1}"
            copy["repeat_index"] = repeat_index + 1
            expanded.append(copy)

    def call(prepared: dict[str, Any]) -> dict[str, Any]:
        try:
            result = ChatCompletionsJudge(config, run_dir=run_dir).judge(prepared)
            result["repeat_index"] = prepared["repeat_index"]
            return result
        # Provider and transport libraries expose heterogeneous exception types;
        # every call failure must be serialized instead of dropping the task.
        except Exception as exc:  # noqa: BLE001
            section = config[
                "text_judge" if prepared["layer"] in REPORT_TEXT_LAYERS
                else "vision_judge"
            ]
            return {
                "policy_id": EVALUATION_JUDGE_CALL_EVENT["policy_id"],
                "status": "error",
                "request_id": prepared["request_id"],
                "task_id": prepared["task_id"],
                "run_id": prepared["run_id"],
                "family": prepared["family"],
                "layer": prepared["layer"],
                "request_content_sha256": prepared.get("request_content_sha256"),
                "judge_model": section["model"],
                "judge_label": section.get("label", section["model"]),
                "repeat_index": prepared["repeat_index"],
                "error": str(exc),
                "raw_input": prepared,
                "failure_provenance": {
                    "policy_id": EVALUATION_JUDGE_CALL_EVENT["policy_id"],
                    "gate_id": (
                        REPORT_FAILURE_PROVENANCE_GATE["gate_id"]
                        if isinstance(exc, JudgeArtifactInputError)
                        else None
                    ),
                    "kind": (
                        "path_error"
                        if isinstance(exc, JudgeArtifactInputError)
                        else "judge_call_failure"
                    ),
                    "classification": (
                        "artifact_input_failure"
                        if isinstance(exc, JudgeArtifactInputError)
                        else "judge_call_failure"
                    ),
                    "code": getattr(exc, "code", None),
                    "resolved": False,
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                },
                "raw_response": getattr(exc, "raw_response", None),
                "raw_content": getattr(exc, "raw_content", None),
                "recovery_provenance": getattr(
                    exc, "recovery_provenance", {"operations": []}
                ),
                "judged_at": datetime.now(timezone.utc).isoformat(),
            }

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(call, prepared) for prepared in expanded]
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda item: item["request_id"])
