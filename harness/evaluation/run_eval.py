from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.hashing import sha256_file
from core.serialization import atomic_write_text
from evaluation.uniweaver_eval.data import (
    append_jsonl,
    load_matrix_tasks,
    read_json,
    read_jsonl,
    write_json,
)
from evaluation.uniweaver_eval.provider import (
    judge_requests,
    load_config,
    usage_summary,
)
from evaluation.uniweaver_eval.reports import (
    load_report_artifacts,
    report_contract,
)
from evaluation.uniweaver_eval.rubric import ANCHOR_TO_TIER, anchor_from_judgment

REPOSITORY_ROOT = PROJECT_ROOT


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _json_stdout(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=True, indent=2))


def _base_request_id(request_id: Any) -> str | None:
    """Strip the ``:rN`` suffix a repeat adds, matching scoring's own rule."""
    if not isinstance(request_id, str):
        return None
    return re.sub(r":r\d+$", "", request_id)


def _judgment_call_key(row: dict[str, Any]) -> tuple[str, str, int] | None:
    """Identity of one judge call: which request, which judge, which repeat."""
    base = _base_request_id(row.get("request_id"))
    label = row.get("judge_label")
    repeat_index = row.get("repeat_index")
    if base is None or not isinstance(label, str) or not isinstance(repeat_index, int):
        return None
    return (base, label, repeat_index)


def _merge_judgments(
    previous: list[dict[str, Any]], fresh: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Union two judgment files; the newer row wins for the same call."""
    merged: dict[tuple[str, str, int], dict[str, Any]] = {}
    unkeyed: list[dict[str, Any]] = []
    for row in [*previous, *fresh]:
        key = _judgment_call_key(row)
        if key is None:
            unkeyed.append(row)
        else:
            merged[key] = row
    # Ordered by request_id, which is the order judge_requests emits. That keeps
    # a no-op re-run byte-identical to the file it started from instead of
    # reshuffling rows and changing the recorded judgments hash for nothing.
    return sorted(
        [*merged.values(), *unkeyed],
        key=lambda row: str(row.get("request_id") or ""),
    )


def _coverage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """What this judgment file actually covers, by successful call."""
    by_label: dict[str, set[int]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        label = row.get("judge_label")
        repeat_index = row.get("repeat_index")
        if isinstance(label, str) and isinstance(repeat_index, int):
            by_label.setdefault(label, set()).add(repeat_index)
    return {
        "labels": sorted(by_label),
        "repeats_by_label": {
            label: sorted(repeats) for label, repeats in sorted(by_label.items())
        },
        "repeat_count_by_label": {
            label: len(repeats) for label, repeats in sorted(by_label.items())
        },
    }


def _run_dir(args: argparse.Namespace) -> Path | None:
    """Explicit run output root (contains text/ and images/), e.g. from run_umm."""
    return Path(args.run_dir).resolve() if getattr(args, "run_dir", None) else None


REPORT_METRICS = ("TC", "WQ", "VC", "ITA")
TARGET_FIELDS = ("order", "id", "display_name")


def _metric_file(directory: Path, metric: str, kind: str) -> Path:
    return directory / f"report_{metric.lower()}.{kind}.jsonl"


def _read_optional_jsonl(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.is_file() else []


def _merge_object_rows(
    path: Path,
    rows: list[dict[str, Any]],
    object_id: str,
    sort_key,
) -> None:
    """Replace one object's rows and keep every other object untouched."""
    kept = [
        row for row in _read_optional_jsonl(path)
        if row.get("object_id") != object_id
    ]
    append_jsonl(path, sorted([*kept, *rows], key=sort_key))


def _read_targets(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _write_targets(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "\t".join(TARGET_FIELDS) + "\n"
    payload += "".join(
        "\t".join(str(row[field]) for field in TARGET_FIELDS) + "\n"
        for row in rows
    )
    atomic_write_text(path, payload)


def command_collect_inputs(args: argparse.Namespace) -> None:
    """Write the four scoring inputs; no score is computed here."""
    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out_dir)
    benchmark = (
        Path(args.benchmark_file) if args.benchmark_file
        else run_dir / "benchmark_task_matrix.json"
    )
    if not benchmark.is_file():
        raise FileNotFoundError(
            f"benchmark matrix not found: {benchmark}; pass --benchmark-file"
        )
    tasks = load_matrix_tasks(benchmark)
    artifacts = load_report_artifacts(
        REPOSITORY_ROOT,
        args.run,
        args.source,
        Path(args.report_root) if args.report_root else None,
        Path(args.run_registry) if args.run_registry else None,
        run_dir,
    )
    requests_dir = (
        Path(args.requests_dir) if args.requests_dir
        else run_dir / "evaluation" / "scoring" / "requests"
    )
    judgments_dir = (
        Path(args.judgments_dir) if args.judgments_dir
        else run_dir / "evaluation" / "scoring"
    )

    request_index: dict[str, dict[str, dict[str, Any]]] = {
        metric: {} for metric in REPORT_METRICS
    }
    criteria_rows: list[dict[str, Any]] = []
    for metric in REPORT_METRICS:
        path = _metric_file(requests_dir, metric, "requests")
        if not path.is_file():
            raise FileNotFoundError(f"prepared requests not found: {path}")
        for request in read_jsonl(path):
            task_id = request.get("task_id")
            if task_id not in tasks:
                raise ValueError(
                    f"{path}: request for unknown task {task_id!r}")
            if task_id in request_index[metric]:
                raise ValueError(
                    f"{path}: duplicate request for task {task_id!r}")
            if request.get("metric") != metric:
                raise ValueError(
                    f"{path}: request {request.get('request_id')!r} is "
                    f"metric {request.get('metric')!r}, expected {metric}")
            request_index[metric][task_id] = request
            seen = set()
            for criterion in request.get("criteria") or []:
                identifier = criterion.get("id")
                if not isinstance(identifier, str) or not identifier:
                    raise ValueError(
                        f"{path}: task {task_id} has a criterion without an id")
                if identifier in seen:
                    raise ValueError(
                        f"{path}: task {task_id} repeats criterion {identifier}")
                seen.add(identifier)
                criteria_rows.append({
                    "object_id": args.object_id,
                    "task_id": task_id,
                    "metric": metric,
                    "criterion": identifier,
                    "weight": int(criterion.get("weight", 1)),
                })

    judgment_rows: list[dict[str, Any]] = []
    failed_rows = 0
    for metric in REPORT_METRICS:
        path = _metric_file(judgments_dir, metric, "judgments")
        if not path.is_file():
            raise FileNotFoundError(f"judgments not found: {path}")
        seen_calls = set()
        for row in read_jsonl(path):
            if row.get("status") != "ok":
                failed_rows += 1
                continue
            task_id = row.get("task_id")
            request = request_index[metric].get(task_id)
            if request is None:
                raise ValueError(
                    f"{path}: judgment for task {task_id!r} has no request")
            label = row.get("judge_label")
            repeat = row.get("repeat_index")
            if not isinstance(label, str) or not label:
                raise ValueError(f"{path}: judgment without a judge label")
            if not isinstance(repeat, int) or repeat < 1:
                raise ValueError(
                    f"{path}: judgment for task {task_id!r} has no repeat index")
            call = (task_id, label, repeat)
            if call in seen_calls:
                raise ValueError(
                    f"{path}: duplicate judgment call {call!r}")
            seen_calls.add(call)
            if row.get("request_content_sha256") != request.get(
                    "request_content_sha256"):
                raise ValueError(
                    f"{path}: judgment for task {task_id!r} does not match "
                    "the prepared request")
            if _base_request_id(row.get("request_id")) != request.get(
                    "request_id"):
                raise ValueError(
                    f"{path}: judgment for task {task_id!r} names another "
                    "request")
            judgment = row.get("judgment")
            if not isinstance(judgment, dict):
                raise TypeError(
                    f"{path}: judgment for task {task_id!r} is not an object")
            for criterion in request.get("criteria") or []:
                anchor = anchor_from_judgment(
                    judgment, metric, criterion["id"])
                if anchor is None:
                    continue
                judgment_rows.append({
                    "object_id": args.object_id,
                    "task_id": task_id,
                    "metric": metric,
                    "criterion": criterion["id"],
                    "judge": label,
                    "repeat": repeat,
                    "tier": ANCHOR_TO_TIER[anchor],
                })

    task_rows = []
    for task_id in sorted(tasks):
        artifact = artifacts.get(task_id)
        status = (
            report_contract(artifact)["status"] if artifact is not None
            else "MISSING_REPORT"
        )
        task_rows.append({
            "object_id": args.object_id,
            "task_id": task_id,
            "family": tasks[task_id].family,
            "gate": int(status == "PASS"),
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    _merge_object_rows(
        out_dir / "tasks.jsonl", task_rows, args.object_id,
        lambda row: (row["object_id"], row["task_id"]))
    _merge_object_rows(
        out_dir / "criteria.jsonl", criteria_rows, args.object_id,
        lambda row: (row["object_id"], row["task_id"], row["metric"],
                     row["criterion"]))
    _merge_object_rows(
        out_dir / "judgments.jsonl", judgment_rows, args.object_id,
        lambda row: (row["object_id"], row["task_id"], row["metric"],
                     row["criterion"], row["judge"], row["repeat"]))

    targets_path = out_dir / "targets.tsv"
    previous_targets = {
        row.get("id"): row for row in _read_targets(targets_path)
    }
    targets = [
        row for row in previous_targets.values()
        if row.get("id") != args.object_id
    ]
    order = args.order
    if order is None:
        previous = previous_targets.get(args.object_id) or {}
        order = (
            int(previous["order"]) if previous.get("order")
            else max((int(row["order"]) for row in targets), default=0) + 1
        )
    targets.append({
        "order": order,
        "id": args.object_id,
        "display_name": args.display_name or args.object_id,
    })
    targets.sort(key=lambda row: int(row["order"]))
    _write_targets(targets_path, targets)

    _json_stdout({
        "status": "table_inputs_collected",
        "out_dir": str(out_dir.resolve()),
        "object_id": args.object_id,
        "tasks": len(task_rows),
        "gate_pass": sum(row["gate"] for row in task_rows),
        "criteria": len(criteria_rows),
        "judgments": len(judgment_rows),
        "failed_judgment_rows": failed_rows,
        "note": "No score was computed. Run scoring/scripts/score_table.py.",
    })


def command_prepare_run(args: argparse.Namespace) -> None:
    """Prepare all deterministic UniWeaver-Bench inputs from a transferred run directory."""
    from baselines.umm_evaluation_prepare import prepare_requests

    run_dir = Path(args.run_dir).resolve()
    if (run_dir / "review_variants.json").is_file():
        raise ValueError(
            "UniWeaver parent runs contain two report variants; run prepare-run "
            "separately on no_review/ and reviewed/"
        )
    snapshot_path = run_dir / "benchmark_task_matrix.json"
    if not snapshot_path.is_file():
        raise SystemExit(
            f"benchmark matrix not found: {snapshot_path}\n"
            "A run directory must keep the task matrix it was generated from."
        )
    snapshot = read_json(snapshot_path)
    benchmark_name = snapshot.get("benchmark_name")
    manifest = prepare_requests(run_dir, benchmark_name)
    _json_stdout({
        "status": "run_prepared",
        "run_dir": str(run_dir),
        "evaluation_manifest": str(
            (run_dir / "evaluation" / "scoring" / "manifest.json").resolve()
        ),
        "request_counts": {
            metric: record["request_count"]
            for metric, record in manifest["outputs"].items()
        },
    })


def command_judge(args: argparse.Namespace) -> None:
    request_path = Path(args.requests)
    config_path = Path(args.config)
    prepared = read_jsonl(request_path)
    config = load_config(config_path)
    output = Path(args.output)
    meta_path = output.with_suffix(".meta.json")
    previous: list[dict[str, Any]] = []
    completed: set[tuple[str, str, int]] = set()
    previous_meta: dict[str, Any] = {}
    prior_config_sha256s: list[str] = []
    if output.exists() or meta_path.exists():
        if not args.resume:
            raise FileExistsError(
                f"refusing to append to or overwrite existing judge output: {output}"
            )
        if not output.is_file() or not meta_path.is_file():
            raise ValueError("judge resume requires both judgments and metadata")
        previous_meta = read_json(meta_path)
        # Only the request file must match; config and repeat may differ, since
        # adding a judge or raising the repeat count is exactly that.
        if previous_meta.get("request_sha256") != sha256_file(request_path):
            raise ValueError(
                "judge resume request mismatch: the request file is not the one "
                "this output was built from"
            )
        # Judgments may be appended to, but not edited behind the metadata's
        # back. The hash is rewritten after every pass, so this still catches an
        # external modification while allowing the file to grow.
        recorded_hash = previous_meta.get("judgments_sha256")
        if recorded_hash and recorded_hash != sha256_file(output):
            raise ValueError(
                "judge resume judgments hash does not match metadata: the output "
                "file was modified outside this command"
            )
        previous = read_jsonl(output)
        # Only successful calls are reusable. A recorded transport failure is
        # retried on the next pass: "we tried once and it failed" is not the
        # same as "this call is answered".
        retry_ids = {str(item) for item in (getattr(args, "retry", None) or [])}
        # Only labels this pass can issue are eligible; other judges keep their
        # rows and stay countable as reused.
        covered_labels = {
            section.get("label", section["model"])
            for section in (config["text_judge"], config["vision_judge"])
        }
        for row in previous:
            if row.get("status") != "ok":
                continue
            key = _judgment_call_key(row)
            if key is None:
                continue
            # Only the calls this pass issues get replaced; verdicts of judges it
            # does not cover are kept.
            if (
                retry_ids
                and key[1] in covered_labels
                and (key[0] in retry_ids or row.get("task_id") in retry_ids)
            ):
                continue
            completed.add(key)
        prior_config_sha256s = [
            value for value in (previous_meta.get("config_sha256s") or [])
            if isinstance(value, str)
        ]
        if not prior_config_sha256s and isinstance(
                previous_meta.get("config_sha256"), str):
            prior_config_sha256s = [previous_meta["config_sha256"]]

    results = judge_requests(
        prepared, config, args.workers, args.repeat, run_dir=_run_dir(args),
        completed_calls=completed,
    )
    merged = _merge_judgments(previous, results)
    # Archive exactly the verdicts this pass overwrote: a superseded answer is
    # evidence, not a duplicate.
    replaced = {
        key for key in (_judgment_call_key(row) for row in results) if key is not None
    }
    superseded = [
        row for row in previous if _judgment_call_key(row) in replaced
    ]
    if superseded:
        append_jsonl(output.parent / f"{output.stem}.retired.jsonl", superseded)
    # append_jsonl is an atomic whole-file write, not a shell-style append, so
    # this rewrites the file with the union rather than risking a torn result.
    append_jsonl(output, merged)
    current_config_sha256 = sha256_file(config_path)
    metadata = {
        "created_at": previous_meta.get("created_at")
        or datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "request_path": str(request_path.resolve()),
        "request_sha256": sha256_file(request_path),
        "config_path": str(config_path.resolve()),
        "config_sha256": current_config_sha256,
        # Every config this file has ever been built with, so growing a run
        # across judges stays auditable.
        "config_sha256s": sorted({*prior_config_sha256s, current_config_sha256}),
        "repeat": args.repeat,
        "workers": args.workers,
        "request_count": len(prepared),
        "expanded_call_count": len(merged),
        "calls_this_pass": len(results),
        "calls_reused": len(completed),
        "successful_calls": sum(item.get("status") == "ok" for item in merged),
        "failed_calls": sum(item.get("status") != "ok" for item in merged),
        "judge_labels": sorted(
            {
                item["judge_label"]
                for item in merged
                if isinstance(item.get("judge_label"), str)
            }
        ),
        "judge_models": sorted(
            {
                item["judge_model"]
                for item in merged
                if isinstance(item.get("judge_model"), str)
            }
        ),
        "endpoint": config["base_url"].rstrip("/")
        + config.get("request_path", "/chat/completions"),
        "usage": usage_summary(merged),
        # How much of the intended grid this file covers. A partial run is a
        # legitimate intermediate state; saying so explicitly is what keeps it
        # from being mistaken for a finished one downstream.
        "coverage": _coverage_summary(merged),
    }
    metadata["judgments_sha256"] = sha256_file(output)
    write_json(output.with_suffix(".meta.json"), metadata)
    _json_stdout({"status": "judged", "output": str(output.resolve()), **metadata})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UniWeaver-Bench evaluation runner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_dir = argparse.ArgumentParser(add_help=False)
    run_dir.add_argument(
        "--run-dir",
        help=(
            "Explicit run output root containing text/ and images/; required by "
            "portable visual judge packets after transfer."
        ),
    )

    collect = subparsers.add_parser(
        "collect-inputs",
        help=(
            "Write tasks, criteria, judgments, and targets for the score-table "
            "script; no score is computed"
        ),
        parents=[run_dir],
    )
    collect.add_argument("--run", required=True)
    collect.add_argument(
        "--source", choices=["bagel", "sequential"], required=True)
    collect.add_argument("--report-root")
    collect.add_argument("--run-registry")
    collect.add_argument(
        "--benchmark-file",
        help="default: <run-dir>/benchmark_task_matrix.json",
    )
    collect.add_argument(
        "--requests-dir",
        help="default: <run-dir>/evaluation/scoring/requests",
    )
    collect.add_argument(
        "--judgments-dir",
        help="default: <run-dir>/evaluation/scoring",
    )
    collect.add_argument(
        "--object-id",
        required=True,
        help="row name in the final table, e.g. bagel or uw_bagel",
    )
    collect.add_argument(
        "--display-name",
        help="table label; defaults to --object-id",
    )
    collect.add_argument(
        "--order",
        type=_positive_int,
        help="table row order; defaults to after the existing rows",
    )
    collect.add_argument("--out-dir", required=True)
    collect.set_defaults(func=command_collect_inputs)

    prepare_run = subparsers.add_parser(
        "prepare-run",
        help="Prepare all UniWeaver-Bench report judge packets from a transferred run",
    )
    prepare_run.add_argument("--run-dir", required=True)
    prepare_run.set_defaults(func=command_prepare_run)

    judge = subparsers.add_parser(
        "judge", help="Call a configured judge endpoint", parents=[run_dir]
    )
    judge.add_argument("--requests", required=True)
    judge.add_argument("--config", required=True)
    judge.add_argument("--output", required=True)
    judge.add_argument("--workers", type=_positive_int, default=1)
    judge.add_argument("--repeat", type=_positive_int, default=2)
    judge.add_argument(
        "--retry", nargs="+", default=[], metavar="ID",
        help=(
            "Request id or task id to judge again even though a successful "
            "judgment already exists. Everything else is reused; the superseded "
            "verdict is kept beside the output."
        ),
    )
    judge.add_argument(
        "--resume", action="store_true",
        help="Reuse only a complete hash-matched immutable judge stage.",
    )
    judge.set_defaults(func=command_judge)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
