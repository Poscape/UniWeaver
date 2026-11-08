#!/usr/bin/env python3
"""Check that an ablation run and its baseline use the same production sources."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ablation.manifest import TRACKED_PRODUCTION_FILES, release_commit
from core.hashing import sha256_file


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def find_manifests(root: Path) -> list[Path]:
    return sorted(root.rglob("ablation_manifest.json"))


def find_source_hashes(root: Path) -> dict[str, set[str]]:
    """Collect ``path -> {sha256}`` from the runner's recorded provenance."""
    found: dict[str, set[str]] = {}
    for path in root.rglob("provenance.json"):
        payload = _read_json(path)
        if not isinstance(payload, dict):
            continue
        identity = payload.get("image_execution_identity")
        files = identity.get("files") if isinstance(identity, dict) else None
        if not isinstance(files, list):
            continue
        for entry in files:
            if not isinstance(entry, dict):
                continue
            name = entry.get("path")
            digest = entry.get("sha256")
            if isinstance(name, str) and isinstance(digest, str) and digest:
                found.setdefault(name, set()).add(digest)
    return found


def find_model_binding(root: Path) -> dict:
    """Collect the model/checkpoint facts the source-hash gate cannot see."""
    found: dict[str, set] = {"model_path": set(), "sft_checkpoint": set()}
    for path in root.rglob("provenance.json"):
        payload = _read_json(path)
        if not isinstance(payload, dict):
            continue
        model_path = payload.get("model_path")
        if isinstance(model_path, str) and model_path:
            found["model_path"].add(model_path)
        # ``sft_checkpoint`` is only written when one was used, so a missing
        # key means the run used the base checkpoint (None).
        found["sft_checkpoint"].add(payload.get("sft_checkpoint") or None)
    return {key: value for key, value in found.items() if value}


def _normalise_model_path(value):
    """Compare paths as resolved strings; ``None`` means "no SFT checkpoint"."""
    if value in (None, "", "none", "None"):
        return None
    return str(Path(value).expanduser().resolve())


def find_commit_records(root: Path) -> list[tuple[Path, str]]:
    """Collect (path, commit) from whatever provenance the runner wrote."""
    records = []
    for name in ("run_metadata.json", "provenance.json", "summary.json",
                 "training_provenance.json"):
        for path in root.rglob(name):
            payload = _read_json(path)
            if not isinstance(payload, dict):
                continue
            commit = None
            for key in ("git_commit", "commit", "source_commit",
                        "release_commit"):
                value = payload.get(key)
                if isinstance(value, str) and value:
                    commit = value
                    break
            if commit is None:
                git = payload.get("git")
                if isinstance(git, dict):
                    commit = git.get("commit")
            if isinstance(commit, str) and commit:
                records.append((path, commit))
    return records


def _source_hashes_of_tree() -> dict[str, set[str]]:
    """The sha256 of each tracked production file in this checkout."""
    found: dict[str, set[str]] = {}
    for name in TRACKED_PRODUCTION_FILES:
        path = PROJECT_ROOT / name
        if path.is_file():
            found.setdefault(name, set()).add(sha256_file(path))
    return found


def _report(formal_sources, subject_sources, label, commits, formal_commits):
    """Compare ``subject_sources`` against the formal run's hashes."""
    mismatched: list[str] = []
    missing: list[str] = []
    for name in TRACKED_PRODUCTION_FILES:
        formal_hashes = formal_sources.get(name)
        subject_hash = (subject_sources.get(name) or {None})
        if not formal_hashes or not subject_hash or None in subject_hash:
            missing.append(name)
        elif not (subject_hash & formal_hashes):
            mismatched.append(name)
    inconsistent = sorted(
        name for name in TRACKED_PRODUCTION_FILES
        if len(formal_sources.get(name) or ()) > 1)
    comparison_ok = not mismatched and not missing and not inconsistent

    commit_match = None
    if commits and formal_commits:
        commit_match = commits == formal_commits
    return {
        "subject": label,
        "ablation_commits": sorted(commits),
        "formal_commits": sorted(formal_commits),
        "commit_match": commit_match,
        "commit_note": (
            None if commit_match is not None else
            "production source hashes are the comparison record"),
        "compared_files": list(TRACKED_PRODUCTION_FILES),
        "source_mismatch": mismatched,
        "source_missing": missing,
        "source_inconsistent_within_formal_run": inconsistent,
        "source_match": comparison_ok,
        "match": bool(comparison_ok),
    }


def _print_report(report):
    print(f"compared: {report['subject']}")
    print(f"ablation commit(s): {report['ablation_commits']}")
    print(f"formal   commit(s): {report['formal_commits']}")
    if report["commit_note"]:
        print(f"commit comparison: {report['commit_note']}")
    print(f"production files compared: {len(TRACKED_PRODUCTION_FILES)}")
    if report["source_mismatch"]:
        print(f"MISMATCH on {len(report['source_mismatch'])} file(s): "
              f"{report['source_mismatch']}")
    if report["source_missing"]:
        print(f"MISSING on {len(report['source_missing'])} file(s): "
              f"{report['source_missing']}")
    if report["source_inconsistent_within_formal_run"]:
        print("formal run is internally inconsistent on: "
              f"{report['source_inconsistent_within_formal_run']}")
    print("MATCH: the ablation and the baseline share one production "
          "revision (source hashes are identical)"
          if report["match"] else
          "MISMATCH: do not compare these runs; re-run one side")


def _base_model_facts(model_path: Path) -> dict | None:
    """Names and sizes of the base BAGEL artifacts, or None if unusable."""
    if model_path is None or not model_path.is_dir():
        return None
    facts = {}
    for entry in sorted(model_path.iterdir()):
        if entry.is_file() and not entry.name.startswith("."):
            facts[entry.name] = entry.stat().st_size
    return facts


def _sft_checkpoint_sha256(path: Path) -> str | None:
    """Hash the one file that actually varies between experiments."""
    if path is None or not path.is_file():
        return None
    return sha256_file(path)


def _model_binding_findings(formal_run_dir: Path, args) -> list[str]:
    """Compare the model this run will use with the baseline, by content."""
    local_model_path = _normalise_model_path(
        getattr(args, "expect_model_path", None))
    local_sft_path = _normalise_model_path(
        getattr(args, "expect_sft_checkpoint", None))
    if local_model_path is None and local_sft_path is None:
        return []          # nothing was declared; the gate stays source-only

    binding = find_model_binding(formal_run_dir)
    identity = _read_baseline_model_identity(formal_run_dir)
    recorded_artifacts = (identity or {}).get("artifacts")
    if not isinstance(recorded_artifacts, dict):
        recorded_artifacts = None

    findings: list[str] = []

    # --- base model -------------------------------------------------------
    recorded_paths = sorted(binding.get("model_path") or ())
    if not recorded_paths:
        findings.append("model_path is absent from the baseline provenance")
    elif len(recorded_paths) > 1:
        findings.append("the baseline used more than one base model: "
                        f"{recorded_paths}")
    elif recorded_artifacts is None:
        findings.append("model_identity is absent from the baseline provenance")
    else:
        recorded_base = {name: value
                         for name, value in recorded_artifacts.items()
                         if name != "sft_checkpoint"}
        local_facts = _base_model_facts(Path(local_model_path))
        if local_facts is None:
            # The gate may simply not be able to see the model from where it
            # runs; that is a fact about the gate, not about the model.
            findings.append(
                f"the base model could not be inspected at {local_model_path}; "
                f"the baseline used {recorded_paths[0]}")
        else:
            missing = sorted(set(recorded_base) - set(local_facts))
            extra = sorted(set(local_facts) - set(recorded_base))
            sized = sorted(
                name for name in set(recorded_base) & set(local_facts)
                if recorded_base[name].get("size") != local_facts[name])
            if missing:
                findings.append("base model is missing artifacts the baseline "
                                f"used: {missing}")
            if extra:
                findings.append("base model has artifacts the baseline did not "
                                f"record: {extra}")
            if sized:
                findings.append(
                    "base model artifacts differ in size from the baseline: "
                    f"{[(n, recorded_base[n].get('size'), local_facts[n]) for n in sized]}")

    # --- SFT checkpoint ---------------------------------------------------
    recorded_sft = (recorded_artifacts or {}).get("sft_checkpoint")
    recorded_digest = (recorded_sft or {}).get("sha256")
    local_digest = _sft_checkpoint_sha256(
        Path(local_sft_path) if local_sft_path else None)
    if recorded_digest and recorded_sft.get("size") is not None and local_digest is None:
        findings.append("the baseline applied an SFT checkpoint but this run "
                        "applies none; set BAGEL_SFT_CHECKPOINT to the same file")
    elif local_digest and not recorded_digest:
        findings.append(
            f"SFT checkpoint presence differs from the baseline: this run uses "
            f"{local_sft_path}")
    elif recorded_digest and local_digest and recorded_digest != local_digest:
        findings.append("SFT checkpoint differs from the baseline: sha256 "
                        f"{local_digest[:12]} vs the baseline's "
                        f"{recorded_digest[:12]}")
    return findings


def _read_baseline_model_identity(run_dir: Path) -> dict | None:
    for path in sorted(run_dir.rglob("provenance.json")):
        payload = _read_json(path)
        if not isinstance(payload, dict):
            continue
        identity = payload.get("model_identity")
        if isinstance(identity, dict):
            return identity
    return None


def _check_current_tree(args) -> int:
    """Compare this checkout with the formal baseline before a batch starts."""
    formal_sources = find_source_hashes(args.formal_run_dir)
    commits = {release_commit()} if release_commit() != "unknown" else set()
    tree_sources = _source_hashes_of_tree()
    report = _report(formal_sources, tree_sources, "current tree", commits,
                     {commit for _path, commit in
                      find_commit_records(args.formal_run_dir)})
    findings: list[str] = []
    if not formal_sources:
        report["source_match"] = False
        findings.append(
            "image_execution_identity is absent from the baseline provenance")
    elif not report["source_match"]:
        findings.append(
            "production sources differ from the baseline: "
            f"{report['source_mismatch'] or report['source_missing']}")
    findings.extend(_model_binding_findings(args.formal_run_dir, args))
    report["match"] = not findings
    report["findings"] = findings
    report["model_binding_match"] = not any(
        "SFT checkpoint" in f or "base model" in f for f in findings)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_report(report)
        for finding in findings:
            print(f"GATE FINDING: {finding}")
        if findings:
            print("MISMATCH: refusing to start.")
    return 2 if findings else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-run-dir", required=True, type=Path)
    parser.add_argument("--ablation-run-dir", type=Path)
    parser.add_argument(
        "--current-tree", action="store_true",
        help=(
            "compare this checkout's production sources against the formal run "
            "instead of an ablation run's manifest. Used as the batch "
            "launcher's pre-start gate: if the sources moved since the baseline "
            "was produced, every variant in the batch would be incomparable."
        ))
    parser.add_argument(
        "--expect-model-path", default=None,
        help=(
            "with --current-tree, require the baseline to have used this base "
            "checkpoint directory. Source hashes cannot see a checkpoint swap."
        ))
    parser.add_argument(
        "--expect-sft-checkpoint", default=None,
        help=(
            "with --current-tree, require the baseline to have used the same "
            "SFT checkpoint. Pass the path, or 'none' for the base checkpoint."
        ))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.current_tree:
        return _check_current_tree(args)
    if args.ablation_run_dir is None:
        print("FAIL: --ablation-run-dir is required unless --current-tree is "
              "given", file=sys.stderr)
        return 2

    ablation = find_manifests(args.ablation_run_dir)
    if not ablation:
        print(f"FAIL: no ablation_manifest.json under {args.ablation_run_dir}")
        return 1
    ablation_commits = set()
    ablation_sources: dict[str, str] = {}
    dirty_manifests: list[str] = []
    for path in ablation:
        payload = _read_json(path) or {}
        commit = payload.get("release_commit")
        if isinstance(commit, str):
            ablation_commits.add(commit)
        if payload.get("release_dirty"):
            dirty_manifests.append(str(path))
        recorded = payload.get("production_file_sha256")
        if isinstance(recorded, dict):
            for name, digest in recorded.items():
                if isinstance(name, str) and isinstance(digest, str) and digest:
                    ablation_sources.setdefault(name, digest)
    if not ablation_sources:
        print(f"FAIL: no production_file_sha256 recorded under "
              f"{args.ablation_run_dir}; re-run the variant to get a "
              f"comparable manifest")
        return 1

    formal_sources = find_source_hashes(args.formal_run_dir)
    if not formal_sources:
        print("FAIL: could not find image_execution_identity in the formal run "
              "provenance; point --formal-run-dir at the run root the operator "
              "archived")
        return 1

    formal_records = find_commit_records(args.formal_run_dir)
    report = _report(
        formal_sources,
        {name: {digest} for name, digest in ablation_sources.items()},
        "ablation run", ablation_commits,
        {commit for _path, commit in formal_records})
    report["formal_evidence"] = [str(path)
                                 for path, _commit in formal_records][:5]
    findings: list[str] = []
    if dirty_manifests:
        findings.append("produced from a dirty worktree: "
                        f"{dirty_manifests}")
    if not report["source_match"]:
        findings.append("production sources differ from the baseline")
    report["match"] = not findings
    report["findings"] = findings
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_report(report)
        for finding in findings:
            print(f"GATE FINDING: {finding}")
        if findings:
            print("MISMATCH: do not put this variant in the table.")
    return 2 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
