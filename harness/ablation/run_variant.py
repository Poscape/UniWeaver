#!/usr/bin/env python3
"""Run one ablation variant through the production runner."""
from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ablation import manifest as M
from ablation import patches
from ablation import profiles as P


def _split(argv: list[str]) -> tuple[str | None, list[str]]:
    """Peel our own flags off; everything else goes to the production runner."""
    variant = None
    describe = False
    rest: list[str] = []
    index = 0
    while index < len(argv):
        item = argv[index]
        if item in ("--variant", "--ablation-variant"):
            variant = argv[index + 1] if index + 1 < len(argv) else None
            index += 2
            continue
        if item == "--describe":
            describe = True
            index += 1
            continue
        if item == "--list-variants":
            print("\n".join(f"{p.id}\t{p.title}" for p in P.ALL))
            raise SystemExit(0)
        rest.append(item)
        index += 1
    if describe:
        # --describe must not require the rest of a run's arguments.
        if variant is None:
            raise SystemExit("--describe needs --variant")
        print(json.dumps(patches.describe(variant), ensure_ascii=False,
                         indent=2))
        raise SystemExit(0)
    if variant is None:
        raise SystemExit("--variant is required (see --list-variants)")
    return variant, rest


def main(argv: list[str] | None = None) -> int:
    variant, forwarded = _split(list(sys.argv[1:] if argv is None else argv))
    assert variant is not None
    profile = P.profile(variant)
    print(f"[ablation] variant {profile.id}: {profile.title}", flush=True)

    output_dir = None
    for index, item in enumerate(forwarded):
        if item == "--output-dir" and index + 1 < len(forwarded):
            output_dir = Path(forwarded[index + 1])
    patches.apply(variant)

    # ``run_umm.py`` creates and owns the output root. Record a sibling pending
    # manifest before launch, then write the final manifest once the runner owns
    # the root.
    payload = None
    if output_dir is not None:
        payload = M.build(variant, patches.describe(variant),
                          patches.applied(), forwarded)
        pending = M.write_pending(output_dir, payload)
        print(f"[ablation] {variant}: pending manifest -> {pending}", flush=True)
        if payload["release_dirty"]:
            print("[ablation] WARNING: release worktree is dirty; the recorded "
                  "release_commit does not fully describe this run", flush=True)

    try:
        sys.argv = ["run_umm.py", *forwarded]
        runpy.run_module("baselines.run_umm", run_name="__main__")
    finally:
        if output_dir is not None and payload is not None:
            payload["evaluation_package"] = M.evaluation_package_record(
                output_dir)
            path = M.write(output_dir, payload)
            M.clear_pending(output_dir)
            print(f"[ablation] {variant}: manifest -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
