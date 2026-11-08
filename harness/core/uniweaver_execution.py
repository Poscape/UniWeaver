"""Shared Condition/Step stage orchestration for UniWeaver execution."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypeVar

Item = TypeVar("Item")
Result = TypeVar("Result")
Condition = TypeVar("Condition")
Step = TypeVar("Step")


def run_parallel_stage(
    items: Iterable[Item],
    worker: Callable[[Item], Result],
    *,
    max_workers: int,
) -> Iterator[Result]:
    """Run one stage in parallel and yield results as workers complete."""
    pending = list(items)
    if not pending:
        return
    worker_count = min(max(1, max_workers), len(pending))
    if worker_count == 1:
        for item in pending:
            yield worker(item)
        return
    with ThreadPoolExecutor(
        max_workers=worker_count
    ) as executor:
        futures = [executor.submit(worker, item) for item in pending]
        failures = []
        for future in as_completed(futures):
            try:
                yield future.result()
            except BaseException as error:  # noqa: BLE001
                failures.append(error)
        if failures:
            primary = failures[0]
            if len(failures) > 1 and hasattr(primary, "add_note"):
                primary.add_note(
                    f"parallel stage collected {len(failures)} worker failures"
                )
            raise primary


def run_uniweaver_stages(
    *,
    condition_items: Iterable[Condition] = (),
    condition_worker: Callable[[Condition], Condition] | None = None,
    step_items: Iterable[Step] = (),
    step_worker: Callable[[Step], Step] | None = None,
    max_workers: int,
    on_condition: Callable[[Condition], None] | None = None,
    on_step: Callable[[Step], None] | None = None,
) -> None:
    """Run Conditions, wait for the barrier, then run Steps in parallel."""
    conditions = list(condition_items)
    if conditions:
        if condition_worker is None:
            raise ValueError("condition_worker is required for Condition items")
        for condition in run_parallel_stage(
            conditions, condition_worker, max_workers=max_workers
        ):
            if on_condition is not None:
                on_condition(condition)

    steps = list(step_items)
    if steps:
        if step_worker is None:
            raise ValueError("step_worker is required for Step items")
        for step in run_parallel_stage(steps, step_worker, max_workers=max_workers):
            if on_step is not None:
                on_step(step)
