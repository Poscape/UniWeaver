"""Build the goal-only UniWeaver-10K task pool described in the paper."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.text_normalization import normalize_prompt
from sft_prepare_data.trajectory_providers import ApiStageSender

AXES = ("topics", "audiences", "situations", "constraints", "outcomes")
DIFFICULTIES = ("easy", "medium", "hard")
JSON_ARRAY_KEYS = ("result", "items", "data", "verdicts", "candidates")
PRESCRIPTIVE_PATTERNS = (
    r"\b\d+\s+(?:images?|pictures?|illustrations?|photos?)\b",
    r"\b(?:storyboard|shot list|panel layout|output outline)\b",
    r"\b(?:image-by-image|text-image mapping|one image per)\b",
)


def prompt_key(text: str) -> str:
    return hashlib.sha256(normalize_prompt(text).encode("utf-8")).hexdigest()


def has_prescriptive_structure(text: str) -> bool:
    normalized = normalize_prompt(text)
    return any(re.search(pattern, normalized) for pattern in PRESCRIPTIVE_PATTERNS)


def seed_combinations(family: dict) -> list[dict]:
    axes = []
    for axis in AXES:
        values = family.get(axis)
        if not isinstance(values, list) or not values:
            raise ValueError(f"{family.get('name')!r} needs a nonempty {axis} list")
        axes.append(values)
    return [
        {
            "task_family": family["name"],
            "topics": values[0],
            "audience": values[1],
            "situation": values[2],
            "constraint": values[3],
            "outcome": values[4],
        }
        for values in itertools.product(*axes)
    ]


def _parse_json_array(text: str) -> list:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("["), candidate.rfind("]")
        if start < 0 or end <= start:
            raise ValueError("model response is not a JSON array")
        value = json.loads(candidate[start:end + 1])
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in JSON_ARRAY_KEYS:
            if isinstance(value.get(key), list):
                return value[key]
        list_values = [item for item in value.values() if isinstance(item, list)]
        if len(list_values) == 1:
            return list_values[0]
    raise ValueError("model response does not contain a JSON array")


def generation_prompt(family: str, seeds: list[dict]) -> str:
    return f"""Generate one natural user request for each seed in the same order.
The request must state only the user's goal. Do not prescribe an image count,
storyboard, output outline, or explicit text-image mapping. Keep the request
useful for an interleaved text-image answer.

Task family: {family}
Seeds:
{json.dumps(seeds, ensure_ascii=False)}

Return exactly one JSON object with a `result` array of {len(seeds)} strings."""


def filter_prompt(family: str, candidates: list[str]) -> str:
    return f"""Filter these candidate requests for task-family consistency,
naturalness, multimodal usefulness, and goal-only format validity.
Assign one difficulty: easy, medium, or hard.

Task family: {family}
Candidates:
{json.dumps(candidates, ensure_ascii=False)}

Return one JSON object with a `result` array in candidate order. Each item must be:
{{"family_consistent": true, "natural": true, "multimodal_useful": true,
"format_valid": true, "difficulty": "easy|medium|hard"}}."""


def generate_candidates(
    families: list[dict],
    generate: Callable[[str, int], str],
    *,
    raw_per_family: int = 1600,
    batch_size: int = 20,
) -> list[dict]:
    candidates = []
    for family in families:
        name = family["name"]
        combinations = seed_combinations(family)
        generated = []
        position = 0
        while len(generated) < raw_per_family:
            batch = [
                {
                    **combinations[(position + offset) % len(combinations)],
                    "variation": (position + offset) // len(combinations),
                }
                for offset in range(min(batch_size, raw_per_family - len(generated)))
            ]
            position += len(batch)
            values = _parse_json_array(generate(
                generation_prompt(name, batch), 4096))
            if len(values) != len(batch) or not all(
                    isinstance(value, str) and value.strip() for value in values):
                raise ValueError(f"{name}: generation returned the wrong shape")
            generated.extend(value.strip() for value in values)
        candidates.extend(
            {"task_family": name, "user_prompt": value}
            for value in generated
        )
    return candidates


def filter_candidates(
    candidates: list[dict],
    filter_batch: Callable[[str, int], str],
    *,
    batch_size: int = 20,
) -> list[dict]:
    kept, seen = [], set()
    by_family = defaultdict(list)
    for item in candidates:
        by_family[item["task_family"]].append(item)
    for family, items in sorted(by_family.items()):
        for offset in range(0, len(items), batch_size):
            batch = items[offset:offset + batch_size]
            local = [
                item for item in batch
                if not has_prescriptive_structure(item["user_prompt"])
                and prompt_key(item["user_prompt"]) not in seen
            ]
            if not local:
                continue
            verdicts = _parse_json_array(filter_batch(
                filter_prompt(family, [item["user_prompt"] for item in local]),
                4096))
            if len(verdicts) != len(local):
                raise ValueError(f"{family}: filter returned the wrong shape")
            for item, verdict in zip(local, verdicts):
                key = prompt_key(item["user_prompt"])
                difficulty = verdict.get("difficulty")
                if (
                    key in seen
                    or difficulty not in DIFFICULTIES
                    or not all(verdict.get(field) is True for field in (
                        "family_consistent", "natural",
                        "multimodal_useful", "format_valid",
                    ))
                ):
                    continue
                seen.add(key)
                kept.append({**item, "difficulty": difficulty})
    return kept


def stratify(
    candidates: list[dict],
    *,
    per_family: int = 500,
) -> list[dict]:
    easy = per_family // 4
    hard = per_family // 4
    medium = per_family - easy - hard
    target = {"easy": easy, "medium": medium, "hard": hard}
    by_family = defaultdict(list)
    for item in candidates:
        by_family[item["task_family"]].append(item)
    selected = []
    for family, items in sorted(by_family.items()):
        buckets = defaultdict(list)
        for item in items:
            buckets[item["difficulty"]].append(item)
        chosen = []
        for difficulty in DIFFICULTIES:
            ordered = sorted(
                buckets[difficulty], key=lambda item: prompt_key(item["user_prompt"]))
            if len(ordered) < target[difficulty]:
                raise ValueError(
                    f"{family}: need {target[difficulty]} {difficulty} candidates, "
                    f"found {len(ordered)}")
            chosen.extend(ordered[:target[difficulty]])
        slug = re.sub(r"[^a-z0-9]+", "_", family.casefold()).strip("_")
        for index, item in enumerate(chosen, 1):
            task_id = f"uw10k_{slug}_{index:03d}"
            selected.append({
                **item,
                "task_id": task_id,
                "task_key": task_id,
                "task_prompt": item["user_prompt"],
            })
    return selected


def write_task_pool(tasks: Iterable[dict], output: Path) -> None:
    records = list(tasks)
    payload = {
        "schema_version": "uniweaver-10k-task-pool-v1",
        "task_count": len(records),
        "tasks": records,
    }
    if not payload["task_count"]:
        raise ValueError("refusing to write an empty task pool")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api", default="deepseek")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--key")
    parser.add_argument("--base-url")
    parser.add_argument("--raw-per-family", type=int, default=1600)
    parser.add_argument("--per-family", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()

    families = json.loads(args.families.read_text(encoding="utf-8"))
    if not isinstance(families, list) or len(families) != 20:
        parser.error("--families must contain the 20 paper task families")
    names = [family.get("name") for family in families]
    if len(set(names)) != 20 or not all(isinstance(name, str) and name for name in names):
        parser.error("family names must be 20 nonempty unique strings")

    sender = ApiStageSender(args.api, args.model, args.key, args.base_url)

    def generate(prompt: str, max_tokens: int) -> str:
        return sender(
            prompt, label="task_generation", max_tokens=max_tokens,
            do_sample=True, temperature=0.7,
        )

    def filter_batch(prompt: str, max_tokens: int) -> str:
        return sender(
            prompt, label="task_filter", max_tokens=max_tokens,
            do_sample=False, temperature=0,
        )

    candidates = generate_candidates(
        families, generate,
        raw_per_family=args.raw_per_family, batch_size=args.batch_size,
    )
    filtered = filter_candidates(
        candidates, filter_batch, batch_size=args.batch_size,
    )
    selected = stratify(filtered, per_family=args.per_family)
    counts = Counter(item["task_family"] for item in selected)
    if set(counts.values()) != {args.per_family} or len(counts) != 20:
        raise ValueError("selected task pool does not contain 500 tasks per family")
    write_task_pool(selected, args.output)
    print(f"wrote {len(selected)} tasks to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
