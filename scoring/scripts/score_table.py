#!/usr/bin/env python3
"""Rebuild the score table from collected task, criterion, and judgment records.

The script is the only place that turns judge verdicts into a score. Evaluation
runs collect four inputs; they are scored here.

- All three judges score every criterion, each with the same number of passes.
- Each judge's passes reduce to a median without rounding.
- Medians one tier apart score the lower tier; more than one tier apart
  scores the criterion 0 inside its denominator.
- A dimension shares the Total's denominator (gate-passing tasks) and its
  sqrt(gate pass rate) factor, so the four weighted components sum to Total.
- A family Total is its gate-passing mean times sqrt(gate pass rate); a
  category and the overall row average those family Totals with equal weight.
"""

import argparse
import collections
import csv
import json
import math
import os
import re
import statistics

TIER = {"NOT_MET": 0, "PARTIAL": 1, "MET": 2}
METRICS = ("TC", "WQ", "VC", "ITA")
WEIGHTS = {"TC": 0.45, "WQ": 0.10, "VC": 0.35, "ITA": 0.10}
# Rows extracted from older tables keep their original metric names.
METRIC_ALIASES = {"RTC": "TC", "CDQ": "WQ", "VATC": "VC", "RIC": "ITA"}
GROUPS = [
    ("Procedures & Instructions", [
        "Recipe & Food Preparation",
        "Craft & DIY",
        "Assembly & Physical Setup",
        "Care & Maintenance",
        "Before-and-After Transformation",
    ]),
    ("Comparison & Recommendation", [
        "Place / Experience Comparison",
        "Product / Object Comparison",
        "Product Use & Lifestyle Guide",
        "Fashion & Styling",
        "Interior & Lifestyle Inspiration",
    ]),
    ("Observation & Description", [
        "Photography / Visual Showcase",
        "Nature / Species / Environment",
        "Architecture & Design Style",
        "Seasonal / Atmosphere / Environmental Change",
        "Travel & Place Guide",
    ]),
    ("Explanation & Storytelling", [
        "Collection / Visual Catalog Report",
        "Object History & Evolution",
        "Culture & Tradition",
        "Event / Experience Story",
        "Narrative / Slice of Life",
    ]),
]
CATEGORIES = {family: name for name, members in GROUPS for family in members}
CATEGORY_ORDER = tuple(name for name, _ in GROUPS)


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_targets(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def metric_name(value):
    return METRIC_ALIASES.get(value, value)


def panel_judges(judgments):
    judges = sorted({row["judge"] for row in judgments})
    if len(judges) != 3:
        raise SystemExit(
            f"expected exactly three judges, found {len(judges)}: {judges}")
    return tuple(judges)


def score_criteria(criteria, judgments, judges, min_repeats=1):
    passes = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in judgments:
        key = (row["object_id"], row["task_id"],
               metric_name(row["metric"]), row["criterion"])
        passes[key][row["judge"]].append(row["tier"])

    audit = []
    gained = collections.defaultdict(lambda: collections.defaultdict(float))
    weight = collections.defaultdict(lambda: collections.defaultdict(float))
    complete = collections.defaultdict(lambda: collections.defaultdict(lambda: True))
    for row in criteria:
        object_id, task_id = row["object_id"], row["task_id"]
        metric, criterion = metric_name(row["metric"]), row["criterion"]
        key = (object_id, task_id, metric, criterion)
        criterion_weight = float(row["weight"])
        weight[(object_id, task_id)][metric] += criterion_weight

        tiers = passes[key]
        medians = {}
        missing = []
        for judge in judges:
            values = tiers.get(judge) or []
            if len(values) < min_repeats:
                missing.append(judge)
                continue
            medians[judge] = statistics.median(TIER[tier] for tier in values)

        anchor = 0.0
        spread = None
        if missing:
            reason = "MISSING_JUDGE"
        else:
            panel = [medians[judge] for judge in judges]
            spread = max(panel) - min(panel)
            if spread > 1:
                reason = "TWO_TIER_DISAGREEMENT"
            else:
                anchor = min(panel)
                reason = "TIER_USABLE"
        tier_usable = reason == "TIER_USABLE"
        if tier_usable:
            gained[(object_id, task_id)][metric] += criterion_weight * anchor
        else:
            complete[(object_id, task_id)][metric] = False

        audit.append({
            "object_id": object_id,
            "task_id": task_id,
            "family": row.get("family", ""),
            "metric": metric,
            "criterion": criterion,
            "weight": f"{criterion_weight:.4f}",
            "judge_tiers": ";".join(
                f"{judge}={','.join(tiers.get(judge) or ['MISSING'])}"
                for judge in judges),
            "judge_medians": ";".join(
                f"{judge}={medians[judge]:.1f}" for judge in judges
                if judge in medians),
            "missing_judges": ",".join(missing),
            "median_spread": "" if spread is None else f"{spread:.1f}",
            "tier_usable": int(tier_usable),
            "tier_reason": reason,
            "anchor": f"{anchor:.2f}",
            "credit": f"{anchor / 2.0:.4f}",
        })
    return audit, gained, weight, complete


def score_tasks(tasks, gained, weight, complete):
    rows = []
    for task in tasks:
        key = (task["object_id"], task["task_id"])
        totals = weight.get(key) or {}
        values = {
            metric: (100.0 * gained[key][metric] / totals[metric] / 2.0
                     if totals.get(metric) else 0.0)
            for metric in METRICS
        }
        ok = {metric: bool(totals.get(metric))
              and complete.get(key, {}).get(metric, True)
              for metric in METRICS}
        quality = sum(WEIGHTS[metric] * values[metric] for metric in METRICS)
        rows.append({
            "object_id": task["object_id"],
            "task_id": task["task_id"],
            "family": task["family"],
            "category": CATEGORIES[task["family"]],
            "gate": int(task["gate"]),
            **values,
            "task_quality": quality,
            **{f"{metric}_ok": int(ok[metric]) for metric in METRICS},
        })
    return rows


def score_families(task_rows, display_names):
    rows = []
    for object_id, tasks in itertools_group(task_rows):
        families = collections.defaultdict(list)
        for task in tasks:
            families[task["family"]].append(task)
        for family in sorted(families):
            entries = families[family]
            passed = [task for task in entries if task["gate"]]
            validity = len(passed) / len(entries)
            row = {
                "object_id": object_id,
                "display_name": display_names[object_id],
                "family": family,
                "category": CATEGORIES[family],
                "task_count": len(entries),
                "gate_pass_count": len(passed),
                "validity_rate": f"{validity:.4f}",
                "sqrt_validity": f"{math.sqrt(validity):.4f}",
            }
            for metric in METRICS:
                gate_values = [task[metric] for task in passed]
                gate_mean = statistics.mean(gate_values) if gate_values else 0.0
                row[f"{metric}_aligned"] = (
                    f"{gate_mean * math.sqrt(validity):.4f}")
            quality = statistics.mean(task["task_quality"] for task in passed) \
                if passed else 0.0
            row["Total_aligned"] = f"{quality * math.sqrt(validity):.4f}"
            rows.append(row)
    return rows


def itertools_group(task_rows):
    grouped = collections.OrderedDict()
    for task in task_rows:
        grouped.setdefault(task["object_id"], []).append(task)
    return grouped.items()


def score_table(task_rows, family_rows, targets):
    by_object = collections.defaultdict(list)
    for task in task_rows:
        by_object[task["object_id"]].append(task)
    fam_view = collections.defaultdict(dict)
    for row in family_rows:
        fam_view[row["object_id"]][row["family"]] = {
            "Total": float(row["Total_aligned"]),
            **{
                metric: float(row[f"{metric}_aligned"])
                for metric in METRICS
            },
        }

    table = []
    for target in targets:
        object_id = target["id"]
        tasks = by_object[object_id]
        row = {
            "order": int(target["order"]),
            "object_id": object_id,
            "display_name": target["display_name"],
        }
        for name, members in GROUPS + [("Overall", None)]:
            selected = tasks if members is None else [
                task for task in tasks if task["family"] in members]
            families = sorted({task["family"] for task in selected})
            for metric in METRICS:
                row[f"{name}|{metric}"] = (
                    round(statistics.mean(
                        fam_view[object_id][family][metric]
                        for family in families), 4)
                    if families else 0.0)
            row[f"{name}|Total"] = (
                round(statistics.mean(
                    fam_view[object_id][family]["Total"]
                    for family in families), 4)
                if families else 0.0)
        table.append(row)
    return table


def check_tex(table, path):
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    text = "\n".join(
        line for line in text.splitlines()
        if not line.lstrip().startswith("%"))
    block = re.search(
        r"\\begin\{tabular\}\{l\|ccccc.*?\\end\{tabular\}", text, re.DOTALL)
    if not block:
        raise SystemExit(f"{path}: main table not found")
    expected = []
    for line in block.group(0).splitlines():
        line = line.strip()
        if not line.startswith("&"):
            continue
        line = re.sub(r"\\\\\[-?\d+pt\]$", "", line)
        line = re.sub(r"\\\\$", "", line)
        cells = [
            re.sub(r"\\tiny", "", re.sub(
                r"\\gain\{[^}]*\}", "", re.sub(
                    r"\\textbf\{([^}]*)\}", r"\1", re.sub(
                        r"\\underline\{([^}]*)\}", r"\1", cell)))).strip()
            for cell in line.split("&")[1:]
        ]
        if len(cells) == 5 and all(re.fullmatch(r"\d+\.\d", cell) for cell in cells):
            expected.append([float(cell) for cell in cells])
    if len(expected) != len(table) * 5:
        raise SystemExit(
            f"{path}: expected {len(table) * 5} table rows, parsed {len(expected)}")

    names = [name for name, _ in GROUPS] + ["Overall"]
    mismatches = []
    for index, row in enumerate(table):
        for group_index, name in enumerate(names):
            actual = [round(row[f"{name}|{dim}"], 1)
                      for dim in ("TC", "WQ", "VC", "ITA", "Total")]
            wanted = expected[index * 5 + group_index]
            for column, (got, want) in enumerate(zip(actual, wanted)):
                if abs(got - want) > 0.001:
                    mismatches.append(
                        (row["object_id"], name, column, want, got))
    if mismatches:
        for item in mismatches[:10]:
            print("mismatch", item)
        raise SystemExit(f"{len(mismatches)} cells differ from {path}")
    print(f"check: {len(table) * 25} cells match {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--tasks", default=None,
        help="override <inputs>/tasks.jsonl")
    parser.add_argument(
        "--criteria", default=None,
        help="override <inputs>/criteria.jsonl")
    parser.add_argument(
        "--judgments", default=None,
        help="override <inputs>/judgments.jsonl")
    parser.add_argument(
        "--min-repeats", type=int, default=2,
        help=(
            "passes a judge must contribute before its median counts; "
            "defaults to 2, the paper protocol"
        ))
    parser.add_argument("--check-tex", default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    targets = read_targets(args.targets)
    display_names = {row["id"]: row["display_name"] for row in targets}
    tasks = read_jsonl(args.tasks or os.path.join(args.inputs, "tasks.jsonl"))
    criteria = read_jsonl(
        args.criteria or os.path.join(args.inputs, "criteria.jsonl"))
    judgments = read_jsonl(
        args.judgments or os.path.join(args.inputs, "judgments.jsonl"))
    unknown = sorted({row["family"] for row in tasks} - set(CATEGORIES))
    if unknown:
        raise SystemExit(
            "tasks.jsonl uses families outside the UniWeaver-Bench taxonomy: "
            + ", ".join(unknown))
    family_of = {(row["object_id"], row["task_id"]): row["family"] for row in tasks}
    for row in criteria:
        row["family"] = family_of[(row["object_id"], row["task_id"])]

    judges = panel_judges(judgments)
    audit, gained, weight, complete = score_criteria(
        criteria, judgments, judges, args.min_repeats)
    task_rows = score_tasks(tasks, gained, weight, complete)
    family_rows = score_families(task_rows, display_names)
    table = score_table(task_rows, family_rows, targets)

    table_columns = list(table[0])
    write_tsv(os.path.join(args.out_dir, "criterion_audit.tsv"), audit,
              list(audit[0]))
    write_tsv(os.path.join(args.out_dir, "task_scores.tsv"), task_rows,
              list(task_rows[0]))
    write_tsv(os.path.join(args.out_dir, "family_scores.tsv"), family_rows,
              list(family_rows[0]))
    write_tsv(os.path.join(args.out_dir, "main_table.tsv"), table, table_columns)
    write_tsv(os.path.join(args.out_dir, "main_table_rounded.tsv"), [
        {key: (f"{value:.1f}" if key in table_columns[3:] else value)
         for key, value in row.items()} for row in table], table_columns)
    with open(os.path.join(args.out_dir, "main_table.jsonl"), "w",
              encoding="utf-8") as handle:
        for row in table:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary_rows = []
    for target in targets:
        object_id = target["id"]
        tasks_of = [row for row in task_rows if row["object_id"] == object_id]
        audit_of = [row for row in audit if row["object_id"] == object_id]
        summary_rows.append({
            "order": int(target["order"]),
            "object_id": object_id,
            "display_name": target["display_name"],
            "task_count": len(tasks_of),
            "gate_pass_count": sum(row["gate"] for row in tasks_of),
            "gate_fail_count": sum(1 for row in tasks_of if not row["gate"]),
            "criterion_count": len(audit_of),
            "tier_usable_criterion_count": sum(
                row["tier_usable"] for row in audit_of),
            "tier_rejected_criterion_count": sum(
                1 for row in audit_of if not row["tier_usable"]),
        })
    write_tsv(os.path.join(args.out_dir, "object_summary.tsv"), summary_rows,
              list(summary_rows[0]))

    reasons = collections.Counter(row["tier_reason"] for row in audit)
    with open(os.path.join(args.out_dir, "score_summary.json"), "w",
              encoding="utf-8") as handle:
        json.dump({
            "rule": {
                "judges": list(judges),
                "repeat_aggregation": (
                    f"median of at least {args.min_repeats} passes, no rounding"
                ),
                "cross_judge": (
                    "lower median at one tier apart, more than one tier apart "
                    "scores 0"
                ),
                "dimension": (
                    "gate-passing mean, more than one tier apart at 0, times "
                    "sqrt(gate pass rate)"
                ),
                "total": "family gate-passing mean times sqrt(gate pass rate), equal family weight",
            },
            "tasks": len(task_rows),
            "families": len(family_rows),
            "criteria": len(audit),
            "tier_reason_counts": dict(reasons),
        }, handle, indent=2, sort_keys=True)
        handle.write("\n")

    if args.check_tex:
        check_tex(table, args.check_tex)


if __name__ == "__main__":
    main()
