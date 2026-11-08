#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${RUN_DIR:?Set RUN_DIR to one downloaded backend run directory.}"
: "${REPORT_JUDGE_CONFIGS:?Set REPORT_JUDGE_CONFIGS to the three report judge JSON files.}"

PYTHON_EVAL="${PYTHON_EVAL:-python}"
JUDGE_WORKERS="${JUDGE_WORKERS:-8}"
RESUME_EVAL="${RESUME_EVAL:-1}"
REPORT_REPEAT="${REPORT_REPEAT:-2}"

case "$REPORT_REPEAT" in
  2) ;;
  *) echo "REPORT_REPEAT must be 2 for the paper protocol, got '$REPORT_REPEAT'" >&2; exit 2 ;;
esac
case "$RESUME_EVAL" in
  0|1) ;;
  *) echo "RESUME_EVAL must be 0 or 1" >&2; exit 2 ;;
esac
RUN_DIR="$(cd "$RUN_DIR" && pwd)"
RUN_ID="$(basename "$RUN_DIR")"
EVAL_DIR="$RUN_DIR/evaluation/scoring"
export PYTHONPATH="$PROJECT_ROOT"

if [[ ! -f "$EVAL_DIR/manifest.json" ]]; then
  echo "Preparing scoring packets in $EVAL_DIR"
  "$PYTHON_EVAL" "$PROJECT_ROOT/evaluation/run_eval.py" \
    prepare-run --run-dir "$RUN_DIR"
fi

PROTOCOL="$("$PYTHON_EVAL" -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["protocol"])' \
  "$EVAL_DIR/manifest.json")"

read -r -a report_configs <<<"$REPORT_JUDGE_CONFIGS"
if [[ "${#report_configs[@]}" -ne 3 ]]; then
  echo "REPORT_JUDGE_CONFIGS must contain exactly three JSON files" >&2
  exit 2
fi
for report_config in "${report_configs[@]}"; do
  if [[ ! -f "$report_config" ]]; then
    echo "report judge config not found: $report_config" >&2
    exit 2
  fi
done

"$PYTHON_EVAL" - "${report_configs[@]}" <<'PY'
import json
import sys

labels = []
for path in sys.argv[1:]:
    config = json.load(open(path, encoding="utf-8"))
    text = config.get("text_judge", {}).get("label")
    vision = config.get("vision_judge", {}).get("label")
    if not text or text != vision:
        raise SystemExit(
            f"{path}: text_judge and vision_judge must use the same label"
        )
    labels.append(text)
if len(labels) != 3 or len(set(labels)) != 3:
    raise SystemExit("report judge labels must be three unique model names")
PY

report_workers=()
for index in "${!report_configs[@]}"; do
  worker_var="REPORT_JUDGE_WORKERS_$((index + 1))"
  workers="${!worker_var:-$JUDGE_WORKERS}"
  if [[ ! "$workers" =~ ^[1-9][0-9]*$ ]]; then
    echo "$worker_var must be a positive integer, got '$workers'" >&2
    exit 2
  fi
  report_workers+=("$workers")
done

report_source_args=(--source bagel)
if [[ "$PROTOCOL" == "minimal" ]]; then
  report_source_args=(--source sequential --report-root "$RUN_DIR/tasks")
elif [[ "$PROTOCOL" != "uniweaver" ]]; then
  echo "Unsupported prepared protocol: $PROTOCOL" >&2
  exit 2
fi

resume_args=()
if [[ "$RESUME_EVAL" == "1" ]]; then
  resume_args=(--resume)
fi

for index in "${!report_configs[@]}"; do
  report_config="${report_configs[$index]}"
  workers="${report_workers[$index]}"
  for metric in tc wq vc ita; do
    "$PYTHON_EVAL" "$PROJECT_ROOT/evaluation/run_eval.py" judge \
      --run-dir "$RUN_DIR" \
      --requests "$EVAL_DIR/requests/report_${metric}.requests.jsonl" \
      --config "$report_config" --repeat "$REPORT_REPEAT" \
      --workers "$workers" \
      "${resume_args[@]}" \
      --output "$EVAL_DIR/report_${metric}.judgments.jsonl"
  done
done

INPUT_DIR="${INPUT_DIR:-$EVAL_DIR/table_inputs}"
TABLE_DIR="${TABLE_DIR:-$EVAL_DIR/table}"
OBJECT_ID="${OBJECT_ID:-$RUN_ID}"

"$PYTHON_EVAL" "$PROJECT_ROOT/evaluation/run_eval.py" collect-inputs \
  --run-dir "$RUN_DIR" --run "$RUN_ID" \
  "${report_source_args[@]}" \
  --object-id "$OBJECT_ID" \
  --display-name "${DISPLAY_NAME:-$OBJECT_ID}" \
  --out-dir "$INPUT_DIR"

"$PYTHON_EVAL" "$PROJECT_ROOT/../scoring/scripts/score_table.py" \
  --inputs "$INPUT_DIR" --targets "$INPUT_DIR/targets.tsv" \
  --out-dir "$TABLE_DIR" --min-repeats "$REPORT_REPEAT"

echo "Table inputs: $INPUT_DIR"
echo "Score table: $TABLE_DIR/main_table.tsv"
