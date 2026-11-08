#!/usr/bin/env bash
# Run the paper ablations with BAGEL on the 120-task UniWeaver benchmark.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${WORK_ROOT:-}" ]]; then
  WORK_ROOT="$PROJECT_ROOT/output_run"
  if [[ ! -d "$WORK_ROOT/models" ]]; then
    echo "WORK_ROOT is unset and $WORK_ROOT holds no models/ directory;" >&2
    echo "set WORK_ROOT to the parent of repos/, models/, and runs/." >&2
    exit 2
  fi
fi
if [[ -z "${FORMAL_UNIWEAVER_ROOT:-}" ]]; then
  echo "Set FORMAL_UNIWEAVER_ROOT to the full BAGEL run." >&2
  exit 2
fi

RESUME="${RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_abl}"
ABLATION_DATASET="${ABLATION_DATASET:-bench}"
ABLATION_VARIANTS="${ABLATION_VARIANTS:-F1 F2 F4 F3 O32 F5 F5b}"
SMOKE_TASK_KEY="${SMOKE_TASK_KEY:-task_001}"
BENCHMARK_FILE="${BENCHMARK_FILE:-${PROJECT_ROOT}/datasets/benchmarks/uniweaver_bench/bench.json}"

for flag_name in RESUME DRY_RUN; do
  flag_value="${!flag_name}"
  if [[ "$flag_value" != "0" && "$flag_value" != "1" ]]; then
    echo "$flag_name must be 0 or 1" >&2
    exit 2
  fi
done
if [[ "$ABLATION_DATASET" != "bench" ]]; then
  echo "ABLATION_DATASET must be bench, got '$ABLATION_DATASET'" >&2
  exit 2
fi
if [[ ! -d "$FORMAL_UNIWEAVER_ROOT" ]]; then
  echo "formal UniWeaver run not found: $FORMAL_UNIWEAVER_ROOT" >&2
  exit 2
fi

read -r -a variants <<<"$ABLATION_VARIANTS"
if [[ "${#variants[@]}" -eq 0 ]]; then
  echo "ABLATION_VARIANTS is empty" >&2
  exit 2
fi
for variant in "${variants[@]}"; do
  case "$variant" in
    O32|F1|F2|F3|F4|F5|F5b) ;;
    *) echo "unknown ablation variant: $variant" >&2; exit 2 ;;
  esac
done

ablation_root="${ABLATION_OUTPUT_ROOT:-${WORK_ROOT}/runs/abl_uniweaver_120_${RUN_TAG}}"
smoke_root="${ABLATION_SMOKE_ROOT:-${WORK_ROOT}/runs/abl_smoke_${RUN_TAG}}"
runs_dir="${WORK_ROOT}/runs"
LAUNCHER_LOG="${LAUNCHER_LOG:-${WORK_ROOT}/launcher_${RUN_TAG}.log}"
CHAIN_STATUS="${CHAIN_STATUS:-${WORK_ROOT}/chain_status_${RUN_TAG}.tsv}"

if [[ "$RESUME" != "1" ]]; then
  for root in "$ablation_root" "$smoke_root"; do
    if [[ -e "$root" ]]; then
      echo "Refusing to reuse output root: $root" >&2
      exit 2
    fi
  done
elif [[ ! -d "$ablation_root" ]]; then
  echo "RESUME=1 but no ablation output root exists: $ablation_root" >&2
  exit 2
fi

if [[ "$DRY_RUN" != "1" && "${REQUIRED_GPUS:-1}" != "0" ]] \
    && command -v nvidia-smi >/dev/null 2>&1; then
  visible_gpus="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  if [[ -z "$visible_gpus" || "$visible_gpus" -lt "${REQUIRED_GPUS:-1}" ]]; then
    echo "launcher needs ${REQUIRED_GPUS:-1} visible GPU(s), nvidia-smi reports ${visible_gpus:-0}" >&2
    exit 2
  fi
fi

PYTHON_BAGEL="${PYTHON_BAGEL:-python3}"
if ! command -v "$PYTHON_BAGEL" >/dev/null 2>&1 && [[ ! -x "$PYTHON_BAGEL" ]]; then
  echo "BAGEL Python executable not found: $PYTHON_BAGEL" >&2
  exit 2
fi
export PYTHONPATH="$PROJECT_ROOT"
export PYTHONDONTWRITEBYTECODE=1

common_args=(--model-path "$WORK_ROOT/models/BAGEL-7B-MoT")
if [[ -n "${BAGEL_SFT_CHECKPOINT:-}" ]]; then
  common_args+=(--sft-checkpoint "$BAGEL_SFT_CHECKPOINT")
fi

if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$runs_dir"
  : >"$CHAIN_STATUS"
  exec 3>&1 4>&2
  exec > >(tee -a "$LAUNCHER_LOG") 2>&1
fi

record_status() {
  [[ "$DRY_RUN" == "1" ]] && return 0
  local detail="${4:-}"
  detail="${detail//$'\n'/ }"
  detail="${detail//$'\r'/ }"
  detail="${detail//$'\t'/ }"
  printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$detail" >>"$CHAIN_STATUS"
}

run_backend_preflight() {
  if ! "$PYTHON_BAGEL" "$PROJECT_ROOT/scripts/check_backend_ready.py" \
      --backend bagel --require-flash-attn "${common_args[@]}"; then
    record_status bagel preflight FAIL not_ready
    return 1
  fi
  record_status bagel preflight PASS ready
}

run_variant_step() {
  local variant="$1"
  local dataset="$2"
  local root="$3"
  local output_dir="$root/$variant/bagel"
  local log_path="$root/logs/$variant.bagel.log"
  local resume=0
  local -a command=(
    "$PYTHON_BAGEL" "$PROJECT_ROOT/ablation/run_variant.py"
    --variant "$variant"
    --execution online --protocol uniweaver --dataset "$dataset"
    --backend bagel --device cuda
    --model-call-timeout-seconds 3600
    --backend-startup-timeout-seconds 3600
    --task-timeout-seconds 21600
    --max-runtime-conditions 2
    --max-runtime-steps 8
    --max-checks-per-step 3
    --max-steps 8
    --output-dir "$output_dir"
    "${common_args[@]}"
  )
  if [[ "$dataset" == "single" ]]; then
    command+=(--task-key "$SMOKE_TASK_KEY")
  else
    command+=(--benchmark-file "$BENCHMARK_FILE")
  fi
  if [[ "$RESUME" == "1" && -d "$output_dir" ]]; then
    resume=1
    command+=(--resume)
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=0 %s\n' "${command[*]}"
    return 0
  fi
  mkdir -p "$root/logs"
  if [[ "$resume" == "1" ]]; then
    CUDA_VISIBLE_DEVICES=0 "${command[@]}" >>"$log_path" 2>&1
  else
    CUDA_VISIBLE_DEVICES=0 "${command[@]}" >"$log_path" 2>&1
  fi
}

smoke_produced_step() {
  local root="$1"
  local variant="$2"
  [[ -n "$(find "$root/$variant/bagel" -type f -path '*/steps/*.png' \
      -print -quit 2>/dev/null)" ]]
}

formal_progress() {
  "$PYTHON_BAGEL" - "$1" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    print("MISSING summary.json")
    raise SystemExit(3)
summary = json.loads(path.read_text(encoding="utf-8"))
expected = {
    "processed_count": 120,
    "task_count": 120,
    "completed": 120,
    "planner_error": 0,
    "image_error": 0,
    "incomplete": 0,
}
actual = {key: summary.get(key) for key in expected}
print(" ".join(f"{key}={actual[key]}" for key in expected))
raise SystemExit(0 if actual == expected else 4)
PY
}

if [[ "$DRY_RUN" != "1" ]]; then
  if ! PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BAGEL" -m pytest -q -p no:cacheprovider \
      "$PROJECT_ROOT/tests/test_ablation_layer.py"; then
    echo "ablation layer tests failed; refusing to start." >&2
    exit 2
  fi
  if ! "$PYTHON_BAGEL" "$PROJECT_ROOT/ablation/preflight.py" \
      --formal-run-dir "$FORMAL_UNIWEAVER_ROOT" --current-tree \
      --expect-model-path "$WORK_ROOT/models/BAGEL-7B-MoT" \
      --expect-sft-checkpoint "${BAGEL_SFT_CHECKPOINT:-none}"; then
    echo "source, model, or SFT provenance does not match the formal run." >&2
    exit 2
  fi
  if ! run_backend_preflight; then
    echo "BAGEL backend preflight failed." >&2
    exit 1
  fi
else
  echo "=== DRY_RUN: ablation tests, provenance gate, and backend preflight are skipped ==="
fi

echo "=== ABLATION_DATASET=$ABLATION_DATASET variants=${variants[*]} ==="
echo "=== formal root: $ablation_root ==="
echo "=== smoke root:  $smoke_root ==="
echo "=== formal UniWeaver baseline: $FORMAL_UNIWEAVER_ROOT ==="
if [[ "$DRY_RUN" != "1" ]]; then
  echo "=== launcher log: $LAUNCHER_LOG ==="
  echo "=== chain status: $CHAIN_STATUS ==="
fi

overall=0
for variant in "${variants[@]}"; do
  smoke_path="$smoke_root/$variant/bagel"
  formal_path="$ablation_root/$variant/bagel"

  if run_variant_step "$variant" single "$smoke_root"; then
    if [[ "$DRY_RUN" == "1" ]] \
        || smoke_produced_step "$smoke_root" "$variant"; then
      echo "PASS bagel abl-$variant-smoke"
      record_status bagel "abl-$variant-smoke" PASS step_png=present
    else
      echo "FAIL bagel abl-$variant-smoke: no Step PNG under $smoke_path" >&2
      record_status bagel "abl-$variant-smoke" FAIL no_step_png
      record_status bagel "abl-$variant-bench" SKIP smoke_did_not_pass
      overall=1
      continue
    fi
  else
    status=$?
    echo "FAIL bagel abl-$variant-smoke exit=$status" >&2
    record_status bagel "abl-$variant-smoke" FAIL "exit=$status"
    record_status bagel "abl-$variant-bench" SKIP smoke_did_not_pass
    overall=1
    continue
  fi

  if run_variant_step "$variant" "$ABLATION_DATASET" "$ablation_root"; then
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "PASS bagel abl-$variant-bench"
      record_status bagel "abl-$variant-bench" PASS dry_run
      continue
    fi
    if detail="$(formal_progress "$formal_path/summary.json" 2>&1)"; then
      echo "PASS bagel abl-$variant-bench ($detail)"
      record_status bagel "abl-$variant-bench" PASS "exit=0;$detail"
    else
      echo "FAIL bagel abl-$variant-bench: $detail" >&2
      record_status bagel "abl-$variant-bench" FAIL "$detail"
      overall=1
    fi
  else
    status=$?
    echo "FAIL bagel abl-$variant-bench exit=$status" >&2
    record_status bagel "abl-$variant-bench" FAIL "exit=$status"
    overall=1
  fi
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1: no process was started."
  exit "$overall"
fi

echo "Chain status (step, result, detail):"
sort "$CHAIN_STATUS" | awk -F'\t' '{printf "  %-22s %-8s %s\n", $2, $3, $4}'
printf '  summary: pass=%s fail=%s skip=%s\n' \
  "$(awk -F'\t' '$3=="PASS"' "$CHAIN_STATUS" | wc -l | tr -d ' ')" \
  "$(awk -F'\t' '$3=="FAIL"' "$CHAIN_STATUS" | wc -l | tr -d ' ')" \
  "$(awk -F'\t' '$3=="SKIP"' "$CHAIN_STATUS" | wc -l | tr -d ' ')"

echo "Outputs:"
echo "  formal -> $ablation_root"
echo "  smoke  -> $smoke_root"
echo "Launcher log: $LAUNCHER_LOG"
echo "Chain status: $CHAIN_STATUS"

exec 1>&3 2>&4
if [[ -d "$ablation_root" ]]; then
  cp -f "$CHAIN_STATUS" "$ablation_root/chain_status.tsv"
  cp -f "$LAUNCHER_LOG" "$ablation_root/launcher.log"
fi
exit "$overall"
