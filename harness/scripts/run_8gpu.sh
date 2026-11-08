#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# WORK_ROOT holds repos/, models/, and runs/; the ignored harness/output_run/
# tree is used when it is not set, provided that tree already holds models/.
if [[ -z "${WORK_ROOT:-}" ]]; then
  WORK_ROOT="$PROJECT_ROOT/output_run"
  if [[ ! -d "$WORK_ROOT/models" ]]; then
    echo "WORK_ROOT is unset and $WORK_ROOT holds no models/ directory;" >&2
    echo "set WORK_ROOT to the parent of repos/, models/, and runs/." >&2
    exit 2
  fi
fi

# Default is the complete formal batch. Every backend runs its own smoke
# steps first, then its own UniWeaver and Minimal steps, on its own GPU. When an
# SFT checkpoint is supplied, base and SFT BAGEL run as separate chains.
MODE="${MODE:-bench-both}"
RESUME="${RESUME:-0}"
REVIEW_RETRY="${REVIEW_RETRY:-1}"
SMOKE="${SMOKE:-1}"
PREFLIGHT="${PREFLIGHT:-1}"
DRY_RUN="${DRY_RUN:-0}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
SMOKE_TASK_KEY="${SMOKE_TASK_KEY:-task_001}"
: "${BENCHMARK_FILE:?Set BENCHMARK_FILE to the task-matrix JSON.}"

case "$MODE" in
  smoke-minimal) waves=(smoke-minimal) ;;
  smoke-uniweaver)    waves=(smoke-uniweaver) ;;
  bench-uniweaver)       waves=(bench-uniweaver) ;;
  bench-minimal)    waves=(bench-minimal) ;;
  bench|bench-both)    waves=(bench-uniweaver bench-minimal) ;;
  *)
    echo "MODE must be smoke-minimal, smoke-uniweaver, bench-both (default), bench-uniweaver, or bench-minimal" >&2
    exit 2
    ;;
esac

for flag_name in REVIEW_RETRY SMOKE PREFLIGHT RESUME DRY_RUN; do
  flag_value="${!flag_name}"
  if [[ "$flag_value" != "0" && "$flag_value" != "1" ]]; then
    echo "$flag_name must be 0 or 1" >&2
    exit 2
  fi
done

is_bench_mode=0
case "$MODE" in
  bench|bench-both|bench-uniweaver|bench-minimal) is_bench_mode=1 ;;
esac
if [[ "$RESUME" == "1" && "$is_bench_mode" != "1" ]]; then
  echo "RESUME=1 is supported by this launcher only for benchmark modes" >&2
  exit 2
fi

# A resumed batch already passed (or deliberately skipped) its own smoke.
run_smoke=0
if [[ "$is_bench_mode" == "1" && "$SMOKE" == "1" && "$RESUME" != "1" ]]; then
  run_smoke=1
fi
smoke_waves=()
if [[ "$run_smoke" == "1" ]]; then
  case "$MODE" in
    bench-uniweaver)    smoke_waves=(smoke-uniweaver) ;;
    bench-minimal) smoke_waves=(smoke-minimal) ;;
    *)          smoke_waves=(smoke-minimal smoke-uniweaver) ;;
  esac
fi

uniweaver_output_root="${UNIWEAVER_OUTPUT_ROOT:-${WORK_ROOT}/runs/bench_120_uniweaver_${RUN_TAG}}"
minimal_output_root="${MINIMAL_OUTPUT_ROOT:-${WORK_ROOT}/runs/bench_120_minimal_${RUN_TAG}}"
smoke_minimal_root="${SMOKE_MINIMAL_ROOT:-${WORK_ROOT}/runs/smoke_minimal_${RUN_TAG}}"
smoke_uniweaver_root="${SMOKE_UNIWEAVER_ROOT:-${WORK_ROOT}/runs/smoke_uniweaver_${RUN_TAG}}"
runs_dir="${WORK_ROOT}/runs"
# Batch-level records sit beside runs/, which holds one directory per run.
LAUNCHER_LOG="${LAUNCHER_LOG:-${WORK_ROOT}/launcher_${RUN_TAG}.log}"
CHAIN_STATUS="${CHAIN_STATUS:-${WORK_ROOT}/chain_status_${RUN_TAG}.tsv}"

# Second argument marks a launcher-driven preflight smoke, which always uses the
# dedicated smoke root so it can never collide with the formal step's root.
wave_output_root() {
  local step="$1"
  local auto_smoke="${2:-0}"
  case "$step" in
    smoke-minimal)
      if [[ "$auto_smoke" == "1" ]]; then
        printf '%s\n' "$smoke_minimal_root"
      else
        printf '%s\n' "${OUTPUT_ROOT:-$smoke_minimal_root}"
      fi
      ;;
    smoke-uniweaver)
      if [[ "$auto_smoke" == "1" ]]; then
        printf '%s\n' "$smoke_uniweaver_root"
      else
        printf '%s\n' "${OUTPUT_ROOT:-$smoke_uniweaver_root}"
      fi
      ;;
    bench-uniweaver) printf '%s\n' "${OUTPUT_ROOT:-$uniweaver_output_root}" ;;
    bench-minimal) printf '%s\n' "${OUTPUT_ROOT:-$minimal_output_root}" ;;
  esac
}

if [[ -n "${OUTPUT_ROOT:-}" && "${#waves[@]}" != "1" ]]; then
  echo "OUTPUT_ROOT applies to single-step modes only; for MODE=$MODE set" >&2
  echo "UNIWEAVER_OUTPUT_ROOT and MINIMAL_OUTPUT_ROOT instead." >&2
  exit 2
fi
if [[ -n "${OUTPUT_ROOT:-}" && "${#smoke_waves[@]}" -gt 0 ]]; then
  echo "OUTPUT_ROOT applies to the formal step only; preflight smoke writes to" >&2
  echo "SMOKE_MINIMAL_ROOT/SMOKE_UNIWEAVER_ROOT, so the two never share a root." >&2
fi

if [[ "$RESUME" == "1" ]]; then
  any_existing=0
  for wave in "${waves[@]}"; do
    if [[ -d "$(wave_output_root "$wave")" ]]; then
      any_existing=1
    fi
  done
  if [[ "$any_existing" != "1" ]]; then
    echo "RESUME=1 but no benchmark output root exists yet for MODE=$MODE" >&2
    exit 2
  fi
else
  for wave in "${waves[@]}"; do
    root="$(wave_output_root "$wave")"
    if [[ -e "$root" ]]; then
      echo "Refusing to reuse output root: $root" >&2
      exit 2
    fi
  done
  for wave in "${smoke_waves[@]:-}"; do
    if [[ -n "$wave" ]]; then
      root="$(wave_output_root "$wave" 1)"
      if [[ -e "$root" ]]; then
        echo "Refusing to reuse smoke output root: $root" >&2
        exit 2
      fi
    fi
  done
fi

# The base batch uses GPU indexes 0-6. A supplied SFT checkpoint adds a second
# BAGEL chain on GPU 7. Refuse to start instead of letting a missing GPU fail
# later during preflight.
required_gpus=7
if [[ -n "${BAGEL_SFT_CHECKPOINT:-}" ]]; then
  required_gpus=8
fi
REQUIRED_GPUS="${REQUIRED_GPUS:-$required_gpus}"
if [[ "$DRY_RUN" != "1" && "$REQUIRED_GPUS" != "0" ]] \
    && command -v nvidia-smi >/dev/null 2>&1; then
  visible_gpus="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  if [[ -z "$visible_gpus" || "$visible_gpus" -lt "$REQUIRED_GPUS" ]]; then
    echo "launcher needs $REQUIRED_GPUS visible GPUs, nvidia-smi reports" \
      "${visible_gpus:-0}" >&2
    echo "set REQUIRED_GPUS=0 to bypass this check explicitly" >&2
    exit 2
  fi
fi

# Persist the launcher's own decisions: the per-backend logs only contain what
# run_umm.py printed, not which step the launcher judged passed or skipped.
if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$runs_dir"
  : >"$CHAIN_STATUS"
  # Keep the caller's stdout/stderr so the tee can be retired before the log is
  # copied into the run roots (see the archive step).
  exec 3>&1 4>&2
  exec > >(tee -a "$LAUNCHER_LOG") 2>&1
  export PYTHONPATH="$PROJECT_ROOT"
  export PYTHONDONTWRITEBYTECODE=1
fi

pids=()
names=()

cleanup_children() {
  local pid
  for pid in "${pids[@]:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${pids[@]:-}"; do
    if [[ -n "$pid" ]]; then
      wait "$pid" 2>/dev/null || true
    fi
  done
}

on_exit() {
  local status="$?"
  trap - EXIT INT TERM
  cleanup_children
  exit "$status"
}

on_int() {
  trap - EXIT INT TERM
  cleanup_children
  exit 130
}

on_term() {
  trap - EXIT INT TERM
  cleanup_children
  exit 143
}

trap on_exit EXIT
trap on_int INT
trap on_term TERM

# backend <tab> step <tab> result <tab> detail
record_status() {
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  # One physical line per record. The detail comes from whatever a child
  # printed, so it can be multi-line (a python traceback out of formal_progress,
  # a multi-line preflight reason); writing it verbatim would split the row and
  # leave the status file unparsable. A tab would add columns, so flatten that
  # too.
  local detail="${4:-}"
  detail="${detail//$'\n'/ }"
  detail="${detail//$'\r'/ }"
  detail="${detail//$'\t'/ }"
  printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$detail" >>"$CHAIN_STATUS"
}

set_scope_args() {
  case "$1" in
    smoke-minimal)
      STEP_SCOPE_ARGS=(--execution online --protocol minimal --dataset single --task-key "$SMOKE_TASK_KEY")
      ;;
    smoke-uniweaver)
      STEP_SCOPE_ARGS=(--execution online --protocol uniweaver --allow-unreported-backbone --dataset single --task-key "$SMOKE_TASK_KEY")
      ;;
    bench-uniweaver)
      STEP_SCOPE_ARGS=(--execution online --protocol uniweaver --allow-unreported-backbone --dataset bench --benchmark-file "$BENCHMARK_FILE")
      ;;
    bench-minimal)
      STEP_SCOPE_ARGS=(--execution online --protocol minimal --dataset bench --benchmark-file "$BENCHMARK_FILE")
      ;;
  esac
}

review_applicable() {
  case "$1" in
    smoke-uniweaver|bench-uniweaver) return 0 ;;
    *) return 1 ;;
  esac
}

# Run one step of a chain in the foreground and return its exit status.
run_step() {
  local gpu="$1"
  local python_bin="$2"
  local run_label="$3"
  local backend="$4"
  local mode="$5"
  local root="$6"
  shift 6
  local resume=0
  local output_dir="$root/$run_label"
  local log_path="$root/logs/$run_label.log"
  set_scope_args "$mode"
  # Resume only a step that already owns an output directory: a backend that
  # never produced one (for example it failed preflight in the previous attempt)
  # must start fresh, and run_umm.py rejects --resume without an existing root.
  if [[ "$RESUME" == "1" && -d "$output_dir" ]]; then
    resume=1
  fi
  local command=("$python_bin" "$PROJECT_ROOT/baselines/run_umm.py" \
    "${STEP_SCOPE_ARGS[@]}" \
    --backend "$backend" --device cuda \
    --model-call-timeout-seconds 3600 \
    --backend-startup-timeout-seconds 3600 \
    --task-timeout-seconds 21600 \
    # UniWeaver training and evaluation use the Plan3 0-2 reference contract.
    --max-runtime-conditions 2 \
    --max-runtime-steps 8 \
    --max-checks-per-step 3 \
    --max-steps 8 \
    --output-dir "$output_dir")
  if [[ "$resume" == "1" ]]; then
    command+=(--resume)
  fi
  if review_applicable "$mode"; then
    if [[ "$REVIEW_RETRY" == "1" ]]; then
      command+=(--review-retry)
    else
      command+=(--no-review-retry)
    fi
  fi
  command+=("$@")
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%s %s\n' "$gpu" "${command[*]}"
    return 0
  fi
  mkdir -p "$root/logs"
  echo "Launching $run_label $mode on GPU $gpu -> $output_dir"
  if [[ "$resume" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "${command[@]}" >>"$log_path" 2>&1
  else
    CUDA_VISIBLE_DEVICES="$gpu" "${command[@]}" >"$log_path" 2>&1
  fi
}

# A smoke only counts when a Step PNG exists; process completion alone is
# insufficient evidence that image generation ran.
smoke_produced_step() {
  local root="$1"
  local run_label="$2"
  [[ -n "$(find "$root/$run_label" -type f -path '*/steps/*.png' -print -quit 2>/dev/null)" ]]
}

# A formal step only counts when the run processed the whole panel, so completion
# is read from the persisted task counts rather than the process exit code alone.
# Prints "processed=<n>/<total> completed=.. planner_error=.. image_error=.. step_error=..
# preflight_error=.. incomplete=.."; UniWeaver uses the planner/image pair and the minimal
# protocol uses the step/preflight pair, so the inapplicable pair reads 0/None.
# and succeeds only when every task was processed and none was left incomplete.
formal_progress() {
  local python_bin="$1"
  local root="$2"
  local run_label="$3"
  "$python_bin" - "$root/$run_label/summary.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    print("MISSING summary.json")
    raise SystemExit(3)
summary = json.loads(path.read_text(encoding="utf-8"))
processed = summary.get("processed_count")
total = summary.get("task_count")
counts = {key: summary.get(key) for key in
          ("completed", "planner_error", "image_error", "step_error",
           "preflight_error", "incomplete")}
print(
    f"processed={processed}/{total} " + " ".join(
        f"{key}={value}" for key, value in counts.items()))
raise SystemExit(0 if processed == total and not counts["incomplete"] else 4)
PY
}

# Lightweight readiness gate: the backend's own interpreter must see torch and CUDA,
# and every declared path must exist. Never loads weights. BAGEL additionally
# requires the official flash_attn, which its modeling code imports directly.
run_preflight() {
  local python_bin="$1"
  local backend="$2"
  shift 2
  # A missing interpreter must say so: otherwise the probe's own output is
  # empty and the chain fails with a reason nobody can act on.
  if ! command -v "$python_bin" >/dev/null 2>&1 && [[ ! -x "$python_bin" ]]; then
    echo "PREFLIGHT FAILED $backend: interpreter not found: $python_bin" >&2
    echo "  set the PYTHON_<BACKEND> variable for this backend to its own" >&2
    echo "  environment's python (see the Command Settings section in README.md)" >&2
    return 2
  fi
  local -a probe_args=(--backend "$backend")
  local argument
  for argument in "$@"; do
    probe_args+=("$argument")
  done
  if [[ "$backend" == "bagel" ]]; then
    probe_args+=(--require-flash-attn)
  fi
  local reason
  if reason="$("$python_bin" "$PROJECT_ROOT/scripts/check_backend_ready.py" \
      "${probe_args[@]}" 2>&1)"; then
    echo "PREFLIGHT OK $backend"
    # Readiness warnings (optional-but-large artifacts) must stay visible even
    # though the probe itself passed.
    if [[ "$reason" == *WARNING* ]]; then
      printf '%s\n' "$reason" | grep '^WARNING' >&2 || true
    fi
    return 0
  fi
  echo "PREFLIGHT FAILED $backend: $reason" >&2
  return 2
}

# One backend owns one GPU and runs smoke, then UniWeaver, then Minimal.
run_chain() {
  local gpu="$1"
  local python_bin="$2"
  local run_label="$3"
  local backend="$4"
  shift 4
  local overall=0
  local blocked=0
  local mode
  local root
  local status
  if [[ "$PREFLIGHT" == "1" && "$DRY_RUN" != "1" ]]; then
    if ! run_preflight "$python_bin" "$backend" "$@"; then
      record_status "$run_label" "preflight" FAIL "not_ready"
      for mode in "${smoke_waves[@]:-}" "${waves[@]}"; do
        if [[ -n "$mode" ]]; then
          record_status "$run_label" "$mode" SKIP "preflight_failed"
        fi
      done
      return 1
    fi
    record_status "$run_label" "preflight" PASS "ready"
  fi
  if [[ "${#smoke_waves[@]}" -gt 0 ]]; then
    for mode in "${smoke_waves[@]}"; do
      root="$(wave_output_root "$mode" 1)"
      if run_step "$gpu" "$python_bin" "$run_label" "$backend" "$mode" "$root" "$@"; then
        if [[ "$DRY_RUN" == "1" ]] || smoke_produced_step "$root" "$run_label"; then
          echo "PASS $run_label $mode"
          record_status "$run_label" "$mode" PASS "step_png=present"
        else
          echo "SMOKE FAILED $run_label $mode: no Step PNG under $root/$run_label" >&2
          record_status "$run_label" "$mode" SMOKE_FAILED "no_step_png"
          blocked=1
          overall=1
          break
        fi
      else
        status="$?"
        echo "FAIL $run_label $mode exit=$status log=$root/logs/$run_label.log" >&2
        record_status "$run_label" "$mode" FAIL "exit=$status"
        blocked=1
        overall=1
        break
      fi
    done
  fi
  if [[ "$blocked" == "1" ]]; then
    for mode in "${waves[@]}"; do
      echo "SKIP $run_label $mode (smoke did not pass)" >&2
      record_status "$run_label" "$mode" SKIP "smoke_did_not_pass"
    done
    return "$overall"
  fi
  for mode in "${waves[@]}"; do
    root="$(wave_output_root "$mode")"
    if run_step "$gpu" "$python_bin" "$run_label" "$backend" "$mode" "$root" "$@"; then
      if [[ "$DRY_RUN" == "1" ]]; then
        echo "PASS $run_label $mode"
        record_status "$run_label" "$mode" PASS "dry_run"
        continue
      fi
      if detail="$(formal_progress "$python_bin" "$root" "$run_label" 2>&1)"; then
        echo "PASS $run_label $mode ($detail)"
        record_status "$run_label" "$mode" PASS "exit=0;$detail"
      else
        echo "INCOMPLETE $run_label $mode: $detail log=$root/logs/$run_label.log" >&2
        record_status "$run_label" "$mode" INCOMPLETE "$detail"
        overall=1
      fi
    else
      status="$?"
      echo "FAIL $run_label $mode exit=$status log=$root/logs/$run_label.log" >&2
      record_status "$run_label" "$mode" FAIL "exit=$status"
      overall=1
    fi
  done
  return "$overall"
}

launch_chain() {
  local gpu="$1"
  local python_bin="$2"
  local run_label="$3"
  local backend="$4"
  shift 4
  if [[ "$DRY_RUN" == "1" ]]; then
    run_chain "$gpu" "$python_bin" "$run_label" "$backend" "$@"
    return 0
  fi
  run_chain "$gpu" "$python_bin" "$run_label" "$backend" "$@" &
  pids+=("$!")
  names+=("$run_label")
}

echo "=== MODE=$MODE smoke=$run_smoke formal=${waves[*]} REVIEW_RETRY=$REVIEW_RETRY RESUME=$RESUME DRY_RUN=$DRY_RUN ==="
echo "=== launcher log: $LAUNCHER_LOG ==="
if [[ "$DRY_RUN" != "1" ]]; then
  echo "=== chain status: $CHAIN_STATUS ==="
fi
for wave in "${smoke_waves[@]:-}"; do
  if [[ -n "$wave" ]]; then
    echo "=== $wave root: $(wave_output_root "$wave" 1) ==="
  fi
done
for wave in "${waves[@]}"; do
  if review_applicable "$wave"; then
    echo "=== $wave root: $(wave_output_root "$wave") (review per REVIEW_RETRY=$REVIEW_RETRY) ==="
  else
    echo "=== $wave root: $(wave_output_root "$wave") (review disabled) ==="
  fi
done

bagel_python="${PYTHON_BAGEL:-python}"
launch_chain 0 "$bagel_python" bagel bagel \
  --model-path "$WORK_ROOT/models/BAGEL-7B-MoT"
if [[ -n "${BAGEL_SFT_CHECKPOINT:-}" ]]; then
  launch_chain 7 "$bagel_python" bagel-sft bagel \
    --model-path "$WORK_ROOT/models/BAGEL-7B-MoT" \
    --sft-checkpoint "$BAGEL_SFT_CHECKPOINT"
fi
launch_chain 1 "${PYTHON_OMNIGEN2:-python}" omnigen2 omnigen2 \
  --repo-path "$WORK_ROOT/repos/OmniGen2" \
  --model-path "$WORK_ROOT/models/OmniGen2"
launch_chain 2 "${PYTHON_SHOWO2:-python}" show-o2 show-o2 \
  --repo-path "$WORK_ROOT/repos/Show-o" \
  --model-path "$WORK_ROOT/models/Show-o2-7B" \
  --vq-path "$WORK_ROOT/models/Wan2.1-VAE/Wan2.1_VAE.pth" \
  --llm-path "$WORK_ROOT/models/Qwen2.5-7B-Instruct"
launch_chain 3 "${PYTHON_TOKENFLOW:-python}" tokenflow tokenflow \
  --repo-path "$WORK_ROOT/repos/TokenFlow" \
  --model-path "$WORK_ROOT/models/TokenFlow-7B" \
  --vq-path "$WORK_ROOT/models/TokenFlow-tokenizer/tokenflow_clipb_32k_enhanced.pt" \
  --understanding-vision-path "$WORK_ROOT/models/TokenFlow-tokenizer/tokenflow_siglip_32k.pt" \
  --llm-path "$WORK_ROOT/models/TokenFlow-understanding-14B"
launch_chain 4 "${PYTHON_ILLUME_PLUS:-python}" illume-plus illume-plus \
  --repo-path "$WORK_ROOT/repos/ILLUME_plus" \
  --model-path "$WORK_ROOT/models/ILLUME-plus-7B-hf" \
  --vq-path "$WORK_ROOT/models/DualViTok" \
  --llm-path "$WORK_ROOT/models/DualViTok-SDXL-Decoder"
launch_chain 5 "${PYTHON_SEED_X:-python}" seed-x seed-x \
  --repo-path "$WORK_ROOT/repos/SEED-X" \
  --model-path "$WORK_ROOT/models/SEED-X-17B" \
  --vq-path "$WORK_ROOT/models/QwenViT/qwen_vit_G.pt" \
  --llm-path "$WORK_ROOT/models/SDXL-base"
launch_chain 6 "${PYTHON_JANUS_PRO:-python}" janus-pro janus-pro \
  --repo-path "$WORK_ROOT/repos/Janus" \
  --model-path "$WORK_ROOT/models/Janus-Pro-7B"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1: no process was started."
  exit 0
fi

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "CHAIN OK ${names[$index]}"
  else
    status="$?"
    echo "CHAIN FAILED ${names[$index]} exit=$status" >&2
    failed=1
  fi
  pids[$index]=""
done

echo "Chain status (backend, step, result, detail):"
sort "$CHAIN_STATUS" | awk -F'\t' '{printf "  %-12s %-14s %-13s %s\n", $1, $2, $3, $4}'
printf '  summary: pass=%s fail=%s smoke_failed=%s skip=%s\n' \
  "$(awk -F'\t' '$3=="PASS"' "$CHAIN_STATUS" | wc -l | tr -d ' ')" \
  "$(awk -F'\t' '$3=="FAIL"' "$CHAIN_STATUS" | wc -l | tr -d ' ')" \
  "$(awk -F'\t' '$3=="SMOKE_FAILED"' "$CHAIN_STATUS" | wc -l | tr -d ' ')" \
  "$(awk -F'\t' '$3=="SKIP"' "$CHAIN_STATUS" | wc -l | tr -d ' ')"

echo "Outputs:"
for wave in "${smoke_waves[@]:-}"; do
  if [[ -n "$wave" ]]; then
    echo "  $wave -> $(wave_output_root "$wave" 1)"
  fi
done
for wave in "${waves[@]}"; do
  echo "  $wave -> $(wave_output_root "$wave")"
done
echo "Launcher log: $LAUNCHER_LOG"
echo "Chain status: $CHAIN_STATUS"

# Keep the launcher decision record with this run's artifacts. Both formal roots
# get the chain-status copy (the documented Minimal layout lists it too); the
# launcher log goes to the first formal root, or to the smoke root when this
# invocation produced no formal root at all.
#
# Retire the tee first: restoring the original stdout/stderr closes its pipe, and
# the short wait lets tee drain. Announce the archive through ``tee -a`` before
# copying, so the canonical log is closed afterwards and the archived copy is
# byte-identical to it.
exec 1>&3 2>&4
sleep 0.2
archive_roots=()
for wave in "${waves[@]}"; do
  candidate="$(wave_output_root "$wave")"
  if [[ -d "$candidate" ]]; then
    archive_roots+=("$candidate")
  fi
done
if [[ "${#archive_roots[@]}" -eq 0 ]]; then
  for wave in "${smoke_waves[@]:-}"; do
    if [[ -n "$wave" ]]; then
      candidate="$(wave_output_root "$wave" 1)"
      if [[ -d "$candidate" ]]; then
        archive_roots+=("$candidate")
      fi
    fi
  done
fi
if [[ "${#archive_roots[@]}" -gt 0 ]]; then
  printf 'Archived launcher log and chain status under: %s\n' \
    "${archive_roots[0]}" | tee -a "$LAUNCHER_LOG"
  for candidate in "${archive_roots[@]:1}"; do
    printf 'Archived chain status under: %s\n' "$candidate" \
      | tee -a "$LAUNCHER_LOG"
  done
  for candidate in "${archive_roots[@]}"; do
    cp -f "$CHAIN_STATUS" "$candidate/chain_status.tsv"
  done
  cp -f "$LAUNCHER_LOG" "${archive_roots[0]}/launcher.log"
fi
exit "$failed"
