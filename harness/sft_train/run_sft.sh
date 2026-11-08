#!/usr/bin/env bash
set -euo pipefail

# BAGEL text SFT launcher.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_CONFIG="${SCRIPT_DIR}/configs/lora_full_native40k_lr5e-5.json"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=8
export TORCH_CUDNN_SDPA_ENABLED=1

if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
  echo "UniWeaver text SFT requires exactly one visible GPU: $CUDA_VISIBLE_DEVICES" >&2
  exit 2
fi

DATA_DIR=""
OUTPUT=""
RESUME=""
CONFIG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume)
      RESUME="--resume"
      shift
      ;;
    --config)
      if [[ $# -lt 2 ]]; then
        echo "--config requires a path" >&2
        exit 2
      fi
      CONFIG="$2"
      shift 2
      ;;
    -*)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 DATA_DIR [OUTPUT_DIR] [--config CONFIG_JSON] [--resume]" >&2
      exit 2
      ;;
    *)
      if [[ -z "$DATA_DIR" ]]; then
        DATA_DIR="$1"
      elif [[ -z "$OUTPUT" ]]; then
        OUTPUT="$1"
      else
        echo "Unexpected positional argument: $1" >&2
        echo "Usage: $0 DATA_DIR [OUTPUT_DIR] [--config CONFIG_JSON] [--resume]" >&2
        exit 2
      fi
      shift
      ;;
  esac
done

if [[ -z "$DATA_DIR" ]]; then
  echo "Usage: $0 DATA_DIR [OUTPUT_DIR] [--config CONFIG_JSON] [--resume]" >&2
  exit 2
fi
OUTPUT="${OUTPUT:-${PROJECT_ROOT}/output_run/runs/training/sft_real}"
CONFIG="${CONFIG:-$DEFAULT_CONFIG}"
if [[ ! -f "$CONFIG" ]]; then
  echo "Training config does not exist: $CONFIG" >&2
  exit 2
fi

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Dataset directory does not exist: $DATA_DIR" >&2
  exit 2
fi
shopt -s nullglob
DATA_FILES=("$DATA_DIR"/*.json)
if (( ${#DATA_FILES[@]} == 0 )); then
  echo "Dataset contains no trajectory JSON files: $DATA_DIR" >&2
  exit 2
fi

echo "=== BAGEL Real-Data SFT ==="
echo "GPU:   $(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo "Data:   $DATA_DIR (${#DATA_FILES[@]} files)"
echo "Output: $OUTPUT"
echo "Mode:   lora"
echo "Policy: values supplied by the training config"
echo "Holdout: UniWeaver-Bench benchmark tasks are held out"
echo "Resume: ${RESUME:---no-resume}"
echo "Config: $CONFIG"
echo ""

PYTHON_BIN="${BAGEL_PYTHON:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi
TRAIN_ARGS=(
  --data-dir "$DATA_DIR"
  --output-dir "$OUTPUT"
  --config "$CONFIG"
)
if [[ -n "$RESUME" ]]; then
  TRAIN_ARGS+=(--resume)
fi
"${PYTHON_BIN}" "${SCRIPT_DIR}/train_sft.py" "${TRAIN_ARGS[@]}"
