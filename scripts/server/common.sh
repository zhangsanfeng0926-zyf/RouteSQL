#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${SCRIPT_DIR}/server.env"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "[ERROR] Missing ${ENV_FILE}. Copy server.env.example to server.env first."
  exit 1
fi

# shellcheck disable=SC1090
source "${ENV_FILE}"

if [[ -z "${ROOT_DIR:-}" ]]; then
  ROOT_DIR="${DEFAULT_ROOT}"
fi

cd "${ROOT_DIR}"

export HF_ENDPOINT="${HF_ENDPOINT:-}"
export HUGGINGFACE_HUB_BASE_URL="${HUGGINGFACE_HUB_BASE_URL:-}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"

mkdir -p "${ROOT_DIR}/${LOG_DIR}"

require_var() {
  local name="$1"
  local value="${!name:-}"
  if [[ -z "${value}" ]]; then
    echo "[ERROR] ${name} is empty in server.env"
    exit 1
  fi
}

require_var PYTHON_BIN
require_var OPENAI_API_KEY
require_var OPENAI_BASE_URL
require_var MODEL

PREFIX="${DATA_TYPE^^}-${SPLIT^^}_${PROMPT_REPR}_${K_SHOT}-SHOT"
Q1_DIR="dataset/process/${PREFIX}_${SELECTOR_STAGE1}_${EXAMPLE_TYPE}-EXAMPLE_CTX-200_ANS-${MAX_SEQ_LEN}"
Q2_DIR="dataset/process/${PREFIX}_${SELECTOR_STAGE2}_${EXAMPLE_TYPE}-EXAMPLE_CTX-200_ANS-${MAX_SEQ_LEN}"
Q1_RESULT="${Q1_DIR}/RESULTS_MODEL-${MODEL}.txt"
Q2_RESULT="${Q2_DIR}/RESULTS_MODEL-${MODEL}.txt"

run_step() {
  local step="$1"
  shift
  local out_log="${ROOT_DIR}/${LOG_DIR}/${step}.log"
  local err_log="${ROOT_DIR}/${LOG_DIR}/${step}.err.log"

  echo "[$(date '+%F %T')] START ${step}" | tee -a "${out_log}"
  "$@" > >(tee -a "${out_log}") 2> >(tee -a "${err_log}" >&2)
  echo "[$(date '+%F %T')] DONE ${step}" | tee -a "${out_log}"
}
