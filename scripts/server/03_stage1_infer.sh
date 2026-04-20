#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

infer_cmd=(
  "${PYTHON_BIN}" -W ignore -u ./scripts/python_tools/ask_llm.py
  --question "./${Q1_DIR}"
  --openai_api_key "${OPENAI_API_KEY}"
  --openai_base_url "${OPENAI_BASE_URL}"
  --model "${MODEL}"
  --n "${N}"
  --db_dir "./${DB_DIR}"
  --temperature "${TEMPERATURE}"
)

if [[ "${SQL_FRAMEWORK_FILL:-0}" == "1" ]]; then
  infer_cmd+=(--two_stage_framework)
fi

run_step "03_stage1_infer" "${infer_cmd[@]}"