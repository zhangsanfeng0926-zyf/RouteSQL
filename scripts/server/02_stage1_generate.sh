#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

run_step "02_stage1_generate" \
  "${PYTHON_BIN}" -W ignore -u ./scripts/python_tools/generate_question.py \
  --data_type "${DATA_TYPE}" \
  --split "${SPLIT}" \
  --tokenizer "${TOKENIZER}" \
  --max_seq_len "${MAX_SEQ_LEN}" \
  --prompt_repr "${PROMPT_REPR}" \
  --k_shot "${K_SHOT}" \
  --example_type "${EXAMPLE_TYPE}" \
  --selector_type "${SELECTOR_STAGE1}"