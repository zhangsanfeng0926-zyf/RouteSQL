#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

source venv/bin/activate
set -a
source scripts/server/server.env
set +a

export PYTHON_BIN="/home/zhaoyanfeng/DAIL-SQL/venv/bin/python"
export SQL_FRAMEWORK_FILL=1

Q1_DIR="dataset/process/${DATA_TYPE^^}-${SPLIT^^}_${PROMPT_REPR}_${K_SHOT}-SHOT_${SELECTOR_STAGE1}_${EXAMPLE_TYPE}-EXAMPLE_CTX-200_ANS-${MAX_SEQ_LEN}"
Q1_OUT="${Q1_DIR}/RESULTS_MODEL-${MODEL}.txt"

RESUME_INDEX=0
if [[ -f "${Q1_OUT}" ]]; then
  RESUME_INDEX=$(wc -l < "${Q1_OUT}")
fi

echo "[$(date '+%F %T')] RESUME_START index=${RESUME_INDEX}"

"${PYTHON_BIN}" -W ignore -u ./scripts/python_tools/ask_llm.py \
  --question "./${Q1_DIR}" \
  --openai_api_key "${OPENAI_API_KEY}" \
  --openai_base_url "${OPENAI_BASE_URL}" \
  --model "${MODEL}" \
  --n "${N}" \
  --db_dir "./${DB_DIR}" \
  --temperature "${TEMPERATURE}" \
  --two_stage_framework \
  --fill_candidates 3 \
  --repair_rounds 2 \
  --start_index "${RESUME_INDEX}"

echo "[$(date '+%F %T')] RESUME_STAGE1_DONE"

bash ./scripts/server/04_stage2_generate.sh
bash ./scripts/server/05_stage2_infer.sh
bash ./scripts/server/06_stage3_finalize.sh
bash ./scripts/server/07_eval.sh

echo "[$(date '+%F %T')] RESUME_ALL_DONE"
