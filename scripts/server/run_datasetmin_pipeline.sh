#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_ROOT_DIR="${ROOT_DIR}"
ENV_FILE="${SCRIPT_DIR}/server.env"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "[ERROR] Missing ${ENV_FILE}"
  exit 1
fi

# shellcheck disable=SC1090
source "${ENV_FILE}"

if [[ -z "${ROOT_DIR:-}" ]]; then
  ROOT_DIR="${DEFAULT_ROOT_DIR}"
fi

cd "${ROOT_DIR}"

PYTHON_BIN="${ROOT_DIR}/venv/bin/python"
RUN_ID="${1:-$(date +%Y%m%d_%H%M%S)}"

export NLTK_DATA="${ROOT_DIR}/nltk_data"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HUGGINGFACE_HUB_BASE_URL="${HUGGINGFACE_HUB_BASE_URL:-${HF_ENDPOINT}}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export SENTENCE_TRANSFORMERS_HOME="${ROOT_DIR}/.cache/sentence_transformers"
export TRANSFORMERS_CACHE="${ROOT_DIR}/.cache/transformers"

DATA_ROOT="dataset/dataset_min"
PROCESS_ROOT="dataset/process_datasetmin"
DATASET_DIR="${DATA_ROOT}/spider"
DB_DIR="${DATASET_DIR}/database"
TABLE_PATH="${DATASET_DIR}/tables.json"
GOLD_PATH="${DATASET_DIR}/dev_gold.sql"

DATA_TYPE="${DATA_TYPE:-spider}"
SPLIT="${SPLIT:-test}"
TOKENIZER="${TOKENIZER:-gpt-3.5-turbo}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-10000}"
K_SHOT="${K_SHOT:-9}"
PROMPT_REPR="${PROMPT_REPR:-SQL}"
EXAMPLE_TYPE="${EXAMPLE_TYPE:-QA}"
SELECTOR_STAGE1="${SELECTOR_STAGE1:-EUCDISQUESTIONMASK}"
SELECTOR_STAGE2="${SELECTOR_STAGE2:-EUCDISMASKPRESKLSIMTHR}"
MODEL="${MODEL:-gpt-5}"
TEMPERATURE="${TEMPERATURE:-1.0}"
N="${N:-1}"

PREFIX="${DATA_TYPE^^}-${SPLIT^^}_${PROMPT_REPR}_${K_SHOT}-SHOT"
Q1_DIR="${PROCESS_ROOT}/${PREFIX}_${SELECTOR_STAGE1}_${EXAMPLE_TYPE}-EXAMPLE_CTX-200_ANS-${MAX_SEQ_LEN}"
Q2_DIR="${PROCESS_ROOT}/${PREFIX}_${SELECTOR_STAGE2}_${EXAMPLE_TYPE}-EXAMPLE_CTX-200_ANS-${MAX_SEQ_LEN}"

STAGE1_RESULT="results/RESULTS_MODEL-${MODEL}_datasetmin_stage1_${RUN_ID}.txt"
STAGE2_RESULT="results/RESULTS_MODEL-${MODEL}_datasetmin_stage2_${RUN_ID}.txt"
FINAL_RESULT="results/RESULTS_MODEL-${MODEL}_datasetmin_final_${RUN_ID}.txt"
MERGE_TRACE="results/RESULTS_MODEL-${MODEL}_datasetmin_merge_trace_${RUN_ID}.jsonl"
LATEST_RESULT="results/RESULTS_MODEL-${MODEL}_datasetmin_latest.txt"
LOCAL_EVAL_JSON="results/datasetmin_eval_${RUN_ID}.json"
OFFICIAL_EVAL_TXT="results/datasetmin_official_eval_${RUN_ID}.txt"
OFFICIAL_EVAL_JSON="results/datasetmin_official_eval_${RUN_ID}_metrics.json"

mkdir -p results logs/server

log_step() {
  echo "[$(date '+%F %T')] $1"
}

run_cmd() {
  log_step "START $1"
  shift
  "$@"
  log_step "DONE $1"
}

run_cmd "stage1_generate" \
  "${PYTHON_BIN}" scripts/python_tools/generate_question.py \
    --data_type "${DATA_TYPE}" \
    --split "${SPLIT}" \
    --tokenizer "${TOKENIZER}" \
    --max_seq_len "${MAX_SEQ_LEN}" \
    --prompt_repr "${PROMPT_REPR}" \
    --k_shot "${K_SHOT}" \
    --example_type "${EXAMPLE_TYPE}" \
    --selector_type "${SELECTOR_STAGE1}" \
    --data_root "${DATA_ROOT}" \
    --process_root "${PROCESS_ROOT}"

run_cmd "stage1_infer" \
  "${PYTHON_BIN}" scripts/python_tools/ask_llm.py \
    --openai_api_key "${OPENAI_API_KEY}" \
    --openai_base_url "${OPENAI_BASE_URL}" \
    --model "${MODEL}" \
    --question "${Q1_DIR}" \
    --db_dir "${DB_DIR}" \
    --n "${N}" \
    --temperature "${TEMPERATURE}" \
    --output_suffix "datasetmin_stage1_${RUN_ID}" \
    --result_output_dir results

run_cmd "stage2_generate" \
  "${PYTHON_BIN}" scripts/python_tools/generate_question.py \
    --data_type "${DATA_TYPE}" \
    --split "${SPLIT}" \
    --tokenizer "${TOKENIZER}" \
    --max_seq_len "${MAX_SEQ_LEN}" \
    --prompt_repr "${PROMPT_REPR}" \
    --k_shot "${K_SHOT}" \
    --example_type "${EXAMPLE_TYPE}" \
    --selector_type "${SELECTOR_STAGE2}" \
    --pre_test_result "${STAGE1_RESULT}" \
    --data_root "${DATA_ROOT}" \
    --process_root "${PROCESS_ROOT}"

run_cmd "stage2_infer" \
  "${PYTHON_BIN}" scripts/python_tools/ask_llm.py \
    --openai_api_key "${OPENAI_API_KEY}" \
    --openai_base_url "${OPENAI_BASE_URL}" \
    --model "${MODEL}" \
    --question "${Q2_DIR}" \
    --db_dir "${DB_DIR}" \
    --n 3 \
    --temperature 0.3 \
    --two_stage_framework \
    --framework_temperature 0.0 \
    --fill_candidates 5 \
    --repair_rounds 3 \
    --write_framework_outputs \
    --output_suffix "datasetmin_stage2_${RUN_ID}" \
    --result_output_dir results

run_cmd "merge_stage1_stage2" \
  "${PYTHON_BIN}" scripts/python_tools/merge_predictions.py \
    --questions "${Q2_DIR}" \
    --pred_a "${STAGE1_RESULT}" \
    --pred_b "${STAGE2_RESULT}" \
    --db_dir "${DB_DIR}" \
    --out "${FINAL_RESULT}" \
    --trace_out "${MERGE_TRACE}" \
    --prefer a

cp "${FINAL_RESULT}" "${LATEST_RESULT}"

run_cmd "eval_local" \
  "${PYTHON_BIN}" scripts/python_tools/evaluate_spider_em_ex.py \
    --pred "${FINAL_RESULT}" \
    --gold "${GOLD_PATH}" \
    --db_dir "${DB_DIR}" \
    --out "${LOCAL_EVAL_JSON}"

log_step "START eval_official"
NLTK_DATA="${NLTK_DATA}" \
  "${PYTHON_BIN}" third_party/test-suite-sql-eval/evaluation.py \
    --gold "${GOLD_PATH}" \
    --pred "${FINAL_RESULT}" \
    --db "${DB_DIR}" \
    --table "${TABLE_PATH}" \
    --etype all > "${OFFICIAL_EVAL_TXT}"

"${PYTHON_BIN}" - <<PY
import json
import re
from pathlib import Path

src = Path("${OFFICIAL_EVAL_TXT}")
text = src.read_text(encoding="utf-8", errors="ignore").splitlines()
idx_exec = next(i for i, l in enumerate(text) if "EXECUTION ACCURACY" in l)
idx_match = next(i for i, l in enumerate(text) if "EXACT MATCHING ACCURACY" in l)
count_line = text[idx_exec - 1]
exec_line = text[idx_exec + 1]
match_line = text[idx_match + 1]

levels = ["easy", "medium", "hard", "extra", "all"]
counts = [int(x) for x in re.findall(r"\d+", count_line)][-5:]
execs = [float(x) for x in re.findall(r"\d+\.\d+", exec_line)][-5:]
matches = [float(x) for x in re.findall(r"\d+\.\d+", match_line)][-5:]

out = {
    "run_id": "${RUN_ID}",
    "levels": levels,
    "count": dict(zip(levels, counts)),
    "execution_accuracy": dict(zip(levels, execs)),
    "exact_match_accuracy": dict(zip(levels, matches)),
    "pred": "${FINAL_RESULT}",
}
Path("${OFFICIAL_EVAL_JSON}").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("wrote ${OFFICIAL_EVAL_JSON}")
PY
log_step "DONE eval_official"

log_step "PIPELINE_COMPLETE run_id=${RUN_ID}"
log_step "FINAL_RESULT=${FINAL_RESULT}"
log_step "LOCAL_EVAL_JSON=${LOCAL_EVAL_JSON}"
log_step "OFFICIAL_EVAL_JSON=${OFFICIAL_EVAL_JSON}"
