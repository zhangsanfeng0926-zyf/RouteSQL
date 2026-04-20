#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

rm -rf "${Q1_DIR}" "${Q2_DIR}"
rm -f "results/stage3_finalize.log" "results/stage3_final_result_latest.txt" "results/stage3_eval_metrics.json"
rm -f "results/DAIL-SQL+${MODEL}.txt"
rm -f "${ROOT_DIR}/${LOG_DIR}"/*.log "${ROOT_DIR}/${LOG_DIR}"/*.err.log || true

echo "Cleaned outputs for model=${MODEL}"