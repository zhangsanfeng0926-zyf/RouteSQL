#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

"${DIR}/01_preprocess.sh"
"${DIR}/02_stage1_generate.sh"
"${DIR}/03_stage1_infer.sh"
"${DIR}/04_stage2_generate.sh"
"${DIR}/05_stage2_infer.sh"
"${DIR}/06_stage3_finalize.sh"
"${DIR}/07_eval.sh"

echo "All stages completed. Metrics: results/stage3_eval_metrics.json ; Official: results/stage3_official_eval_metrics.json"