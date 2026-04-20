#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

run_step "06_stage3_finalize" bash -lc "
  mkdir -p results
  run_ts=\$(date '+%Y%m%d_%H%M%S')
  model_hist_file=\"results/DAIL-SQL+${MODEL}_\${run_ts}.txt\"
  stage3_hist_file=\"results/stage3_final_result_\${run_ts}.txt\"

  # Persist every run into unique files; keep latest as symlink pointers.
  cp -f '${Q2_RESULT}' \"\${model_hist_file}\"
  cp -f '${Q2_RESULT}' \"\${stage3_hist_file}\"

  ln -sfn \"\$(basename \"\${model_hist_file}\")\" 'results/DAIL-SQL+${MODEL}.txt'
  ln -sfn \"\$(basename \"\${stage3_hist_file}\")\" 'results/stage3_final_result_latest.txt'
  {
    echo 'stage3 finalize done at '$(date '+%F_%T')
    echo 'model=${MODEL}'
    echo 'source=${Q2_RESULT}'
    echo "model_file=\${model_hist_file}"
    echo "stage3_file=\${stage3_hist_file}"
    echo 'latest_link=results/DAIL-SQL+${MODEL}.txt'
    echo 'latest_stage3_link=results/stage3_final_result_latest.txt'
    wc -l < '${Q2_RESULT}' | xargs -I{} echo 'result_lines={}'
  } > results/stage3_finalize.log
"