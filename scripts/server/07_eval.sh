#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

run_step "07_eval" \
  "${PYTHON_BIN}" ./scripts/python_tools/evaluate_spider_em_ex.py \
  --pred ./results/stage3_final_result_latest.txt \
  --gold ./dataset/spider/dev_gold.sql \
  --db_dir ./dataset/spider/database \
  --out ./results/stage3_eval_metrics.json

run_step "07_eval_official" bash -lc "
  set -euo pipefail
  if [[ -f './third_party/test-suite-sql-eval/evaluation.py' ]]; then
    NLTK_DATA=\"${NLTK_DATA:-./nltk_data}\" \
    \"${PYTHON_BIN}\" ./third_party/test-suite-sql-eval/evaluation.py \
      --gold ./dataset/spider/dev_gold.sql \
      --pred ./results/stage3_final_result_latest.txt \
      --db ./dataset/spider/database \
      --table ./dataset/spider/tables.json \
      --etype all > ./results/stage3_official_eval.txt

    \"${PYTHON_BIN}\" - <<'PY'
import json
import re
from pathlib import Path

src = Path('results/stage3_official_eval.txt')
text = src.read_text(encoding='utf-8', errors='ignore').splitlines()
idx_exec = next(i for i, l in enumerate(text) if 'EXECUTION ACCURACY' in l)
idx_match = next(i for i, l in enumerate(text) if 'EXACT MATCHING ACCURACY' in l)
count_line = text[idx_exec - 1]
exec_line = text[idx_exec + 1]
match_line = text[idx_match + 1]

levels = ['easy', 'medium', 'hard', 'extra', 'all']
counts = [int(x) for x in re.findall(r'\d+', count_line)][-5:]
execs = [float(x) for x in re.findall(r'\d+\.\d+', exec_line)][-5:]
matches = [float(x) for x in re.findall(r'\d+\.\d+', match_line)][-5:]

out = {
    'levels': levels,
    'count': dict(zip(levels, counts)),
    'execution_accuracy': dict(zip(levels, execs)),
    'exact_match_accuracy': dict(zip(levels, matches)),
}
Path('results/stage3_official_eval_metrics.json').write_text(
    json.dumps(out, indent=2), encoding='utf-8'
)
print('wrote results/stage3_official_eval_metrics.json')
PY
  else
    echo 'Skip official eval: third_party/test-suite-sql-eval/evaluation.py not found'
  fi
"