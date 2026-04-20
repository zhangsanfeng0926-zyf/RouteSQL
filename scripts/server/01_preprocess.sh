#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

run_step "01_preprocess" \
  "${PYTHON_BIN}" -W ignore -u ./scripts/python_tools/data_preprocess.py --data_type "${DATA_TYPE}"