#!/usr/bin/env bash
# Run a Jupyter notebook headlessly via nbconvert.
#
# Usage:
#   ./scripts/run_notebook.sh notebooks/02_read_gcov.ipynb
#   ./scripts/run_notebook.sh notebooks/02_read_gcov.ipynb --timeout 1800
#
# Requires: jupyter nbconvert (pip install jupyter)
set -euo pipefail

NOTEBOOK="${1:?Usage: run_notebook.sh <notebook.ipynb> [--timeout SECONDS]}"
TIMEOUT="${2:---ExecutePreprocessor.timeout=3600}"

if [[ "$TIMEOUT" == "--timeout" ]]; then
    TIMEOUT="--ExecutePreprocessor.timeout=${3:-3600}"
fi

echo "=== Running notebook: ${NOTEBOOK} ==="
echo "    Timeout: ${TIMEOUT}"

jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.kernel_name=python3 \
    "${TIMEOUT}" \
    "${NOTEBOOK}"

echo "=== Notebook completed: ${NOTEBOOK} ==="
