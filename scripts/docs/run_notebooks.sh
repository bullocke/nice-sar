#!/usr/bin/env bash
# Execute the tutorials in place and refresh the figures used by the docs site.
#
# The notebooks stream NISAR data, so they run locally with NASA Earthdata
# credentials (~/.netrc or EARTHDATA_USERNAME/EARTHDATA_PASSWORD), not in CI.
# CI only checks the saved outputs (scripts/docs/check_notebooks.py).
#
# Usage:
#   scripts/docs/run_notebooks.sh              # all notebooks
#   scripts/docs/run_notebooks.sh 06 11        # only notebooks starting with 06 and 11
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO/notebooks"

if [ "$#" -eq 0 ]; then
    notebooks=( [0-9][0-9]_*.ipynb )
else
    notebooks=()
    for prefix in "$@"; do notebooks+=( "${prefix}"_*.ipynb ); done
fi

for nb in "${notebooks[@]}"; do
    echo "==> $nb"
    start=$SECONDS
    jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=1800 "$nb"
    echo "    done in $(( SECONDS - start )) s, $(du -h "$nb" | cut -f1)"
done

python "$REPO/scripts/docs/export_figures.py"
python "$REPO/scripts/docs/check_notebooks.py"
