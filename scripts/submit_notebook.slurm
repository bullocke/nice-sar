#!/bin/bash
#SBATCH --requeue
#SBATCH --account=wangj-np
#SBATCH --partition=wangj-np
#SBATCH --qos=wangj-np
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --job-name=nice-sar-nb
#SBATCH -o logs/notebook_%j.out
#SBATCH -e logs/notebook_%j.err

# ── Configuration ────────────────────────────────────────────────────────────
# Change NOTEBOOK to the notebook you want to run, or pass via:
#   sbatch --export=NOTEBOOK=notebooks/02_read_gcov.ipynb scripts/submit_notebook.slurm
NOTEBOOK="${NOTEBOOK:-notebooks/02_read_gcov.ipynb}"

# ── Environment Setup ────────────────────────────────────────────────────────
module load miniforge3/latest
conda activate nisar

# Ensure logs directory exists
mkdir -p logs

# Change to the nice-sar project root
cd /uufs/chpc.utah.edu/common/home/dycelab/users/eric/code/NISAR/nice-sar

echo "=== CHPC Notebook Runner ==="
echo "    Node:     $(hostname)"
echo "    CPUs:     ${SLURM_CPUS_PER_TASK}"
echo "    Memory:   ${SLURM_MEM_PER_NODE}MB"
echo "    Notebook: ${NOTEBOOK}"
echo "    Started:  $(date)"

# ── Run Notebook ──────────────────────────────────────────────────────────────
bash scripts/run_notebook.sh "${NOTEBOOK}" --timeout 3600

echo "=== Finished: $(date) ==="
