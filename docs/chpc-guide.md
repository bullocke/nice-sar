# CHPC Guide

Run `nice-sar` on the University of Utah Center for High Performance Computing (CHPC).

## Environment Setup

### Option 1: Micromamba (recommended)

```bash
# Install micromamba
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)

# Create environment
micromamba create -n nisar python=3.12 -y
micromamba activate nisar
pip install nice-sar[dev]
```

### Option 2: Miniforge

```bash
module load miniforge3
conda create -n nisar python=3.12 -y
conda activate nisar
pip install nice-sar[dev]
```

## Earthdata Credentials

Configure `~/.netrc` for non-interactive authentication:

```
machine urs.earthdata.nasa.gov
    login YOUR_USERNAME
    password YOUR_PASSWORD
```

Set permissions:

```bash
chmod 600 ~/.netrc
```

## SLURM Job Submission

The project includes a SLURM template at `scripts/submit_notebook.slurm`.

### Example: Run a notebook

```bash
sbatch scripts/submit_notebook.slurm notebooks/02_read_gcov.ipynb
```

### Custom SLURM script

```bash
#!/bin/bash
#SBATCH --job-name=nisar-processing
#SBATCH --account=wangj-np
#SBATCH --partition=wangj-np
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

module load miniforge3
conda activate nisar

python your_script.py
```

### Headless notebook execution

Use the provided helper script:

```bash
bash scripts/run_notebook.sh notebooks/03_preprocessing.ipynb
```

This runs the notebook via `jupyter nbconvert --execute` with the `Agg` matplotlib backend for
headless rendering.

## S3 Access from CHPC

NISAR data is in AWS us-west-2. CHPC has good connectivity to AWS, so S3 direct reads work well:

```python
from nice_sar.auth import login, get_s3_filesystem

login()
fs = get_s3_filesystem()
# Now open HDF5 files directly from S3
```

!!! note
    S3 credentials from `earthaccess` are temporary (1 hour). For long jobs, call
    `get_s3_filesystem()` periodically to refresh.

## Tips

- Use `--mem=256G` for large GCOV products (full-swath quad-pol)
- Set `dask` chunk sizes to match available memory
- The `wangj-np` partition has priority scheduling for the Wang group
- Store intermediate results in `/scratch/general/vast/YOUR_UNAME/`
