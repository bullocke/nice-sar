# CHPC Guide

Run `nice-sar` on the University of Utah Center for High Performance Computing (CHPC).

## Environment Setup

### Option 1: Micromamba (recommended)

```bash
# Install micromamba
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)

# Create environment
git clone https://github.com/bullocke/nice-sar.git
cd nice-sar
micromamba create -f environment.yml
micromamba activate nisar
pip install -e ".[dev]"
```

### Option 2: Miniforge

```bash
module load miniforge3
git clone https://github.com/bullocke/nice-sar.git
cd nice-sar
conda env create -n nisar -f environment.yml
conda activate nisar
pip install -e ".[dev]"
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

## Notebooks on Open OnDemand

Run the [tutorials](notebooks.md) interactively on a compute node:

1. Open the [Jupyter app on CHPC OnDemand](https://ondemand.chpc.utah.edu/pun/sys/dashboard/batch_connect/sys/jupyter_app/session_contexts/new).
2. Set **Jupyter interface** to *Notebook* and **Jupyter Python version** to *Custom (Environment Setup below)*.
3. In **Environment Setup for Custom Python**, activate the environment and move to the repository (add any `module load` lines your setup needs first):

    ```bash
    conda activate nisar
    cd ~/nice-sar
    ```

4. Choose your cluster, account, partition, cores and hours, then click **Launch**.
5. Open a notebook from `notebooks/`. Make sure `~/.netrc` holds your Earthdata credentials (see above) so the notebooks can stream data without prompting.

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

## Streaming from CHPC

NISAR data live in AWS us-west-2. Direct S3 reads (`get_s3_filesystem`) only work from inside that AWS region, so from CHPC stream over HTTPS and read only the window you need:

```python
from nice_sar.auth import get_https_filesystem, login
from nice_sar.io.products import read_gcov

login()
fs = get_https_filesystem()
hv = read_gcov(url, polarization="HV", filesystem=fs, bbox=(-74.36, 0.76, -74.16, 0.92))
```

For many granules or full frames, download once with `nice-sar download` or `nice-sar subset` and work from local files.

## Tips

- Use `--mem=256G` when reading full GCOV frames (a single band is about 1 GB)
- Set `dask` chunk sizes to match available memory
- The `wangj-np` partition has priority scheduling for the Wang group
- Store intermediate results in `/scratch/general/vast/YOUR_UNAME/`
