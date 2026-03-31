# CHPC Usage Guide for `nice-sar`

Follow these steps to run the [nice-sar notebooks](https://github.com/bullocke/nice-sar/tree/main/notebooks) (e.g., `00_data_discovery.ipynb`) on high-compute nodes using Open OnDemand.

---

## 1. Initial Environment Setup (One-time)
Run these commands in a CHPC terminal (via [Shell Access](https://www.chpc.utah.edu/documentation/software/ondemand.php#shell-access)) to create your persistent `nisar` environment.

```bash
# Clone the repository
git clone [https://github.com/bullocke/nice-sar.git](https://github.com/bullocke/nice-sar.git)
cd nice-sar

# Create the environment (includes notebook + ipykernel for OnDemand compatibility)
conda env create -n nisar -f environment.yml
conda activate nisar

# Install the nice-sar package in editable mode
pip install -e .
```

---

## 2. Launching via Open OnDemand
1.  Navigate to the [Jupyter - CHPC OnDemand](https://ondemand.chpc.utah.edu/pun/sys/dashboard/batch_connect/sys/jupyter_app/session_contexts/new) page.
2.  **Jupyter interface:** Select **Notebook**.
3.  **Jupyter Python version:** Select **Custom (Environment Setup below)**.
4.  **Environment Setup for Custom Python:** Paste the following block (**Note:** I set up my environment using `conda` and in a way that requires loading the 2 modules first, then activating the environment. You may have a different setup, so adjust as needed):
    ```bash
    conda activate nisar
    cd ~/nice-sar
    ```
5.  **Cluster:** Select what is available for you.
6.  **Account, partition, qos:** Select your lab's high-compute allocation, if available.
7.  **Resources:** Set your preferred **Cores** and **Hours**, then click **Launch**.

---

## 3. Running the Notebooks
* Once the session starts, your browser will open to the `nice-sar` directory.
* Open `notebooks/00_data_discovery.ipynb`.
* **Authentication:** Ensure you have your NASA Earthdata credentials in your `~/.netrc` file as described in the [00_data_discovery.ipynb](https://github.com/bullocke/nice-sar/blob/main/notebooks/00_data_discovery.ipynb) setup section to allow for non-interactive data access.
```
