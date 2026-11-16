#!/bin/bash
# Template for a SLURM cluster with one CUDA GPU (>= 16 GB memory). Adjust partition/account.
#SBATCH --job-name=router-modernbert
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/modernbert_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA GPU visible'; print(torch.cuda.get_device_name(0))"
# Development comparison (selection pool only). Resumable: one JSON per rotation.
python scripts/05_dev_compare.py modernbert
