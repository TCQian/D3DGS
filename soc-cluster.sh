#!/bin/bash

#SBATCH --job-name=GaussianFlow                # Job name
#SBATCH --time=2:00:00                   # Time limit hrs:min:sec
#SBATCH --gres=gpu:h100-47:1             # must use this GPU, since pytorch3d relied on it
#SBATCH --mail-type=ALL                  # Get email for all status updates
#SBATCH --mail-user=e0407638@u.nus.edu   # Email for notifications
#SBATCH --mem=16G                        # Request 16GB of memory

source ~/.bashrc
conda activate GaussianFlow

python ./scripts/run_cmu_bench.py