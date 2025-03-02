#!/bin/bash

# Navigate to the target directory
cd /scratch/AzureBlobStorage_CODE/scratch/workspaceblobstore/WebUpload/vllm || {
  echo "Failed to navigate to the directory. Exiting..."
  exit 1
}

# Mark the directory as a safe Git directory
git config --global --add safe.directory /scratch/AzureBlobStorage_CODE/scratch/workspaceblobstore/WebUpload/vllm

# Create the Conda environment
conda create -n vllm python=3.12 -y

# Activate the Conda environment
# This only works in interactive shells, so we source conda.sh first
source $(conda info --base)/etc/profile.d/conda.sh
conda init
source ~/.bashrc
conda activate vllm

# Install vLLM with precompiled binaries
VLLM_USE_PRECOMPILED=1 pip install --editable .