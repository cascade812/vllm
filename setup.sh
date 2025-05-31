#!/bin/bash

# Create the Conda environment
conda create -n vllm python=3.12 -y

#conda env remove --name sequence_env

# Activate the Conda environment
# This only works in interactive shells, so we source conda.sh first
source $(conda info --base)/etc/profile.d/conda.sh
conda init
source ~/.bashrc
conda activate vllm

# Install vLLM with precompiled binaries
# pip install torch==2.7.0 torchvision==0.22.0+cu126 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
VLLM_USE_PRECOMPILED=1 pip install --editable .




pip install pytest
# 
pip install -r requirements/dev.txt --extra-index-url https://download.pytorch.org/whl/cu128
# Linting, formatting and static type checking
pre-commit install --hook-type pre-commit --hook-type commit-msg
# You can manually run pre-commit with
# pre-commit run --all-files

# git commit -s -m "commit message"


git config --global user.email "cascade812@outlook.com"
git config --global user.name "cascade812"



# ===== how to do rebase ============
# First, make sure you're on the branch you want to rebase
# git checkout feature-branch

# Then rebase onto the target branch
# git rebase main
# need rebuilt vllm after rebase


# ===== how to do pick commits from other branch into local changes ============
# First, make sure you're on the branch you want to rebase
# git checkout branch_name
# Cherry-pick without committing (newest last)
# git cherry-pick --no-commit SHA1 SHA2

# git commit --amend --no-edit
# git push --force


# ======= how to do force sync with upstream ============
# # Fetch latest changes
# git fetch upstream
# git fetch origin

# # Switch to main and force sync
# git checkout main
# git reset --hard upstream/main
# git push origin main --force



# ======= VLLM configs ============
# export CUDA_VISIBLE_DEVICES=0,1,2

# VLLM_DISABLE_COMPILE_CACHE=1
# VLLM_USE_V1=1 VLLM_LOGGING_LEVEL=DEBUG

# rm -rf ~/.cache/vllm/torch_compile_cache/