#!/bin/bash

pip install uv

# uv venv --python 3.12 --seed
# source .venv/bin/activate

uv venv --python 3.12 --seed ~/myvllm
source ~/myvllm/bin/activate

# deactivate
# rm -rf .venv

# build from source
VLLM_USE_PRECOMPILED=1 pip install --editable .



## developer
pip install pytest
# 
pip install -r requirements/dev.txt --extra-index-url https://download.pytorch.org/whl/cu128
# Linting, formatting and static type checking
pre-commit install --hook-type pre-commit --hook-type commit-msg
# You can manually run pre-commit with
# pre-commit run --all-files

# git commit -s -m "commit message"


# git config --global user.email ""
# git config --global user.name ""



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


# reru test
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



# ================DEEPEP===============

# ============= NVSHMEM ================
#  Installation Guide https://github.com/deepseek-ai/DeepEP/blob/main/third-party/README.md
# NVSHMEM is a library for high-performance, distributed memory programming.
# wget "https://developer.nvidia.com/downloads/assets/secure/nvshmem/nvshmem_src_3.2.5-1.txz"
# tar -xf nvshmem_src_3.2.5-1.txz 
# cd /scratch/workspaceblobstore/WebUpload/DeepEP/nvshmem_src
# git apply /scratch/workspaceblobstore/WebUpload/DeepEP/third-party/nvshmem.patch

# which nvcc
export CUDA_HOME=/usr/local/cuda
# disable all features except IBGDA
export NVSHMEM_IBGDA_SUPPORT=1

export NVSHMEM_SHMEM_SUPPORT=0
export NVSHMEM_UCX_SUPPORT=0
export NVSHMEM_USE_NCCL=0
export NVSHMEM_PMIX_SUPPORT=0
export NVSHMEM_TIMEOUT_DEVICE_POLLING=0
export NVSHMEM_USE_GDRCOPY=0
export NVSHMEM_IBRC_SUPPORT=0
export NVSHMEM_BUILD_TESTS=0
export NVSHMEM_BUILD_EXAMPLES=0
export NVSHMEM_MPI_SUPPORT=0
export NVSHMEM_BUILD_HYDRA_LAUNCHER=0
export NVSHMEM_BUILD_TXZ_PACKAGE=0
export NVSHMEM_TIMEOUT_DEVICE_POLLING=0

cmake -G Ninja -S . -B build -DCMAKE_INSTALL_PREFIX=$HOME/nvshmem
cmake --build build/ --target install


export NVSHMEM_DIR=$HOME/nvshmem  # Use for DeepEP installation
export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:$LD_LIBRARY_PATH"
export PATH="${NVSHMEM_DIR}/bin:$PATH"


# ### start to install deep_ep
# NVSHMEM_DIR=$HOME/nvshmem python setup.py build
# # You may modify the specific SO names according to your own platform
# ln -s build/lib.linux-x86_64-cpython-312/deep_ep_cpp.cpython-312-x86_64-linux-gnu.so
### execute below under DeepEP root directory

NVSHMEM_DIR=$HOME/nvshmem pip install --no-build-isolation -e .


##
# Run test cases
# NOTES: you may modify the `init_dist` function in `tests/utils.py`
# according to your own cluster settings, and launch into multiple nodes 
python tests/test_intranode.py
python tests/test_internode.py
python tests/test_low_latency.py