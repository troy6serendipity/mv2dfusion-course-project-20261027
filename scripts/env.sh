#!/usr/bin/env bash
MV2D_REPO_ROOT="$(cd "$(dirname "$BASH_SOURCE")/.." && pwd)"
export MV2D_REPO_ROOT
if [ -f "$MV2D_REPO_ROOT/config/local.env.sh" ]; then source "$MV2D_REPO_ROOT/config/local.env.sh"; fi
if [ -z "$MV2D_ENV_PREFIX" ]; then export MV2D_ENV_PREFIX="$MV2D_REPO_ROOT/.runtime/env"; fi
if [ -z "$MV2DFUSION_UPSTREAM" ]; then export MV2DFUSION_UPSTREAM="$MV2D_REPO_ROOT/upstream/MV2DFusion"; fi
if [ -z "$NUSCENES_ROOT" ]; then export NUSCENES_ROOT="$MV2D_REPO_ROOT/data/nuscenes"; fi
if [ -z "$MV2DFUSION_CHECKPOINT" ]; then export MV2DFUSION_CHECKPOINT="$MV2D_REPO_ROOT/weights/mv2dfusion_r50_nusc_latest.pth"; fi
if [ -z "$MV2D_CACHE_ROOT" ]; then export MV2D_CACHE_ROOT="$MV2D_REPO_ROOT/.runtime/cache"; fi
if [ -z "$MV2D_DEMO_OUTPUT" ]; then export MV2D_DEMO_OUTPUT="$MV2D_REPO_ROOT/demo_outputs"; fi
if [ -z "$MV2D_EXPERIMENT_OUTPUT" ]; then export MV2D_EXPERIMENT_OUTPUT="$MV2D_REPO_ROOT/experiment_outputs"; fi
export CUDA_HOME="$MV2D_ENV_PREFIX"
export PATH="$MV2D_ENV_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$MV2D_ENV_PREFIX/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$MV2D_REPO_ROOT/compat:$MV2DFUSION_UPSTREAM"
export MV2DFUSION_NUSCENES_ONLY=1
if [ -z "$TORCH_CUDA_ARCH_LIST" ]; then export TORCH_CUDA_ARCH_LIST=8.6; fi
export MAX_JOBS=4
export OMP_NUM_THREADS=4
export TMPDIR="$MV2D_CACHE_ROOT/tmp"
export PIP_CACHE_DIR="$MV2D_CACHE_ROOT/pip"
export CONDA_PKGS_DIRS="$MV2D_CACHE_ROOT/conda"
export TORCH_HOME="$MV2D_CACHE_ROOT/torch"
export HF_HOME="$MV2D_CACHE_ROOT/huggingface"
export MPLCONFIGDIR="$MV2D_CACHE_ROOT/matplotlib"
export CUDA_CACHE_PATH="$MV2D_CACHE_ROOT/cuda"
export TORCH_EXTENSIONS_DIR="$MV2D_CACHE_ROOT/torch_extensions"
export PYTHONPYCACHEPREFIX="$MV2D_CACHE_ROOT/pycache"
export PYTHONNOUSERSITE=1
if [ -x "$MV2D_ENV_PREFIX/bin/x86_64-conda-linux-gnu-cc" ]; then
 export CC="$MV2D_ENV_PREFIX/bin/x86_64-conda-linux-gnu-cc"
 export CXX="$MV2D_ENV_PREFIX/bin/x86_64-conda-linux-gnu-c++"
fi
if [ -x "$MV2D_ENV_PREFIX/bin/python" ]; then export MV2D_PYTHON="$MV2D_ENV_PREFIX/bin/python"; else export MV2D_PYTHON=python3; fi
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$CUDA_CACHE_PATH" "$TORCH_HOME"
