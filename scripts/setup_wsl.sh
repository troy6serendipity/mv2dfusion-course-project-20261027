#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "$0")/env.sh"
if [ -x "$MV2D_ENV_PREFIX/bin/python" ]; then
 printf 'Reusing existing environment; no packages installed or upgraded.\n'
 "$MV2D_PYTHON" "$MV2D_REPO_ROOT/scripts/check_runtime.py"
 exit 0
fi
if [ -z "$CONDA_EXE" ]; then CONDA_EXE="$(command -v conda || true)"; fi
if [ -z "$CONDA_EXE" ]; then printf 'Install Miniconda in the chosen Linux filesystem and set CONDA_EXE.\n' >&2; exit 1; fi
"$CONDA_EXE" create -y --prefix "$MV2D_ENV_PREFIX" --file "$MV2D_REPO_ROOT/env/conda-linux-64-explicit.txt"
source "$MV2D_REPO_ROOT/scripts/env.sh"
"$MV2D_PYTHON" -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
"$MV2D_PYTHON" -m pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html
"$MV2D_PYTHON" -m pip install torch-scatter==2.1.1 -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
# Intentionally restore the observed package set without resolving incompatible
# legacy numba/networkx/trimesh declarations. The differences are documented.
"$MV2D_PYTHON" -m pip install --no-deps --no-build-isolation -r "$MV2D_REPO_ROOT/env/requirements-runtime.txt"
bash "$MV2D_REPO_ROOT/scripts/clone_upstream.sh"
MMDET_SRC="$MV2D_REPO_ROOT/.runtime/src/mmdetection3d"
MMDET_PIN=c9541b0db89498fdea5cafd05b7b17f7b625b858
if [ ! -d "$MMDET_SRC/.git" ]; then
 mkdir -p "$(dirname "$MMDET_SRC")"
 git clone --no-checkout https://github.com/open-mmlab/mmdetection3d.git "$MMDET_SRC"
 git -C "$MMDET_SRC" checkout --detach "$MMDET_PIN"
elif [ "$(git -C "$MMDET_SRC" rev-parse HEAD)" != "$MMDET_PIN" ]; then
 printf 'Existing mmdetection3d revision differs; use another runtime path.\n' >&2; exit 1
fi
"$MV2D_PYTHON" -m pip install --no-deps --no-build-isolation -e "$MMDET_SRC"
"$MV2D_PYTHON" "$MV2D_REPO_ROOT/scripts/check_runtime.py"
