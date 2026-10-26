#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "$0")/env.sh"
if [ -f "$NUSCENES_ROOT/nuscenes2d_temporal_infos_val_new.pkl" ] && [ -f "$NUSCENES_ROOT/nuscenes2d_temporal_infos_train_new.pkl" ]; then
 printf 'Existing temporal indexes found; no preprocessing repeated.\n'; exit 0
fi
if [ ! -f "$NUSCENES_ROOT/v1.0-mini/sample.json" ]; then
 printf 'Mini data missing. Run download_data.py or configure NUSCENES_ROOT.\n' >&2; exit 1
fi
cd "$MV2DFUSION_UPSTREAM"
"$MV2D_PYTHON" tools/create_data_nusc.py --root-path "$NUSCENES_ROOT" --version v1.0-mini --extra-tag nuscenes2d --max-sweeps 10
