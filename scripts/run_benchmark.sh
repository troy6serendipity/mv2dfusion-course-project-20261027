#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "$0")/env.sh"
"$MV2D_PYTHON" "$MV2D_REPO_ROOT/experiments/course_benchmark.py" "$@"
