#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "$0")/env.sh"
PIN=e4bc2d93f4009ce7210188da79860730676398b6
if [ -d "$MV2DFUSION_UPSTREAM/.git" ]; then
 ACTUAL="$(git -C "$MV2DFUSION_UPSTREAM" rev-parse HEAD)"
 if [ "$ACTUAL" != "$PIN" ]; then printf 'Existing upstream revision differs. Use a new destination; no reset performed.\n' >&2; exit 1; fi
else
 if [ -e "$MV2DFUSION_UPSTREAM" ]; then printf 'Upstream path exists and is not a checkout; refusing overwrite.\n' >&2; exit 1; fi
 mkdir -p "$(dirname "$MV2DFUSION_UPSTREAM")"
 git clone --no-checkout https://github.com/wangzt-halo/MV2DFusion.git "$MV2DFUSION_UPSTREAM"
 git -C "$MV2DFUSION_UPSTREAM" checkout --detach "$PIN"
fi
PATCH="$MV2D_REPO_ROOT/compatibility.patch"
if git -C "$MV2DFUSION_UPSTREAM" apply --reverse --check "$PATCH" >/dev/null 2>&1; then
 printf 'Optional AV2 registration patch already applied.\n'
elif git -C "$MV2DFUSION_UPSTREAM" apply --check "$PATCH"; then
 git -C "$MV2DFUSION_UPSTREAM" apply "$PATCH"
else
 printf 'Patch does not apply; inspect local modifications.\n' >&2; exit 1
fi
git -C "$MV2DFUSION_UPSTREAM" rev-parse HEAD
