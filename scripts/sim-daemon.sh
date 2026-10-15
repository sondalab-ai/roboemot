#!/usr/bin/env bash
# Headless MuJoCo Reachy Mini daemon on localhost:8000 for `pytest -m sim` (SPEC 14).
# Needs: pip install "reachy-mini[mujoco]" plus GStreamer/GObject system packages.
set -euo pipefail
export MUJOCO_GL=${MUJOCO_GL:-egl}
exec reachy-mini-daemon --sim --headless --no-media --no-preload-datasets \
  --dataset-update-interval 0 --no-wake-up-on-start "$@"
