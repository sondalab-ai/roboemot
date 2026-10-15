#!/usr/bin/env bash
# Headless MuJoCo Reachy Mini daemon on localhost:8000 for `pytest -m sim` (SPEC 14).
# Needs: pip install "reachy-mini[mujoco]" plus GStreamer/GObject system packages.
set -euo pipefail
# EGL is Linux-only; macOS uses GLFW.
if [[ "$(uname)" == "Darwin" ]]; then default_gl=glfw; else default_gl=egl; fi
export MUJOCO_GL=${MUJOCO_GL:-$default_gl}
exec reachy-mini-daemon --sim --headless --no-media --no-preload-datasets \
  --dataset-update-interval 0 --no-wake-up-on-start "$@"
