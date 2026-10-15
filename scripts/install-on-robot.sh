#!/usr/bin/env bash
# Manual install of the RoboEmot app on Reachy Mini Wireless, without the HF app store
# (SPEC 4A, Packaging). Run from the repo on a PC on the same LAN as the robot.
#   scripts/install-on-robot.sh [--dev] [host]   default host: reachy-mini.local
# --dev installs configs/dev-unmeasured.yaml as the app config, allowing the app to run
# before the manifest dynamics are measured (SPEC 12). Without it the app refuses to start.
set -euo pipefail
DEV=0
if [ "${1:-}" = "--dev" ]; then DEV=1; shift; fi
HOST=${1:-reachy-mini.local}
cd "$(dirname "$0")/.."
scripts/sync-package-data.sh
D=packages/roboemot-reachymini/src/roboemot_reachymini/data
rm -f "$D/app.yaml"
if [ "$DEV" = 1 ]; then cp configs/dev-unmeasured.yaml "$D/app.yaml"; fi
TMP=/tmp/roboemot-install
ssh "pollen@$HOST" "rm -rf $TMP && mkdir -p $TMP"
scp -r packages/roboemot-core packages/roboemot-reachymini "pollen@$HOST:$TMP/"
ssh "pollen@$HOST" "uname -m; /venvs/apps_venv/bin/pip install $TMP/roboemot-core $TMP/roboemot-reachymini"
echo "Installed. Start it with:"
echo "  curl -X POST http://$HOST:8000/api/apps/start-app/roboemot"
