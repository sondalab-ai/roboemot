#!/usr/bin/env bash
# Copy the configs the Reachy Mini app needs into its package data (installed on the robot).
set -euo pipefail
cd "$(dirname "$0")/.."
D=packages/roboemot-reachymini/src/roboemot_reachymini/data
mkdir -p "$D/presets"
cp configs/manifests/reachy-mini-wireless.yaml "$D/"
cp configs/presets/*.yaml "$D/presets/"
