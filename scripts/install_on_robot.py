#!/usr/bin/env python3
"""Install or uninstall the RoboEmot app on a robot (SPEC 4A, Packaging).

Runs on a PC on the robot's LAN. The flow is robot-agnostic (``roboemot.install``); the robot
model's specifics come from the installer its platform package registers (Reachy Mini Wireless:
``roboemot_reachymini.install``). Every prompt has a flag, so it also runs unattended; without a
terminal, a missing flag stops the run before anything changes (no default is ever taken).

    python scripts/install_on_robot.py                         # interactive wizard
    python scripts/install_on_robot.py --dry-run               # read-only checks
    python scripts/install_on_robot.py --host 192.168.1.20 --user pollen --dev --start --yes
                                    # --yes keeps ~/.cache/uv; add --clear-caches to free it
    python scripts/install_on_robot.py --uninstall --dry-run   # what uninstall would undo
    python scripts/install_on_robot.py --uninstall             # leave the robot as it was

--dev sets the keys of configs/dev-unmeasured.yaml in the app config (app.yaml in the robot's data
folder), so the app runs before the manifest dynamics are measured (SPEC 12). Without it the app
refuses to start. Voice is enabled in the same file (voice.enabled) unless --no-voice; other keys
in the file are kept (SPEC 4A, Installer).
"""

import sys
from pathlib import Path

from roboemot.install import main

if __name__ == "__main__":
    sys.exit(main(root=Path(__file__).resolve().parents[1]))
