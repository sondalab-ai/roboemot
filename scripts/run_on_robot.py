#!/usr/bin/env python3
"""Start the installed RoboEmot app on a robot and print its web UI address (SPEC 4A).

Runs on a PC on the robot's LAN. The flow is robot-agnostic (``roboemot.install.run``); the
robot model's specifics come from the installer its platform package registers.

    python scripts/run_on_robot.py                        # find the robot (mDNS), start the app
    python scripts/run_on_robot.py --host 192.168.1.20 -y
    python scripts/run_on_robot.py --dry-run              # what it would do
"""

import sys
from pathlib import Path

from roboemot.install.run import main

if __name__ == "__main__":
    sys.exit(main(root=Path(__file__).resolve().parents[1]))
