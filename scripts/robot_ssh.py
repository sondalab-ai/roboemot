#!/usr/bin/env python3
"""SSH into the robot, install your key, or fetch its debug report (SPEC 4A, 12A).

Runs on a PC on the robot's LAN. The logic is robot-agnostic (``roboemot.install.ssh``); the
default host and SSH user come from the installer its platform package registers.

    python scripts/robot_ssh.py                          # find the robot (mDNS), interactive shell
    python scripts/robot_ssh.py -- journalctl -n 50      # run one command
    python scripts/robot_ssh.py --host 192.168.1.20 --user pollen
    python scripts/robot_ssh.py --copy-id                # install your SSH key (ssh-copy-id)
    python scripts/robot_ssh.py --debug > report.json    # report built on the robot
    python scripts/robot_ssh.py --dry-run                # print the ssh command only
"""

import sys
from pathlib import Path

from roboemot.install.ssh import main

if __name__ == "__main__":
    sys.exit(main(root=Path(__file__).resolve().parents[1]))
