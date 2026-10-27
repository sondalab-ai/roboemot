---
name: ci-python-3-11-3-12-and-sim-jobs-install-core-without-the-vi
description: CI python (3.11/3.12) and sim jobs install core without the vision extra (no cv2); tests that need cv2 must pytest.importorskip("cv2") and be listed in the python-vision job in .github/workflows/ci.yml — slice U (PR 92) failed CI twice on this.
metadata:
  node_type: memory
  loadout_kind: memory
  scope: repo
  created: 2026-10-22
---
