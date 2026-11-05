---
name: python-3-11-asyncio-wait-for-returns-the-inner-result-and-dr
description: Python 3.11 asyncio.wait_for returns the inner result and drops an outside task.cancel() that lands in the same loop step as the inner future completing (3.12 rewrote wait_for). RoboEmot replaced wait_for with 'async with asyncio.timeout()' on barge-in-cancellable waits in PR 186 (2026-11-05); keep that pattern for new code.
metadata:
  node_type: memory
  loadout_kind: memory
  scope: repo
  created: 2026-11-05
---
