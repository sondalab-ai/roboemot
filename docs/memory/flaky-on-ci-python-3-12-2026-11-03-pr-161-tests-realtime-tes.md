---
name: flaky-on-ci-python-3-12-2026-11-03-pr-161-tests-realtime-tes
description: Realtime CI flakes 2026-11-03 (IndexError on rig.events[-1] in test_realtime_turns.py on PR 161 and test_realtime_tone.py on PR 165): root cause was ConversationWorker counting 'turns' before publishing the conversation.turn record, while rt_rig.turn() waits on that count; fixed in PR 166 (count after publish, regression test). Not a stall-timeout issue.
metadata:
  node_type: memory
  loadout_kind: memory
  scope: repo
  created: 2026-11-03
---
