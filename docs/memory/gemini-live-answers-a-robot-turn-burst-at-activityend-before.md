---
name: gemini-live-answers-a-robot-turn-burst-at-activityend-before
description: Gemini Live answers a robot-turn burst at activityEnd, before response.create, so the per-turn conversation hint (addressed/in_progress/none) must ride the burst's session.update (PR 274, 2026-11-18); GeminiWire never retracts removed lines, so a hint flip piles up context turns.
metadata:
  node_type: memory
  loadout_kind: memory
  scope: repo
  created: 2026-11-18
---
