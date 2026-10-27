---
name: m3-decision-2026-10-17-robot-converses-with-children-overrid
description: M3 decision 2026-10-17: robot converses with children (overrides SPEC 'no conversation while minor'): only allowed_for_minors routes, else built-in child-safe lines; child register max 2 sentences; output filter. Plan: docs/plans/2026-10-17-m3-voice.md on epic/m3-voice.
metadata:
  node_type: memory
  loadout_kind: memory
  scope: repo
  created: 2026-10-17
---

Superseded 2026-10-22 (owner decision, SPEC 8, 12): the route gate is gone (`allowed_for_minors` and `safety.minors_allowed_llm` removed); every route is used while a minor is present. The child register, the minor content rules and the output filter stay; child-safe lines only when no route can answer.
