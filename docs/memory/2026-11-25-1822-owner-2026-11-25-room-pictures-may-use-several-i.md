---
name: 2026-11-25-1822-owner-2026-11-25-room-pictures-may-use-several-i
description: Owner 2026-11-25: room pictures may use several image routes in llm_routes order, priority not round robin (paid Claude only when free Gemini refuses); a 429/400/415/unusable route is skipped until the next day (Anthropic low credit is a 400, so never 'unsupported forever'); the same JPEG is retried on the next route without a new sweep. The first picture after start/switch-on may sweep while engaged or alone, once, with no sweep_wait_s.
metadata:
  node_type: memory
  loadout_kind: decision
  scope: repo
  created: 2026-11-25
  status: active
---

# Owner 2026-11-25: room pictures may use several image routes in llm_routes order, priority not round robin (paid Claude only when free Gemini refuses); a 429/400/415/unusable route is skipped until the next day (Anthropic low credit is a 400, so never 'unsupported forever'); the same JPEG is retried on the next route without a new sweep. The first picture after start/switch-on may sweep while engaged or alone, once, with no sweep_wait_s.

## Context

## Decision

## Alternatives considered

## Rationale
