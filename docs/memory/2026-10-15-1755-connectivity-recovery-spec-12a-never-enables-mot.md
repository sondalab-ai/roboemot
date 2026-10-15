---
name: 2026-10-15-1755-connectivity-recovery-spec-12a-never-enables-mot
description: Connectivity recovery (SPEC 12A) never enables motors: the loop reads driver.motors_enabled and tracks the measured pose while the platform keeps them off, because the app must never wake the robot (SPEC 4A).
metadata:
  node_type: memory
  loadout_kind: decision
  scope: repo
  created: 2026-10-15
  status: active
---

# Connectivity recovery (SPEC 12A) never enables motors: the loop reads driver.motors_enabled and tracks the measured pose while the platform keeps them off, because the app must never wake the robot (SPEC 4A).

## Context

## Decision

## Alternatives considered

## Rationale
