---
name: 2026-11-03-1439-vision-stack-stays-4-models-yunet-sface-nanodet
description: Vision stack stays 4 models (YuNet, SFace, NanoDet, DINOv2): research 2026-11-03 found no single-pass detect+face-ID+objects+ReID net with permissive weights (InsightFace/SapiensID non-commercial, YOLO AGPL). Save CPU by scheduling instead: V-1 per-stage timing, V-2 motion-gated NanoDet with head-stillness gate (no ego-motion warp: head not rotating about camera centre, joint_state not synced to capture).
metadata:
  node_type: memory
  loadout_kind: decision
  scope: repo
  created: 2026-11-03
  status: active
---

# Vision stack stays 4 models (YuNet, SFace, NanoDet, DINOv2): research 2026-11-03 found no single-pass detect+face-ID+objects+ReID net with permissive weights (InsightFace/SapiensID non-commercial, YOLO AGPL). Save CPU by scheduling instead: V-1 per-stage timing, V-2 motion-gated NanoDet with head-stillness gate (no ego-motion warp: head not rotating about camera centre, joint_state not synced to capture).

## Context

## Decision

## Alternatives considered

## Rationale
