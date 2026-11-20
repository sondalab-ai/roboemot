---
name: 2026-11-20-garbled-sentito-after-a-gemini-idle-close-was-vos
description: 2026-11-20: garbled 'Sentito' after a Gemini idle close was Vosk text, not Gemini (no session -> local ASR -> text turn); fixed by PR 291 (connect at speech start, wait within transcript_timeout_s less connect_wait_reserve_s, Live shows robot/cloud source). Over the radio the face rule (PR 289) reads DoA only behind echo_cancelled; Gemini accepts languageCodes it-IT (PR 290) but still wrote 'cigar' over music. test_a_server_response_while_the_turns_answer_runs_is_a_duplicate failed once in CI python-vision (spoken 0 != 1), passed on rerun: suspected timing race.
metadata:
  node_type: memory
  loadout_kind: memory
  scope: repo
  created: 2026-11-20
---
