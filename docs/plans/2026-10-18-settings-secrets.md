# Settings, secrets and web UI on the Sondalab kit — epic plan

Approved by the owner on 2026-10-18. Integration branch `epic/settings-secrets`, one PR per slice, one merge to `main`.
Estimate: 5 slices, ~14 agent runs (Sonnet authors and reviewers; Opus security review on slices 1 and 4).

## Slices
1. **Secret store** (SPEC 8, 12). Keys in a 0600 file in the app data folder (owner choice), set with a CLI command
   and the installer (recorded for uninstall). `${secret:name}` route keys resolve at startup and on reload; keys never
   in logs, the debug report or the web UI; missing secret → the route is unavailable with a clear warning.
2. **Piper hash-pinning.** `piper-tts` and its dependencies locked with hashes for the installer (robot, aarch64) and
   the CI `models` job (x86_64); one source of truth for the version (`roboemot.install.PIPER_TTS_VERSION`).
3. **Nod on "yes", shake on "no"** (SPEC 7, 8 fast path). React to `speech.keyword.yes` / `.no` through
   `ExpressionRenderer.request`, except when the robot itself just asked a question (the reply is an answer, not a
   reaction); a param for the window. Tests for both cases.
4. **Settings API** (SPEC 11A). An allowlist of web-editable fields, with an admin PIN confirmation, applied through the
   config stack's guardrails (runtime layer, audited, conservative rules kept). First fields: `voice.keywords.*.
   wake_phrases`, `voice.wake_word`, `safety.minor_lines.*`, the minors output blocklist.
5. **Web UI on `@sondalab/ui-kit` 0.37.0** with `data-sl-product="roboemot"` (electric magenta). The robot serves the page
   offline: vendor a pinned copy of the kit's CSS (and the effects runtime if used) under `webui/static/` with a sync
   script; light/dark/high-contrast. Migrate the existing pages (`docs: packages/ui-kit/agent/workflows/migrate.md`), then
   add the settings pages for slice 4. Run `sl-audit` on the surface.

## Notes
- Owner decisions so far: `docs/memory/owner-decisions-2026-10-18-*.md`.
- Residuals from M3 follow-ups are in TODO.md.
