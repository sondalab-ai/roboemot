# Web UI settings mirror (1:1, core → robot): epic plan

## Context
The web UI edits only a 4-entry allowlist (`webui/settings.py:84`, SPEC 11A line 899). Owner wants it to mirror every
config field (~481 leaves in `RoboEmotConfig`), laid out Agnostic → Specialized: core sections first, then the robot's
own (Reachy Mini) section contributed by the robot package, and the same hierarchy across the web UI generally.
Owner decisions (2026-10-18): one PIN for operator and admin, risky fields (interpreter path, dirs, bind/hosts,
route base_url) editable and the risk accepted; secrets never shown or set from the web; English-only field texts;
normal edits on the runtime layer, safety loosening on the app YAML with the PIN; nav + sections layout.
Slice D (volume only) was stopped and is folded in here.

## Step 0 — done
The M3 cleanup epic merged to `main` as PR 57 (`fe1a204`); B's node:test + jsdom harness is in place.

## Slices (integration branch `epic/webui-settings-mirror`, one PR each)
1. **Field catalogue (backend).** Schema walker over `RoboEmotConfig` producing per-path descriptors: kind (number,
   int, bool, text, enum from `Literal`, nullable, tuple, string list, string dict, dict-of-model, keyed list), limits
   (`ge/le`), unit, restart, admin_only, opt_in, conservative, description, value, layered default, and **source
   layer** (new `ConfigStack.source_of(path)` from the layer dicts and runtime map; reuse `get_path`, `flatten`,
   `field_meta` in `config/stack.py`). Secret-shaped values redacted (reuse `diagnostics.jsonable` rules); extend
   `test_secret_leak.py` to `/api/settings`. Per-path validator hooks keep today's checks (wake word sanitiser,
   minor lines, blocklist additions only, phrase rules). Replaces `SETTINGS`; `webui/settings.py` joins
   `test_no_magic_numbers.py`.
2. **Generic write path.** Scalars and collections: dict-of-model by key, keyed lists by `name`/`event`
   (`KEYED_LISTS`), add/remove entries where the schema allows (no new language from the web, earlier decision;
   reversed 2026-10-19, see Owner decisions, done in slice 5).
   Write layer: runtime by default; a change refused only for `safety` loosening goes to `layer="app"` with admin
   (writes app YAML via `_persist_app`). `restore()` replays the overrides file as one batch so cross-field
   validators don't depend on order. Responses carry `restart_required`; a "reset to lower layer" clears a runtime
   value (`ConfigStack.clear`). Opus security review (slices 1+2 together if the diff is small).
3. **Robot layer.** New entry-point group `roboemot.settings` keyed by manifest `robot.id`, loaded fail-soft like
   `world/sources.py:load_world_plugin`. The Reachy plugin contributes its section: manifest `behaviour`
   (expression_map, gesture_map, circadian_motion — editable, they are config paths) and read-only hardware limits
   (constraints, dynamics, manifest safety, watchdog ceiling). Core stays robot-agnostic.
4. **Settings page.** `sl-nav` tree (Core → Robot) + `sl-disclosure` groups, generated controls per kind (number with
   unit and `sl-slider` when bounded, `sl-switch`, `sl-select`, text, chip lists, pair inputs, entry editors for
   maps/lists), search, "changed only" filter, source-layer and restart badges, admin tag, PIN per save via `ask()`.
   Apply the same Core → Robot grouping to the other tabs where content is robot-specific (debug sources already
   are). CSP rules hold (CSSOM only, no innerHTML); vendored kit untouched; new chrome strings EN+IT (page test
   requires both), field texts English. JS tests on B's node:test + jsdom harness.
4b. **Restart RoboEmot** (owner, 2026-10-19: no reboot/power-off). Core owns the button, PIN, confirm dialog and
   response timing; the robot plugin provides `restart()` via a `Restartable` protocol (replaces the unused
   `Rebootable`/`REBOOT`) plus its side-effects list, which core renders EN+IT. No plugin hook, no button. Reachy:
   daemon `POST /api/apps/restart-current-app` (present in 1.11.0 and 1.12.0), offered only when
   `GET /api/apps/current-app-status` shows RoboEmot (`install.APP_NAME`) running; side effects: zero pose, possible
   WebRTC drop, SIGINT then kill after 20 s. Respond before the restart fires. The restart badge links to it.
5. **SPEC and exit.** SPEC 11A (allowlist → full mirror, app-YAML write for safety loosening, one PIN, accepted
   risks), SPEC 12 Data notes, TODO.md; exit check: `voice.base.volume_db` edited live from the page and heard on the
   next utterance (sim), a restart field shows the badge, a secret is never in any response.

Order: 1 → 2 → (3 ∥ 4) → 4b → 5 → 6 (6: CI `models` barge-in flake, may run alongside 4b). Estimate: 6 slices, ~14 agent runs (Sonnet authors + one Sonnet review each, Opus
security review on 1–2). Re-confirm if it passes ~21.

## Verification
Per slice: `ruff check . && ruff format --check .`, full `pytest`, node tests, `scripts/audit_webui.sh`, CI green on
each PR. End to end: sim daemon + app, open the web UI, edit a number, a bool, an enum, a map entry, a route model,
a safety tighten (runtime) and a safety loosen (app YAML written, audit entry), restart a restart-field, confirm
`roboemot config dump` matches the page.

## Status (2026-10-19 01:40)
Merged into the epic branch: slice 1 (PR 58), 2 (PR 59), 3 (PR 60), 4 (PR 61). Agent runs used: 21 of the ~21 cap
(extra runs: slice 1–2 security fix rounds, an Opus re-verify, and a CI latency investigation that proved the
`models` barge-in failure is runner noise, not this epic). Paused for owner OK before 4b and 5.
- 4b: superseded by the 4b slice text and Owner decisions below (Restart only, no reboot).
- 5: SPEC 11A/12, TODO.md, exit check as above.
Owner decision during the run: `privacy.*` and minor paths are admin-only (readable only via a PIN-checked reveal).

## Owner decisions (2026-10-19 08:50)
Go-ahead for 4b, 5 and a flake slice (estimate now ~31 runs).
- Kept: `circadian.location` private; robot plugins cannot claim safety/webui/privacy/llm_routes or private/admin
  paths; raw-JSON fields read-only; only web-added entries removable; only safety loosening persists to the app YAML;
  "default" = value after reset.
- Reboot: dropped. reachy-mini-os >= 0.2.4 (`stage2/05-reachy-mini/00-run.sh`, `/etc/sudoers.d/010-pollen-reachy`)
  allows `pollen` only a fixed command list (no `systemctl reboot`; `shutdown -h now` is there); older images allow
  bare `systemctl` (CVE-2026-53605). Owner chose Restart only: remove the unused `Rebootable`/`REBOOT` hook.
- Robot commands: derive from Pollen sources/docs first; ask the owner to run them only for what stays unknown.
- Restart RoboEmot: daemon route accepted. The PIN dialog lists zero pose, possible WebRTC drop, kill after 20 s (EN+IT).
- Folded into slice 5 (reverses agent defaults):
  - Remove is shown only where the server would accept it.
  - A save keeps other rows' unsaved edits.
  - Debug report: revert `scheme://user@host` redaction (only user:pass@ stays hidden).
  - Keyed routes may change scheme/host/port, with the PIN and a dialog naming the new host the key will go to.
  - base_url with credentials: shown masked, replaced whole and write-only.
  - New language keys can be added (validated against the supported voice languages).
  - Minors blocklist words removable/resettable under the admin PIN, audited.
- New slice 6, before epic -> main: fix the CI `models` barge-in flake (listener lag, GC pauses).
- Robot TODOs: SSH user `pollen` and Python 3.12 (reachy-mini-os `uv venv --python 3.12`) settled from source;
  nod/shake on the hardware still needs the owner.
