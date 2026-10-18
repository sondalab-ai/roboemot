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
   (`KEYED_LISTS`), add/remove entries where the schema allows (no new language from the web, earlier decision).
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
4b. **Restart and reboot.** Same Core → Robot split, PIN per action, confirm dialog. Core: "Restart RoboEmot" (app
   restarts itself, needed for `restart=True` fields; the restart badge links to it). Robot: "Reboot robot" via a
   new `reboot()` hook on the `roboemot.settings` robot plugin, shown only when the plugin offers it. For Reachy Mini
   the author first checks what the daemon / OS offers (daemon API vs `systemctl reboot` and the privilege it needs;
   the installer must not leave a sudoers change unrecorded). If nothing clean exists, report before building.
5. **SPEC and exit.** SPEC 11A (allowlist → full mirror, app-YAML write for safety loosening, one PIN, accepted
   risks), SPEC 12 Data notes, TODO.md; exit check: `voice.base.volume_db` edited live from the page and heard on the
   next utterance (sim), a restart field shows the badge, a secret is never in any response.

Order: 1 → 2 → (3 ∥ 4) → 4b → 5. Estimate: 6 slices, ~14 agent runs (Sonnet authors + one Sonnet review each, Opus
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
- 4b: "Restart RoboEmot" via the daemon's `POST /api/apps/restart-current-app` (no privilege, no installer change).
  "Reboot robot": no daemon route; `sudo -n systemctl reboot` likely works for `pollen` but is unverified (run
  `sudo -n -l` on the robot); fallback is a recorded sudoers drop-in. Robot plugin gets the `Rebootable` protocol
  (already in core, `actions: ["reboot"]`).
- 5: SPEC 11A/12, TODO.md, exit check as above.
Owner decision during the run: `privacy.*` and minor paths are admin-only (readable only via a PIN-checked reveal).
Agent defaults, pending owner OK: `circadian.location` is private too; the minors blocklist only grows from the web
(reset refused too); robot plugins cannot claim safety/webui/privacy/llm_routes or private/admin paths.
