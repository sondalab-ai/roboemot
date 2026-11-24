# Skill Packs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** People add text-only skill packs (`SKILL.md`) to the robot. An admin turns them on, and the voice model (any route with tools) starts and follows them from natural speech.

**Architecture:** A new core package `roboemot.skill_packs` reads and checks the folders under `<data_dir>/skill_packs/`, installs uploads, and holds the one active skill pack. Two voice tools (`skill_pack_start`, `skill_pack_end`) come from core's tool pack. The conversation worker appends the catalogue block and the active block to every turn's instructions. The web UI Tools tab gets an admin-only Skills section.

**Tech Stack:** Python 3.11, pydantic config schema, pytest, the vanilla JS web UI (`node --test` in `tests/webui_js`).

**Spec:** `docs/specs/2026-11-24-skill-packs-design.md` (approved 2026-11-24). Read it before every task.

## Global Constraints

- One branch `feat/skill-packs`, one PR, one commit per task (owner rule 2026-11-20). Work in a worktree with its own venv (CLAUDE.md, Commands).
- Every behavioural number is a `skill_packs.*` config field (SPEC 11A). `tests/test_no_magic_numbers.py` must pass.
- Core names no robot (`tests/test_robot_hierarchy.py`).
- Logs carry skill pack names, sizes and reasons only. Never a body, a transcript or exception text (SPEC 12).
- "skill pack" in code and SPEC. "Skills" / "Abilità" in the web UI only.
- Name rule: `^[a-z0-9]+(-[a-z0-9]+)*$`, 1 to 64 characters, equal to the folder name.
- Config defaults (design 5.8): `enabled` `[]`; `max_enabled` 20 (1–100); `description_max_chars` 200 (20–500); `body_max_chars` 4000 (200–8000); `upload_max_bytes` 262144 (1024–4194304); `upload_max_files` 50 (1–500); `idle_end_s` 300 (30–3600). All admin-only except `idle_end_s`.
- Files 0600, folders 0700. Atomic install: extract to a temporary folder inside `<data_dir>/skill_packs/.incoming/`, check, then `os.replace`.
- Commit messages cite SPEC sections and end with the `Co-Authored-By` line.

## Review Focus

1. A zip with `../x`, an absolute path, a symbolic link, or a "zip bomb" (small compressed, huge uncompressed) must be refused and leave no file behind. Owner: Task 3.
2. A `SKILL.md` with no front matter, with Windows line endings, with a UTF-8 BOM, or that is not UTF-8 must give a clear invalid reason, never a crash. Owner: Task 2.
3. The model calls `skill_pack_start("Quiz Night")` or a name that was turned off a moment ago: normalised match for the first, `not_found` for the second. Owner: Task 6.
4. A Gemini session reconnects in the middle of an active skill pack: the next turn's instructions still carry the body. Owner: Task 7.
5. The admin removes or turns off the active skill pack during a conversation: it ends at once, and the next turn carries the end line once. Owner: Task 5 (state) and Task 8 (API call).

---

### Task 1: Config fields

**Files:**
- Modify: `packages/roboemot-core/src/roboemot/config/schema.py` (new `SkillPacksConfig`, root field `skill_packs`; add `skill_pack_start` and `skill_pack_end` to the default `ConversationConfig.llm_skills`)
- Modify: `packages/roboemot-core/src/roboemot/webui/descriptions_it.json` (Italian descriptions for the new fields, like the other fields)
- Test: `packages/roboemot-core/tests/test_skill_packs_config.py`

**Interfaces:**
- Produces: `cfg.skill_packs.enabled: list[str]`, `.max_enabled`, `.description_max_chars`, `.body_max_chars`, `.upload_max_bytes`, `.upload_max_files`, `.idle_end_s`, with the defaults and ranges in Global Constraints. `enabled` items match the name rule. A list longer than `max_enabled` or with duplicates fails validation.

- [ ] **Step 1: Write failing tests.** `test_defaults` (every default as listed), `test_ranges_reject_out_of_range` (one value below and one above each range), `test_enabled_rejects_bad_name` (`"Quiz"`, `"a/b"`, `""`), `test_enabled_rejects_duplicates_and_over_max`, `test_admin_only_fields` (every field but `idle_end_s` has `admin_only`), `test_default_llm_skills_hold_the_two_tools`.
- [ ] **Step 2: Run** `pytest packages/roboemot-core/tests/test_skill_packs_config.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** with `param(...)` like `ToolsConfig`, citing "SPEC 11, 11A: skill packs".
- [ ] **Step 4: Run** the new test, `test_no_magic_numbers.py`, the settings and description tests (`pytest -k "schema or descriptions or settings"`). Expected: PASS.
- [ ] **Step 5: Commit** `feat(config): skill_packs fields (SPEC 11, 11A)`.

### Task 2: Reading and checking skill packs

**Files:**
- Create: `packages/roboemot-core/src/roboemot/skill_packs/__init__.py`, `.../skill_packs/library.py`
- Modify: `packages/roboemot-core/src/roboemot/paths.py` (`skill_packs_dir(configured: str = "") -> Path`, `<data_dir>/skill_packs`)
- Test: `packages/roboemot-core/tests/test_skill_packs_library.py` (folders built in `tmp_path`)

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class SkillPack: name: str; description: str; body: str; meta: dict[str, str]; ignored_files: tuple[str, ...]; size_bytes: int`
  - `@dataclass(frozen=True) class InvalidPack: name: str; reason: str` with `reason` in `{"no_skill_md", "not_utf8", "no_front_matter", "bad_name", "name_mismatch", "no_description", "description_too_long", "body_empty", "body_too_long"}`
  - `def read_pack(folder: Path, cfg: RoboEmotConfig) -> SkillPack | InvalidPack`
  - `class SkillPackLibrary: __init__(self, root: Path, config: Callable[[], RoboEmotConfig])`, `scan() -> None`, `packs() -> list[SkillPack | InvalidPack]` (name order), `get(name: str) -> SkillPack | None`, `enabled(cfg) -> list[SkillPack]` (names in `cfg.skill_packs.enabled` that are valid, name order; a missing name is logged once per scan).
- Front matter: the first line is `---`, a closing `---` line follows, and each line in between is `key: value`. Strip a UTF-8 BOM and normalise `\r\n`. No YAML library (keys and values are plain strings).

- [ ] **Step 1: Write failing tests**: one test per invalid reason; `test_valid_pack_fields` (name, description, body stripped, `meta == {"author": "x"}`, `ignored_files == ("run.py",)`); `test_bom_and_crlf_are_accepted`; `test_scan_skips_dot_folders` (`.incoming`); `test_enabled_skips_missing_and_invalid`; `test_body_never_logged` (caplog holds no body text).
- [ ] **Step 2: Run** `pytest packages/roboemot-core/tests/test_skill_packs_library.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `read_pack`, `SkillPackLibrary` and `skill_packs_dir`.
- [ ] **Step 4: Run** the test again. Expected: PASS.
- [ ] **Step 5: Commit** `feat(skill-packs): read and check SKILL.md folders (SPEC 11)`.

### Task 3: Upload install and remove

**Files:**
- Create: `packages/roboemot-core/src/roboemot/skill_packs/install.py`
- Test: `packages/roboemot-core/tests/test_skill_packs_install.py`

**Interfaces:**
- Consumes: `read_pack`, `SkillPackLibrary` (Task 2).
- Produces:
  - `class InstallError(Exception)` with `.code` in `{"too_large", "too_many_files", "bad_path", "link", "not_zip", "no_skill_md", "exists", "invalid"}`. For `invalid`, `.reason` holds the `InvalidPack.reason`.
  - `def install(root: Path, data: bytes, filename: str, cfg: RoboEmotConfig, *, replace: bool = False) -> str` returns the name. A `.md` filename is one `SKILL.md`, and its folder name comes from its `name`. A `.zip` filename is an archive.
  - `def remove(root: Path, name: str) -> bool`.
- Zip rules (design 5.2): compressed size and the sum of `ZipInfo.file_size` are each at most `upload_max_bytes`; entry count at most `upload_max_files`; refuse absolute paths, `..` parts and entries whose Unix mode is a symbolic link; accept one top folder or files at the root. Read entries with a byte cap, never `extractall`.

- [ ] **Step 1: Write failing tests**: one per error code (build zips in memory with `zipfile`); `test_zip_bomb_refused` (one entry declared under the limit that inflates past it must fail with `too_large`); `test_failed_install_leaves_nothing` (no folder in `root` or `.incoming`); `test_exists_without_replace`; `test_replace_swaps_folder`; `test_modes_0600_0700`; `test_md_upload_uses_front_matter_name`; `test_remove`.
- [ ] **Step 2: Run** `pytest packages/roboemot-core/tests/test_skill_packs_install.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `install` and `remove`. The temporary folder is removed in a `finally` block.
- [ ] **Step 4: Run** the test again. Expected: PASS.
- [ ] **Step 5: Commit** `feat(skill-packs): install uploads safely, remove (SPEC 11, 12)`.

### Task 4: Prompt blocks and budget fit

**Files:**
- Create: `packages/roboemot-core/src/roboemot/skill_packs/prompt.py` (the fixed English texts and the block builders)
- Modify: `packages/roboemot-core/src/roboemot/llm/prompt.py` (`skill_packs_fit`)
- Test: `packages/roboemot-core/tests/test_skill_packs_prompt.py`

**Interfaces:**
- Consumes: `SkillPack` (Task 2), `system_tokens_worst_case`, `estimate_tokens` (existing).
- Produces:
  - `def catalogue_block(packs: list[SkillPack]) -> str`: `""` for no packs, else the header and one `name: description` line per pack. The header tells the model to call `skill_pack_start` only when the person asks for something that matches, and never on its own initiative.
  - `def active_block(pack: SkillPack) -> str`: the frame from design 5.5 (three points), then the body.
  - `def ended_line(name: str) -> str`: exactly `f"The skill pack {name} has ended. Do not follow its instructions any more."`
  - `def skill_packs_fit(cfg: RoboEmotConfig, packs: list[SkillPack]) -> str | None` in `llm/prompt.py`: the name of the first route where `system_tokens_worst_case(cfg) + estimate_tokens(catalogue_block(packs)) + max(estimate_tokens(active_block(p)) for p in packs) + cfg.conversation.turn_reserve_tokens` exceeds `context_budget_tokens`, else None. An empty `packs` always fits.

- [ ] **Step 1: Write failing tests**: `test_catalogue_empty`, `test_catalogue_lines_in_name_order`, `test_catalogue_header_forbids_unasked_start` (asserts the "only when the person asks" wording), `test_active_block_frame_then_body` (the three frame points appear before the body), `test_ended_line_exact`, `test_fit_names_the_first_tight_route`, `test_fit_none_when_empty`.
- [ ] **Step 2: Run** `pytest packages/roboemot-core/tests/test_skill_packs_prompt.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the test and `pytest -k "prompt or budget"`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(skill-packs): catalogue and active prompt blocks, budget fit (SPEC 8)`.

### Task 5: Active skill pack state

**Files:**
- Create: `packages/roboemot-core/src/roboemot/skill_packs/session.py`
- Test: `packages/roboemot-core/tests/test_skill_packs_session.py`

**Interfaces:**
- Consumes: `SkillPackLibrary`, block builders, `skill_packs_fit` (Tasks 2 and 4).
- Produces: `class SkillPackSession` (thread-safe with one lock):
  - `__init__(self, library: SkillPackLibrary, config: Callable[[], RoboEmotConfig], clock: Callable[[], float])`
  - `usable(cfg) -> list[SkillPack]`: `library.enabled(cfg)`, minus packs that make `skill_packs_fit` fail (each skipped name logged once with the route name).
  - `start(name: str) -> tuple[str, str | None]`: normalise (lower case, trim, spaces and underscores to hyphens), then `("started", body)` or `("not_found", None)`. It replaces any active pack, and the replaced one's end line is queued.
  - `end(reason: str) -> bool`, `reason` in `{"tool", "conversation_end", "asleep", "resting", "disabled", "removed", "admin", "idle"}`.
  - `heard_addressed() -> None` restarts the idle timer. `tick() -> None` ends with `idle` after `idle_end_s`.
  - `on_config(cfg) -> None` ends with `disabled` when the active name left `usable`.
  - `instructions() -> str`: catalogue block, active block, then each queued end line once (the queue is cleared by this call).
  - `snapshot() -> dict` with `active`, `started_at`, and `counts` (starts, and ends per reason) for diagnostics.
  - `enabled_names() -> list[str]` (names of `usable`).

- [ ] **Step 1: Write failing tests** with a fake clock: `test_start_normalises_name`, `test_start_unknown_or_disabled_is_not_found`, `test_start_replaces_and_queues_end_line`, `test_each_end_reason`, `test_end_line_sent_once`, `test_idle_end_after_idle_end_s`, `test_heard_addressed_resets_idle`, `test_on_config_disable_ends_active`, `test_too_large_pack_skipped`, `test_instructions_empty_when_nothing_enabled`.
- [ ] **Step 2: Run** `pytest packages/roboemot-core/tests/test_skill_packs_session.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the test again. Expected: PASS.
- [ ] **Step 5: Commit** `feat(skill-packs): active skill pack state and end reasons (SPEC 8)`.

### Task 6: The two voice tools

**Files:**
- Create: `packages/roboemot-core/src/roboemot/tools/skill_packs.py`
- Modify: `packages/roboemot-core/src/roboemot/skills/host.py` (`skill_packs: SkillPackSession | None = None` field and docstring)
- Modify: `packages/roboemot-core/pyproject.toml` (`skill_packs = "roboemot.tools.skill_packs:skills"` under `roboemot.skills`)
- Test: `packages/roboemot-core/tests/test_tool_skill_packs.py`

**Interfaces:**
- Consumes: `SkillPackSession` (Task 5), `ToolHost`, `@skill`, `SkillResult` (existing; see `tools/sleep.py` for the pattern).
- Produces: `def skills(host: ToolHost) -> list[Callable[..., Any]]` with:
  - `skill_pack_start(name: str)` → `SkillResult({"status": "started", "instructions": body})` or `{"status": "not_found", "enabled": [...names]}`. `follow_up` stays True.
  - `skill_pack_end()` → `{"status": "ended"}` or `{"status": "none_active"}`. `follow_up=False`.
  - Both `available` only while `host.skill_packs` is set and `enabled_names()` is not empty.
  - English descriptions: start only when the person asks for an activity that matches a catalogue line; end when the person wants to stop or the activity is over.

- [ ] **Step 1: Write failing tests**: `test_start_returns_body`, `test_start_not_found_lists_enabled`, `test_end_statuses`, `test_unavailable_without_session_or_enabled_packs`, `test_entry_point_registered` (loads through `load_skill_plugins`), `test_end_is_fire_and_forget`.
- [ ] **Step 2: Run** `pytest packages/roboemot-core/tests/test_tool_skill_packs.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement**, then `pip install -e packages/roboemot-core` so that the entry point registers.
- [ ] **Step 4: Run** the test and `pytest -k "skills or plugins or tool"`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(tools): skill_pack_start and skill_pack_end (SPEC 11, Voice tools)`.

### Task 7: Runtime wiring

**Files:**
- Modify: `packages/roboemot-core/src/roboemot/runtime/startup.py` (build `SkillPackLibrary(skill_packs_dir(cfg.data_dir), ...)` and `SkillPackSession`, `scan()` at start, set `host.skill_packs`, pass the session to the worker, call `on_config` on every config change, end with `asleep` on `robot.asleep` and with `resting` when the circadian state enters `resting`, call `tick()` from the behaviour tick)
- Modify: `packages/roboemot-core/src/roboemot/conversation/worker.py`: append `session.instructions()` to the realtime instructions (`_base_instructions` near line 1098, `_burst_instructions` near line 1115, the `response.create` instructions near line 1173) and to the chat `system_prompt` near line 2585. Call `heard_addressed()` where the wake decision holds. Call `end("conversation_end")` where the conversation state turns to `NO_CONVERSATION`.
- Test: `packages/roboemot-core/tests/test_skill_packs_worker.py` (with the worker's existing fakes; see `test_conversation_worker.py`)

**Interfaces:**
- Consumes: `SkillPackSession` (Task 5).
- Produces: `ConversationWorker(..., skill_packs: SkillPackSession | None = None)`.

- [ ] **Step 1: Write failing tests**: `test_chat_system_prompt_carries_catalogue_and_active_block`; `test_realtime_instructions_carry_active_block`; `test_reconnect_replay_still_has_active_block` (after a reconnect the next turn's `response.create` instructions hold the body); `test_fallback_to_chat_keeps_active_block`; `test_conversation_end_ends_pack`; `test_no_skill_text_when_session_none`.
- [ ] **Step 2: Run** `pytest packages/roboemot-core/tests/test_skill_packs_worker.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** Keep `instructions()` text after the existing parts, so the cached prefix stays stable.
- [ ] **Step 4: Run** the full suite `pytest`. Expected: PASS (compare any failure with `main` before you change it).
- [ ] **Step 5: Commit** `feat(runtime): wire skill packs into turns and end triggers (SPEC 8)`.

### Task 8: Web UI API, Tools state, diagnostics

**Files:**
- Modify: `packages/roboemot-core/src/roboemot/runtime/tools_state.py` (skill pack list in `state()`; new methods)
- Modify: `packages/roboemot-core/src/roboemot/webui/api.py` (`ToolsPort` methods, routes, `body_limit` case for the upload)
- Modify: `packages/roboemot-core/src/roboemot/runtime/diagnostics.py` (`FnSource("roboemot.skill_packs", session.snapshot)` plus installed, enabled and invalid counts with reasons)
- Test: `packages/roboemot-core/tests/test_webui_tools.py` (extend), `packages/roboemot-core/tests/test_skill_packs_diagnostics.py`

**Interfaces:**
- Consumes: `install`, `remove`, `InstallError` (Task 3); `SkillPackLibrary`, `SkillPackSession` (Tasks 2 and 5); `SettingsPort.change` for `skill_packs.enabled`; `_check_pin` and `v.confirmed` for the admin confirmation (see `_forget`, `api.py:824`).
- Produces (all admin only: a live session plus `pin` and `confirm` on writes):
  - `GET /api/tools/skill_packs` → `{"packs": [{"name", "description", "state": "on"|"off"|"invalid"|"missing"|"too_large", "reason", "route", "meta", "ignored_files", "size_bytes"}], "active": name|null}`
  - `GET /api/tools/skill_packs/<name>` → adds `"body"`.
  - `POST /api/tools/skill_packs/upload` `{filename, data_b64, replace, pin, confirm}`. A replacement drops the name from `enabled` (design 5.2: a replaced skill pack is off). Body limit: `skill_packs.upload_max_bytes` × 4/3 plus 4096 bytes, only with a live session (same rule as `CATALOGUE_IMPORT`).
  - `POST /api/tools/skill_packs/<name>/enable` `{on: bool, pin, confirm}`. Turning on is refused with `409` `llm_skills_missing` (field `conversation.llm_skills`) or `too_large` (field `route`), or with `max_enabled`.
  - `POST /api/tools/skill_packs/<name>/remove` `{pin, confirm}` → also drops the name from `enabled` and ends it with `removed`.
  - `POST /api/tools/skill_packs/rescan`, `POST /api/tools/skill_packs/end` → `session.end("admin")`.

- [ ] **Step 1: Write failing tests**: one per route for success; `test_writes_need_pin_and_confirm`; `test_enable_refusals` (each code); `test_upload_body_limit_only_with_session`; `test_upload_error_codes_map_to_422`; `test_state_never_carries_body` (list route); `test_diagnostics_snapshot_has_no_body`.
- [ ] **Step 2: Run** `pytest packages/roboemot-core/tests/test_webui_tools.py packages/roboemot-core/tests/test_skill_packs_diagnostics.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the tests and `pytest -k webui`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(webui): skill packs API and debug report (SPEC 11A, 12, 12A)`.

### Task 9: Web UI Skills section

**Files:**
- Modify: `packages/roboemot-core/src/roboemot/webui/static/tools.js`, `.../static/index.html`, `.../static/style.css`, `.../webui/strings.py` (English and Italian strings)
- Modify: `packages/roboemot-core/src/roboemot/webui/static/live.js` (active skill pack name)
- Test: `packages/roboemot-core/tests/webui_js/skill_packs.test.mjs` (with `harness.mjs`), `packages/roboemot-core/tests/test_webui_strings_cache.py` (extend if strings are listed there)

**Interfaces:**
- Consumes: the routes of Task 8.
- Produces: the Skills section of design 5.6: the list with states and reasons, view text, on/off (PIN and confirm dialog), replace, remove, "Add skill" (file input, `.md` or `.zip`, base64), "Rescan", the active pack with "End". The ignored-files note reads "not used: RoboEmot runs no code from skill packs" (Italian: "non usati: RoboEmot non esegue codice dai pacchetti").

- [ ] **Step 1: Write failing JS tests**: `renders states and reasons`, `turning on asks for PIN and confirm`, `shows refusal message with route name`, `upload sends filename and base64`, `active pack shows End`.
- [ ] **Step 2: Run** `cd packages/roboemot-core/tests/webui_js && node --test`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement** with the vendored UI kit components only (never edit `vendor/`).
- [ ] **Step 4: Run** `node --test` (all pass), `pytest -k strings`, and `scripts/audit_webui.sh`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(webui): Skills section in the Tools tab (SPEC 11A, 12)`.

### Task 10: SPEC, examples, robot checks

**Files:**
- Modify: `docs/specs/kickoff.md`: Glossary (SPEC 1, "Skill pack"); SPEC 8 (Conversation state: the active skill pack and its end reasons; LLM router: the two blocks); SPEC 11 (Voice tools: the two tools and the `llm_skills` default; Plugin interfaces: none added); SPEC 11A (the `skill_packs.*` fields); SPEC 12 (Data: the folder and modes; Behaviour and content: untrusted text inside a fixed frame, admin review); SPEC 12A (the debug report section); SPEC 15 "To verify on the robot" (design section 8).
- Create: `configs/skill_packs/quiz-night/SKILL.md`, `configs/skill_packs/bedtime-story/SKILL.md` (examples, not installed by default)
- Modify: `CLAUDE.md` repository layout row for `roboemot-core` (add "skill packs")
- Test: `packages/roboemot-core/tests/test_skill_packs_examples.py` (both examples pass `read_pack` with the default config)

- [ ] **Step 1: Write the failing example test.**
- [ ] **Step 2: Run** it. Expected: FAIL (no files).
- [ ] **Step 3: Write** the examples and the SPEC text. Every number in the SPEC must equal the schema default (check against `config/schema.py`).
- [ ] **Step 4: Run** `pytest` and `ruff check . && ruff format --check .`. Expected: PASS.
- [ ] **Step 5: Commit** `docs(specs): skill packs in SPEC 1, 8, 11, 11A, 12, 12A, 15`, then open the PR. The PR description lists design section 8 as unverified on the robot.
