# Body Prompt Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every LLM that speaks as the robot gets a short, per-robot body description that the owner can edit in the web UI.

**Architecture:** New admin-only config field `persona.body_prompt` (library default `""`). Each robot manifest sets its default under `behaviour.persona.body_prompt` (config layer 2). `llm/prompt.py::_prompt` places it right after `persona.prompt` (part 1b).

**Tech Stack:** Python 3.11, pydantic schema (`roboemot.config.schema`), pytest, ruff.

**Spec:** `docs/specs/2026-11-16-body-prompt-design.md`

## Global Constraints

- Core never names a robot or its DOFs (`tests/test_robot_hierarchy.py`). Robot words appear only in `configs/manifests/*.yaml`.
- No magic numbers (SPEC 11A): the length cap is a schema field.
- Prompt text is `admin_only` (SPEC 11A, Guardrails); never reachable through per-person overrides (`PERSON_SECTIONS` stays unchanged).
- Each default body text is at most 200 characters and states only what the manifest declares.

## Review Focus

- A manifest `behaviour.persona` with any key other than `body_prompt` is refused (prompt text from a manifest must not replace `persona.prompt`).
- A body text longer than `persona.body_prompt_max_chars` is refused at load, as `persona.prompt` is.
- `body_prompt: ""` gives the same prompt as before this change (no blank line, no stray part).
- `check_route_budgets` still passes for `configs/companion.yaml` with the Reachy manifest.
- A runtime change of `persona.body_prompt` without `admin=True` is refused.

---

### Task 1: Schema field and prompt part 1b

**Files:**
- Modify: `packages/roboemot-core/src/roboemot/config/schema.py` (`PersonaConfig`, ~line 5989)
- Modify: `packages/roboemot-core/src/roboemot/llm/prompt.py` (`_prompt`, `system_tokens_worst_case`, `check_route_budgets` message)
- Test: `packages/roboemot-core/tests/test_body_prompt.py` (create)

**Interfaces:**
- Produces: `PersonaConfig.body_prompt: str`, `PersonaConfig.body_prompt_max_chars: int`.

- [ ] **Step 1: Write the failing tests** in `test_body_prompt.py`:
  - `test_body_prompt_follows_the_persona_prompt`: cfg with `persona.body_prompt="BODY"`; `system_prompt(cfg, cfg.register.params)` splits on `"\n"`; line 0 is `cfg.persona.prompt`, line 1 is `"BODY"`. Same check on `realtime_instructions(...)`.
  - `test_empty_body_prompt_leaves_the_prompt_unchanged`: default cfg; no line equals `""` and line 1 is not empty-derived (prompt equals the one built with the field absent, i.e. compare to `_prompt` output where parts are filtered).
  - `test_body_prompt_too_long_is_rejected`: `PersonaConfig(body_prompt="x" * 301)` raises `ValidationError` with `"persona.body_prompt"` in the message.
  - `test_worst_case_counts_the_body_prompt_cap`: `system_tokens_worst_case` grows by at least `body_prompt_max_chars // CHARS_PER_TOKEN` against a cfg where the body prompt is empty (it counts the cap, not the current text).
  - `test_body_prompt_change_needs_the_admin`: through `ConfigStack.set_runtime("persona.body_prompt", "x")` (no admin) raises `ConfigError`; with `admin=True` it applies. Follow the existing `persona.prompt` admin test pattern in `test_persona.py` / `test_config_stack*.py`.
- [ ] **Step 2: Run** `pytest packages/roboemot-core/tests/test_body_prompt.py -v`. Expected: FAIL (no field).
- [ ] **Step 3: Implement**
  - `body_prompt: str = param("", "The robot's body as the LLM is told it, after the persona prompt; the robot manifest sets the default; changes need the admin", admin_only=True)`.
  - `body_prompt_max_chars: int = param(300, "Longest accepted body prompt", ge=50, le=2000, advanced=True)`.
  - Extend `_prompt_length` with the same check and message shape for `body_prompt`.
  - `_prompt`: `parts = [cfg.persona.prompt, cfg.persona.body_prompt, register.persona_fragment, ...]` (the existing filter drops empty parts).
  - `system_tokens_worst_case`: build the prompt from a cfg copy whose `persona.body_prompt` is `"x" * body_prompt_max_chars`.
  - `check_route_budgets` error text: add `persona.body_prompt` to the list of fields to shorten.
  - Update the module docstring order line and `_prompt`/`system_prompt` docstrings.
- [ ] **Step 4: Run** the new tests plus `pytest packages/roboemot-core/tests/test_no_magic_numbers.py packages/roboemot-core/tests/test_persona.py packages/roboemot-core/tests/test_llm_history.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(persona): body prompt as part 1b of the system prompt (SPEC 8, 11A)`.

### Task 2: Manifest default per robot

**Files:**
- Modify: `packages/roboemot-core/src/roboemot/config/stack.py` (`MANIFEST_BEHAVIOUR_KEYS` ~line 707; new `check_manifest_persona`; call it next to `check_manifest_voice` ~line 1185)
- Modify: `configs/manifests/reachy-mini-wireless.yaml` (`behaviour:` block, line 61)
- Modify: `configs/manifests/desk-head-v1.yaml` (add a `behaviour:` block)
- Test: `packages/roboemot-core/tests/test_body_prompt.py`

**Interfaces:**
- Consumes: `PersonaConfig.body_prompt` (Task 1).
- Produces: `MANIFEST_PERSONA_KEYS = frozenset({"body_prompt"})`, `check_manifest_persona(persona: Any) -> None` (raises `ConfigError`).

- [ ] **Step 1: Write the failing tests**:
  - `test_manifest_sets_the_body_prompt`: load `configs/manifests/reachy-mini-wireless.yaml` the way `test_manifest_voice_name.py` does; merged `cfg.persona.body_prompt` equals the manifest text and is non-empty.
  - `test_manifest_persona_may_set_only_the_body_prompt`: behaviour `{"persona": {"prompt": "x"}}` raises `ConfigError` naming `body_prompt`.
  - `test_manifest_body_prompt_must_be_text`: `{"persona": {"body_prompt": 3}}` raises `ConfigError`.
  - `test_every_shipped_manifest_has_a_short_body_prompt`: for each `configs/manifests/*.yaml`, `behaviour.persona.body_prompt` is a non-empty string of at most 200 chars.
  - `test_runtime_override_wins_and_reset_restores_the_manifest_text`: admin override replaces it; removing the override gives the manifest text again.
  - `test_shipped_configs_pass_the_route_budget`: `configs/companion.yaml` with the Reachy manifest passes `check_route_budgets`.
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement** the allowlist and the check (pattern of `check_manifest_motion`). Manifest texts (≤200 chars, only what the manifest declares):
  - Reachy: `"Your body: a head that tilts and turns on a base that rotates, two antennas you can move, a camera, microphones and a speaker. You have no arms or legs and cannot walk."`
  - desk-head: `"Your body: a head on a neck that pans and tilts, LED eyes, a camera, microphones and a speaker. You have no arms or legs and cannot walk."`
- [ ] **Step 4: Run** the new tests and `pytest packages/roboemot-core/tests/test_manifest*.py packages/roboemot-core/tests/test_robot_hierarchy.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(manifest): per-robot body prompt default (SPEC 4, 11A)`.

### Task 3: SPEC and full validation

**Files:**
- Modify: `docs/specs/kickoff.md` (SPEC 8 prompt parts ~line 1032; SPEC 8 free-text list ~line 943; SPEC 11A Guardrails ~line 1311 and Persona; SPEC 4 manifest `behaviour` keys)

- [ ] **Step 1:** Add part 1b (`persona.body_prompt`, manifest default) to the parts list; add it beside `persona.prompt` in the free-text lists at ~943 and ~1311; add `persona.body_prompt` to the manifest behaviour keys.
- [ ] **Step 2: Run** `ruff check . && ruff format --check . && pytest -q`. Expected: all pass (compare any failure against `main` before dismissing).
- [ ] **Step 3: Commit** `docs(spec): body prompt (SPEC 4, 8, 11A)`.
