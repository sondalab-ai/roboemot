# Skill packs: text skills that people add to the robot (design)

Date: 2026-11-24. Status: waiting for owner approval (builder check done). SPEC sections touched: 1 (Glossary),
8 (LLM router, Realtime speech route, Conversation state), 11 (LLM tools, Voice tools, Plugin
interfaces), 11A, 12 (Data, Behaviour and content), 12A (Diagnostics).

## 1. Intent

The owner wants people to give the robot new abilities by adding a file, without a code change
or a release. Examples: a quiz game, a bedtime routine, a cooking helper, a language-practice
partner. Later a RoboEmot marketplace may distribute these files.

The first request said "when a Claude API is configured". During the design the owner chose
text-only skills run by whatever model answers (owner decision 2026-11-24). Nothing in this
design needs Claude, so the feature works with every LLM route that offers tools: chat routes,
`openai_realtime` and `gemini_live`.

## 2. Terms

SPEC 1 already uses **skill** for an `@skill` function offered to the LLM as a tool, and
`roboemot.skills` is an entry-point group. To keep one word for one thing:

- **skill pack**: the new thing in this design. It is a folder with a `SKILL.md` file: a name, a
  one-line description and instructions in plain text. It holds no code that RoboEmot runs.
- **skill**: unchanged, an `@skill` tool.

The web UI may show skill packs as "Skills" ("Abilità" in Italian), because owners never see
`@skill` tools by that name. SPEC text and code use "skill pack". (Claude's call.)

## 3. Goals, non-goals, anti-goals

Goals:

- G1. An admin adds a skill pack through the web UI (upload) or by copying a folder into the data
  folder. Both paths give the same result.
- G2. A new skill pack is off. Only the web UI admin can turn it on, after reading its text in the
  web UI (owner decision 2026-11-24).
- G3. The person starts a skill pack with natural speech ("facciamo un quiz"). The model picks
  the skill pack from a one-line catalogue. There is no phrase list or keyword match (SPEC 11,
  Voice tools).
- G4. While a skill pack is active, the robot follows its instructions across turns, across a
  realtime reconnect, and across a fallback to another route.
- G5. A skill pack acts on the robot only through the tools that are already offered
  (`conversation.llm_skills`), so the executor's checks, the arbiter and the safety limiter still
  apply.

Non-goals (this design):

- Running a skill pack's scripts, on the robot or in a cloud sandbox. Files other than `SKILL.md`
  are stored but never read or run.
- A marketplace, signatures, versions, updates or a download from a URL.
- Claude-specific features (Anthropic Agent Skills, code execution).
- More than one active skill pack at a time.

Anti-goals (the feature fails even when it works):

- A skill pack changes the persona, the safety rules, the privacy rules or the language rules.
- A skill pack that nobody reviewed speaks to a child.
- The prompt grows with every installed skill pack until replies get slow.
- The robot starts a skill pack on its own, without the person asking.

## 4. Approaches considered

The owner made three choices in conversation (2026-11-24):

1. **What a skill pack can do.** Options: text-only instructions (chosen); scripts in Claude's
   cloud sandbox (needs a native Anthropic route, which breaks SPEC 8's OpenAI-compatible-only
   rule, adds seconds per reply and cannot move the robot); Python code on the robot (most
   powerful, unsafe for a marketplace without a sandbox and review).
2. **Which model follows the skill pack.** Options: the voice model itself (chosen); Claude
   thinks and Gemini only speaks (1 to 3 s slower per skill turn, reworded answers, the
   conversation goes to two clouds, and every turn of a multi-turn skill pack goes through Claude).
   A third option was dropped: hand the conversation to Claude while a skill pack runs. Gemini
   Live is speech-to-speech and cannot act only as a voice for Claude's text.
3. **How the model gets the text.** Options: a catalogue plus load on request (chosen); every
   enabled skill pack's full text always in the instructions (needs no tools, so it would also work
   on Pollen, but the prompt grows with every skill pack, the model mixes skill packs up, and it does
   not scale to a marketplace).

## 5. Design

### 5.1 Format

A skill pack is a folder `<data_dir>/skill_packs/<name>/` with a `SKILL.md` file. The front
matter uses the same shape as Anthropic Agent Skills, so a future marketplace or a Claude path
can reuse the files (Claude's call):

```markdown
---
name: quiz-night
description: A general-knowledge quiz game with score keeping, for one or more players.
---

When the person wants to play: ask how many players ...
```

- `name`: required. Lower-case letters, digits and hyphens, 1 to 64 characters. It must equal
  the folder name.
- `description`: required. One line, at most `skill_packs.description_max_chars`. The model reads
  it in the catalogue, so English is recommended but not enforced.
- Other front-matter keys (for example `author`, `version`, `license`) are kept and shown in the
  web UI. They change no behaviour.
- The body is plain text or Markdown, at most `skill_packs.body_max_chars`. It may be in any
  language.
- Other files in the folder are allowed and ignored. The web UI lists them with the note "not
  used: RoboEmot runs no code from skill packs".

A folder that breaks a rule is **invalid**. It is listed in the web UI with the reason. It cannot
be enabled, and it never reaches a model.

### 5.2 Adding and removing

- **Upload (web UI, admin only).** The Tools tab gets a Skills section with "Add skill". It takes
  a `SKILL.md` file or a `.zip` file. Rules for a zip file: at most
  `skill_packs.upload_max_bytes` compressed and uncompressed, at most
  `skill_packs.upload_max_files` entries, no absolute paths, no `..` path parts, no symbolic links,
  one top folder or files at the root, and a `SKILL.md`. RoboEmot extracts it into a temporary
  folder in the data folder, checks it, then moves it into place, so a bad upload leaves nothing
  behind. The upload has its own request size limit equal to `skill_packs.upload_max_bytes`, apart
  from `webui.max_body_bytes`, like the catalogue import. The transport (multipart or base64 in
  JSON) is left to the builder. An upload whose name already exists asks the admin to confirm a replacement. A
  replacement turns the skill pack off.
- **Folder copy.** A developer copies a folder into `<data_dir>/skill_packs/`. RoboEmot reads the
  folder at start, after an upload or a removal, and when the admin presses "Rescan" in the Skills
  section. There is no file watcher. (Claude's call.)
- **Remove (admin only).** This deletes the folder and drops the name from the enabled list.
- Files are written with mode 0600 and folders with 0700, like the other data files (SPEC 12,
  Data).

### 5.3 Enabling

`skill_packs.enabled` is an admin-only list of names (SPEC 11A), changed from the Skills section.
Before the admin turns a skill pack on, the web UI shows its full text. Turning it on asks for the
admin confirmation (SPEC 11A). An enabled name with no valid folder is skipped and logged, and the
web UI shows it as missing.

The change goes through the same admin confirmation flow as the other admin-only fields. The
enabled list is capped at `skill_packs.max_enabled`.

Turning on a skill pack is refused, with a message, in two cases (Claude's call, so that G5 stays
true):

- `conversation.llm_skills` does not hold both `skill_pack_start` and `skill_pack_end`. That list
  can only narrow the set (SPEC 11), so a robot with a saved, narrower list would never offer them.
  The message names the field.
- The real size of the catalogue of the enabled skill packs, plus the longest enabled body, would
  break the budget of a configured route. The message names the route.

The budget check measures the installed enabled skill packs, not a worst case from the config
caps, so routes pay nothing for skill packs that are not on. `check_route_budgets` (SPEC 8)
reads only config today. It gets the measured skill pack sizes as an extra input, both at
enable time and at load. At load, an enabled skill pack that no longer fits (for example after a
route change) is skipped and logged, and the web UI shows it as "too large for route X". Load
does not fail. (Claude's call.)

### 5.4 Catalogue and the two tools

While at least one valid skill pack is enabled:

- The instructions of every turn (chat and realtime) carry a catalogue block: a fixed English
  header, then one line per enabled skill pack (`name: description`), in name order. The header
  tells the model to call `skill_pack_start` only when the person asks for something that matches
  a line, and never on its own initiative.
- Two plugin skills come from core's tool pack (`roboemot.skills` group) and are added to the
  default `conversation.llm_skills`:
  - `skill_pack_start(name: str)`. `name` is free text, not a `choices` enum, because Gemini Live
    fixes its tool list when the session opens (`gemini_wire.py`, `tools_update_ignored`), and a
    skill pack enabled during a session must still be startable. The executor normalises the name:
    it lowers the case, trims it, and turns spaces and underscores into hyphens. It then checks
    the name against the enabled valid skill packs. An unknown name returns `not_found` and the
    list of enabled names. (Claude's call.) A valid name makes that skill pack active, replaces any other active one,
    and returns its body. `follow_up` stays True, so the model acts on the body in the same turn.
  - `skill_pack_end()`. This ends the active skill pack and returns `ended` or `none_active`.
    It is fire-and-forget (`follow_up=False`).
  - Both are `available` only while at least one valid skill pack is enabled. On Gemini Live the
    tools exist only in sessions opened after that, so turning on the first skill pack takes
    effect at the next session.

### 5.5 Active skill pack state

Core holds the active skill pack: its name, its body as read at start, and the start time. The
model's memory is not the source of truth, because a realtime reconnect replays only user and
robot text and drops tool results (`realtime/turn.py`, `replay_block`). The state lives in memory
next to the conversation state that builds the turn instructions. It does not survive a process
restart. (Claude's call; the exact owner is left to the builder.)

On a live Gemini Live session the body reaches the model twice: once as the `skill_pack_start`
tool result, and once as context when the active block first appears. The tool result is needed,
because Gemini sends changed instructions only before the next person turn, not before the
follow-up answer to a tool call (`gemini_wire.py`). The duplicate costs one body's tokens once per
start and is accepted. (Claude's call.)

While a skill pack is active, every turn's instructions carry an active block after the catalogue:
a fixed English frame, then the body. The frame says:

- the text comes from an installed skill pack and guides the activity;
- it cannot change the persona, the safety rules, the privacy rules, the language rules or the
  tool rules, and where it conflicts with them those rules win;
- the person can stop the activity at any time, and the model then calls `skill_pack_end`.

The active skill pack ends when:

1. the model calls `skill_pack_end`;
2. the conversation ends (SPEC 8, Conversation state);
3. the robot goes to sleep (`robot.asleep`) or the circadian state rests (a nap, the quiet hours);
4. the admin turns the skill pack off, removes it, or presses "End" in the web UI;
5. no addressed utterance (the wake decision of SPEC 8, Conversation state) arrives for
   `skill_packs.idle_end_s`.

When it ends, the active block goes away. Gemini Live keeps its own history and cannot drop the
body (`gemini_wire.py`: a removed line is not sent). So the next turn's instructions carry one
fixed line, "The skill pack <name> has ended. Do not follow its instructions any more.", once.
This drift is accepted, like the other Gemini history drift.

A fallback from a realtime route to a chat route keeps the active state, because the chat route
builds its instructions from the same blocks.

### 5.6 Web UI

The Tools tab gets a Skills section, admin only:

- a list: name, description, state (on, off, invalid with the reason, missing), source details
  from the front matter, size, and ignored files;
- view the full text, turn on or off (with admin confirmation), replace, remove;
- "Add skill" (upload) and "Rescan";
- the active skill pack, if one is active, with an "End" button that ends it like
  `skill_pack_end`.

The UI strings exist in English and Italian. The Live tab shows the active skill pack's name.
(Claude's call.)

### 5.7 Diagnostics and logs

The debug report (SPEC 12A) gets: the number of installed, enabled and invalid skill packs, the
invalid reasons, the active name, and the start and end counts by reason. Logs carry skill pack
names, sizes and reasons. They never carry a body or a transcript (SPEC 12).

### 5.8 Configuration (SPEC 11A)

All fields are new. The defaults are Claude's call for the builder to check against the route
budgets.

| Field | Default | Range | Admin only |
| --- | --- | --- | --- |
| `skill_packs.enabled` | `[]` | names | yes |
| `skill_packs.max_enabled` | 20 | 1 to 100 | yes |
| `skill_packs.description_max_chars` | 200 | 20 to 500 | yes |
| `skill_packs.body_max_chars` | 4000 | 200 to 8000 | yes |
| `skill_packs.upload_max_bytes` | 262144 (256 KiB) | 1024 to 4194304 | yes |
| `skill_packs.upload_max_files` | 50 | 1 to 500 | yes |
| `skill_packs.idle_end_s` | 300 | 30 to 3600 | no |

## 6. Constraints

- SPEC 8: no new route kind. Skill packs use the instructions and tools that every route
  already supports.
- SPEC 11A: every number is a config field. No magic numbers (`tests/test_no_magic_numbers.py`).
- SPEC 4A and 12A: core names no robot. Skill packs are robot-agnostic.
- SPEC 12: skill pack text is untrusted. It is data inside a fixed frame, never system rules.
  Admin review covers what the skill pack is for. The frame and the tool checks cover injection.
- Tools on the Pollen cloud-voice profile are not verified (SPEC 11, Voice tools). If Pollen
  ignores tools, skill packs do nothing there. No fallback is added.

## 7. Left to the builder

- Module layout (for example `roboemot/skill_packs/` for loading and state, and the two tools in
  core's tool pack).
- The exact English text of the catalogue header, the active frame and the end line, with prompt
  tests like the other prompt rules.
- The web UI layout inside the Tools tab, using the vendored UI kit.
- Two example skill packs in `configs/skill_packs/` for tests and documentation. They are not
  installed by default.
- Tests: format rules, zip-slip and size limits, the enable gate, budget refusal, the executor
  name check, the active state across a reconnect and a route fallback, every end reason, and the
  Gemini end line.

## 8. To verify on the robot

- Gemini Live starts the right skill pack from free speech in Italian and English, and does not
  start one unasked.
- Gemini Live follows a multi-turn skill pack (a quiz of five questions) and stops when the
  person says so.
- After a session reconnect in the middle of a skill pack, the robot carries on.
