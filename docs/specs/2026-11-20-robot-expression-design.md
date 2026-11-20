# Robot expression: LLM-written reactions (design, 2026-11-20)

Status: approved by the owner 2026-11-20. Implements on SPEC 7 (appraisal events), SPEC 8
(conversation state, robot-initiated turn, non-verbal cues, Radio playback) and SPEC 11 (Voice
tools: Due items, the announcer).

## Intent

The robot reacts in its own words when something happens around it. Today it either stays
silent or plays a fixed cue (a gesture plus a short sound, `voice/cues.py`). The LLM writes a
short line that fits the moment, so the robot seems to notice and care.

## Situations (owner choice, 2026-11-20)

1. **Didn't catch it / half-understood.** Someone spoke to the robot and it did not get it: the
   server's transcript never came (today's `not_heard` cue), a local decode heard nothing
   (`decodes_lost`), or the words came only from the robot's own recognizer and their meaning is
   unclear. The robot asks one short clarifying question («Scusa, non ho capito: volevi la musica
   o la radio?»).
2. **Failed action.** Something the robot tried outside a live answer failed: for example a
   radio station that does not load or drops. The robot says so in one sentence. (A tool failure
   inside a live answer is already explained by the model in that answer: out of scope.)
3. **Shown object.** Someone shows the robot an object or animal (`object.shown`, SPEC 7). The
   robot comments on it or asks about it.
4. **Returning person.** A known person (`person.known`) comes back after an absence. The robot
   greets them briefly. **It never mentions anything from an earlier conversation** (owner).

## Hard rules (owner)

- **Never interrupt a conversation.** A reaction speaks only when the conversation is idle
  (`conversation_idle`): no turn running, no one speaking, no answer playing.
- **Never touch the music.** While the robot's radio plays (audible, or held by its own player),
  a reaction never pauses, lowers or stops it. It shows **only a gesture** and the line is never
  said later (owner choice "Gesture only"). This differs from the timer and reminder announcer,
  which pauses the radio.

## Approaches

- **A. The announcer path.** Reuse what due timers and reminders use: a queue, a wait for an
  idle conversation, a robot-initiated turn (`ConversationWorker.event_turn`) whose event text
  the model turns into a line in the live session (so it fits the context and language).
- **B. Pre-written lines.** The LLM writes a few lines per situation in advance; the robot picks
  one locally. Fast and free per event, but generic: it cannot name the station or the object.
- **C. Model instructions only.** Tell the cloud model to ask instead of ignoring. Covers only
  situation 1, and only when the audio reached the cloud.

**Chosen: A, with B as the offline fallback** (owner, 2026-11-20). With a cloud route up, the
model writes the line (A). With no route, the robot says a fixed line of the situation with the
local voice (`prompts/<situation>.<lang>.txt`, like `event_fallback`), else plays a cue, as the
announcer does today. A robot with no local voice (the cloud-voice profile) gets the gesture and
cue only.

## Design

- **Expressor.** A runtime part beside the announcer, sharing its idle wait and turn start but
  not its media pause. It takes reaction requests `(situation, data, created_at)` from the four
  triggers and handles one at a time.
- **Per request:**
  1. Drop it if it is older than its situation's max age (a clarifying question is useless after
     the moment: proposed default 10 s; failed action 30 s; shown object while the object is
     still shown; returning person 30 s).
  2. If the radio plays (audible or held by the player): play the situation's gesture, count it,
     done. No sound, no line, no later line.
  3. If the conversation is not idle: wait for it, up to the max age; then drop.
  4. Start a robot-initiated turn with an event text built from fixed templates and data only
     (category, station name, the person's first name if enrolled), never free text heard from
     the room. The model answers in one short sentence.
  5. Offline: the fixed line with the local voice, else the cue.
- **Rate limits** (settings): at most one reaction per situation per cooldown; the same person
  is greeted again only after a minimum absence (proposed 30 min); the same object category only
  once per look (as `object.shown` already does).
- **Half-understood also in the model's instructions:** when the model hears audio clearly meant
  for the robot but cannot tell what is wanted, it asks a short question instead of calling
  `ignore_utterance`.
- **Settings:** `expression.enabled` (on), per-situation switches, max ages, cooldowns, absence
  for greeting. Every number is a schema field (SPEC 11A).
- **Report and Live tab:** counters per situation and outcome (`spoken`, `gesture_only_music`,
  `dropped_stale`, `dropped_busy`, `offline_line`, `cue`), and a Live event per reaction. No
  heard text in logs (SPEC 12).
- **Privacy:** nothing new leaves the robot: the event text holds no audio and no transcript;
  a returning person's name goes only where the profile summary already sends it.

## Decisions after the builder check (2026-11-20)

Owner:
- **Model asks instead of ignoring:** yes, behind a switch (`expression.model_asks`, on), shipped
  last. Only for audio that clearly reached the robot (the name, or the face rule); the expressor's
  limits do not apply inside the live session.
- **Returning person, both cases greet:** (1) waking from sleep or the night counts as a return
  («Buongiorno, Marcello»); (2) someone away at least `expression.greet_absence_s` (proposed 30
  min) while the robot was awake. An app restart does not greet people seen in the
  `greet_absence_s` before it (no record kept across restarts: the first sighting after a restart
  is not a return).

Coordinator's calls (owner may correct):
- **Existing reactions stay.** The appraisal rules for `person.known` and `object.shown` keep
  their emotion reaction; the line comes after it. While music plays that existing reaction is
  the whole response: no extra gesture. For "didn't catch it" the spoken question replaces the
  `not_heard` cue's sound (its gesture still shows); with music or offline the cue stays as
  today.
- **Asleep, resting, quiet hours:** a reaction never wakes the robot; it is dropped (counted).
  The wake-up greeting fires once the robot is awake and outside quiet hours.
- **Music test:** "music plays" means the radio is audible now (`media_audible`) or the player
  holds a playing station. A station that failed or is silent is not music: the failure can be
  said, and saying it must not cancel the player's retry.
- **Language:** `conversation.language`. Name: the enrolled person's name from the registry, as
  the profile summary already uses it.
- **Memory:** cooldowns per person and per object category, in memory only (a restart clears
  them).

## Goals, non-goals, anti-goals

- Goals: the four situations, the two hard rules, a short natural line in the robot's language.
- Non-goals: chatting on its own initiative, reacting to TV or strangers' speech, any reaction
  during music other than a gesture.
- Anti-goals (a failure even if it works): the robot talks over a conversation; the radio
  pauses or dips because of a reaction; repeated greetings or comments that feel like nagging;
  a greeting that recalls a past conversation.

## Left to the builder

- The exact trigger for "half-understood" from the robot's own words (a confidence signal, or
  only the `not_heard` / `decodes_lost` cases at first).
- Which media-player failures count as "failed action".
- The gesture per situation (manifest `gesture_map` entries, robot-agnostic).
