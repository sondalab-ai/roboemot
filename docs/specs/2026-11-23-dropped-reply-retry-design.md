# Dropped or slow cloud reply: retry, wait and spoken status (design, 2026-11-23)

Status: draft for the owner, 2026-11-23. Builds on SPEC 8 (Realtime speech route, conversation
state, robot-initiated turn) and SPEC 12/12A (privacy, debug report).

## Intent

A cloud reply can stop before it says anything. Seen on the robot 2026-11-23: the robot asked
about a book, the person answered, the server sent the transcript at 14:22:52, no reply came, the
watchdog closed the session at 14:22:56 (`stall`), the robot played a fallback sound and the
session came back at 14:22:59. The person's answer was lost and they never learned why.

On a robot with the cloud-voice profile the cloud is the only voice: today's backup (the chat
route answers the turn, Piper says it, `ConversationWorker._realtime_lost`) has no voice to speak
with, so the person hears only a cue.

The robot should try again, tell the person it is doing so, and say sorry if it cannot.

A second case, seen the same day (14:40 to 14:44): «Ciao Neve» said twice, no answer. Debug
report: Gemini Live, robot turns, 5 utterances sent, 2 transcripts (1.5 s and 8.9 s after the
audio), 3 transcript timeouts at `transcript_timeout_s` 4.0 s; the robot then asked to repeat
(its own recognizer heard «ciò neve», unused because the audio had gone to the cloud). Whether
the 3 transcripts came late or never is not counted anywhere.

## Owner decisions (2026-11-23)

- **Retry by resending the person's words** in the new session, with the conversation so far.
- **At most 2 retries** (3 attempts in all).
- **The first retry is announced** («Un attimo, riprovo») so the person knows why the robot is
  waiting. The second retry is not announced.
- **After the last attempt fails, the robot apologizes** («Scusa, ho perso la connessione»).
- **The announcement and the apology are saved clips of the cloud voice:** recorded once while
  the cloud works, kept on disk, and played when the cloud does not answer. **The cache is
  invalidated when the voice or the robot's language changes** (owner: "so that the cloud voice
  is not lost for nothing").
- **Robots with a local voice retry the cloud first** too; only after the retries fail does
  today's backup (chat route answers, Piper says it) run. The voice stays the same as long as
  possible.
- **Slow transcript: wait longer and fill the silence.** The robot waits longer for the cloud's
  transcript (about 9 s) instead of asking to repeat after 4 s, and fills the wait with a saved
  phrase of its own voice from the same clip cache («Fammi pensare…»).
- **Total wait about 15 s:** from the first loss to the apology at most
  `retry.total_budget_s` (default 15). A retry that cannot start before the budget ends is
  skipped.

## Goals, non-goals, anti-goals

- Goals: a reply lost before it started gets up to two more tries; the person hears in the
  robot's own voice that it is retrying and, if all fails, that it is sorry.
- Non-goals: resuming a reply that already played in part; retrying a turn in which a tool ran;
  retries for chat routes (they have the route fallback chain); retrying robot-initiated turns
  (reactions, due items: they have their own drop rules).
- Anti-goals (a failure even if it works): a tool runs twice (the radio starts again, a timer is
  set twice); the robot repeats half an answer; a clip plays over someone speaking; a clip in a
  different voice or language than the robot's current one; the clip recording is heard through
  the speaker.

## Approaches

- **A. The conversation turn retries (chosen).** `ConversationWorker` owns the turn: it knows
  whether any audio played, whether a tool ran, and the history. On a lost reply it waits for the
  route's reconnect and runs the turn again, the person's words sent as a text item.
- **B. The route retries inside the link.** Hidden from the worker, but the route does not know
  about tools: a retry could run a tool twice. Rejected.
- **C. Keep today's backup (chat route + Piper).** No voice on a cloud-voice robot. Rejected as
  the answer; on robots with Piper it runs after the retries (owner).

## Design

### When a turn is retried

A failure is any end of a realtime turn's attempt with no answer: the reply `closed` (any
reason: `stall`, loss, stuck send), a server `error` on the turn's `response.create`, and the
failures before the request is sent (`no_session`, a failed `session.update` or text send,
`_realtime_rounds_held` returning False early). All must hold:

1. Nothing of the reply played (`speech.played` false for every job).
2. No skill ran in the turn (`turn.rt_tools` false, worker.py `rt_tools` is set only when a real
   skill ran). Tone, skip and over-cap calls do not run anything and do not block a retry; their
   settled outputs are discarded with the attempt.
3. The turn is a person's utterance (not a robot-initiated event turn).
4. No newer utterance is queued for the worker (the person spoke again: that turn wins, this one
   ends without apology). New check: the worker has no such test today.
5. Retries so far < `voice.realtime_route.retry.max_retries` (default 2), and the next attempt
   can start before `retry.total_budget_s` from the first failure.

Otherwise the existing path runs unchanged (`_realtime_lost`).

Both turn modes are in scope. In `turns: server` mode the first attempt adopts the server's
response; a retry is always a robot-asked turn (text item plus `response.create`).

### One retry

1. Before the first retry only: play the `retrying` clip (see below), if cached; else the
   existing listening cue.
2. Wait for the route to be ready, at most `retry.reconnect_wait_s` (and within the total
   budget). Not ready in time counts as a failed attempt. A stall does not always close the
   session: after `response.created` the request is abandoned and the same session (same
   generation) may stay up, so "ready" may be immediate and the retry runs on the live session.
3. Before the resend, the failed attempt's server items are deleted on the session that holds
   them, if it is still up (`route.delete_items` on that generation), so the same words are not
   twice in the conversation. A new session gets the history it lacks through the existing
   replay (`connect_session` / next-turn replay).
4. The person's words go as one `input_text` user item (`turn.utterance.text`), then
   `response.create`. The retry always forces this text path: the audio branch (the server's own
   items hold the words when `audio[0] == route.generation`, worker.py) must not be taken, since
   those items were deleted.
5. The attempt is followed exactly as a first attempt (same watchdogs, same tool handling). A
   tool call in a retried attempt runs normally: from then on condition 2 blocks further retries.

### After the last attempt

- Robot with Piper: today's backup runs (the chat route answers, Piper says it). No apology
  clip: the person gets an answer. If the backup also fails, Piper says the `sorry` line.
- Robot without a local voice: play the `sorry` clip, if cached; else the cue. The turn ends
  (the person can ask again).

### Slow transcript

- `voice.realtime_route.transcript_timeout_s` default goes from 4.0 to 9.0 s (range unchanged).
- When an addressed utterance's transcript has not come after `retry.filler_after_s` (default
  2.5 s), the `thinking` clip plays once for that utterance (if cached; else nothing).
- New counter: transcripts that arrive after the robot stopped waiting
  (`transcripts_after_timeout`), so the report tells late from missing.

### Slow reply

- The same `thinking` clip plays when a turn's first reply audio has not come
  `retry.filler_after_s` after the request, at most once per turn (not when it already played
  for the transcript). Reply audio arriving while it plays queues after it.

### Voice clips cache

- **Lines:** three fixed lines per language, `prompts/retrying.<lang>.txt`,
  `prompts/sorry.<lang>.txt` and `prompts/thinking.<lang>.txt` (Italian and English at first).
- **Key:** provider host, model, the voice name actually sent in `session.update` (route `voice`
  or the platform profile's), `conversation.language` (the robot's language, not a turn's detected language: a clip is
  always in the language the robot speaks by default), and a hash of the line text. Any change is
  a cache miss: the old file is deleted when its replacement is stored.
- **Recording:** when a clip is missing and the conversation is idle (no turn, no one speaking,
  no answer playing, no media audible), the robot asks the open session for one response:
  "Say exactly this sentence and nothing else: <line>". The audio is captured to memory, never
  played. It is kept only if the response's output transcript matches the line (case and
  punctuation ignored); otherwise retried at the next idle moment, at most
  `retry.clip_record_attempts` per key per day. The recording item is deleted from the server
  conversation afterwards so it does not enter the history.
- **Storage:** WAV files in the app data folder (`~/.local/share/roboemot/voice-clips/`), mode
  0600, at the speaker's sample rate. They hold the robot's own synthetic voice only, never
  room audio (SPEC 12 holds).
- **Playback:** through the existing speech output, like a cue, with the speaking state set so
  the listener ignores the robot's own voice.

### Settings (SPEC 11A)

Under `voice.realtime_route.retry`: `enabled` (on), `max_retries` (2, 0 to 3),
`total_budget_s` (15), `filler_after_s` (2.5), `reconnect_wait_s`, `announce_first_retry` (on), `clip_record_attempts`. Each with a unit,
range and an Italian description in `webui/descriptions_it.json`.

### Report and Live tab (SPEC 12A)

- `roboemot.realtime_route.retry`: attempts, recovered (by attempt number), gave_up, skipped (by
  reason: played, tool_ran, newer_utterance, disabled), clips (per line: cached, missing,
  recorded, rejected_mismatch).
- Fillers: played (by cause: transcript, reply), missing clip.
- Live events: «Risposta persa: riprovo (1 di 2)», «Risposta recuperata», «Risposta persa:
  mi scuso». No transcript text.

## Constraints

- No magic numbers (SPEC 11A); core stays robot-agnostic (no robot names, no `getattr` duck
  typing); nothing logged beyond counts and states (SPEC 12).
- The resent words are the text of the utterance the turn already sent (as audio or text) in
  this conversation: the same content goes to the same provider again, nothing new leaves the
  robot. `utterance.text` can come from the robot's own recognizer (`words_source`); that is
  still the same utterance.

## Coordinator's calls (owner may correct)

- Failures before the request is sent count as failures (same silence for the person).
- Both turn modes are in scope; a retry is always robot-asked.
- Tone, skip and over-cap calls do not block a retry; only a skill that ran does.
- A clip uses `conversation.language`, not a turn's detected language.
- The `thinking` filler also covers a slow first reply, not only a slow transcript (same
  silence for the person).
- `transcript_timeout_s` default raised for every realtime route, not only Gemini.
- With Piper, no apology clip before the backup answer: an answer beats an apology.

## Left to the builder

- How the worker waits for readiness (route event vs. polling `ready`).
- The exact output-transcript comparison.
- Whether the recording response uses `tool_choice: "none"` or an empty tool list (no tool may
  run during a recording).
- Gemini Live specifics (its session sends no `response.create`; the resend is a client
  content turn).
