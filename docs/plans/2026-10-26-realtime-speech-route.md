# Realtime speech route (optional, off by default): epic plan

## Context
Owner request (2026-10-26). On the CM4 the local voice pipeline (SPEC 8: Silero VAD, Vosk with the grammar pass, a
chat-completions route, Piper or `voice/openai_speech.py`) hears poorly and sounds flat. The owner liked the voice of
Pollen's conversation app (Apache-2.0), which streams microphone audio over a WebSocket speaking the OpenAI realtime
protocol; the backend does ASR, the LLM and TTS.

Owner decisions (binding):
- An alternative to the local pipeline, configurable and **off by default**. The local pipeline stays and is the fallback.
- Hierarchy pattern (SPEC 4A, Hierarchy gate): core gets a generic realtime client, configured with a direct URL or a
  session provider. The Reachy package registers the provider that allocates a session through Pollen's proxy with the
  robot's `hardware_id`. Any other robot uses direct mode.
- Keys only from the secret store (`${secret:name}`). Audio leaving the robot is a new kind of egress: its own explicit
  opt-in, shown in the debug report; audio and text are never logged.

Paths: `core/` = `packages/roboemot-core/src/roboemot/`, `rm/` = `packages/roboemot-reachymini/src/roboemot_reachymini/`.

### Verified facts (sources read 2026-10-26)
Pollen app, commit `b731ea1` (`src/reachy_mini_conversation_app/huggingface_realtime.py`, `config.py`, README):
- Deployed mode: `POST https://pollen-robotics-reachy-mini-realtime-url.hf.space/session`, header
  `X-Reachy-Mini-Authorization: Bearer <token>` only when a token exists, JSON body `{"hardware_id": ...}` only when the
  daemon reports one. Reply: `connect_url` (must end in `/realtime`, may carry a query), `session_id`. The app then sends
  the **same bearer token to the allocated `connect_url`** as the realtime API key (`_build_openai_compatible_client_
  from_realtime_url(connect_url, bearer_token)`), a host chosen by the allocator.
- Local mode: `ws://<host>:8765/v1/realtime` (default port 8765), served by Hugging Face `speech-to-speech`.
- Session: PCM in and out at the server's native rate through an HF extension (`{"type": "audio/pcm", "rate": null}`,
  16 kHz), `server_vad` with `interrupt_response: true`, `tools` with `tool_choice: "auto"`, a voice from a fixed
  Qwen3-TTS list (default `Aiden`); Italian and English are among the offered languages.
- It streams **every** microphone frame, continuously. A send stuck for 5 s aborts the socket (`AUDIO_SEND_TIMEOUT_S`).
- Local barge-in (README): a Silero candidate ducks playback to 25%; playback stops only when the server VAD confirms.
- `response.done` may carry `metadata["speech_to_speech.turn_latency"]` (vad_decision, stt, llm, tts_ttfa, e2e).
- It uses the `openai` SDK's realtime client (`openai==2.28.0`), which needs `websockets`.
- The allocator Space page states no terms, rate limits, retention, logging, SLA, privacy policy or models.

`huggingface/speech-to-speech`, commit `bf39b6c` (`src/speech_to_speech/api/openai_realtime/`: README, `service.py`,
`handlers_*.py`, `runtime_config.py`, `websocket_router.py`; `VAD/vad_handler.py`), Apache-2.0:
- Client events accepted: `input_audio_buffer.append`, `input_audio_buffer.commit`, `session.update`,
  `conversation.item.create` (`input_text` or `function_call_output` only), `conversation.item.delete`,
  `conversation.item.truncate` (accepted, no-op), `response.create` (per-response `instructions` and `tool_choice`),
  `response.cancel`. Server events: `session.created/updated`, `error`, `input_audio_buffer.speech_started/stopped`,
  `conversation.item.input_audio_transcription.delta/completed`, `response.created`, `response.output_audio.delta/done`,
  `response.output_audio_transcript.delta/done`, `response.function_call_arguments.done`, `response.done`
  (`completed|cancelled|incomplete|failed`, with usage). "A tested core subset, not a claim of full equivalence."
- Audio: the pipeline is 16 kHz PCM16 in 512-sample chunks; the client rate comes from `audio.input.format.rate`.
- **Turns always end by the server VAD.** `input_audio_buffer.commit` only flushes the resampler; there is no
  `turn_detection: null` path. `session.update` sets `turn_detection.threshold` and `silence_duration_ms`
  (`vad_handler.py:229-239`). The VAD handler's constructor defaults turn on a smart-turn model that can hold an
  incomplete turn up to 2000 ms (`smart_turn_max_wait_ms`, `vad_handler.py:79-83`); the server's CLI may differ.
- `turn_detection.create_response: false` is honoured. On the STT path (the default: STT, then an LLM) the transcript
  still goes into the server's chat as text, `transcription.completed` is still sent, and a later `response.create`
  answers it (`service.py:844`). On the direct-audio-input path (an audio LLM, no STT) the utterance is kept as a WAV
  item instead (`service.py:919`). This plan relies on the STT path, because the wake gate needs the transcript.
- **No authentication and no throttling.** It binds to loopback by default; the LLM can be local (Qwen3.5-4B by
  default, which parses tool calls from `<code>` blocks) or a remote API configured on the server host.

RoboEmot code read (this branch, `127abdf`): `voice/pipeline.py` (`build_listener`), `voice/listener.py` (ASR seam
`ASRBackend`, hand-off with the grammar job and deadline), `voice/speech.py` (`SpeechOutput._speak/_push`, the timeline,
the echo reference), `voice/envelope.py`, `voice/openai_speech.py` (route-borrowing speech backend, budget charge),
`conversation/worker.py`, `llm/router.py` (`eligible`, `charge`, `auth_headers`; every route serves chat),
`llm/prompt.py`, `health/network.py`, `runtime/assembly.py`, `runtime/startup.py`, `runtime/diagnostics.py`,
`plugins.py` (`GROUPS`), `config/schema.py` (`param` with `opt_in`, `conservative`, `admin_only`, `restart`).
Dependencies: `reachy-mini 1.12` requires `websockets>=12,<16`, so the robot's apps venv already has it; `openai`'s
`realtime` extra needs `websockets>=13,<16`. `websockets 15.0.1` is 1.5 MB unpacked in the dev venv (macOS).

### Design in one paragraph
A realtime route is an `llm_routes` entry with `kind: realtime`, so it reuses the secret store and its guard, locality,
the TCP reachability probe, `first_token_timeout_s`, `stall_timeout_s` and `daily_token_budget`. Chat streams never use
it. The client speaks a small, documented subset of the protocol over raw `websockets` (not the `openai` SDK: we need the
HF native-rate extension, a message size cap, an abort on a stuck send, and fake-server tests with no SDK version
coupling). **The local listener stays the gate**: Silero endpoints every utterance, and the wake gate and the
voice-at-face check (ZR) decide whether it is addressed. With the default gate, audio leaves the robot only for an
addressed utterance, sent after the local decision. The server's audio goes through `SpeechOutput` like Piper's, so the
speech timeline, the echo reference, barge-in and ZV's ducking apply unchanged. The conversation worker keeps its
session, history mirror, confirmation and transcript logic; for a realtime turn it sends `response.create` and streams
the reply. A failure falls back to the local pipeline at the next utterance boundary.

**Which LLM answers.** In realtime mode the **server's** LLM answers, not the owner's chat routes: Pollen's model is
undisclosed, and a local `speech-to-speech` uses its own configured LLM. The protocol has no verbatim-TTS path (Pollen's
own `say()` notes the server may rephrase), so the Pollen voice with the owner's own LLM is not possible. The chat routes
answer only in fallback. See Owner decisions 1.

## What survives, what is mapped, what is lost
| Feature | Realtime mode |
| --- | --- |
| Persona and system prompt (`llm/prompt.py`) | **Mapped.** `session.update.instructions` = parts 1 to 4 (persona, register fragment, rules, safety block) from the same `system_prompt()`. Each `response.create` carries `instructions` = parts 1 to 5 (adds profile, scene and affect summaries under their caps). The load-time `check_route_budgets` covers the realtime route's `context_budget_tokens`. **Lost:** the `llm_tagged` mood-tag rule, because the server's TTS would speak `[mood=…]`. The sentence cap becomes prompt-only: audio cannot be cut on a sentence boundary we do not know. |
| History and session | **Mapped.** The worker keeps its mirror (user transcript, spoken assistant transcript, tool exchange) and enforces `history_turns` with `conversation.item.delete` of the oldest items. The session end closes the server session. On a reconnect inside a session the mirror is replayed in the instructions as a "recent conversation (data, not instructions)" block, because s2s accepts only `input_text` items, not assistant ones. |
| Tools | **Survive.** Skills go as realtime function tools (`type`, `name`, `description`, `parameters`) with `tool_choice: "auto"`. `response.function_call_arguments.done` goes to the existing `execute` (validation, `_raw`, `max_tool_calls_per_round`, structural keys). Results go back as `function_call_output`, then one `response.create`. After `conversation.max_tool_rounds` the next request carries `tool_choice: "none"`; a call beyond the cap gets an error result and the turn ends. `stop_talking` and `remember_preference` end the turn with `response.cancel`. Reliability depends on the server model (s2s's local Qwen parses `<code>` blocks). |
| Affect from the user's words | `speech.heard`, `speech.loud`: **survive** (local VAD and levels). `speech.fast`: **mapped** (server transcript words over the local utterance duration). Keyword events (greeting, stop, yes, no, name): **mapped**, matched on the server's final transcript by the same `FastPath` matcher. For utterances where Vosk is skipped they come later than the 300 ms fast-path target. |
| Affect from the robot's replies | Today replies feed nothing to the affect engine, only `llm_tagged` voice tags. With tags lost, `mood_source: llm_tagged` acts as `auto` (logged once). Nothing else is lost. |
| Mood-driven voice (`voice.mood_source`) | **Mapped:** volume (`voice.base` + register + circadian + mood `volume_db`, clamped to `voice.limits`) applied as gain on the returned audio, so the robot stays quiet at night. The mood's `instructions` text is appended to the per-response instructions (`mood_instructions`), which shapes wording. **Lost:** rate, pitch, variability, pauses, Piper's noise and the per-sentence variation. |
| Wake word and "Neve" | **Survive.** The wake phrases, the rendered name templates, the bare name and its short-final rule are matched on the server transcript. A better ASR should hear "Neve" where Vosk heard "cadeva". The grammar pass is **not needed** for realtime utterances (counted `skipped: realtime`). `missing_words` is None, so the name is always added. Non-addressable speech still goes through local Vosk and the grammar pass (below). |
| ZR voice-at-face gate | **Survives and decides egress.** Default gate `addressed`: an utterance that is addressable without the wake word at its start (engaged, enrolled in view, follow-up window) is buffered with Vosk skipped. At its end the voice-at-face check runs locally; only if it passes is the audio sent. One that fails is decoded by Vosk from the buffer for the wake word, as today, and sent only if the wake word is found. Speech that is not addressable runs the local Vosk path; if the wake word is heard, its buffered audio is sent. Nothing is streamed while the robot speaks or sleeps. A `live` gate (streaming during speech) is not built (Owner decisions 2). |
| Barge-in | **Ours, not the server's.** `interrupt_response: false`, and nothing is streamed while the robot speaks. The local `BargeIn` (Silero, echo guard, onset path, keyword-stop) cuts the `SpeechOutput` as today, then sends `response.cancel` and `conversation.item.truncate(audio_end_ms = played)`. The listener restarts from the pre-roll and that audio becomes the next utterance. ZV's ducking sits on the same push path, so it applies to realtime audio. |
| Speech timeline and head motion while speaking | **Survives.** A new stream job in `SpeechOutput` pushes the returned PCM through `_push(..., speech=True)`, so energy frames, phrase gaps and emphasis come from the real audio. There is one timeline sentence per response audio item; the question flag is set from `output_audio_transcript.done`, which arrives before playback ends, for `question_end` and yes/no. `prebuffer_ms` absorbs network jitter; underruns are counted as today. |
| Latency (SPEC 13) | Fast-path reaction ≤ 300 ms and barge-in ≤ 200 ms: **local, unchanged**. End of speech to first audio ≤ 2.0 s (cm4): **at risk.** It adds the upload, the server's turn end (`close_silence_ms` of zeros sent in a burst, plus a smart-turn hold of up to 2000 ms on s2s defaults), STT, LLM, TTS and the round trip. The report measures it; SR-5 decides. |
| Route budget and accounting | **Mapped.** `response.done.usage.total_tokens` is charged to the route's `daily_token_budget`, also for cancelled and failed responses. Without usage, a chars/4 estimate of instructions and transcripts is charged, as the router does. New: `daily_upload_budget_s` caps the microphone seconds sent per day. A spent budget makes the route ineligible, and the local pipeline answers. |
| Fallback at start | Missing `realtime` extra, opt-in off with `input: audio`, route ineligible, pseudonyms conflict (Owner decisions 3) or connect failure within `connect_timeout_s`: the local pipeline runs. The link `realtime` goes lost and is retried with `connectivity.network` backoff. A switch happens only at an utterance boundary, never mid-turn. |
| Fallback during a session | Socket closed, stuck send (`send_timeout_s`), or no transcript or no first audio within `transcript_timeout_s`, then the route's `first_token_timeout_s`: before any audio played, that turn is answered locally, from the server transcript if it arrived, otherwise from a Vosk decode of the buffered audio, through the chat routes and Piper (`fallback_mid_turn`). After audio played, the reply is cut like a barge-in (history marked interrupted) and not repeated. |
| `network` link | It is offline when every route that can answer a turn is offline: the chat routes, plus the realtime route while realtime mode is on. A live realtime route therefore cannot mask dead chat routes in local mode, and the puzzled gesture and offline phrase keep working. |
| Debug report (SPEC 12A) | New section `roboemot.realtime_route`, listed in its own row below. |
| Web UI (SPEC 11A) | The settings mirror picks up the new fields (opt-ins need the admin). The Routes tab shows `kind`. A chat-route check of a realtime route answers 409 `route_kind`; SR-5 adds a realtime check. The Live state gets `audio_egress` (a banner while audio may leave). The Reachy robot section adds read-only rows for the provider. |
| Installer | Core extra `realtime = ["websockets>=13,<16"]`. On Reachy it is already present (reachy-mini 1.12 requires `websockets>=12,<16`), so it is pinned at the robot's version and adds about 0 MB. Elsewhere about 1.5 MB unpacked (aarch64 wheel figure measured in SR-4, as on 2026-10-22). |

### Debug report section `roboemot.realtime_route`
Numbers and states only: no text, audio, keys, query strings or identifier values.
- state: `off`, `blocked` with a reason (`no_opt_in`, `no_extra`, `pseudonyms`, `budget`, `upload_budget`), `connecting`, `connected` or `fallback_local`;
- mode and gate;
- endpoint as `scheme://host:port` only;
- `audio_egress` and `platform_id_sent` as booleans;
- counters: sessions, reconnects, failures by reason, utterances sent, rejected or deleted, transcripts and transcript timeouts, responses by status, unrequested responses, tool calls, fallbacks at start and mid-turn, unknown events, oversize messages;
- audio seconds sent today against the budget;
- latency p50/p95/max: endpoint to transcript, endpoint to first audio, and the server's own stages when sent.

## Slices (integration branch `epic/robot-plugin-hierarchy`, one PR each)
The owner made `epic/robot-plugin-hierarchy` the only integration branch (2026-10-26). Prerequisite: ZV (barge-in
ducking) has merged into it.

SR-1. **Protocol client, route kind, session-provider seam. No runtime wiring.**
   - Files:
     - new `core/realtime/__init__.py`;
     - new `core/realtime/protocol.py`: builders for the client subset and a parser for the server subset, size and type checks, base64 PCM16 decode with an even-length check, unknown events counted and ignored, server `error` reduced to its `code`, never its `message`;
     - new `core/realtime/client.py`: `RealtimeClient`, a thread with its own event loop like `SpeechOutput`; it connects with `Authorization` only per the URL rules below, sets `max_size`, never follows redirects, pings, aborts on a send stuck past `send_timeout_s`, and gives a snapshot;
     - new `core/realtime/session.py`: the `SessionProvider` protocol (`allocate(AllocationRequest) -> Allocation(connect_url, key_hosts)`), `DirectSession` (the WS URL derived from the route `base_url`: `http`→`ws`, `https`→`wss`, plus `/realtime`), `ENTRY_POINT_GROUP = "roboemot.realtime"`, and `load_session_provider(robot_id)`;
     - `core/plugins.py`: register the group;
     - `core/config/schema.py`;
     - `core/llm/router.py`: chat `eligible()` skips `kind: realtime`, a new `realtime_route(name)`;
     - `core/llm/conformance.py` and `core/webui/routes.py` (route list and check): 409 `route_kind`;
     - `core/webui/api.py` (`/api/routes/add`), `core/webui/strings.py` and the Routes tab page: the `kind` field;
     - `core/config/stack.py` (`secret_violations`, line 929): a `kind` change of a keyed route needs the admin;
     - `packages/roboemot-core/pyproject.toml`: extras `realtime`, and `websockets` in `dev`;
     - `packages/roboemot-core/tests/test_no_magic_numbers.py`: the new modules.
   - URL rules, in one helper used by both modes:
     - `ws` or `wss` only; `wss` required unless the route's effective locality is local;
     - no userinfo, at most 2048 characters;
     - no loopback or link-local host unless local (reuse the `url_local_refused` check);
     - the query is kept for the connection but never logged or reported;
     - the key goes only to the route's own host, or to a host matching a `key_hosts` suffix the provider declares statically (shown in the report).
   - Config:
     - `llm_routes[].kind`: `chat | realtime`, default `chat`.
     - `voice.realtime_route.connect_timeout_s`: 5.0 s, 1 to 30.
     - `voice.realtime_route.send_timeout_s`: 5.0 s, 1 to 30.
     - `voice.realtime_route.max_message_bytes`: 1048576 bytes, 65536 to 16777216.
   - Tests (no network):
     - `packages/roboemot-core/tests/realtime/fake_server.py`: a `websockets.serve` on `127.0.0.1:0`, scripted to the s2s subset. All new core tests go in `packages/roboemot-core/tests/realtime/`;
     - `test_realtime_protocol.py`: build and parse, malformed, oversize and unknown events, odd-length PCM;
     - `test_realtime_client.py`: connect, `session.update` and `updated`, the receive loop, close mid-stream, stall, stuck send (the fake stops reading), oversize message;
     - `test_realtime_urls.py`: derivation and every refusal; the key never reaches a non-approved host, checked on the fake's request headers; the query is not in logs;
     - router, conformance and web UI tests for `kind`;
     - the hierarchy gate passes with the new group (contract: factory `(manifest)` returning a `SessionProvider`).
   - SPEC: SPEC 8 new subsection "Realtime speech route" (design, protocol subset, URL and key rules, the "which LLM answers" note). SPEC 8 "Every LLM route speaks the OpenAI Chat Completions API" gets "(chat routes; a `kind: realtime` route speaks the realtime subset below)". SPEC 4A Hierarchy gate lists the group. SPEC 11A catalog row. CLAUDE.md: "LLM routes are OpenAI-compatible only" gets a realtime-subset sentence, which also names the one exception to `tool_choice="auto"`: `tool_choice: "none"` on the request after the tool-round cap. It forbids no tool, so it is not forced tool choice; and the repository-layout row names `realtime/`.

SR-2. **Realtime replies: text in, server voice out.**
   - Files:
     - new `core/realtime/route.py`: `RealtimeRoute` owns the client lifecycle (connect at start, close at sleep and at session end, reopen on wake and on need), the link `realtime` in `NetworkMonitor`, eligibility, the budget charge and the snapshot;
     - new `core/realtime/turn.py`: one turn with instructions, `response.create`, audio to `SpeechOutput`, the assistant transcript, tools and usage;
     - `core/voice/speech.py`: `SpeechOutput.stream(chunks, utterance, mood, volume_db)`, a stream job in the same queue, epoch and `_push(speech=True)` path, with `prebuffer_ms`, a late question flag and `stop()` cutting it;
     - `core/conversation/worker.py`: a realtime branch in `_answer`, the history mirror, the instructions replay, fallback to the chat path;
     - `core/llm/prompt.py`: `realtime_instructions()`, which drops the tag rule;
     - fixed lines (confirmation question, fallback, offline cue, health) go to the server voice as a `response.create` asking for that line while the route is up, and to Piper only when it is down (Owner decisions 4);
     - `core/health/network.py`: the aggregate rule above;
     - `core/runtime/assembly.py`, `startup.py`, `diagnostics.py` (new section), `sleep.py`, `live.py` (`audio_egress`).
   - Input is the local Vosk transcript as `conversation.item.create` `input_text`. Only text leaves the robot, as with a chat route, so no audio opt-in is needed. Before the response, `session.update` carries the instructions.
   - Barge-in: `response.cancel` + `truncate`. ZV ducking applies to the stream job (a test asserts it).
   - Config (`voice.realtime_route.*`):
     - `enabled`: false, restart;
     - `route`: "", restart; must name a `kind: realtime` route, checked at load;
     - `session`: `direct | platform`, default `direct`, restart;
     - `input`: `text | audio`, default `text`;
     - `voice`: "" (server default);
     - `prebuffer_ms`: 200 ms, 0 to 2000;
     - `mood_instructions`: true.
   - Tests (fake server):
     - a whole turn: timeline frames written, echo reference fed, volume gain applied, question flag;
     - a tool round trip, the tool-round cap, `stop_talking`;
     - barge-in sends cancel and truncate and marks the history interrupted;
     - the budget is charged from usage, and estimated without usage;
     - fallback at start: no extra, connect refused, budget spent;
     - fallback mid-turn: close before first audio goes to the chat path, close after first audio cuts the reply;
     - the session end closes the server session; history beyond the cap deletes items;
     - an unrequested `response.created` is cancelled and counted;
     - the network aggregate;
     - privacy: a turn runs with log capture and the real debug report is built; neither holds the transcript, the reply text or PCM bytes (pattern of `test_secret_leak.py`).
   - SPEC: SPEC 8 Voice/expressivity: what maps and what is lost in realtime mode. SPEC 8 Conversation state: the history mirror, item deletion and replay; the sentence cap is prompt-only in realtime. SPEC 12A Links and Diagnostics. SPEC 11A Live tab `audio_egress`.

SR-3. **Audio in: addressed audio leaves the robot.**
   - Files:
     - new `core/realtime/input.py`: `RealtimeInput` buffers an utterance, sends it with `close_silence_ms` of zeros in one burst, collects **every** `transcription.completed` the burst produces (a server that still splits it gives several items; their texts are joined in order), and enforces the upload budget;
     - `core/voice/listener.py`: at utterance start decide `addressable` (the same predicates as the wake gate); then either skip Vosk and buffer, or run Vosk as today. A deferred final reuses the `_Handoff` job and deadline machinery (`transcript_timeout_s`), so the listener thread never blocks on the network. A rejected or failed realtime utterance is decoded by Vosk from its buffer;
     - `core/voice/pipeline.py`, `core/voice/fastpath.py` (rate and keywords from the server text);
     - `core/realtime/turn.py`: audio turns;
     - `core/config/schema.py`.
   - Server session: `server_vad` with `silence_duration_ms = server_silence_ms`, `interrupt_response: false`. Only addressed audio is sent, and the instructions are refreshed just before it, so a backend that ignores `create_response: false` is harmless.
   - Config:
     - `privacy.realtime_audio`: false, `opt_in`: "Allow addressed microphone audio to leave the robot to the realtime route";
     - `voice.realtime_route.audio_gate`: `addressed` only (the field exists so a later gate is a config value, not a migration), default `addressed`;
     - `transcript_timeout_s`: 4.0 s, 0.5 to 15;
     - `server_silence_ms`: 600 ms, 64 to 3000. It must be at least `voice.vad.end_silence_ms` (500 ms), checked at load. Otherwise the server VAD could end a turn on a pause the local endpointer kept inside one utterance: it would answer the first fragment, and the second would collide with that response (`conversation_already_has_active_response`);
     - `close_silence_ms`: 900 ms, 100 to 5000. It must exceed `server_silence_ms`, checked at load. The zeros go in the same burst, so they cost upload, not wall time;
     - `daily_upload_budget_s`: 1800 s, 0 to 86400.
   - Tests in `packages/roboemot-core/tests/realtime/` (fake server, test clips in `packages/roboemot-core/tests/data`):
     - no audio frame reaches the fake for an unaddressed utterance (no face, no wake word), nor while the robot speaks, sleeps or the opt-in is off;
     - wake word on a non-addressable utterance: the Vosk path, then the buffer is sent;
     - a voice-at-face reject is never sent;
     - a pause inside one utterance (shorter than `end_silence_ms`) gives one answer; a fake that splits it anyway gives joined transcripts and no second `response.create`;
     - transcript timeout goes to the local decode and the chat path;
     - the upload budget is enforced;
     - the listener `share` does not grow on the fake;
     - keywords and `speech.fast` come from the server text;
     - privacy log and report scan, as in SR-2.
     - Opus security review: new egress.
   - SPEC changes, these sentences amended:
     - SPEC 8 router: "Only text leaves the robot: transcripts and the scene summary. Audio and images are never sent unless an app enables a multimodal route explicitly." gets "…, or the admin opts in to the realtime speech route's audio (`privacy.realtime_audio`; addressed utterances only by default)".
     - SPEC 8 routes: "Only text is sent: transcript, scene summary, and the speaker's display name and preferences. Never embeddings or audio." gets the same exception.
     - SPEC 12 Data: "Only text (transcript + scene summary) can reach an off-board LLM route, and that can be disabled entirely" gets "…; with `privacy.realtime_audio` the addressed utterances' audio reaches the realtime route, whoever speaks, a child included; the consent text says so".
     - SPEC 8 stage 2 notes how ZR decides egress. SPEC 13 latency note.

SR-4. **Reachy session provider (Pollen allocator), settings row, installer.**
   - Files:
     - new `rm/realtime.py`: `PlatformSession` implements `SessionProvider`. In platform mode the route's `base_url` **is the allocator URL**, so the secret guard pins it, the TCP probe checks it and locality reads it. The plugin keeps the allocator URL only as the documented example in `configs/companion.yaml` and refuses a route whose `base_url` path is not `/session` (`provider_url`). It POSTs to that `base_url` with `httpx` and no redirects, and sends the `X-Reachy-Mini-Authorization` header only when the route has a key. `hardware_id` goes in the body only with `privacy.realtime_platform_id`, read through `rm/daemon.py`. The `connect_url` must end in `/realtime` and pass the SR-1 URL rules; `key_hosts` is declared statically. Timeout: `connect_timeout_s`;
     - `packages/roboemot-reachymini/pyproject.toml`: entry point `roboemot.realtime` → `reachy-mini-wireless`;
     - `rm/settings.py`: read-only rows `robot.reachy-mini-wireless.realtime_session` (allocator host, identifier sent yes or no);
     - `rm/install.py` and `core/install/__init__.py`: `EXTRA_DISK_MB["realtime"]`, the extra added with voice, the dry-run line `would install the realtime extra (websockets <ver>, already present)`;
     - `configs/companion.yaml`: a commented example of a `kind: realtime` route for each mode;
     - Locality for pseudonyms, `wss`, and the loopback rule are computed on **both** the allocator host and the connect host. Both must be local for the route to count as local.
   - Config: `privacy.realtime_platform_id`: false, `opt_in`: "Let the platform's session provider send the robot's platform identifier". Help text stays generic: no vendor name in core.
   - Tests in `packages/roboemot-reachymini/tests/test_realtime_provider.py` and `test_install_wizard.py`, using `httpx.MockTransport` for the allocator (no network):
     - the identifier is absent unless opted in, and its value is never in logs or the report;
     - the token goes only to the allocator and the declared hosts;
     - a redirect is refused;
     - a bad `connect_url` (no `/realtime`, `ws` to a cloud host, userinfo, loopback) is refused;
     - an allocator HTTP error falls back to local with reason `provider_http_<code>`;
     - the installer dry run adds no package on a venv that already has websockets.
     - Opus security review: key forwarding, identifiers.
   - SPEC: SPEC 4A (provider, installer extra, sizes), SPEC 12A Reachy diagnostics row, SPEC 12 Data (what the allocator receives).

SR-5. **Robot validation (HIL) and realtime check.**
   - Files: `core/cli.py`, `core/realtime/check.py`, `core/webui/routes.py`, `packages/roboemot-core/tests/realtime/test_realtime_check.py`, `packages/roboemot-reachymini/tests/test_realtime_hil.py`, `docs/specs/kickoff.md`.
   - Config: none (the check reuses `llm_check.deadline_s`).
   - `roboemot realtime check <route>`, also from the Routes tab: connect, `session.updated`, one `input_text` turn, first-audio latency. It sends no microphone audio, and its budget is charged.
   - On the robot, gate `addressed`, against Pollen deployed. Measure:
     - endpoint to transcript and to first audio, p50/p95, against 2.0 s;
     - listener `share` and CPU, Vosk skipped versus local;
     - underruns;
     - barge-in ≤ 200 ms;
     - nod-to-phrase sync on the returned audio;
     - "Neve" recognition;
     - client-chosen item ids (`input_text`, `function_call_output`): `conversation.item.delete` of them works against Pollen's server (SR-2 sets `item.id`; unverified until now), and `conversation.item.truncate` is honoured;
     - TV with nobody engaged sends nothing.
   - Tests `@pytest.mark.hil`. SPEC 13 table rows with the figures, SPEC 15 open questions, owner A/B listen.

Order: SR-1 → SR-2 → SR-3 → SR-4 → SR-5. SR-4 may start after SR-1 if it mocks `RealtimeRoute`, but it merges after SR-2.
Estimate: 5 slices, about 15 to 17 agent runs:
- 5 Sonnet authors and 5 Sonnet reviews;
- Opus security reviews on SR-3 (audio egress) and SR-4 (key forwarding, identifier), plus SR-1 if the URL and key helper grows past the rules above;
- 2 or 3 fix rounds;
- SR-5 needs the owner at the robot.

Re-confirm past about 24.

## Risks
- **Pollen allocator: undocumented, no terms, no SLA, no privacy statement.** It can change, rate-limit or disappear without notice. It receives addressed audio of anyone in the home, a child included, and with the opt-in the robot's `hardware_id`. The models and retention are unknown. Mitigation: off by default, two admin opt-ins, local fallback, an upload cap, the report shows egress. The owner chose this deployed service as the target (Owner decisions 1).
- **s2s has no authentication or throttling.** On the LAN anyone can use it, and a `ws://` LAN endpoint carries audio in clear. Local only: `wss` is required off-LAN.
- **Protocol drift.** Both servers implement a subset ("not full equivalence"); the deployed one is not inspectable. The fake server pins our subset; unknown events are counted, not fatal.
- **Latency.** The smart-turn hold and cloud round trips may break the 2.0 s budget. SR-5 measures before the owner relies on it.
- **Behaviour differs from local mode.** The server LLM, sentence cap and tool reliability are not ours. Fixed lines change voice when spoken locally.
- **ZV dependency.** SR-2 must use the same push path, or ducking silently stops applying (test enforced).

## Owner decisions (2026-10-26)
1. **Which LLM answers: Pollen cloud only.** The product target is Pollen's deployed allocator (SR-4). Core keeps the generic direct-URL client, because the provider split needs it and the fake-server tests and the manual sim run use it, but no LAN-server features or docs are built beyond that.
2. **Audio gate: `addressed`.** Audio leaves the robot only after the local wake and voice-at-face decision. `live` is not built.
3. **Pseudonyms: block.** With `privacy.pseudonyms_to_cloud` on, a cloud realtime route is unavailable and the local pipeline runs (fail closed).
4. **Fixed lines: server voice.** The confirmation question, fallback, offline cue and health lines are spoken by the server voice when the route is up, accepting that the server may rephrase them. The local Piper line is used only when the route is down.
5. **`hardware_id`: off by default** (controller ruling, plan recommendation). `privacy.realtime_platform_id` stays an opt-in, turned on only if the allocator refuses anonymous sessions.

## Rules for authors
- Worktree per slice from `origin/epic/robot-plugin-hierarchy`, its own `.venv` with the `dev` extra (it carries `websockets`, so fake-server tests never skip). Check `roboemot.__file__`.
- Core names no robot, vendor or Space URL: `pollen`, `reachy` and the allocator URL live only in `rm/`. `test_robot_hierarchy.py` must pass with an empty baseline.
- Every new behavioural constant is a schema field. New modules go into `test_no_magic_numbers.py`.
- Never log, report or raise with audio, transcripts, replies, keys, query strings or identifier values. Server `error.message` is dropped.
- No network in unit tests: the fake WS server and `httpx.MockTransport` only.
- Cite SPEC 8, 11A, 12, 12A, 13 (and 4A for SR-4) in commits.

## Verification
- Per slice: `ruff check . && ruff format --check .`, full `pytest`, CI green, including the `models` and `sim` jobs.
- SR-2 and SR-3: also a manual run against the sim daemon (`scripts/sim-daemon.sh`) with a local `speech-to-speech serve` on the Mac (direct mode, `locality: local`) and the web UI.
- SR-4: `install_on_robot.py --dry-run` shows the realtime extra and no new package.
- SR-5: the HIL checklist above on the Wireless, figures recorded in SPEC 13.
