# Configuration guide for the robot owner

This guide covers the settings an owner is most likely to change: the cloud voice, the local
voice, the LLM routes, safety, the circadian rhythm and the privacy switches. It ends with a
troubleshooting list for the Live tab.

Every default, unit and range below comes from the configuration schema,
`packages/roboemot-core/src/roboemot/config/schema.py`. The schema is the source of truth. If this
guide and the schema ever disagree, the schema wins. The design is in the engineering spec,
[`docs/specs/kickoff.md`](../specs/kickoff.md) (SPEC); each section cites the part it follows.

In the tables, "Restart" means the change applies only after RoboEmot restarts. Every other
change applies within about one second (hot reload, SPEC 11A).

## Where a value lives and which one wins

SPEC 11A, Layers. The effective configuration is a merge of layers. A later layer overrides an
earlier one, key by key (`config/stack.py`, `LAYER_ORDER`).

| Order | Layer | Where it lives | Who writes it |
| --- | --- | --- | --- |
| 1 (lowest) | Library defaults | The schema, plus `configs/defaults/roboemot.yaml` (empty today) | RoboEmot releases |
| 2 | Robot manifest | The `behaviour` block of `configs/manifests/<robot>.yaml` | RoboEmot releases. Reachy Mini sets the default name, its name aliases, the voice pitch and a 50 Hz arbiter |
| 3 | Personality preset | `configs/presets/<name>.yaml` (`default`, `calm`, `curious`) | RoboEmot releases; you choose one with `personality` |
| 4 | App config | `~/.local/share/roboemot/app.yaml` on the robot (mode 0600) | The installer, and you by hand. Kept across reinstalls |
| 5 | Per-person overrides | The registry (person records) | The enrollment and the robot itself |
| 6 (highest) | Runtime overrides | `~/.local/share/roboemot/webui_settings.json` (mode 0600) | The web UI **Settings** and **Routes** tabs |

What this means in practice:

- **A value saved in the web UI wins over `app.yaml`.** If you edit `app.yaml` by hand and nothing
  changes, check whether the web UI holds the same key. Reset it in the web UI.
- **The installer writes `app.yaml`, not the web UI file.** Before it changes anything, the
  installer reads `webui_settings.json` and stops when a saved value there would widen what leaves
  the robot. It names the key, never the value. Reset that key in the web UI and run it again.
- **Lists of routes merge by name.** An `llm_routes` entry in `app.yaml` with the same `name` as one
  below it replaces that one; other routes stay.
- **The data folder** is `~/.local/share/roboemot`, or `$ROBOEMOT_DATA_DIR` when set. `app.yaml`
  is always found there (it cannot move itself through `data_dir`).

### Three ways to change a value

1. **Web UI, Settings tab** (**Impostazioni**). It mirrors every field of the schema, with its
   default, unit, range and the layer it comes from. Edits need the PIN. They go to the runtime
   layer and are saved in `webui_settings.json`. A privacy opt-in, a safety loosening and prompt
   text also need the admin confirmation (SPEC 11A, Guardrails). A safety loosening is the one
   edit the web UI writes into `app.yaml` instead.
2. **Installer flags** (`python scripts/install_on_robot.py ...`). They set only the installer's own
   keys in `app.yaml`: `voice.enabled`, the dev content (`--dev`) and the cloud-voice profile
   (`--cloud-voice`). A reinstall with the opposite flag removes exactly those keys. Add
   `--dry-run` to see the keys without writing anything.
3. **CLI on the robot.** `roboemot secrets set <name>` stores an API key. `roboemot config dump
   --app ~/.local/share/roboemot/app.yaml` prints the merged config from the files. The CLI does
   not see the web UI's saved values; the web UI does.

## 1. Cloud voice

SPEC 4A (Installer), SPEC 8 (Realtime speech route, Non-verbal cues), SPEC 12.

### What it does

The robot answers by voice through a realtime speech server instead of its own Vosk, LLM route and
Piper pipeline. On Reachy Mini this is Pollen's server on Hugging Face (route `pollen`). The local
listener still decides who is talking to the robot. Only an utterance it accepts (an "addressed"
utterance) goes to the server.

The library default is off. Audio leaving the robot is an admin opt-in (SPEC 12).

### The quick way: the installer profile

```bash
python scripts/install_on_robot.py --host <robot> --cloud-voice --dry-run   # the plan, writes nothing
python scripts/install_on_robot.py --host <robot> --cloud-voice
```

The profile writes these keys into `app.yaml` (`cloud_voice_config` in
`roboemot/install/__init__.py`; Reachy values from `roboemot_reachymini/install.py`):

| Key | Value written | Library default |
| --- | --- | --- |
| `llm_routes[pollen]` | `kind: realtime`, `base_url: https://pollen-robotics-reachy-mini-realtime-url.hf.space/session`, `api_key: ${secret:hf_token}`, `model: s2s` | no route |
| `voice.realtime_route.enabled` | `true` | `false` |
| `voice.realtime_route.route` | `pollen` | `""` |
| `voice.realtime_route.mode` | `platform` | `direct` |
| `voice.realtime_route.input` | `audio` | `text` |
| `voice.realtime_route.skip_tool` | `true` | `false` |
| `voice.realtime_route.transcript_timeout_s` | `10.0` | `4.0` |
| `privacy.realtime_audio` | `true` | `false` |
| `voice.backend` | `none` | `piper` |
| `voice.fallback_backend` | `none` | `piper` |
| `voice.wake_word_policy` | `never` | `unless_enrolled_in_view` |
| `voice.direction.enabled` | `true` | `true` |
| `voice.facing.enabled` | `true` | `false` |

It also asks for the `hf_token` key when the robot has none (no key: it stops and changes
nothing), skips Piper and removes a Piper virtualenv it installed earlier, and deletes the local
TTS voice files. `-y` never turns the profile on: an unattended run must say `--cloud-voice` or
`--no-cloud-voice`. A reinstall with `--no-cloud-voice` removes exactly these keys; local voice
then needs Piper again (`--piper`). The step-by-step manual setup is in
[Realtime voice through Hugging Face](../realtime-voice-setup.md).

**Privacy.** With this profile, the robot listens without a wake word. The audio of every
utterance spoken while a person faces the robot and looks at it goes to the server, phone calls
included. The skip tool only decides whether the robot answers: the audio has already left.

### The route: `voice.realtime_route.*`

| Key | Default | Unit, range | What it does |
| --- | --- | --- | --- |
| `voice.realtime_route.enabled` | `false` | on/off | Answer through the realtime route. When the route cannot answer, the local pipeline answers (if there is one). Restart. |
| `voice.realtime_route.route` | `""` | route name | The `llm_routes` entry (`kind: realtime`) to use. Restart. |
| `voice.realtime_route.mode` | `direct` | `direct`, `platform` | `direct`: connect to the route's own URL. `platform`: the robot's session provider asks the route's `base_url` for a session first (Reachy Mini). Restart. |
| `voice.realtime_route.input` | `text` | `text`, `audio` | What leaves the robot for an addressed utterance. `text`: the local Vosk transcript. `audio`: the utterance's audio, transcribed by the server; needs `privacy.realtime_audio`, else the route is blocked (`no_opt_in`). |
| `voice.realtime_route.audio_gate` | `addressed` | `addressed` only | Which audio may leave: only an utterance the wake gate and the voice-at-face check accepted, sent once it ended, never live. |
| `voice.realtime_route.transcript_timeout_s` | `4.0` | s, 0.5 to 15 | Longest wait for the server's transcript of the audio. Past it the robot shows the did-not-get-it cue and does not answer. |
| `voice.realtime_route.server_silence_ms` | `600.0` | ms, 64 to 3000 | Silence that ends a turn for the server's VAD. Must be at least `voice.vad.end_silence_ms` (500 ms by default). |
| `voice.realtime_route.close_silence_ms` | `900.0` | ms, 100 to 5000 | Silence appended after the utterance, so the server ends the turn at once. Must be more than `server_silence_ms`. |
| `voice.realtime_route.local_decode_timeout_s` | `10.0` | s, 1 to 60 | Longest wait for a local Vosk decode that the robot still needs. Past it the utterance is dropped (counted). |
| `voice.realtime_route.wake_decode_max_s` | `3.0` | s, 0.5 to 30 | For an utterance the voice-at-face check rejected, Vosk decodes only this much speech, looking for the wake word. |
| `voice.realtime_route.daily_upload_budget_s` | `1800.0` | s per day, 0 to 86400 | Microphone audio the robot may send per day. Past it, utterances go as text. The counter survives a restart (`realtime_upload.json` in the data folder). |
| `voice.realtime_route.voice` | `""` | name | The server's voice. Empty: the server's default. |
| `voice.realtime_route.transcription_language` | `""` | language code | Language the server transcribes in (audio input only). Empty: `conversation.language`. |
| `voice.realtime_route.transcription_model` | `""` | model name | Transcription model sent to the server (audio input only). Empty: the server's default. |
| `voice.realtime_route.prebuffer_ms` | `200.0` | ms, 0 to 2000 | Reply audio held before playback starts, against network jitter. |
| `voice.realtime_route.mood_instructions` | `true` | on/off | Add the mood's voice instructions to each request. |
| `voice.realtime_route.skip_tool` | `false` | on/off | Let the server's model call `ignore_utterance` and stay silent for speech not meant for the robot (a phone call, people talking to each other, TV). The audio still reached the server. |
| `voice.realtime_route.conversation_span_s` | `30.0` | s, 1 to 600 | Conversation start. With `skip_tool`, a conversation counts as in progress while the robot's last spoken answer ended at most this long ago (or a confirmation waits, or the wake word or name was heard). Outside it, each turn carries the strict start rule: answer only words clearly meant for the robot. |
| `voice.realtime_route.connect_timeout_s` | `5.0` | s, 1 to 30 | Longest time for session allocation, WebSocket handshake and `session.updated` together. |
| `voice.realtime_route.send_timeout_s` | `5.0` | s, 1 to 30 | A send stuck this long aborts the socket. Also the keepalive ping interval and timeout. |

A lost session reconnects in the background, paced by the `connectivity.network` backoff. The
status clears without waiting for the next utterance.

### Budgets

Two daily caps apply to the cloud voice:

- **Tokens:** the route's `daily_token_budget` (see LLM routes; default `200000` tokens per day).
  The installer's `pollen` entry does not set it, so the default applies. When the day's count
  reaches it, the route is blocked with reason `budget` until the next day.
- **Upload:** `voice.realtime_route.daily_upload_budget_s` (default 1800 s, 30 minutes of audio per
  day). Past it, utterances go as the local text transcript.

### The key: `roboemot secrets set`

SPEC 8 (Secrets), SPEC 12. A route's `api_key` should be `${secret:<name>}`. The value lives in the
robot's secret store (`secrets.json` in the data folder, mode 0600). A route whose secret cannot be
resolved is unavailable; it is never called with an empty key.

Three ways to store a key:

- On the robot: `/venvs/apps_venv/bin/roboemot secrets set hf_token`. It reads the value from the
  prompt or stdin, never from an argument. `secrets list` shows names only; `secrets remove`
  deletes one.
- Installer: `--secret hf_token` (asks without echo, over SSH).
- Web UI: **Routes** tab (**Modelli**), **API keys** card, with the PIN. The web UI has no TLS yet,
  so prefer SSH.

### Who counts as addressed

SPEC 8, stage 2 and stage 3. These settings decide which utterances the robot answers. With the
cloud voice and audio input, they also decide which audio leaves the robot. They apply to the
local voice too.

| Key | Default | Unit, range | What it does |
| --- | --- | --- | --- |
| `voice.wake_word_policy` | `unless_enrolled_in_view` | `always`, `never`, `unless_enrolled_in_view` | When the wake word is needed. `unless_enrolled_in_view`: anyone speaking while an enrolled person is in view is addressed without it; a stranger alone must say it. |
| `voice.fastpath.addressed_window_s` | `15.0` | s, 1 to 600 | Follow-up window. After an addressed utterance that needed the wake word, the next one starting within this time is addressed too, without the wake word. |
| `voice.fastpath.window_max_s` | `120.0` | s, 1 to 3600 | The follow-up window never reaches further than this from the wake word that opened it. |
| `voice.direction.enabled` | `true` | on/off | Voice-at-face check. An utterance addressed without the wake word must come from the direction of a visible face. Off: music and TV can be answered. |
| `voice.direction.tolerance_deg` | `25.0` | deg, 1 to 90 | How far a sound direction reading may be from a face and still match it. |
| `voice.direction.min_match_share` | `0.5` | 0 to 1 | Share of the readings that must match a face. |
| `voice.direction.degraded_rule` | `auto` | `auto`, `engaged`, `reject` | What to do with too few direction readings. `reject` needs the wake word; `engaged` passes when attention is on a visible person; `auto` picks between the two by reason. |
| `voice.facing.enabled` | `false` | on/off | Facing gate. On top of the voice-at-face check, the face the voice came from must look at the robot. Stops a person watching a screen near the robot from being answered. Needs `voice.direction.enabled`. |
| `voice.facing.max_yaw_deg` | `30.0` | deg, 1 to 90 | A face looks at the robot when its head turn is at most this. |
| `voice.facing.min_share` | `0.5` | 0 to 1 | Share of the samples in which the face must look at the robot. |
| `voice.facing.cpu_max_percent` | `90.0` | %, 1 to 100 | Above this CPU load for `cpu_hold_s`, the head turn is not trusted and the voice-at-face check decides alone. |
| `voice.facing.cpu_hold_s` | `10.0` | s, 0 to 600 | How long the CPU must stay above `cpu_max_percent`. |

An utterance with the wake word always passes both checks.

### Feedback without a voice: non-verbal cues

SPEC 8, Non-verbal cues. With `voice.backend: none` the robot has no local voice. When it cannot
speak, it plays a short gesture and sound instead: `listening` (audio left, waiting for the answer),
`not_heard` (no transcript came back), `fallback` (the route cannot answer and the network is up)
and `offline` (the network is down).

| Key | Default | Unit, range | What it does |
| --- | --- | --- | --- |
| `voice.cues.enabled` | `true` | on/off | Show the non-verbal cues. |
| `voice.cues.listening_onset_s` | `0.3` | s, 0 to 3 | Delay before the listening gesture; an answer that starts sooner shows none. |
| `voice.cues.listening_hold_max_s` | `10.0` | s, 1 to 60 | Longest the listening gesture is held without an answer. |
| `voice.cues.listening_sound` | `true` | on/off | Play a short, quiet blip when the listening gesture shows. |
| `voice.cues.volume_db` | `-6.0` | dB, -40 to 0 | Level of the cue sounds (the circadian volume offset adds to it). |

### Privacy consequences

- `input: text` sends the local transcript as text. `input: audio` sends the speaker's voice, a
  child's included, and the server transcribes it. Audio needs `privacy.realtime_audio`.
- With `privacy.pseudonyms_to_cloud` on, a cloud realtime route is blocked (`pseudonyms`): the
  server would hear real names. Turn one of the two off, or set the route's `locality` correctly.
- With `voice.direction.enabled` off, or on a robot with no sound direction, a TV can be addressed
  and its audio sent.

## 2. Local voice

SPEC 8 (Stages, Voice, expressivity and mood).

### What it does

Everything stays on the robot: Silero VAD finds speech, Vosk transcribes it, an LLM route writes
the answer and Piper speaks it. Piper runs in its own process from a separate virtualenv (it is
GPL-3.0). The installer turns the voice on (`voice.enabled: true`), creates the Piper virtualenv and
fetches the voice models (`roboemot models fetch --voice`). `--no-voice` leaves voice off;
`--no-piper` skips Piper.

| Key | Default | Unit, range | What it does |
| --- | --- | --- | --- |
| `voice.enabled` | `false` | on/off | Run the voice pipeline. The installer sets it on. Restart. |
| `voice.backend` | `piper` | `piper`, `openai_speech`, `kokoro`, `none` | Speech synthesis. `none`: no local speech; only the realtime route's voice speaks. Restart. |
| `voice.fallback_backend` | `piper` | `piper`, `kokoro`, `none` | Offline fallback voice. Restart. |
| `voice.voices` | `{it: it_IT-paola-medium, en: en_US-amy-medium}` | per language | Piper voice per language. Restart. |
| `voice.asr.models` | `{it: vosk-small-it, en: vosk-small-en}` | per language | Vosk model per language. Restart. |
| `voice.models_dir` | `""` | path | Folder of the voice models. Empty: `voice-models/` in the data folder. Restart. |
| `voice.piper.python` | `""` | path | Interpreter of the Piper virtualenv. Empty: `piper-venv/bin/python` in the data folder. Needs the admin PIN. Restart. |
| `conversation.language` | `it` | language code | Default conversation language. The web UI offers only languages with installed voice models. |
| `voice.wake_word` | `null` | name | The robot's name, for the prompt and wake detection. Unset: the name the platform reports (on Reachy Mini, the name the owner gave it). A value here always wins. |
| `voice.default_name` | `""` | name | Set by the manifest (Reachy Mini: `Reachy`). Never spoken; it only decides when the manifest's name aliases apply. |
| `voice.barge_in.enabled` | `true` | on/off | Speech while the robot talks stops it. Off: only the stop keyword is heard while it speaks. |

### Language

The language of an utterance is the speaker's registry preference (`preferences.language`) when a
Vosk model exists for it, else `conversation.language`, else the first language in
`voice.asr.models`.

### Wake word and the platform name

Small Vosk models cannot spell a made-up name such as "Reachy". So the robot wakes on phrases of
common words in the final text, per language (`voice.keywords.<language>.wake_phrases`):

- Italian: `ehi robot`, `ciao robot`
- English: `hey robot`, `hello robot`

When the robot has a name (`voice.wake_word`, or the platform's name), each language adds its
`name_wake_templates` (`ciao {name}`, `ehi {name}`; `hey {name}`, `hello {name}`), but only when
every word of the name is in that recognizer's vocabulary. The Reachy Mini manifest adds the
spellings Vosk produces for "Reachy" (`reati` in Italian, `riyadh she` in English) as name
aliases. The bare name wakes only in a short final text (at most
`voice.fastpath.alias_wake_max_words`, default 4 words).

Changing the name from the web UI needs the admin confirmation. The robot sanitises the name before
it goes into the prompt.

**Privacy.** Nothing leaves the robot, unless an LLM route is a cloud route (next section).

## 3. LLM routes

SPEC 8 (LLM router, OpenAI-compatible routes, Secrets).

### What it does

`llm_routes` is an ordered list of OpenAI-compatible endpoints. The router tries them in order and
falls back to the next one on a timeout or an error. A `kind: realtime` route is only for the cloud
voice; chat requests never use it. When no route answers, the robot says
`conversation.fallback_lines` (or, with no local voice, plays the fallback cue).

The library default is an empty list. Add routes in `app.yaml` or on the web UI **Routes** tab.

```yaml
llm_routes:
  - name: cloud
    base_url: https://api.example.com/v1
    api_key: ${secret:cloud_key}
    model: some-model
    locality: cloud
```

| Key | Default | Unit, range | What it does |
| --- | --- | --- | --- |
| `llm_routes[].name` | `""` | letters, digits, `_`, `-`; at most 64 | Route name (required). |
| `llm_routes[].base_url` | `""` | http(s) URL | OpenAI-compatible base URL (required). |
| `llm_routes[].api_key` | `none` | text or `${secret:name}` | The key. Use `${secret:name}`; a literal key sits in `app.yaml` in clear. |
| `llm_routes[].model` | `""` | model id | Model id (required). |
| `llm_routes[].kind` | `chat` | `chat`, `realtime` | `chat`: Chat Completions. `realtime`: the realtime speech route. A change on a keyed route needs the admin. |
| `llm_routes[].locality` | `auto` | `auto`, `local`, `cloud` | Whether the route may see real names. `auto`: local for loopback, private addresses, `*.local` and names without a dot; cloud for everything else. A LAN proxy that forwards to a cloud provider must be set to `cloud`. |
| `llm_routes[].first_token_timeout_s` | `3.0` | s, 0.1 to 30 | Fall back to the next route when the first token takes longer. |
| `llm_routes[].stall_timeout_s` | `5.0` | s, 0.1 to 60 | Longest gap between chunks once the answer has started. |
| `llm_routes[].max_tokens` | `300` | tokens, 16 to 8192 | Reply token cap. |
| `llm_routes[].daily_token_budget` | `200000` | tokens per day, at least 0 | Daily token cap. Past it the route is skipped until the next day. |
| `llm_routes[].context_budget_tokens` | `8192` | tokens, 256 to 2000000 | Prompt budget per request (4 characters per token). The oldest turns are dropped to fit. |
| `llm_routes[].supports_images` | `false` | on/off | The route accepts images (`image_url` parts). |
| `conversation.turn_reserve_tokens` | `256` | tokens, 16 to 8192 | Room every route's context budget must leave beyond the system prompt, for the newest turn. Checked at load. |

### Secrets and locality guardrails

- A route with a `${secret:name}` key is pinned to its approved `base_url` and `api_key`. Changing
  either, or the `kind`, needs the admin PIN. A new endpoint host needs an explicit confirmation in
  the web UI, because the key will be sent there.
- Making a route local (its `locality`, its `base_url` or a new route) needs the admin: a local
  route sees real names.

### How to check a route

- `roboemot llm check <route>` (chat routes) and `roboemot realtime check <route>` (realtime routes),
  on the robot.
- Web UI **Routes** tab, **Check**. Prefer it: the CLI does not see the web UI's saved values.

**Privacy.** A cloud route receives the conversation text, the speaker summary and the scene
summary. With `privacy.pseudonyms_to_cloud` on, it receives aliases instead of registry names.

## 4. Safety

SPEC 11A (Guardrails), SPEC 12 (Physical safety).

### What it does

The safety settings cap how the robot moves and what happens when commands stop. Two rules:

- **At runtime, safety only gets more conservative.** The web UI and the API may tighten a
  `safety.*` value. Loosening one needs the admin PIN and is written to `app.yaml`, not to the web
  UI file. The same rule covers every field with a conservative direction outside `safety.*` (for
  example `privacy.pseudonyms_to_cloud`, `voice.echo_tail_ms`).
- **Manifest limits are ceilings.** The joint limits and dynamics in the robot manifest
  (`pose_groups`, `dynamics`, `constraints`) cannot be exceeded by any layer. A manifest without
  limits is rejected.

| Key | Default | Unit, range | Safe direction | What it does |
| --- | --- | --- | --- | --- |
| `safety.watchdog_ms` | `200.0` | ms, 20 to 2000 | lower | Return home slowly when no command arrives for this long. |
| `safety.watchdog_return_factor` | `0.3` | 0.01 to 1 | lower | Speed factor of the watchdog's return home. |
| `safety.soft_stop` | `hold_position` | `hold_position` | none | Soft stop behaviour. |
| `safety.hard_stop` | `disable_motors` | `disable_motors` | none | Hard stop behaviour. |
| `safety.constraint_margin_deg` | `3.0` | deg, 0 to 20 | higher | Coupled constraints (head and body yaw) are planned this far inside their limit. |
| `safety.allow_unmeasured_dynamics` | `false` | on/off | off | Run with manifest dynamics still `null`. Development only; `--dev` on the installer sets it. |
| `safety.unmeasured_dynamics_fallback_vel` | `0.5` | rad/s, 0.01 to 10 | lower | Velocity cap for unmeasured dynamics (dev only). |
| `safety.unmeasured_dynamics_fallback_acc` | `2.0` | rad/s^2, 0.01 to 100 | lower | Acceleration cap for unmeasured dynamics (dev only). |
| `safety.unmeasured_dynamics_fallback_lin_vel` | `0.02` | m/s, 0.001 to 1 | lower | Linear velocity cap for unmeasured translations (dev only). |
| `safety.unmeasured_dynamics_fallback_lin_acc` | `0.1` | m/s^2, 0.001 to 10 | lower | Linear acceleration cap for unmeasured translations (dev only). |
| `safety.extra_prompt_rules` | `[]` | list of text | none | Rules added to the system prompt's safety block. Changes need the admin. |

The Reachy Mini manifest's dynamics are still `null` (not measured yet). Until they are, the robot
runs only with `safety.allow_unmeasured_dynamics: true` (the installer's `--dev`), and the limiter
uses the fallback caps above.

The stop keyword, the soft stop and the hard stop work in every state, asleep or awake.

## 5. Circadian rhythm, quiet hours and presence

SPEC 7A.

### What it does

The robot simulates a day: it is `awake`, `drowsy` or `resting`. Sleep pressure builds while it is
awake; an internal clock, entrained by the camera's light level, sets when it is most alert.
Resting is a RoboEmot behaviour with the motors on: smaller motion, quieter voice, slower vision.
The robot still answers in every state, only more quietly.

| Key | Default | Unit, range | What it does |
| --- | --- | --- | --- |
| `circadian.enabled` | `true` | on/off | Run the circadian rhythm. Off also turns off quiet hours. |
| `circadian.quiet_hours` | `["22:30", "07:00"]` | local time, start and end | Window in which resting is forced (unless people present keep the robot awake). |
| `circadian.presence_keeps_awake` | `true` | on/off | People present and active prevent resting, inside quiet hours too. |
| `circadian.presence_window_s` | `60.0` | s, 1 to 3600 | People seen, recognised or attended to within this time count as present. |
| `circadian.wake_hold_s` | `300.0` | s, 10 to 3600 | How long the wake word keeps a resting robot awake. |
| `circadian.peak_hour` | `15.0` | h, 0 to 24 | Hour of the day at which alertness peaks. |
| `circadian.drowsy_threshold` | `0.55` | 0 to 1 | Sleepiness above which the robot is drowsy. |
| `circadian.resting_threshold` | `0.75` | 0 to 1 | Sleepiness above which the robot rests. |
| `circadian.light_source` | `camera` | `camera`, `clock_only` | Entrainment source. `clock_only` uses the clock (and `location`, if set). |
| `circadian.location` | `null` | `{lat, lon}` | Optional location for clock-only mode. Private: masked in the web UI. |

Quiet hours also stop the robot's look-around scans.

**Not the same as the robot's sleep.** When the owner puts the robot to sleep from the platform
(motors off), RoboEmot pauses perception, the listener and the behaviour tick
(`runtime.pause_when_asleep`, default `true`). Nothing is sent to the cloud while the robot sleeps.

**Privacy.** `circadian.light_source: camera` reads only a light level from frames; no frame is
stored.

## 6. Privacy switches

SPEC 12 (Data). Every privacy option is off by default. Turning one on is an opt-in and needs the
admin PIN. `privacy.*` values are masked in the web UI's list and shown only on request, with the
PIN.

| Key | Default | Unit, range | What it does |
| --- | --- | --- | --- |
| `privacy.realtime_audio` | `false` | opt-in | Let addressed microphone audio leave the robot for the realtime route. The installer's `--cloud-voice` sets it. |
| `privacy.realtime_platform_id` | `false` | opt-in | Let the platform's session provider send the robot's stable platform identifier with each session request (platform mode only). The server can then link all sessions of this robot. |
| `privacy.pseudonyms_to_cloud` | `false` | on/off, safe direction on | Send aliases (`Persona-A`, `Person-A`) instead of registry names to cloud routes, and restore the names in the answer. A remote speech voice still hears the names it speaks. Blocks a cloud realtime route. Turning it off again needs the admin. |
| `privacy.multimodal_frames` | `false` | opt-in | Allow single camera frames to be sent to LLM routes for image questions. |
| `privacy.keep_enrollment_images` | `false` | opt-in | Keep the face crops taken at enrollment. |
| `privacy.debug_recording` | `false` | opt-in | Allow time-limited debug recording. |
| `webui.live.enabled` | `false` | opt-in | Serve the Live tab (mood, who is in view by name, conversation phase). |
| `webui.live.transcript` | `false` | opt-in | Keep the recent conversation, as text, in memory for the Live tab. Never saved, logged or reported. |
| `webui.live.video` | `false` | opt-in | Serve a live camera preview to the Live tab. Needs `webui.live.enabled`. Every open page shows a banner while someone watches. |

Rules that hold whatever the switches say:

- Frames, audio and face embeddings are never written to logs.
- API keys are never logged, shown or returned. A route whose `${secret:name}` cannot be resolved
  is not called.
- The debug report holds codes and numbers, never text, audio or names.

## Troubleshooting

The Live tab's **What's happening** panel (**Cosa succede**) shows one status line and the last
decisions with their reasons. It needs `webui.live.enabled` (and `webui.live.activity`, on by
default). Status lines below are given in English and Italian.

| Live tab shows | What it means | What to check |
| --- | --- | --- |
| "Cloud voice down (connection_lost)" / "Voce cloud non disponibile (connection_lost)" | The socket to the server dropped. | Nothing, usually: the route reconnects in the background, paced by the network backoff. If it keeps coming back, check the robot's internet access. |
| "Cloud voice down (connect_failed)" or "(connect_timeout)" | The robot could not open a session within `connect_timeout_s`. | Internet access; the server's status. The route keeps retrying. |
| "Cloud voice down (closed_by_server)" | The server closed the session. | The route reconnects in the background. If it repeats, run the route's **Check** on the Routes tab: a `provider_http_401` or `403` there means the key is wrong or expired; store a new one with `roboemot secrets set`. |
| "Cloud voice down (no_opt_in)" | `input: audio` without `privacy.realtime_audio`. | Turn the opt-in on (admin PIN), or set `input: text`. |
| "Cloud voice down (pseudonyms)" | `privacy.pseudonyms_to_cloud` is on and the route is a cloud route. | Turn one of the two off. |
| "Cloud voice down (budget)" | The route's `daily_token_budget` is used up for today. | Wait for the next day, or raise the budget. |
| "Cloud voice down (unavailable)" | The route is not usable now: its key cannot be resolved, or its link is offline. | `roboemot secrets list` shows the key's name; check the network. |
| "Cloud voice down (no_route)" | `voice.realtime_route.route` names no `llm_routes` entry. | The route name; a web UI override of it. |
| "Cloud voice down (no_extra)" or "(provider)" | The `realtime` extra (websockets) or the robot's session provider is missing. | Run the installer again. |
| "Offline: the network is down" / "Offline: la rete non c'è" | The network link is down. | The robot's Wi-Fi. The robot plays the offline cue or phrase when spoken to. |
| "Not answered: no wake word" | The utterance needed the wake word. | `voice.wake_word_policy`; say a wake phrase. |
| "Not answered: the voice did not come from a face" | The voice-at-face check rejected it. | Face the robot, or say the wake word. `voice.direction.tolerance_deg`. |
| "Not answered: no facing data (no head turn measured)" or a not-facing reason | The facing gate rejected it. | Look at the robot. `voice.facing.max_yaw_deg`. |
| "No transcript from the cloud within N s" | The server did not return a transcript in `transcript_timeout_s`. | The server's latency; raise `voice.realtime_route.transcript_timeout_s` (up to 15 s). |
| "Fallback sound: the cloud voice could not answer and there is no local voice" | The route was down and `voice.backend` is `none`. | The route's reason in the status line. |
| "Asleep" / "Dorme" | The platform holds the motors off. | Wake the robot from the platform's app. |

When the web UI and `app.yaml` disagree, the web UI wins: look at the field's source layer on the
Settings tab, and reset it there.
