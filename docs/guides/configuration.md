# Configuration guide for the robot owner

This guide covers the settings an owner is most likely to change: the cloud voice, the local
voice, the LLM routes, safety, the circadian rhythm and the privacy switches. It ends with a
troubleshooting list for the Live tab.

Every default, unit and range below comes from the configuration schema,
`packages/roboemot-core/src/roboemot/config/schema.py`. The schema is the source of truth. If this
guide and the schema disagree, the schema wins. The design is in the engineering spec,
[`docs/specs/kickoff.md`](../specs/kickoff.md) (SPEC). Each section cites the SPEC sections it
follows.

How to read the tables:

- "Needs restart" in the "What it does" column means the change applies only after RoboEmot
  restarts. Every other change applies without a restart (hot reload, within 1 s, SPEC 11A, Rules).
- "Opt-in" in the "What it does" column marks a privacy option. It is off by default, and turning
  it on needs the admin confirmation.
- `""` is an empty text value.

## Glossary

SPEC 1 (Glossary) defines the project's own terms. This list explains the technical words used
below.

| Term | Meaning |
| --- | --- |
| Admin PIN | The web UI's admin PIN. You type it to confirm a guarded change. |
| Admin confirmation | The web UI step that asks for the admin PIN before a guarded change: a privacy opt-in, a safety loosening, prompt text, a route's key or endpoint. |
| Backoff | Waiting longer after each failed attempt before trying again. |
| Hot reload | A changed setting takes effect while RoboEmot runs, with no restart. |
| Keepalive | A small message sent at fixed times to check that a connection is still alive. |
| Loopback | The robot's own address (`127.0.0.1`, `localhost`). Traffic to it never leaves the robot. |
| Mode 0600 | A file only its owner (the robot's user) can read and write. |
| Piper | The local text-to-speech program. It runs in its own process. |
| Pseudonyms | Neutral aliases (`Persona-A`, `Person-A`) sent to a cloud service instead of real names. |
| Silero | The local voice activity detector (VAD) model. |
| Socket | The network connection (a WebSocket) between the robot and the cloud voice server. |
| TLS | Encryption of a network connection (`https`, `wss`). The web UI has none (SPEC 14, M5). |
| VAD | Voice activity detection: finding where speech starts and ends in the microphone audio. |
| Vosk | The local speech recognizer. It turns speech into text on the robot. |

## Where a value lives and which one wins

SPEC 11A, Layers.

The effective configuration is a merge of layers. A later layer overrides an earlier one, key by
key. The order comes from `LAYER_ORDER` in `packages/roboemot-core/src/roboemot/config/stack.py`.

| Order | Layer | Where it lives | Who writes it |
| --- | --- | --- | --- |
| 1 (lowest) | `defaults` | The schema, plus `configs/defaults/roboemot.yaml` (it holds no values). | RoboEmot releases. |
| 2 | `manifest` | The `behaviour` block of `configs/manifests/<robot>.yaml`. | RoboEmot releases. Reachy Mini sets the default name, its name aliases, the voice pitch and a 50 Hz arbiter. |
| 3 | `preset` | `configs/presets/<name>.yaml` (`default`, `calm`, `curious`). | RoboEmot releases. You choose one with `personality`. |
| 4 | `app` | `~/.local/share/roboemot/app.yaml` on the robot (mode 0600). | The installer, and you by hand. It is kept across reinstalls. |
| 5 | `derived` | Memory only. | The library: values it computes in place of app values, such as persona traits rated by an LLM (SPEC 11A, Persona). The owner never writes it. The Settings tab shows it as a field's source layer. |
| 6 | `person` | The registry (person records). | Enrollment and the robot itself. |
| 7 (highest) | `runtime` | `~/.local/share/roboemot/webui_settings.json` (mode 0600). | The web UI **Settings** and **Routes** tabs. |

What this means in practice:

- A value saved in the web UI wins over `app.yaml`. If you edit `app.yaml` by hand and nothing
  changes, check whether the web UI holds the same key. Reset it in the web UI.
- The installer writes `app.yaml`, not the web UI file. Before it changes anything, the installer
  reads `webui_settings.json`. It stops when a saved value there would widen what leaves the robot.
  It names the key, never the value. Reset that key in the web UI and run the installer again.
- Lists of routes merge by name. An `llm_routes` entry in `app.yaml` with the same `name` as an
  entry below it replaces that entry. Other routes stay.
- The data folder is `~/.local/share/roboemot`, or `$ROBOEMOT_DATA_DIR` when that is set. `app.yaml`
  is always read from there. The `data_dir` setting cannot move it.

### Three ways to change a value

1. **Web UI, Settings tab** (**Impostazioni**). It mirrors every field of the schema, with its
   default, unit, range and source layer. Every edit needs the PIN. Edits go to the runtime layer
   and are saved in `webui_settings.json`. A privacy opt-in, a safety loosening and prompt text also
   need the admin confirmation (SPEC 11A, Guardrails). A safety loosening is the one edit that the
   web UI writes into `app.yaml` instead.
2. **Installer flags** (`python scripts/install_on_robot.py ...`). They set only the installer's own
   keys in `app.yaml`: `voice.enabled`, the dev content (`--dev`) and the cloud-voice profile
   (`--cloud-voice`). A reinstall with the opposite flag removes exactly those keys. Add
   `--dry-run` to see the keys without writing anything.
3. **CLI on the robot.** `roboemot secrets set <name>` stores an API key. `roboemot config dump
   --app ~/.local/share/roboemot/app.yaml` prints the merged configuration from the files. The CLI
   does not see the values saved in the web UI. The web UI does.

To restart RoboEmot after a change that needs a restart, run `python scripts/run_on_robot.py
--restart` from the repo checkout. It asks first (`-y` skips the question), and the robot's head
moves.

## 1. Cloud voice

SPEC 4A (Installer), SPEC 8 (Realtime speech route, Non-verbal cues), SPEC 12.

### What it does

The robot answers by voice through a realtime speech server instead of its own Vosk, LLM route and
Piper pipeline. On Reachy Mini this is Pollen's server on Hugging Face (route `pollen`). The local
listener still decides who is talking to the robot. Only an utterance it accepts (an "addressed"
utterance) goes to the server.

The cloud voice is off by default. Audio leaving the robot is an admin opt-in (SPEC 12).

### Set it up with the installer

1. Create a Hugging Face access token. Sign in at https://huggingface.co, open
   https://huggingface.co/settings/tokens, select **Create new token**, then **Fine-grained**.
   Leave every write permission off: the route needs a read-only token. Copy the token; Hugging
   Face shows it only once.
2. From the repo checkout, look at the plan. It writes nothing:

   ```bash
   python scripts/install_on_robot.py --host <robot> --cloud-voice --dry-run
   ```

3. Run it:

   ```bash
   python scripts/install_on_robot.py --host <robot> --cloud-voice
   ```

   The installer asks for the token (`hf_token`) when the robot has none. It reads it without
   echo. With no token, it stops and changes nothing.

Passing `--cloud-voice` accepts the `privacy.realtime_audio` opt-in. `-y` alone never turns the
profile on. An unattended run must say `--cloud-voice` or `--no-cloud-voice`.

The profile needs the voice pipeline, so it keeps `voice.enabled: true`, which the installer sets.
The installer refuses `--cloud-voice` together with `--no-voice` or `--piper`, before it touches the
robot.

The profile writes these keys into `app.yaml`. The values come from `cloud_voice_config` in
`packages/roboemot-core/src/roboemot/install/__init__.py` and from
`packages/roboemot-reachymini/src/roboemot_reachymini/install.py`.

| Key | Value written | Default |
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

The profile also skips Piper, removes a Piper virtualenv that the installer added earlier, and
deletes the local text-to-speech voice files. The step-by-step manual setup is in
[Realtime voice through Hugging Face](../realtime-voice-setup.md).

**Privacy.** With this profile, the robot listens without a wake word. The audio of every
utterance spoken while a person faces the robot and looks at it goes to the server. Phone calls
are included. The skip tool only decides whether the robot answers. By then the audio has already
left the robot.

### Verify

1. Check the route. On the robot, run `/venvs/apps_venv/bin/roboemot realtime check pollen --app
   ~/.local/share/roboemot/app.yaml --robot reachy-mini-wireless`. Or use the **Check** button on
   the web UI **Routes** tab (**Modelli**). Prefer the button: the CLI does not see values saved in
   the web UI.
2. Turn on `webui.live.enabled` on the Settings tab. It is a privacy opt-in, so it needs the admin
   confirmation. Open the **Live** tab. The status line should not say "Cloud voice down" /
   "Voce cloud non disponibile".

### Go back to local voice

1. Run the installer again with `--no-cloud-voice`. It removes exactly the profile keys listed above
   and keeps every other key. Add `--piper` to install Piper again; the models step fetches the
   text-to-speech voices again.
2. If the web UI holds a saved value that keeps audio leaving the robot
   (`privacy.realtime_audio: true`, `voice.realtime_route.enabled: true` or
   `voice.realtime_route.input: audio`), the installer stops and names it. Reset it in the web UI
   and run the installer again.
3. Restart RoboEmot (`python scripts/run_on_robot.py --restart`), if the installer did not start
   it.

### The settings an owner uses: `voice.realtime_route.*`

| Key | Default | Unit, range | What it does |
| --- | --- | --- | --- |
| `voice.realtime_route.enabled` | `false` | on/off | Answer through the realtime route. When the route cannot answer, the local pipeline answers, if there is one. Needs restart. |
| `voice.realtime_route.route` | `""` | route name | The `llm_routes` entry (`kind: realtime`) to use. Needs restart. |
| `voice.realtime_route.input` | `text` | `text`, `audio` | What leaves the robot for an addressed utterance. `text` sends the local Vosk transcript. `audio` sends the utterance's audio, and the server transcribes it. `audio` needs `privacy.realtime_audio`; without it the route is blocked (`no_opt_in`). |
| `voice.realtime_route.skip_tool` | `false` | on/off | Lets the server's model stay silent for speech not meant for the robot. Examples are a phone call, people talking to each other, and TV. A skipped utterance gets no answer. Its audio still reached the server. |
| `voice.realtime_route.daily_upload_budget_s` | `1800.0` | s per day, 0 to 86400 | Microphone audio the robot may send per day. Past it, utterances go as text. The count survives a restart. |
| `voice.realtime_route.transcript_timeout_s` | `4.0` | s, 0.5 to 15 | Longest wait for the server's transcript of the audio. Past it, the robot shows the did-not-get-it cue and does not answer. |
| `voice.realtime_route.voice` | `""` | name | The server's voice. Empty uses the server's default. |

After a change of `voice.realtime_route.enabled`, `voice.realtime_route.route` or
`voice.realtime_route.mode`, restart RoboEmot (`python scripts/run_on_robot.py --restart`).

A lost session reconnects in the background. The reconnect waits longer after each failure
(backoff, `connectivity.network.*`). The status clears without waiting for the next utterance.

### Advanced: transport and decoding

| Key | Default | Unit, range | What it does |
| --- | --- | --- | --- |
| `voice.realtime_route.mode` | `direct` | `direct`, `platform` | `direct` connects to the route's own URL. `platform` asks the route's `base_url` for a session first; Reachy Mini uses it. Needs restart. |
| `voice.realtime_route.audio_gate` | `addressed` | `addressed` | Which audio may leave. Only an utterance the wake gate and the voice-at-face check accepted is sent. It is sent once it ended, never live. |
| `voice.realtime_route.transcription_language` | `""` | language code | Language the server transcribes in. It is sent with audio input only. Empty uses `conversation.language`. |
| `voice.realtime_route.transcription_model` | `""` | model name | Transcription model sent to the server. It is sent with audio input only. Empty uses the server's default. |
| `voice.realtime_route.conversation_span_s` | `30.0` | s, 1 to 600 | Used with `voice.realtime_route.skip_tool`. A conversation counts as in progress while the robot's last spoken answer ended at most this long ago. It also counts while a confirmation waits, or after the wake word or the name. Outside a conversation, each turn tells the server to answer only words clearly meant for the robot. |
| `voice.realtime_route.mood_instructions` | `true` | on/off | Adds the mood's voice instructions to each request. |
| `voice.realtime_route.server_silence_ms` | `600.0` | ms, 64 to 3000 | Silence that ends a turn for the server's VAD. It must be at least `voice.vad.end_silence_ms` (default 500 ms). |
| `voice.realtime_route.close_silence_ms` | `900.0` | ms, 100 to 5000 | Silence added after the utterance, so the server ends the turn at once. It must be more than `voice.realtime_route.server_silence_ms`. |
| `voice.realtime_route.local_decode_timeout_s` | `10.0` | s, 1 to 60 | Longest wait for a local Vosk decode that the robot still needs. Past it, the utterance is dropped and counted. |
| `voice.realtime_route.wake_decode_max_s` | `3.0` | s, 0.5 to 30 | For an utterance the voice-at-face check rejected, Vosk decodes only this much speech. It looks for the wake word. |
| `voice.realtime_route.prebuffer_ms` | `200.0` | ms, 0 to 2000 | Reply audio held before playback starts, against network delays. |
| `voice.realtime_route.connect_timeout_s` | `5.0` | s, 1 to 30 | Longest time to open a session: allocation, socket handshake and the server's first reply together. |
| `voice.realtime_route.send_timeout_s` | `5.0` | s, 1 to 30 | A send stuck this long closes the socket. It is also the keepalive interval and timeout. |

### Budgets

Two daily caps apply to the cloud voice. They count different things, and either one can run out
first:

- **Upload budget** (`voice.realtime_route.daily_upload_budget_s`, default 1800 s, so 30 minutes
  of audio per day). It counts the microphone audio sent, the added closing silence included. When
  it runs out, the robot keeps answering. It sends the local Vosk transcript as text instead of the
  audio. The count is kept in `realtime_upload.json` in the data folder, so a restart does not reset
  it.
- **Token budget** (`llm_routes[].daily_token_budget` of the route, default 200000 tokens per day).
  It counts the tokens the server reports for each answer, or an estimate when the server reports
  none. The installer's `pollen` entry does not set it, so the default applies. When it runs out,
  the route is blocked with reason `budget` until the next day. With the profile there is no local
  voice, so the robot plays the fallback cue instead of answering.

The debug report's `roboemot.realtime_route` section shows the upload used
(`upload_s_today`) next to `upload_budget_s`.

### The key: `roboemot secrets set`

SPEC 8 (Secrets), SPEC 12.

A route's `llm_routes[].api_key` should be `${secret:<name>}`. The value lives in the robot's secret
store, `secrets.json` in the data folder (mode 0600). A route whose secret cannot be read is
unavailable. It is never called with an empty key.

There are three ways to store a key:

- On the robot, run `/venvs/apps_venv/bin/roboemot secrets set hf_token`. It reads the value from
  the prompt or stdin, never from an argument. `roboemot secrets list` shows names only.
  `roboemot secrets remove` deletes one.
- Run the installer with `--secret hf_token`. It asks without echo, and sends the key over SSH.
- Use the web UI **Routes** tab, **API keys** card, with the admin confirmation. The web UI has no
  TLS, so the key travels in clear on the local network. Prefer SSH.

### Who counts as addressed

SPEC 8, Stages (stages 2 and 3).

These settings decide which utterances the robot answers. With the cloud voice and audio input,
they also decide which audio leaves the robot. They apply to the local voice too.

| Key | Default | Unit, range | What it does |
| --- | --- | --- | --- |
| `voice.wake_word_policy` | `unless_enrolled_in_view` | `always`, `never`, `unless_enrolled_in_view` | When the wake word is needed. With `unless_enrolled_in_view`, anyone speaking while an enrolled person is in view is addressed without it. A stranger alone must say it. |
| `voice.fastpath.addressed_window_s` | `15.0` | s, 1 to 600 | Follow-up window. After an addressed utterance that needed the wake word, the next one that starts within this time is addressed too. It needs no wake word. |
| `voice.fastpath.window_max_s` | `120.0` | s, 1 to 3600 | The follow-up window never reaches further than this from the wake word that opened it. |
| `voice.direction.enabled` | `true` | on/off | Voice-at-face check. An utterance addressed without the wake word must come from the direction of a visible face. When it is off, music and TV can be answered. |
| `voice.direction.tolerance_deg` | `25.0` | deg, 1 to 90 | How far a sound direction reading may be from a face and still match it. |
| `voice.direction.min_match_share` | `0.5` | 0 to 1 | Share of the readings that must match a face. |
| `voice.direction.degraded_rule` | `auto` | `auto`, `engaged`, `reject` | What to do with too few direction readings. `reject` needs the wake word. `engaged` passes when attention is on a visible person. `auto` picks one of the two by the reason. |
| `voice.facing.enabled` | `false` | on/off | Facing gate. On top of the voice-at-face check, the face the voice came from must look at the robot. It stops a person watching a screen near the robot from being answered. It needs `voice.direction.enabled`. |
| `voice.facing.max_yaw_deg` | `30.0` | deg, 1 to 90 | A face looks at the robot when its head turn is at most this. |
| `voice.facing.min_share` | `0.5` | 0 to 1 | Share of the samples in which the face must look at the robot. |
| `voice.facing.cpu_max_percent` | `90.0` | %, 1 to 100 | Above this CPU load for `voice.facing.cpu_hold_s`, the head turn is not trusted. The voice-at-face check then decides alone. |
| `voice.facing.cpu_hold_s` | `10.0` | s, 0 to 600 | How long the CPU must stay above `voice.facing.cpu_max_percent`. |

An utterance with the wake word always passes both checks.

### Feedback without a voice: non-verbal cues

SPEC 8, Non-verbal cues.

With `voice.backend: none` the robot has no local voice. When it cannot speak, it plays a short
gesture and sound instead:

- `listening`: the audio left, and the robot waits for the answer.
- `not_heard`: no transcript came back.
- `fallback`: the route cannot answer, and the network is up.
- `offline`: the network is down.

| Key | Default | Unit, range | What it does |
| --- | --- | --- | --- |
| `voice.cues.enabled` | `true` | on/off | Shows the non-verbal cues. |
| `voice.cues.listening_onset_s` | `0.3` | s, 0 to 3 | Delay before the listening gesture. An answer that starts sooner shows none. |
| `voice.cues.listening_hold_max_s` | `10.0` | s, 1 to 60 | Longest the listening gesture is held without an answer. |
| `voice.cues.listening_sound` | `true` | on/off | Plays a short, quiet blip when the listening gesture shows. |
| `voice.cues.volume_db` | `-6.0` | dB, -40 to 0 | Level of the cue sounds. The circadian volume offset adds to it. |

### Privacy consequences

- `voice.realtime_route.input: text` sends the local transcript as text.
  `voice.realtime_route.input: audio` sends the speaker's voice, a child's included, and the server
  transcribes it. Audio needs `privacy.realtime_audio`.
- With `privacy.pseudonyms_to_cloud` on, a cloud realtime route is blocked (`pseudonyms`), because
  the server would hear real names. Turn one of the two off, or set the route's
  `llm_routes[].locality` correctly.
- With `voice.direction.enabled` off, or on a robot with no sound direction, a TV can be addressed
  and its audio sent.

## 2. Local voice

SPEC 8 (Stages; Voice, expressivity and mood).

### What it does

Everything stays on the robot. Silero finds speech, Vosk transcribes it, an LLM route writes the
answer and Piper speaks it. Piper runs in its own process from a separate virtualenv, because it is
GPL-3.0. The installer turns the voice on (`voice.enabled: true`), creates the Piper virtualenv and
fetches the voice models (`roboemot models fetch --voice`). `--no-voice` leaves the voice off.
`--no-piper` skips Piper.

| Key | Default | Unit, range | What it does |
| --- | --- | --- | --- |
| `voice.enabled` | `false` | on/off | Runs the voice pipeline. The installer turns it on. Needs restart. |
| `voice.backend` | `piper` | `piper`, `openai_speech`, `kokoro`, `none` | Speech synthesis. `none` means no local speech: only the realtime route's voice speaks. Needs restart. |
| `voice.fallback_backend` | `piper` | `piper`, `kokoro`, `none` | Offline fallback voice. Needs restart. |
| `voice.voices` | `{it: it_IT-paola-medium, en: en_US-amy-medium}` | per language | Piper voice per language. Needs restart. |
| `voice.asr.models` | `{it: vosk-small-it, en: vosk-small-en}` | per language | Vosk model per language. Needs restart. |
| `voice.models_dir` | `""` | path | Folder of the voice models. Empty uses `voice-models/` in the data folder. Needs restart. |
| `voice.piper.python` | `""` | path | Interpreter of the Piper virtualenv. Empty uses `piper-venv/bin/python` in the data folder. A change needs the admin confirmation. Needs restart. |
| `conversation.language` | `it` | language code | Default conversation language. The web UI offers only languages with installed voice models. |
| `voice.wake_word` | `null` | name | The robot's name, for the prompt and wake detection. When unset, the robot uses the name the platform reports (on Reachy Mini, the name the owner gave it). A value here always wins. |
| `voice.default_name` | `""` | name | Set by the manifest (Reachy Mini: `Reachy`). It is never spoken. It only decides when the manifest's name aliases apply. |
| `voice.barge_in.enabled` | `true` | on/off | Speech while the robot talks stops it. When it is off, only the stop keyword is heard while the robot speaks. |

### Language

The language of an utterance is the speaker's registry preference (`preferences.language`) when a
Vosk model exists for it. Otherwise it is `conversation.language`. Otherwise it is the first
language in `voice.asr.models`.

### Wake word and the platform name

The small Vosk models cannot spell a made-up name such as "Reachy". So the robot wakes on phrases
of common words in the final text, per language (`voice.keywords.<language>.wake_phrases`):

- Italian: `ehi robot`, `ciao robot`
- English: `hey robot`, `hello robot`

When the robot has a name (`voice.wake_word`, or the platform's name), each language adds its
`voice.keywords.<language>.name_wake_templates`: `ciao {name}` and `ehi {name}` in Italian,
`hey {name}` and `hello {name}` in English. This happens only when every word of the name is in
that recognizer's vocabulary. The Reachy Mini manifest adds the spellings Vosk produces for
"Reachy" as name aliases: `reati` in Italian, `riyadh she` in English. The bare name wakes the
robot only in a short final text, at most `voice.fastpath.alias_wake_max_words` words (default 4).

Changing the name from the web UI needs the admin confirmation. The robot cleans the name before
it goes into the prompt.

**Privacy.** Nothing leaves the robot, unless an LLM route is a cloud route (next section).

## 3. LLM routes

SPEC 8 (LLM router, OpenAI-compatible routes, Secrets).

### What it does

`llm_routes` is an ordered list of OpenAI-compatible endpoints. The router tries them in order. It
moves to the next one after a timeout or an error. A `kind: realtime` route is only for the cloud
voice; chat requests never use it. When no route answers, the robot says
`conversation.fallback_lines`. With no local voice, it plays the fallback cue instead.

By default the list is empty. Add routes in `app.yaml` or on the web UI **Routes** tab.

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
| `llm_routes[].name` | `""` | letters, digits, `_`, `-`; at most 64 | Route name. Required. |
| `llm_routes[].base_url` | `""` | http(s) URL | OpenAI-compatible base URL. Required. |
| `llm_routes[].api_key` | `none` | text or `${secret:name}` | The key. Use `${secret:name}`. A literal key sits in `app.yaml` in clear. The default is the literal text `none`. |
| `llm_routes[].model` | `""` | model id | Model id. Required. |
| `llm_routes[].kind` | `chat` | `chat`, `realtime` | `chat` is a Chat Completions route. `realtime` is the realtime speech route. A change on a route with a key needs the admin confirmation. |
| `llm_routes[].locality` | `auto` | `auto`, `local`, `cloud` | Whether the route may see real names. `auto` treats loopback, private addresses, `*.local` and names without a dot as local. Everything else is cloud. A local proxy that forwards to a cloud provider must be set to `cloud`. |
| `llm_routes[].first_token_timeout_s` | `3.0` | s, 0.1 to 30 | Moves to the next route when the first token takes longer. |
| `llm_routes[].stall_timeout_s` | `5.0` | s, 0.1 to 60 | Longest gap between chunks once the answer has started. |
| `llm_routes[].max_tokens` | `300` | tokens, 16 to 8192 | Reply token cap. |
| `llm_routes[].daily_token_budget` | `200000` | tokens per day, at least 0 | Daily token cap. Past it, the route is skipped until the next day. |
| `llm_routes[].context_budget_tokens` | `8192` | tokens, 256 to 2000000 | Prompt budget per request, at 4 characters per token. The oldest turns are dropped to fit. |
| `llm_routes[].supports_images` | `false` | on/off | The route accepts images (`image_url` parts). |
| `conversation.turn_reserve_tokens` | `256` | tokens, 16 to 8192 | Room every route's context budget must leave beyond the system prompt, for the newest turn. It is checked when the configuration loads. |

### Secrets and locality guardrails

- A route with a `${secret:name}` key is pinned to its approved `llm_routes[].base_url` and
  `llm_routes[].api_key`. Changing either, or `llm_routes[].kind`, needs the admin confirmation. A
  new endpoint host needs one more confirmation in the web UI, because the key will be sent there.
- Making a route local needs the admin confirmation, because a local route sees real names. This
  covers `llm_routes[].locality`, `llm_routes[].base_url` and a new route.

### How to check a route

- On the robot, run `roboemot llm check <route>` for a chat route and `roboemot realtime check
  <route>` for a realtime route.
- Or use the web UI **Routes** tab, **Check**. Prefer it: the CLI does not see values saved in the
  web UI.

**Privacy.** A cloud route receives the conversation text, the speaker summary and the scene
summary. With `privacy.pseudonyms_to_cloud` on, it receives aliases instead of registry names.

## 4. Safety

SPEC 11A (Guardrails), SPEC 12 (Physical safety).

### What it does

The safety settings cap how the robot moves and what happens when commands stop. Two rules apply:

- **At runtime, safety only gets more conservative.** The web UI and the API may tighten a
  `safety.*` value. Loosening one needs the admin confirmation and is written to `app.yaml`, not to
  the web UI file. The same rule covers every field with a safe direction outside `safety.*`, for
  example `privacy.pseudonyms_to_cloud` and `voice.echo_tail_ms`.
- **Manifest limits are ceilings.** No layer can exceed the joint limits and dynamics in the robot
  manifest (`pose_groups`, `dynamics`, `constraints`). A manifest without limits is rejected.

| Key | Default | Unit, range | Safe direction | What it does |
| --- | --- | --- | --- | --- |
| `safety.watchdog_ms` | `200.0` | ms, 20 to 2000 | lower | The robot returns home slowly when no command arrives for this long. |
| `safety.watchdog_return_factor` | `0.3` | 0.01 to 1 | lower | Speed factor of the watchdog's return home. |
| `safety.soft_stop` | `hold_position` | `hold_position` | none | Soft stop behaviour. |
| `safety.hard_stop` | `disable_motors` | `disable_motors` | none | Hard stop behaviour. |
| `safety.constraint_margin_deg` | `3.0` | deg, 0 to 20 | higher | Coupled constraints (head and body yaw) are planned this far inside their limit. |
| `safety.allow_unmeasured_dynamics` | `false` | on/off | off | Runs with manifest dynamics still `null`. It is for development only. The installer's `--dev` turns it on. |
| `safety.unmeasured_dynamics_fallback_vel` | `0.5` | rad/s, 0.01 to 10 | lower | Velocity cap for unmeasured dynamics (development only). |
| `safety.unmeasured_dynamics_fallback_acc` | `2.0` | rad/s^2, 0.01 to 100 | lower | Acceleration cap for unmeasured dynamics (development only). |
| `safety.unmeasured_dynamics_fallback_lin_vel` | `0.02` | m/s, 0.001 to 1 | lower | Linear velocity cap for unmeasured translations (development only). |
| `safety.unmeasured_dynamics_fallback_lin_acc` | `0.1` | m/s^2, 0.001 to 10 | lower | Linear acceleration cap for unmeasured translations (development only). |
| `safety.extra_prompt_rules` | `[]` | list of text | none | Rules added to the system prompt's safety block. A change needs the admin confirmation. |

The Reachy Mini manifest's dynamics are `null`: they are to be measured in milestone M0
(SPEC 4A, Reachy Mini manifest). While they are `null`, the robot runs only with
`safety.allow_unmeasured_dynamics: true` (the installer's `--dev`), and the limiter uses the
fallback caps above.

The stop keyword, the soft stop and the hard stop work in every state, asleep or awake.

## 5. Circadian rhythm, quiet hours and presence

SPEC 7A.

### What it does

The robot simulates a day. It is `awake`, `drowsy` or `resting`. Sleep pressure builds while it is
awake. An internal clock, set by the camera's light level, decides when it is most alert. Resting
is a RoboEmot behaviour with the motors on: smaller motion, a quieter voice, slower vision. The
robot still answers in every state, only more quietly.

| Key | Default | Unit, range | What it does |
| --- | --- | --- | --- |
| `circadian.enabled` | `true` | on/off | Runs the circadian rhythm. Turning it off also turns off quiet hours. |
| `circadian.quiet_hours` | `["22:30", "07:00"]` | local time, start and end | Window in which resting is forced, unless people present keep the robot awake. |
| `circadian.presence_keeps_awake` | `true` | on/off | People present and active prevent resting, inside quiet hours too. |
| `circadian.presence_window_s` | `60.0` | s, 1 to 3600 | People seen, recognised or attended to within this time count as present. |
| `circadian.wake_hold_s` | `300.0` | s, 10 to 3600 | How long the wake word keeps a resting robot awake. |
| `circadian.nap_after_s` | `1800.0` | s, 300 to 86400, or `null` | Day nap: with no face in view and no speech addressed to the robot for this long, outside quiet hours, the robot rests (reason `nap`) until a face or addressed speech wakes it for `wake_hold_s`. A television does not count. Only time outside quiet hours adds up. `null` turns naps off. |
| `circadian.peak_hour` | `15.0` | h, 0 to 24 | Hour of the day at which alertness peaks. |
| `circadian.drowsy_threshold` | `0.55` | 0 to 1 | Sleepiness above which the robot is drowsy. |
| `circadian.resting_threshold` | `0.75` | 0 to 1 | Sleepiness above which the robot rests. |
| `circadian.light_source` | `camera` | `camera`, `clock_only` | What sets the internal clock. `clock_only` uses the clock, and `circadian.location` if set. |
| `circadian.location` | `null` | `{lat, lon}` | Optional location for clock-only mode. It is private: the web UI masks it. |

Quiet hours also stop the robot's look-around scans.

| Key | Default | Unit, range | What it does |
| --- | --- | --- | --- |
| `world.scan.alone_after_s` | `120.0` | s, 10 to 3600 | With no face in view and no sound for this long, the robot is alone and holds still: no look-around scans or glances. Breathing and reflexes go on. A face or a sound ends it at once. Never shorter than `circadian.presence_periods` face-detection periods, so a robot whose vision runs slowly is not fooled by a still face. With reflexes off, a loud sound does not count, only speech the listener hears. |

The Live tab says why the robot rests (a nap, quiet hours, sleepiness, an app hold) and when it is
alone.

**Not the same as the robot's sleep.** The owner can put the robot to sleep from the platform
(motors off). With the default `runtime.pause_when_asleep: true`, RoboEmot then pauses perception,
the listener and the behaviour tick, and nothing is sent to the cloud while the robot sleeps.

**Privacy.** With `circadian.light_source: camera`, the robot reads only a light level from the
frames. No frame is stored.

## 6. Privacy switches

SPEC 12 (Data).

Every privacy option is off by default. Turning one on is an opt-in and needs the admin
confirmation. The web UI masks `privacy.*` values in its list. It shows them only on request, with
the PIN.

| Key | Default | Unit, range | What it does |
| --- | --- | --- | --- |
| `privacy.realtime_audio` | `false` | on/off | Opt-in. Lets addressed microphone audio leave the robot for the realtime route. The installer's `--cloud-voice` turns it on. |
| `privacy.realtime_platform_id` | `false` | on/off | Opt-in. Lets the platform's session provider send the robot's stable platform identifier with each session request (platform mode only). The server can then link all sessions of this robot. |
| `privacy.pseudonyms_to_cloud` | `false` | on/off | Sends pseudonyms instead of registry names to cloud routes, and puts the names back in the answer. A remote speech voice still hears the names it speaks. It blocks a cloud realtime route. Its safe direction is on: turning it off again needs the admin confirmation. |
| `privacy.multimodal_frames` | `false` | on/off | Opt-in. Allows single camera frames to be sent to LLM routes for image questions. |
| `privacy.keep_enrollment_images` | `false` | on/off | Opt-in. Keeps the face crops taken at enrollment. |
| `privacy.debug_recording` | `false` | on/off | Opt-in. Allows time-limited debug recording. |
| `webui.live.enabled` | `false` | on/off | Opt-in. Serves the Live tab: mood, who is in view by name, conversation phase. |
| `webui.live.transcript` | `false` | on/off | Opt-in. Keeps the recent conversation, as text, in memory for the Live tab. It is never saved, logged or reported. |
| `webui.live.video` | `false` | on/off | Opt-in. Serves a live camera preview to the Live tab. It needs `webui.live.enabled`. Every open page shows a banner while someone watches. |

These rules hold whatever the switches say:

- Frames, audio and face embeddings are never written to logs.
- API keys are never logged, shown or returned. A route whose `${secret:name}` cannot be read is
  not called.
- The debug report holds codes and numbers, never text, audio or names.

## Troubleshooting

SPEC 8 (Realtime speech route), SPEC 12A (Diagnostics).

The Live tab's **What's happening** panel (**Cosa succede**) shows a status line, the conditions
that hold now (notes), and the last decisions with their reasons. It needs `webui.live.enabled` and
`webui.live.activity` (on by default). The labels follow the UI language. The tables quote them in
English and Italian where both are given; decisions are quoted in English.

The status line shows one status: the first that holds, in this order (`_status` in
`packages/roboemot-core/src/roboemot/runtime/live.py`):

| Order | Status line (English / Italian) |
| --- | --- |
| 1 | "Asleep" / "Dorme" |
| 2 | "Answering" / "Sta rispondendo" |
| 3 | "Cut off by a barge-in" / "Interrotto da chi parlava" |
| 4 | "Waiting for the answer" / "Aspetta la risposta" |
| 5 | "Sending what you said to the cloud" / "Sta mandando al cloud quello che hai detto" |
| 6 | "Listening" / "In ascolto" |
| 7 | "Offline: the network is down" / "Offline: la rete non c'è" |
| 8 | "Cloud voice down" / "Voce cloud non disponibile" |
| 9 | "Idle" / "A riposo" |

So "Offline" hides "Cloud voice down", and so does a conversation in progress. The reason code is
not in the status line. It is in a separate note: "The cloud voice is down ({reason})." /
"La voce cloud non è disponibile ({reason})." The network note is "The network is down." /
"La rete non c'è."

### Cloud voice reason codes

The note "The cloud voice is down ({reason})." / "La voce cloud non è disponibile ({reason})." shows
one of these codes:

| Reason | What it means | What to check |
| --- | --- | --- |
| `connection_lost` | The socket to the server dropped. | Usually nothing. The route reconnects in the background, with backoff. If it keeps coming back, check the robot's internet access. |
| `connect_failed`, `connect_timeout` | The robot could not open a session within `voice.realtime_route.connect_timeout_s`. | Internet access and the server's status. The route keeps retrying. |
| `closed_by_server` | The server closed the session. | The route reconnects in the background. If it repeats, run the route's **Check** on the Routes tab. A `provider_http_401` or `provider_http_403` there means the token is wrong or expired. Store a new one with `roboemot secrets set`. |
| `no_opt_in` | `voice.realtime_route.input: audio` without `privacy.realtime_audio`. | Turn the opt-in on (admin confirmation), or set `voice.realtime_route.input: text`. |
| `pseudonyms` | `privacy.pseudonyms_to_cloud` is on and the route is a cloud route. | Turn one of the two off. |
| `budget` | The route's `llm_routes[].daily_token_budget` is used up for the day. | Wait for the next day, or raise the budget. |
| `unavailable` | The router has no usable route of that name. The causes: the route's `${secret:name}` cannot be read (the router marks the route unavailable), the route's link is offline by the reachability checks, or the router is missing. | `roboemot secrets list` for the key's name; the network; the debug report. |
| `no_route` | `voice.realtime_route.route` names no `llm_routes` entry. | The route name, and a web UI value saved for it. |
| `no_extra`, `provider` | The `realtime` extra (websockets) or the robot's session provider is missing. | Run the installer again. |

### Symptoms

| Symptom | Live tab shows | What to check |
| --- | --- | --- |
| The robot does not answer, and the network is down. | "Offline: the network is down" / "Offline: la rete non c'è" | The robot's Wi-Fi. When spoken to, the robot plays the offline cue or says the offline phrase. |
| It does not answer without the wake word. | "Not answered: the voice did not come from a face", "Not answered: not facing the robot (head turned … deg, limit … deg)", "Not answered: no facing data (no head turn measured)" or "Not answered: the cloud model skipped it (conversation: …)" | The profile sets `voice.wake_word_policy: never`, so the voice-at-face check and the facing gate decide. Face the robot and look at it, or say the wake phrase. Outside a conversation (`voice.realtime_route.conversation_span_s`), the server answers only words clearly meant for the robot. With policy `always` or `unless_enrolled_in_view`, the decision is "Not answered: no wake word". |
| It answers the TV. | "Answered (…)" | Check that `voice.direction.enabled` and `voice.facing.enabled` are on, and turn on `voice.realtime_route.skip_tool`. Known limits (SPEC 8, Stages): a face on the screen that looks into the camera passes, and a sound from behind the robot can be mirrored onto a face in front. Its audio still reaches the server. |
| No sound at all when the cloud voice fails. | "Fallback sound: the cloud voice could not answer and there is no local voice" | With `voice.backend: none`, the cues are the only feedback. If `voice.cues.enabled` is off, the robot stays silent. Turn it on. Cues are also muted while the robot sleeps. |
| The cloud does not return what you said. | "No transcript from the cloud within … s" | The server's delay. Raise `voice.realtime_route.transcript_timeout_s` (at most 15 s). |
| The robot is still. | "Asleep" / "Dorme" | The platform holds the motors off. Wake the robot from the platform's app. |

When the web UI and `app.yaml` disagree, the web UI wins. Look at the field's source layer on the
Settings tab, and reset it there.
