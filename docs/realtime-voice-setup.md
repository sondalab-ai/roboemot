# Realtime voice through Hugging Face (Reachy Mini Wireless)

This guide sets up the realtime speech route on the robot. With it, Pollen's server does the
transcription, the answer and the voice (SPEC 8, Realtime speech route). The robot sends the
audio of addressed utterances to the server. The route is off by default (SPEC 12).

## With the installer: one flag

From the repo checkout, one flag replaces steps 2 to 4 below:

```bash
python scripts/install_on_robot.py --host 192.168.88.198 --cloud-voice --dry-run   # the plan, writes nothing
python scripts/install_on_robot.py --host 192.168.88.198 --cloud-voice
```

The installer writes the `pollen` route and its settings into `app.yaml` (route on with audio
input, `privacy.realtime_audio`, the skip tool, no local voice, wake policy `never`, the
voice-at-face and facing checks, a transcript timeout of 10 s), asks for `hf_token` before it
changes anything when the robot has none (no token: it stops with nothing changed;
`--secret hf_token` replaces an expired one), and removes Piper and its voice files (SPEC 4A,
Installer). Settings saved in the web UI win over `app.yaml`: the installer stops when one would
widen what leaves the robot (for example the facing check off) and names it; reset it in the
web UI and run again. Read its privacy text before you answer: the audio of every utterance
spoken while a person faces the robot and looks at it leaves the robot for Pollen's server,
phone calls included. `-y` does not turn it on. A reinstall with `--no-cloud-voice` removes
exactly these keys; local voice then needs Piper again (`--piper`). Then start RoboEmot (step 5)
and check the route (step 6). The manual steps below stay for a robot set up by hand.

The server is reached in two hops:

1. The robot asks Pollen's allocator for a session:
   `https://pollen-robotics-reachy-mini-realtime-url.hf.space/session`.
2. The allocator answers with the WebSocket address of the session, on a connect host under
   `endpoints.huggingface.cloud` (the host changes: the robot saw two different ones in one day).

The robot sends your Hugging Face token only to the allocator and to hosts under
`.endpoints.huggingface.cloud`, over wss on port 443 (`KEY_HOSTS` in
`packages/roboemot-reachymini/src/roboemot_reachymini/realtime.py`).

## Before you start

- RoboEmot is installed on the robot and runs (see [On the robot](../README.md#on-the-robot)).
  The installer adds the `realtime` extra (`websockets`) by default.
- You know the operator PIN of the web UI (`http://<robot>:8042`).
- Close the Reachy mobile app: the daemon runs one app at a time.

## 1. Create a Hugging Face token

1. Sign in at https://huggingface.co and open https://huggingface.co/settings/tokens.
2. Select **Create new token**, then **Fine-grained**.
3. Leave every write permission off. The route needs a read-only token.
4. Copy the token. Hugging Face shows it only once.

## 2. Store the token on the robot

Use one of these ways. The robot stores the token in its secret store (`secrets.json`, mode 0600)
and never shows it again.

- **Web UI:** **Routes** tab (**Modelli** in Italian), **API keys** card (**Chiavi API**). Name
  `hf_token`, paste the token, confirm with the PIN.
- **Installer:** add `--secret hf_token` to `scripts/install_on_robot.py`. It asks for the value
  without echo.
- **Shell on the robot:** `/venvs/apps_venv/bin/roboemot secrets set hf_token`. It reads the value
  from the prompt.

The web UI has no TLS yet: the PIN and a typed key travel in clear on the LAN. The installer and
the shell use SSH.

## 3. Add the `pollen` route

**Routes** tab, **Add a route** card (**Aggiungi un modello**):

| Field | Value |
| --- | --- |
| Name | `pollen` |
| Kind | Realtime speech (`realtime`) |
| Base URL | `https://pollen-robotics-reachy-mini-realtime-url.hf.space/session` |
| Key | `hf_token` |
| Model | `s2s` |

The page asks you to confirm the hosts that receive the key. In a file, the same route is the
`pollen` example in `configs/companion.yaml`.

## 4. Turn on the route

**Settings** tab (**Impostazioni**), **Search** field (**Cerca**):

1. Search `realtime_route` and set:
   - `voice.realtime_route.enabled`: true
   - `voice.realtime_route.route`: `pollen`
   - `voice.realtime_route.mode`: `platform`
   - `voice.realtime_route.input`: `audio`
2. Search `realtime_audio`. The field is private: select **Show with PIN** (**Mostra con il PIN**),
   then turn on `privacy.realtime_audio`.

Optional:
- `voice.realtime_route.voice`: the server's voice. Empty uses the server's default, `Aiden`
  (male). Pollen's server (Qwen3-TTS) lists `Aiden`, `Ryan`, `Dylan`, `Eric`, `Ono_Anna`,
  `Serena`, `Sohee`, `Uncle_Fu` and `Vivian` (from Pollen's conversation app); `Serena` and
  `Vivian` are female voices. Their Italian has not been tried on the robot.
- `voice.realtime_route.transcript_timeout_s`: how long the robot waits for the server's
  transcript before it decodes the utterance locally (default 4 s; the `--cloud-voice` profile
  writes 10 s, to drop to the measured p95 plus about 1 s once the listener lag fix, CV-0, is
  measured on the robot). The server took about 3 s on
  2026-10-29; a larger value avoids falling back to the local pipeline on a slow transcript.

`privacy.realtime_audio` sends the audio of every addressed utterance to the server, whoever
speaks, a child included. Leave `privacy.realtime_platform_id` off unless the allocator refuses
anonymous sessions: with it on, the allocator can link every session of this robot.

## 5. Restart RoboEmot

`enabled`, `route` and `mode` take effect only at a restart. From the repo checkout:

```bash
python scripts/run_on_robot.py --restart   # asks first; the head moves
```

## 6. Check the route

**Routes** tab, `pollen`, **Check** (**Controlla**). A working route shows:

| Field | Expected |
| --- | --- |
| Mode | `platform` |
| Allocator host | `pollen-robotics-reachy-mini-realtime-url.hf.space` |
| Connect host | a host under `endpoints.huggingface.cloud` |
| Key to the connect host | `sent` |
| session.updated seen | `true` |
| First audio (ms) | a figure |
| Item delete | `accepted` |

From a shell, the same check is:

```bash
python scripts/robot_ssh.py -- /venvs/apps_venv/bin/roboemot realtime check pollen \
  --app ~/.local/share/roboemot/app.yaml --robot reachy-mini-wireless --json
```

The CLI does not see the settings saved from the web UI. Prefer the web UI check.

## Troubleshooting

| Result | Cause and fix |
| --- | --- |
| Mode `direct` | The app did not restart after the `mode` edit, or runs a version before PR #131. Update the robot and restart. |
| `url_key_host` | The allocator handed out a host outside `.endpoints.huggingface.cloud`. The robot does not send the token there, and the local pipeline answers. Check the connect host, then decide whether to add it to `KEY_HOSTS`. |
| `provider_http_401` or `provider_http_403` | The allocator refused the request. The token may be wrong or expired: store a new one (step 2). |
| `provider_unreachable` | The robot cannot reach the allocator. Check the robot's internet access. |
| "The realtime extra (websockets) is not installed." | Run the installer again. |

While the route is down, the local pipeline answers (Vosk and Piper, when installed).

## Gemini Live (Google): a second provider

**Not ready yet.** This section documents the configuration of the `gemini_live` protocol (SPEC 8,
Gemini Live wire). The robot speaks it from slice S3 of the Gemini Live epic on; the route
**Check** (web UI and `roboemot realtime check`) learns it in slice S4, so until then it may fail
on a Gemini route that works.

Gemini Live is a separate route beside `pollen`. Pollen's route stays as it is; you switch
between the two with `voice.realtime_route.route` and a restart.

### Privacy and terms

Read this before you add the route.

- On the free tier, Google may use the content you send (audio, transcripts, instructions) to
  improve its products, and people may review it.
- The Gemini API terms require users aged 18 or over, and forbid apps that are likely to be used by
  people under 18. A robot at home can hear children.
- The owner accepted both risks on 2026-11-11 (SPEC 12). `privacy.realtime_audio` still decides
  whether any audio leaves the robot.
- Every Gemini connect, reconnects included, carries the current profile summary, scene summary
  and the replay of the conversation so far in its setup (owner decision 2026-11-12).

### 1. Create a key and store it

Create an API key in Google AI Studio (<https://aistudio.google.com/apikey>). Store it on the
robot as `gemini_key`:

```bash
/venvs/apps_venv/bin/roboemot secrets set gemini_key   # reads the value from the prompt or stdin
```

Never pass the key as an argument: it would stay in the shell history. The web UI **API keys**
card works too. The robot sends the key only in the `x-goog-api-key` header, never in the URL.

### 2. Add the `gemini` route

**Routes** tab, **Add a route** card:

| Field | Value |
| --- | --- |
| Name | `gemini` |
| Kind | Realtime speech (`realtime`) |
| Realtime protocol | Gemini Live (`gemini_live`) |
| Base URL | `wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent` |
| Key | `gemini_key` |
| Model | `gemini-3.8-live` |
| Voice | one of `Kore`, `Aoede`, `Leda`, `Puck`, `Charon` |

The same route in `app.yaml`:

```yaml
llm_routes:
  - name: gemini
    kind: realtime
    protocol: gemini_live
    base_url: wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent
    model: gemini-3.8-live
    api_key: ${secret:gemini_key}
    locality: cloud
    voice: Kore
    connect_timeout_s: 15
```

- `voice`: the route's own voice. It wins over `voice.realtime_route.voice`, which Pollen's
  route keeps using. The five voices above were tried in Italian on 2026-11-11 and all sounded
  fine. Google's documentation gives no gender for them.
- `connect_timeout_s`: the route's own connect budget. Some first connects took 10.6 s until
  Gemini confirmed the session, more than the shared 5 s. The web UI form has no field for it: set
  it in `app.yaml`, or search `llm_routes` in the **Settings** tab.
- The base URL has no query: the web UI refuses one.

### 3. Switch to Gemini

A `gemini_live` route needs `voice.realtime_route.mode: direct` and `turns: robot` (the default);
the configuration refuses anything else, one edit at a time. In the **Settings** tab, edit in this
order:

1. `voice.realtime_route.mode`: `direct` (first: Pollen's route accepts it, a Gemini route
   refuses `platform`).
2. `voice.realtime_route.route`: `gemini`.
3. `voice.realtime_route.enabled`: true, if it is not already on.
4. Restart RoboEmot (section 5, Restart RoboEmot).

With `voice.realtime_route.input: audio` the utterances' audio goes to Google, and that needs the
opt-in `privacy.realtime_audio` (section 4). With `input: text` only the local transcript goes.

To go back, edit `route`: `pollen` first, then `mode`: `platform`, and restart.

### What to expect

- Gemini closes a session that has been quiet for about two minutes. The robot does not
  reconnect at once: the next turn connects, so the first answer after a quiet spell waits for
  the connect, usually 0.5 to 0.7 s, rarely up to about 10 s (owner decision 2026-11-12). The
  Live tab shows it as an idle close.
- If Google refuses the connects again and again (the key, the free-tier quota or access), the
  route reports `server_refused` after `voice.realtime_route.refusals_to_report` refusals in a
  row (3) and keeps retrying with a growing wait; the local pipeline answers meanwhile.

Or edit `voice.realtime_route` in `app.yaml` (all fields at once) and restart.
