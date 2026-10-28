# Realtime voice through Hugging Face (Reachy Mini Wireless)

This guide sets up the realtime speech route on the robot. With it, Pollen's server does the
transcription, the answer and the voice (SPEC 8, Realtime speech route). The robot sends the
audio of addressed utterances to the server. The route is off by default (SPEC 12).

The server is reached in two hops:

1. The robot asks Pollen's allocator for a session:
   `https://pollen-robotics-reachy-mini-realtime-url.hf.space/session`.
2. The allocator answers with the WebSocket address of the session, on the connect host
   `llgpsjdk3p4ayeud.endpoints.huggingface.cloud`.

The robot sends your Hugging Face token to these two hosts only. The connect host is declared
exactly in `KEY_HOSTS` (`packages/roboemot-reachymini/src/roboemot_reachymini/realtime.py`).

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
| Connect host | `llgpsjdk3p4ayeud.endpoints.huggingface.cloud` |
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
| `url_key_host` with a new connect host | Pollen moved the endpoint. The robot does not send the token to an undeclared host, and the local pipeline answers. Declare the new host exactly in `KEY_HOSTS` and update the robot. |
| `provider_http_401` or `provider_http_403` | The allocator refused the request. The token may be wrong or expired: store a new one (step 2). |
| `provider_unreachable` | The robot cannot reach the allocator. Check the robot's internet access. |
| "The realtime extra (websockets) is not installed." | Run the installer again. |

While the route is down, the local pipeline answers (Vosk and Piper, when installed).
