# Voice tools

The robot can act on what you say: take a nap, start a timer, remind you of something, tell you
the weather and play internet radio. You speak normally. There are no fixed phrases to learn. The
language model of the conversation route understands the request and calls the matching tool
(SPEC 11, Voice tools).

| Tool | Say, for example | On by default |
| --- | --- | --- |
| Sleep | "Go to sleep", "Vai a dormire" | Yes |
| Timers | "Set a timer for ten minutes for the pasta", "Metti un timer di 5 minuti" | Yes |
| Reminders | "Remind me to call Anna tomorrow at six", "Ricordami ogni giorno alle 8 di prendere le medicine" | Yes |
| Weather | "What's the weather tomorrow?", "Che tempo fa?" | No: an admin turns it on |
| Radio | "Play some jazz", "Metti Radio Deejay", "Louder", "Stop the radio" | No: an admin turns it on |

## Before you start

- **A conversation route that can call tools.** Tools work only through a language model: a chat
  route, or a realtime route (OpenAI realtime or Gemini Live). With no route reachable, the robot
  says it cannot do that now. There is no offline understanding of commands.
- **Pollen's cloud voice is not verified yet.** It is unknown whether Pollen's speech-to-speech
  server passes tool calls. If it does not, no tool works on that profile. Test it with "set a
  timer for one minute" and look for the announcement.
- **The voice listener must run.** Timers and reminders need it. Without it, the Tools tab shows
  them as not available.
- **Set the time zone.** Reminders use `circadian.timezone`. The installer asks for it
  (`--timezone Europe/Rome`). A wrong zone makes reminders ring at the wrong hour.

## Quick start (web UI)

1. Open the web UI and log in with the PIN.
2. Open the **Tools** tab (**Strumenti** in Italian). It has one card per tool. Each card shows
   the tool's status: Ready, Off or Not available, with the reason.
3. Sleep, timers and reminders work at once. Try "set a timer for one minute".
4. For weather:
   1. On the Weather card, search for your town and pick it from the list. This saves the
      location and the place name.
   2. Turn the switch on and confirm with the PIN.
5. For radio:
   1. On the Radio card, add a station from the **Station catalogue**, or add your own: a name,
      its stream URL and optional genres.
   2. Press the test play button to check that the stream plays.
   3. Turn the switch on and confirm with the PIN.

The Settings tab shows the same settings under the "Voice tools" group, including the advanced
ones.

## Sleep

Say "go to sleep". The robot says good night, then takes a nap when its reply has ended.

- For 10 minutes (`circadian.requested_nap_min_s`, 60 to 7200 s), only the robot's name, a wake
  phrase, or a due timer or reminder wakes it. Your face, movement, light, a loud sound and speech
  without its name do not.
- After that time, the normal day-nap rules apply.
- During quiet hours, or when the robot already rests, it answers that it is already resting.
- If you keep talking after the request ("go to sleep, but first tell me the time"), the robot
  answers first and naps after that turn.
- A requested nap stops the radio.

## Timers and reminders

- **Timers.** "Set a timer for 20 minutes", optionally with a label ("for the oven"). The longest
  timer is 24 hours (`tools.scheduler.timer_max_s`).
- **Reminders.** A text, a clock time and a date: "remind me to water the plants tomorrow at
  nine". They can repeat every day, on weekdays, or on chosen weekdays ("every Monday and
  Thursday at 7").
- **List and cancel.** "What timers do I have?", "Cancel the oven timer", "Cancel the plants
  reminder". You can also cancel them in the Tools tab.
- **Nothing to cancel.** A timer that rang is no longer active. If you ask to cancel it, the
  robot says that it already ended or that no such timer is active.

When an item is due:

1. The robot wakes from a nap if it naps. During platform sleep (motors off), the item waits and
   fires when the robot is awake again, however late.
2. It waits until nobody is talking, at most 60 s (`tools.scheduler.announce_max_wait_s`). Then it
   waits only for its own current sentence to end.
3. It pauses the radio, nods, and says the reminder in its own words. You can answer without
   saying its name.
4. A timer that nobody answers repeats after 30 s, at most 2 times
   (`tools.scheduler.repeat_after_s`, `tools.scheduler.repeat_max`). A reminder does not repeat.
5. If the language model does not answer within 8 s (`tools.scheduler.announce_timeout_s`), the
   robot says a fixed sentence with its local voice, or plays a cue sound when it has no local
   voice.

Timers and reminders survive a restart. They are kept in `tools_schedule.json` in the app data
folder (`~/.local/share/roboemot/`, mode 0600). An item that fell due while RoboEmot was not
running is announced at start if it is less than 15 minutes late
(`tools.scheduler.missed_grace_s`). An older item is listed under "Missed" in the Tools tab.

Limits: at most 20 active items (`tools.scheduler.max_items`) and texts of at most 200
characters (`tools.scheduler.text_max_chars`).

## Weather

The robot asks Open-Meteo (no account, no API key) for the current weather and a daily forecast
of up to 3 days (`tools.weather.forecast_days`, 1 to 7), in metric units.

- It needs a location: `circadian.location` (lat, lon). The place search in the Tools tab sets
  it.
- If weather is on but no place is set, the robot still knows the tool. When you ask for the
  weather, it says that no place is set up. It tells you to set the place in the Tools tab. It
  sends no request.
- Only the coordinates and the number of days leave the robot. The place name stays on the robot
  and is never sent to the language model.
- A forecast is reused for 10 minutes (`tools.weather.cache_s`), so a follow-up question costs no
  new request.
- If Open-Meteo does not answer within 3 s (`tools.weather.timeout_s`), the robot says the
  weather is unreachable.

## Radio

The robot plays internet radio streams from your station list through its speaker.

- **Stations.** Each station has a name, a stream URL (http or https), optional genres and an
  optional language. The model picks a station by its name or by a genre ("play some jazz" plays
  the first jazz station). It can never play a URL that is not on your list.
- **Station catalogue.** The Radio card has a closed **Station catalogue** section with checked
  streams that ship with RoboEmot, in six groups: Jazz; Ambient, chillout; Funk, soul, R&B;
  Hip-hop, rap; Rock; Metal, progressive. **Add** puts one station on your list, and **Add all**
  puts every station of a group on it after one PIN. An added station is a normal station: you
  can edit or remove it. Nothing in the catalogue plays until you add it. A station already on
  your list, by name (case ignored) or by URL, shows as added. Radio France (FIP, Mouv') and
  SomaFM streams carry an "unofficial stream" note: their operators publish no permission for
  players like this one. Each station links its operator's page.
- **No station yet.** If the radio is on but the list is empty, the robot still knows the tool.
  When you ask for music, it says that no station is set up. It tells you to add one in the Tools
  tab. It plays nothing.
- **Stream URLs.** Use the direct stream URL (often ending in `.mp3`, `.aac` or `/stream`).
  Redirects are not followed, and playlist formats (HLS `.m3u8`, DASH) are refused. If the test
  play fails, find the final stream URL of the station.
- **Volume.** The music volume is separate from the voice: -12 dB by default
  (`tools.media.volume_db`). "Louder" and "quieter" change it by 3 dB (`tools.media.volume_step_db`).
- **It pauses when someone speaks.** The radio pauses while the robot speaks, and while you talk
  to it from its name until its reply ends. Then the radio rejoins the live stream after 1.5 s
  (`tools.media.resume_after_s`).
- **It gets quieter while you speak over it.** On a robot with a hardware echo canceller (Reachy
  Mini Wireless), the music drops by 18 dB (`tools.media.duck_db`) as soon as someone starts
  speaking over it, so the robot hears you better. It comes back up 1.5 s after you stop
  (`tools.media.duck_release_s`). Turn it off with `tools.media.duck_on_speech`.
- **Say the name first.** While the radio plays, only the robot's name (or a wake phrase)
  addresses it, so it never answers song lyrics. The only exception is the reply to an
  announcement.
- **Dancing.** The robot may dance to its own radio (`tools.radio.dance`, on by default).
- **When it stops.**
  - "stop the radio"
  - the admin turns the radio off
  - the playing station is removed from the list
  - a requested nap, platform sleep, or the circadian rest (night, quiet hours)
  - after 2 hours (`tools.media.max_play_s`)
- **Naps.** While the radio plays, the robot does not take idle naps.

The robot decodes streams with `gst-launch-1.0`, which is already on Reachy Mini.

## Configure without the web UI

The same settings can go in `~/.local/share/roboemot/app.yaml` on the robot. A value saved in the
web UI wins over `app.yaml` (see [configuration.md](configuration.md)). Example:

```yaml
circadian:
  timezone: Europe/Rome
  location: {lat: 45.46, lon: 9.19}
tools:
  weather:
    enabled: true
    place_name: Milano
  radio:
    enabled: true
    stations:
      - name: Radio Swiss Jazz
        url: https://stream.srg-ssr.ch/m/rsj/mp3_128
        genres: [jazz]
      - name: Radio Paradise
        url: https://stream.radioparadise.com/mp3-128
        genres: [rock, eclectic]
        language: en
  media:
    volume_db: -15
```

The two station URLs are examples. Check each station with the test play button in the Tools tab
before you rely on it. After you edit `app.yaml` by hand, restart RoboEmot so it reads the file:
`python scripts/run_on_robot.py --restart`.

## Turn a tool off

- **Weather and radio:** switch them off in the Tools tab.
- **Any tool, sleep and timers included:** remove its names from `conversation.llm_skills`
  (Settings tab, Conversation). The model is then not offered that tool. The tool names are:
  - `go_to_sleep`
  - `timer_set`, `timer_cancel`, `reminder_set`, `reminder_cancel`, `schedule_list`
  - `weather`
  - `radio_play`, `radio_stop`, `radio_volume`

If you saved your own `conversation.llm_skills` list in an older version, it does not contain the
new tools. At startup the log then says `conversation.llm_skills leaves out core voice tools,
the model is not offered: ...`. Add the names, or reset the setting to its default.

## Privacy (SPEC 12)

- Weather and radio are off until an admin turns them on with the PIN. They are the only tools
  that use the internet.
- Weather sends only the coordinates and the number of days. The weather URLs are admin-only and
  must be https.
- Radio connects only to the URLs on your station list. The station list is admin-only.
- Logs never contain reminder texts, timer labels, coordinates, the place name or station URLs.
- What you say to set a reminder is conversation text. It reaches the language model like any
  other sentence, so it leaves the robot when the route is a cloud route.

## Troubleshooting

| What you see | Likely cause | What to do |
| --- | --- | --- |
| The robot talks about a timer but none is set | The route does not pass tool calls (possibly Pollen's server) | Use a chat route, OpenAI realtime or Gemini Live |
| "Timers need the voice listener" in the Tools tab | Voice is off | Turn voice on |
| Weather: "Set a place first" | No `circadian.location` | Use the place search in the Tools tab |
| The robot says no station or no place is set up | Radio or weather is on, but the station list is empty or no place is set | Add a station from the Station catalogue on the Radio card (or your own), or set the place in the Tools tab |
| Weather: "unreachable" | No internet, or Open-Meteo slower than 3 s | Check the network |
| The test play says "could not be reached" | Wrong URL, a redirect, or a playlist URL | Use the final direct stream URL |
| Radio: "this robot's speaker cannot play music" | The robot platform has no media output | Radio needs a speaker that supports media (Reachy Mini does) |
| The robot ignores you while music plays | You did not say its name | Start with its name: "Reachy, stop the radio" |
| Reminders ring at the wrong hour | Wrong `circadian.timezone` | Set the zone in Settings, or reinstall with `--timezone` |
| A new tool is never used | A saved `conversation.llm_skills` list hides it | See "Turn a tool off" |
| "Too soon" in the Tools tab | Searches or test plays closer than 3 s apart (`webui.tools.min_interval_s`) | Wait a moment |

## For developers: add your own tool

Tools are plugins in the `roboemot.skills` entry-point group (SPEC 11, Voice tools). It is the one
plugin group not keyed by `robot.id`, so a tool works on every robot.

1. Write a factory that takes a `ToolHost` and returns a list of `@skill` functions:

   ```python
   from roboemot.skills.host import ToolHost
   from roboemot.skills.registry import SkillResult, skill

   def skills(host: ToolHost):
       @skill(
           llm_visible=True,
           description="Tell the person a joke",
           available=lambda: host.config().conversation.language in ("it", "en"),
       )
       def tell_joke() -> SkillResult:
           return SkillResult({"status": "ok"})

       return [tell_joke]
   ```

2. Register it in your package's `pyproject.toml`:

   ```toml
   [project.entry-points."roboemot.skills"]
   jokes = "my_package.jokes:skills"
   ```

3. Add the skill names to `conversation.llm_skills`. Only listed skills are offered to the model.

Rules:

- **Side-effect free factory.** A factory only builds functions. The realtime check calls it with
  a throwaway host.
- **Live checks.** `available` and the skill read `host.config()` on every call, never a
  captured config.
- **Parameter types.** Parameters may be `str`, `int`, `float`, `bool`, `Literal[...]`,
  `X | None`, or `Annotated` with `Range`, `MaxLen` and `Desc`. Arguments are validated strictly.
- **Empty choices.** A skill with an empty `choices` list is not offered. Set
  `offer_without_choices=True` to keep it offered. The parameter is then a plain string within
  its length cap, and your skill decides. Return `{"error": "not_configured"}` when the tool is
  not set up, and say in the description what the person must do.
- **Timeout.** A skill that takes longer than `conversation.tool_timeout_s` (5 s) is abandoned.
- **What `ToolHost` gives you.** `config`, `wall_now` (time in the robot's zone), `bus`, and,
  when the run has them, `nap`, `scheduler`, `http` (a client without redirects or environment
  proxies) and `media`.
- **Logging.** Log names and counts, never user text.
- **Broken plugins.** A plugin that fails to load is logged and skipped. It never stops the run.
