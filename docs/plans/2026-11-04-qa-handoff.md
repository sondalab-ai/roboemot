# QA handoff: one robot round for everything since 2026-10-30 (2026-11-04)

This file collects every robot check that is still open from the recent sessions:
- this session (vision, wake path, reflexes, soothe, web UI);
- the music-dance session (`roboemot-6a`);
- the nap session (`roboemot-da`);
- the cloud-voice epic.

Run the round once, in the order below. Send the debug reports back to a fresh session. That session updates SPEC 13 and SPEC 15 and opens tuning slices.

It replaces these older lists, which stay as background:
- `docs/plans/2026-10-27-qa-handoff.md`;
- the "QA handoff" section of `docs/plans/2026-10-28-cloud-voice-default.md`;
- section 1 and the wake path item of `docs/plans/2026-11-03-vision-cost-handoff.md`;
- Q1 of `docs/plans/2026-11-04-music-dance.md`.

**Last full robot run:** 2026-10-30 on #138 (local pipeline numbers, before the cloud-voice epic). On 2026-11-04 the owner checked the cup only, with three debug reports. Everything merged after #138 is untested on the robot unless a check below says otherwise.

**2026-11-05 (owner, informal, before a reinstall, so on code before PRs 188 and 190):** two debug reports, at 09:41 and 13:27.
- The cup was recognised slowly and not followed (2 episodes, both `lost`, 2.4 s).
- Music was not detected (`music.confidence` 0.14).
- The robot answered 6 of 51 utterances. `rejected_not_facing` was 29, and the facing yaw read a median 66.8° while the owner faced the robot.
- One answer was spoken three times, because of skill rounds.
- The server transcribed "ciao Neve" as "Ciao, Reachy Mini" and "hey ciao neve" as "si deve".
- One reply ended with a stray spoken word, "odyssey".

The fixes are listed in the code follow-ups below.

## 0. Setup

Owner rules: an agent does not move the motors, does not start or restart RoboEmot, and asks before any SSH write to the robot. The SSH password is never stored.

1. On the Mac: `git pull --ff-only` on `main` (at least `d93f4df`, PR 190; take the newest `main`), then `pip install --no-deps -e packages/roboemot-core -e packages/roboemot-reachymini`.
2. Run `python scripts/install_on_robot.py --dry-run --cloud-voice --host 192.168.88.198`, then the same command without `--dry-run`. Optional, once: add `--overclock` (PR 189). The wizard checks the hardware, writes a trial config, and asks before it reboots the robot into it. Read `system.cpu.freq_max_mhz` (1800) and `temperatures.throttling` afterwards, and run the B step 3 cost check once at 1.5 GHz and once at 1.8 GHz. If the Mac says "No route to host", run `sudo arp -s 192.168.88.198 88:a2:9e:8d:58:b5`.
3. Start with `python scripts/run_on_robot.py --host 192.168.88.198`. Do not use `--restart` right after an install. Wait about 2 minutes before you talk: the listener lags while the models load.
4. Check in the web UI Settings: `vision.objects_enabled` is on, and `voice.barge_in.onset_enabled` is on (it was turned off during an earlier QA). Until the facing fix lands, turn `voice.facing.enabled` off for every section except C.
5. Take each debug report from the web UI Diagnostics tab. The SSH report (`robot_ssh.py --debug`) has no runtime sections. Name each file after its check (for example `B-cup.json`).

## A. Cloud voice baseline (epic PR 160, CV-0 to CV-21)

PRs: https://github.com/sondalab-ai/roboemot/pull/160, plus CV-20/21 https://github.com/sondalab-ai/roboemot/pull/161 and https://github.com/sondalab-ai/roboemot/pull/165.

1. Hold a 10-minute conversation facing the robot. Include one phone call in front of the robot and one TV clip.
2. Pass (debug report):
   - `roboemot.realtime_route.state` is `connected`; `speech.backend` is `none`; no Piper process runs.
   - `local_fallbacks` is 0.
   - listener `lag_max_s` is under 1 s, and `dropped_s` is close to 0 (CV-0; it was 35.6 s of 275 s on 2026-10-30).
   - `latency_ms.transcript` p95 is far below 17.6 s, and `first_audio` p95 is far below 20.7 s (the 2026-10-30 figures). Record both in SPEC 13.
   - `skipped` counts the phone call. The TV gets no answer.
   - `tone_retry` against `tones`: how often the model calls the tone tool without speaking (CV-21).
3. Record the transcript p95 so that `voice.realtime_route.transcript_timeout_s` can be retuned. The profile writes 10 s, and the code default is 4 s.
4. Listen for these four things:
   - Is the server voice better than Piper? This is the owner A/B item in SPEC 15.
   - The listening/thinking cue.
   - Answers cut at the sentence cap.
   - False barge-in cuts (`bargein.triggers` with nobody talking).
5. Interrupt the robot while it uses a tool (for example during a long answer with a tool call).
   - Pass: the next turn answers normally, and the debug report shows no server error after the cut.
   - Why: nobody has verified that the platform server accepts tool results sent right after `response.cancel` (CV-8, https://github.com/sondalab-ai/roboemot/pull/148).
6. **Name hint** (https://github.com/sondalab-ai/roboemot/pull/190). Say "ciao Neve" 5 times.
   - Pass: the Live transcript reads "Neve", not "Reachy Mini".
   - Read `roboemot.realtime_route.transcription_hint` `sent` and `refused`, and the client `update_fallback`. A non-zero `refused` means that the server does not accept the hint.
   - Watch for a hallucinated "Neve" in the transcript of silence or noise. It could wake the robot falsely.
7. **One answer per question.** Pass: `realtime_route.responses.completed` is close to `conversation.turns` minus `skipped`, and no sentence is spoken twice. The fire-and-forget slice ends the turn after a spoken reply with `play`, `look_at` or `express_emotion`; read `conversation.follow_ups_skipped`.
8. **Stray words.** Write down every stray word at the end of a reply ("odyssey" on 2026-11-05), with the time. If it repeats, report it to Pollen: the server model generates it.

## B. Vision: objects, cup, cost (PRs 162, 163, 167, 174, 176, 177, 178)

PRs:
- https://github.com/sondalab-ai/roboemot/pull/162
- https://github.com/sondalab-ai/roboemot/pull/163
- https://github.com/sondalab-ai/roboemot/pull/167
- https://github.com/sondalab-ai/roboemot/pull/174
- https://github.com/sondalab-ai/roboemot/pull/176
- https://github.com/sondalab-ai/roboemot/pull/177
- https://github.com/sondalab-ai/roboemot/pull/178

1. **Cup at arm's length.** Sit at about 1 m from the robot and hold a coffee cup at chest height. Move it slowly left and right, then hold it still for 30 s.
   - Pass: the Live preview shows a box on the cup the whole time. The robot looks at the cup and follows it while it moves (V-6 shown object).
   - Debug keys, under `roboemot.vision`:
     - `object_input` (how many runs used the crop near faces);
     - `objects.low_score_continued` (tracks kept by weaker detections);
     - `object_gate` (runs and skips, with the reason);
     - `entity_embeds` (`runs` and `skips` by reason, V-4).
   - Debug key, under `roboemot.attention`: `shown_object` (episodes, how each one ended, and its length).
   - On 2026-11-04 the cup was seen only very close and briefly, with 2 episodes, both `lost`. Episodes should now last while the cup is in view.
2. **Still cup.** Use a cup that the robot already knows. Put the cup on the table in view for 2 minutes, then pick it up and show it.
   - Pass: no second long novelty look, neither after about 10 s on the table nor when you pick it up. PR 185 (https://github.com/sondalab-ai/roboemot/pull/185) keeps a recognised object's identity while it sits still, so a repeat look is now a fail.
   - Known limit, not a fail: a known cup that is still from the moment it comes into view may never be recognised while the object gate is on (see the follow-ups).
3. **Cost.** After about 10 minutes with people and objects in view, save the report. Read `roboemot.vision.stage_timing`: the CPU share of YuNet, SFace, DINOv2 and NanoDet. These numbers decide V-3 (lower face detection rate while nobody is in view) in `docs/plans/2026-11-03-vision-cost-handoff.md`.
4. **Fewer DINOv2 embeds** (https://github.com/sondalab-ai/roboemot/pull/188). In the cluttered room, read `entity_embeds.skips.young` and `skips.episode`, `stage_timing.appearance_embed.runs_per_s`, `object_detect.runs_per_s` and `stage_drops.objects`.
   - Baseline on 2026-11-05 (before PR 188): `appearance_embed` 0.23 runs/s at 1174 ms, `object_detect` 0.87 runs/s (3 wanted while a cup was shown), `stage_drops.objects` 1033 in 3.5 min, `entity_embeds.runs.new` 30.
   - Pass: `object_detect.runs_per_s` close to `objects_active_hz` (3) while a cup is shown, and the cup episodes last while the cup is in view.
   - Embed worker (https://github.com/sondalab-ai/roboemot/pull/191): read `roboemot.vision.embed_worker` (queue wait p50/p90, dropped, failed) and `process.wall_ms.max`, which should no longer reach about 1.2 s. Compare `vision.appearance_threads` 1 against 2 on `object_detect.runs_per_s` and the listener `realtime.share`.
5. Optional: run `scripts/bench/run_cm4.py --only nanodet_plus_m_320,yolox_nano_416` on the robot. This is a SPEC 15 item: object detector p50 and p95 on the CM4.

## C. Wake path with the name "Neve" (PRs 179, 180)

PR: https://github.com/sondalab-ai/roboemot/pull/180. The 2026-11-04 diagnosis: the facing check blocked the answers, not ASR. The face yaw read 40 to 73 degrees against the `voice.facing.max_yaw_deg` limit of 30.

Do each step 5 times, then save one report for the whole set:
1. Alone at 50 cm, looking at the robot, without the name.
2. The same, looking away.
3. Looking away: "Neve, come stai?"
4. Looking away: "Neve" followed by 5 or more words.
5. Looking away: "ciao Neve" followed by a long sentence.
6. As step 1, with a second person 1 m to the side.
7. Turn away from the other person and speak at once. Then face the robot for 3 s first and speak.

On 2026-11-05 the yaw read a median 66.8° (49 checks, min 14°) while the owner faced the robot. A Mac experiment ruled out the yaw formula and small faces: LFW frontal faces at 4–40 px eye span read 5–10°. The photographs in the room are behind the robot. The facing diagnostics slice adds the data that decides the cause. Before the steps, turn `webui.live.video_landmarks` on (it needs `webui.live.enabled` and `webui.live.video`) and check in the Live preview that the 5 magenta dots sit on your eyes, nose and mouth, and read the yaw label under your box. After the steps, read `roboemot.voice` `direction.facing.recent[]` (`face_source`, `faces[]` with `entry_id`, `eye_px`, `yaw_deg`; `bearing_head_deg`, where about ±180 means a front/back confusion) and `roboemot.vision` `faces[]` and `frame` (`eye_px` in detector px, `letterbox_scale`, `channel_order`). PR: https://github.com/sondalab-ai/roboemot/pull/193.

Read `roboemot.voice`:
- the wake `decisions` (`wake_word` and the `rejected_*` outcomes);
- `grammar` `name_first`;
- `direction.facing.yaw_deg` while you speak.

Steps 3 to 5 must answer, because "Neve" as the first word wakes the robot (`voice.grammar.name_first_max_start_s` 1.5 s). The first-word wake also needs a voice at a face, so speak with your face in view.

How to read the results:
- Step 1 over 30 degrees: the yaw has a bias. Raise `voice.facing.max_yaw_deg` or recalibrate `vision.facing_nose_depth_ratio`.
- Step 1 passes and step 6 fails: the check reads the wrong face. The fix is to keep the yaw per face track.
- Only step 7 fails: the timing is wrong. The fix is to judge the yaw after the speech starts.

## D. Reflexes in a quiet room (PR 182)

PR: https://github.com/sondalab-ai/roboemot/pull/182.

1. Sit quietly in a quiet room for 2 minutes. Do not talk.
   - Pass: about 0 "girarsi verso una voce" (turn to a voice) and about 0 "sussulto per un rumore forte" (startle at a loud noise).
2. Clap once, loudly. Pass: exactly one startle.
3. Speak from the side, out of view. Pass: one turn toward your voice.
4. Let the robot move, for example during an expression or the cup follow. Pass: no startle and no turn caused by its own motion.
5. Read `roboemot.reflexes`. For each rule, read `fired`, `dropped_cooldown`, and `gate`:
   - `offered` is what the rule alone would fire on;
   - `triggered` is what passed the gate;
   - `suppressed`: `motion` and `rearm` for loud onsets, `no_vad`, `own_speech` and `motion` for a new voice.
   Also read `armed` and the onset figures, which tune `min_rms` and the ratio.
6. Also in SPEC 15:
   - the reflex latency within 100 ms;
   - `withdraw` raises the nose;
   - the startle under load. The TODO item says that a bang shorter than one chunk can be missed. That item waits for an owner decision.

## E. Soothe sounds (PRs 172, 180)

PRs: https://github.com/sondalab-ai/roboemot/pull/172 and https://github.com/sondalab-ai/roboemot/pull/180.

1. Leave the robot idle until it hums. The defaults are now 12 dB lower: -24 dBFS for the calm and joy hums, -26 for the blirps, -27 for sleepy.
   - Pass: the hum is audible but quiet.
   - If it is not, change `soothe.variants[].shape.gain_db` and report the value.
2. Start talking while it hums. Pass: the hum stops at once, and no new hum starts while you speak.
3. Read the soothe counters `played`, `held_by_speech` and `cut_by_speech`, and the barge-in outcome `cut_own` (a hum cut, not a sentence).

## F. Alone, day nap, nap wake, Live badges (PRs 158, 170, 171)

PRs:
- https://github.com/sondalab-ai/roboemot/pull/158
- https://github.com/sondalab-ai/roboemot/pull/170
- https://github.com/sondalab-ai/roboemot/pull/171

1. **Alone.** Leave the room for more than 120 s (`world.scan.alone_after_s`).
   - Pass: the scans and the idle gaze shifts stop. Breathing and micro-motions stay.
   - Pass: a face or a sound resumes them at once.
   - Read the `roboemot.alone` section.
2. **Day nap.** Outside quiet hours, give the robot no face and no addressed speech for 1800 s (`circadian.nap_after_s`).
   - Pass: `roboemot.circadian` state is `resting` with reason `nap`, and the robot takes the rest posture.
   - Pass: a TV left on does not block the nap.
3. **Nap wake stimuli.** Start a new nap for each test:
   a. Wave a hand in front of the camera, at least 10 s after the nap starts (`circadian.nap_motion_grace_s`).
   b. Make a loud noise.
   c. Turn on the room light, at least 300 s after the nap starts (`circadian.nap_light_grace_s`). The wake comes 60 to 80 s after the light change.

   Pass:
   - Each test ends the nap, and the robot leaves the rest posture.
   - `nap.wakes.<reason>` (`motion`, `loud`, `light`) goes up by 1 for each test.
   - `roboemot.vision` `scene_motion.confirmed` goes up on the hand test.
   - `scene_motion.no_pose` does not keep rising during a nap. If it rises, the frames carry no head pose.
   - No wake comes from a still room during a 10-minute nap.

   Risks:
   - A TV in view wakes the nap through motion. The remedy is `circadian.nap_wake_on_motion: false`.
   - The motion thresholds are first guesses: `vision.object_gate` `pixel_diff` 25, `changed_share` 2 %, `vision.scene_motion.confirm_samples` 2.
   - The CPU cost during a nap is never measured. Compare the CPU with the nap on and with the nap off.
4. **Live tab badges.** Open the Live tab during a day nap, while resting, while drowsy, and with the motors off (platform sleep).
   - Pass: the motor badge reads "Motori attivi" or "Motori spenti" to match the motors.
   - Pass: the modes card shows "Pisolino", "Riposo" or "Assonnato" to match the state.
   - Pass: the status line agrees with the modes badge.
   - Source: `/api/live` `modes.rest`.

## G. Music and the wake word (PR 184: slices 181 and 183)

PR: https://github.com/sondalab-ai/roboemot/pull/184. The dance slices (music-dance 2 to 4) start only after Q1 passes.

1. **Q1.** Play a song for 3 minutes, talk over it, and say the wake word 10 times. Then do 3 minutes of speech only and 3 minutes of TV.
   - Report `roboemot.voice.music.cost_ms_per_audio_s` (median and p90).
   - Report the listener `realtime` stages `music`, `decode` and `asr_*`.
   - Report the wake `decisions` `rejected_music` and `wake_word`, and the wake hits out of 10.
   - Pass:
     - the music cost median is at most 1 % of one core (at most 2 % with an owner OK);
     - there is no `rejected_music` during speech only;
     - the robot does not answer the lyrics.
   - If the cost is above 2 %, retry with a longer hop and fewer bands first.
2. **CPU timer.** During the Q1 song, read `cost_ms_per_audio_s`.
   - Pass: the samples are more than 0, and the median is more than 0 and less than 10 ms per audio second.
   - A median of 0 means that `time.thread_time` does not work on the CM4 kernel. The Q1 cost number then cannot be trusted.
3. **Bass noise is not music.** Use 2 minutes of a washing machine spin cycle, or a neighbour's bass through a wall. Talk to the robot without the wake word.
   - Pass: `roboemot.voice.music.starts` stays 0, and there is no `rejected_music`.
   - Why: the tests used only synthetic signals.
4. **Ordinary music confirms.** Play 1 minute each of 2 normal songs with drums, vocals and other instruments, at a normal room level.
   - Pass: `roboemot.voice.music.playing` becomes true within about 4 s for each song.
   - If a song never confirms, report it with `music.min_band_periodicity` (default 0.35) for tuning. The tests reached only about 0.45, a margin of 0.10.

## H. Web UI (PRs 159, 164, 173, 175)

PRs:
- https://github.com/sondalab-ai/roboemot/pull/159
- https://github.com/sondalab-ai/roboemot/pull/164
- https://github.com/sondalab-ai/roboemot/pull/173
- https://github.com/sondalab-ai/roboemot/pull/175

1. **Impostazioni (Settings).** Pass: there are no empty rows, and each accordion opens and closes. Pass: the reference settings offer choices, and the "only null" filter works.
2. **Tab switch.** Switch tabs fast while a view loads. Pass: the new tab never shows the old tab's content or an error from the old tab.
3. **Language.** Pass: the dashboard shows one language per page. Pass: the Emotions tab works, and its writes ask for the PIN.

## I. Older open robot items (not required for this round)

These items are in SPEC 15 ("To verify on the robot in M0") or in `TODO.md`. Do them if time allows:
- the installer `--uninstall` leaves an empty diff;
- the Python version on the robot (the Piper hash locks);
- `secrets.json` is owned by the user the app runs as;
- the listener real time with `ps -L` nice values;
- the barge-in tuning and ducking with music;
- the nod and shake signs;
- the M2 and M3 exits on the robot;
- the recorded clips;
- the light sensing log;
- the connectivity gestures;
- the #133 checks: no freeze with a person behind the robot, and a photo stops holding the gaze after about 20 s;
- the Wi-Fi move: the robot sits at -66 to -82 dBm on a DFS 5 GHz channel.

The realtime HIL suite (`test_realtime_hil.py`, section 2 of `docs/plans/2026-10-27-qa-handoff.md`) is still open. It was impractical on 2026-10-30. Section A gives the main figures without it.

## What to send back

- One debug report per section (A to H), named after the section.
- One line per check: pass, fail, or not done, with a note where the result is a judgement.
- The values you changed in Settings during the round.

## Not robot QA (code work)

- Done 2026-11-04: sticky identity for still objects, https://github.com/sondalab-ai/roboemot/pull/185.
- Done 2026-11-04: flaky tests and the Python 3.11 `asyncio.wait_for` race on a barge-in, https://github.com/sondalab-ai/roboemot/pull/186.
- Proposed, not approved: put the shown object into the conversation context.

The background is in `docs/plans/2026-11-03-vision-cost-handoff.md`, section 3.

## Follow-ups from the recent sessions (code, not robot QA)

Sources:
- the transcripts of the vision/wake/reflex, cloud-voice/nap and music-dance sessions, and of the small design and research sessions;
- `TODO.md`;
- the open items in the plans of 2026-10-27 and 2026-10-28.

Status terms:
- **approved**: the owner said yes.
- **done**: merged on `main`, with the date.
- **proposed**: it waits for an owner OK.
- **decision**: the owner must choose.
- **limitation**: known and accepted for now.
- **review**: a review finding that was deferred.
- **check first**: it may already be fixed, so read the code before you act.

Items marked "after QA X" wait for that QA section.

### Decisions for the owner

| Item | Where | Status |
| --- | --- | --- |
| With `voice.enabled` off (the default), the `turn-to-voice` reflex never fires, because it needs the speech detector. Accept, or give it a fallback? https://github.com/sondalab-ai/roboemot/pull/182 | `reflex/engine.py`, SPEC 10 | decision |
| Startle under load: a bang shorter than one chunk can be missed when the audio hub runs late. Keep this, or pass the loudest chunk of the pass (a SPEC change)? | `TODO.md`, SPEC 10 | decision |
| CPU headroom: the owner chose an opt-in installer overclock to 1.8 GHz with a tryboot trial (2026-11-05). A Coral or Hailo-8L accelerator was ruled out for this robot. A CM5 swap stays open. https://github.com/sondalab-ai/roboemot/pull/189 | installer, SPEC 4A | done (2026-11-05) |
| Let the server start its answer before the robot's transcript round trip, as Pollen's app does. The owner approved server-driven turns (2026-11-05): a `voice.realtime_route.turns: server` mode with server VAD creating and interrupting responses, speech-gated streaming, the skip tool as the addressing filter, and Pollen-like barge-in (duck to 25 %, stop on the server's `speech_started`). Slices S1–S3. | SPEC 8, 12, 13 | approved |
| Try `voice.mood_source: llm_tagged`, because the owner found the delivery flat. | SPEC 8 | proposed |
| Put the shown object into the conversation context. | `docs/plans/2026-11-03-vision-cost-handoff.md` section 4 | proposed |
| LeRobot: adopt nothing now. Import expression clips in the LeRobot dataset format only if the owner starts recording clips. | `scripts/import_pollen_emotions.py` | proposed |
| Research ideas, not started: voice emotion through the realtime model, a small head-pose net in place of the 5-landmark yaw, a neural echo guard for barge-in, motion flow to predict where a locked face went, a test of the mirror neural net. | `voice/facing.py`, `TODO.md` | proposed |

### Tuning that waits for robot numbers

| Item | Where | Status |
| --- | --- | --- |
| Facing check: the yaw reads about 70° while the owner faces the robot (2026-11-04 and 2026-11-05). The small-face cause is ruled out. Choose the fix from the facing diagnostics (matched entry, eye span, sound bearing, channel order, landmarks in the preview). Do not raise `voice.facing.max_yaw_deg` to hide it. | `voice/facing.py`, `vision/facing.py`, SPEC 8 | after QA C |
| Set `voice.realtime_route.transcript_timeout_s` to the measured p95 plus about 1 s. The profile writes 10 s, and the code default is 4 s. A 2026-11-03 run measured p95 4273 ms. | `config/schema.py`, installer profile | after QA A |
| Entity embeds: PR 188 adds a 3 s minimum age, the episode rule and a rebirth carry (age and last embed). Check `skips.young`, `skips.episode` and `runs.reacquired`. Then try `vision.appearance_input_px` 126 (Slice 2) against 224 for cost and recognition. https://github.com/sondalab-ai/roboemot/pull/188 | `vision/pipeline.py` | after QA B |
| Reflex motion gate: expressive motion above 60 deg/s (clips, antenna wiggles) suppresses reactions for the motion plus a 500 ms tail. Watch `gate.suppressed.motion`. https://github.com/sondalab-ai/roboemot/pull/182 | `reflex/engine.py` | after QA D |
| Tone tool: many `tone_retry` against `tones` means extra server rounds. The appraisal rules for the five tone labels are guesses. https://github.com/sondalab-ai/roboemot/pull/165 | installer profile, SPEC 8 | after QA A |
| Local Vosk decode cap over music: add the cap if `decode` plus `asr_*` is above 15 % of one core with a song playing. | `docs/plans/2026-11-04-music-dance.md` | after QA G |
| A soothe hum opened the speaking window and dropped the start of a sentence (`dropped_speaking` 44 on 2026-11-04). PR 180 holds the hum while someone speaks. Confirm that the drop is gone. | `voice/speech.py` | after QA C/E |
| Face tracking: the rebirth distance p50 is 0.66 against the 0.6 gate. | `vision/tracker.py` | after QA B |
| Barge-in onset-to-stop: 313 ms against the 200 ms budget (2026-10-28). Answers were also cut at the sentence cap (3 of 5 turns, local mode). Recheck both under the realtime route. | SPEC 8, 13 | after QA A |
| A false barge-in on 2026-10-28 has no known cause. CV-9 added per-trigger records. https://github.com/sondalab-ai/roboemot/pull/149 | `voice/bargein.py` | after QA A |

### Code follow-ups

| Item | Where | Status |
| --- | --- | --- |
| Sticky identity for still objects. https://github.com/sondalab-ai/roboemot/pull/185 | `registry/matcher.py` | done (2026-11-04) |
| Make the flaky tests timing-independent, and fix the Python 3.11 `asyncio.wait_for` race that dropped a barge-in cancel. https://github.com/sondalab-ai/roboemot/pull/186 | tests, `realtime/turn.py` | done (2026-11-04) |
| Fewer DINOv2 embeds in a cluttered room and during a look (minimum age, shown-track only by bearing, rebirth carry). https://github.com/sondalab-ai/roboemot/pull/188 | `vision/pipeline.py` | done (2026-11-05) |
| The robot's name as the server's transcription hint, decided at connect. https://github.com/sondalab-ai/roboemot/pull/190 | `realtime/` | done (2026-11-05) |
| Slice 2: DINOv2 on its own worker thread, with `vision.appearance_threads`, `vision.appearance_input_px`, `vision.embed_retry_s` and `vision.appearance_spinning`. Watch the peak vision CPU (about 4 cores while an embed runs), and check with `ps -L -o tid,ni,comm` that `roboemot-embed` runs at nice 5. https://github.com/sondalab-ai/roboemot/pull/191 | `vision/embed_worker.py`, `vision/pipeline.py` | done (2026-11-05) |
| Opt-in installer overclock (`--overclock`, tryboot trial, verify, promote; revert on reinstall or uninstall). https://github.com/sondalab-ai/roboemot/pull/189 | installer, SPEC 4A | done (2026-11-05) |
| Fire-and-forget skills: a spoken reply with only `play`, `look_at` or `express_emotion` calls ends the turn, with no repeat round. The tone tool is offered in the first response only. https://github.com/sondalab-ai/roboemot/pull/192 | `skills/`, `conversation/worker.py`, SPEC 8 | done (2026-11-05) |
| Facing diagnostics: per-check records, per-face yaw and eye span, frame channel order, landmarks in the Live preview (`webui.live.video_landmarks`). https://github.com/sondalab-ai/roboemot/pull/193 | `vision/`, `voice/direction.py`, web UI | done (2026-11-05) |
| Server-driven turns S1a: `voice.realtime_route.turns: server` streams speech live, gated at onset by `server_gates`, under the new consent `privacy.realtime_audio_unaddressed` (the `--cloud-voice` profile sets it). Test `server` mode on the robot only once S1b has merged: before it, the robot asks for the answer itself and can answer a paused sentence twice. https://github.com/sondalab-ai/roboemot/pull/194 | `realtime/`, `voice/listener.py`, installer | done (2026-11-05) |
| Server-driven turns S1b: server-created responses (the server answers a stream that ended whole on its own; fragments and duplicates are cancelled and never play), `latency_ms.turn`. `server` mode can now be tested on the robot (after a reinstall); read `realtime_route.server_responses`, `server_fragments`, `server_duplicates`, `server_unanswered` and `latency_ms.turn` in QA A. https://github.com/sondalab-ai/roboemot/pull/196 | `realtime/`, `conversation/worker.py`, `voice/listener.py` | done (2026-11-06) |
| Server-driven turns S2 (barge-in like Pollen's), S3 (latency and cues). | `realtime/`, `conversation/worker.py`, `voice/` | approved |
| Nap resume: after a stimulus wake (`loud`, `light`, `motion`) with no confirmed face, addressed speech or conversation after it, the next nap comes after `circadian.nap_resume_after_s` (180 s) instead of `nap_after_s` (1800 s). Presence alone keeps the short resume (owner decision 2026-11-05). Read `nap.resumes` and `nap.resuming` in QA F. https://github.com/sondalab-ai/roboemot/pull/195 | `circadian/nap.py`, `circadian/runner.py`, SPEC 7A | done (2026-11-05) |
| Detect-then-track for the shown object: a cheap single-object tracker (OpenCV VitTrack, 48.6 ms on a Pi 4B) between NanoDet runs. | `vision/tracker.py` | approved |
| A known object that is still from the moment it comes into view is never recognised while the object gate is on: its votes land about 10 s apart, so `votes_required` live votes never exist. Possible fix: hold the gate while a new track's vote burst is open against an active record (the V-4 "denser object runs" note). Found in PR 185. | `vision/object_gate.py`, `vision/pipeline.py` | proposed |
| An object that moves and then stops can lose its track: the constant-velocity filter carries the old track away across the 10 s gate gap, and a new unknown track is born. Found in PR 185. | `vision/tracker.py` | proposed |
| `Registry.delete_entity` does not notify the forget listeners. The matcher covers this with a gallery check on each observation. Found in PR 185. | `registry/` | proposed |
| `reflexes.rules` and `soothe.variants` are missing from `KEYED_LISTS`, so every config layer replaces them whole. | `config/stack.py` | proposed |
| Driver angle contract: every driver reports each angle on the commands' continuous scale, never wrapped. Add a contract test over every registered driver. | SPEC 4A, `PoseGroupDriver` | approved (owner, 2026-10-28) |
| After a cut in a response that called a tool, the history has no interrupted assistant entry. https://github.com/sondalab-ai/roboemot/pull/148 | `conversation/worker.py` | limitation |
| The transcription language follows `conversation.language`, so a speaker whose `preferences.language` differs is transcribed in the wrong language on the server. Document this in SPEC 8, or fix it. https://github.com/sondalab-ai/roboemot/pull/139 | `realtime/route.py` | review |
| A text turn already queued can reopen the realtime session while the robot sleeps (with `runtime.pause_when_asleep`). | `realtime/` | limitation |
| The barge-in echo buffer may drop the playing audio for sentences over 2 s. | `voice/bargein.py` | check first |
| Object low-score pass: one frame with nothing at a score of 0.2 or more breaks the track chain until a 0.4 detection arrives. https://github.com/sondalab-ai/roboemot/pull/176 | `vision/tracker.py` | limitation |
| Object gate: the 80x60 motion test can miss a small moving object, such as a 40 px ball. Gate gaps can also expire tracks and churn track IDs. https://github.com/sondalab-ai/roboemot/pull/163 | `vision/object_gate.py`, `vision/tracker.py` | check first |
| Object gate review nits: a 3-channel assumption, state written without the lock, and a float32 full-frame copy on each granted tick. Some of the code may have moved since. https://github.com/sondalab-ai/roboemot/pull/163 | `vision/object_gate.py` | check first |
| Crop near faces: the validator checks only `full_interval_s < object_lost_s`. Far tentative tracks hold the gate in `live_track`. https://github.com/sondalab-ai/roboemot/pull/178 | `vision/object_crop.py` | check first |
| Day nap review items. A nap in progress may keep `nap.napping` true when `robot.asleep` fires. The motion streak has no time-gap limit. A light rise inside `nap_light_grace_s` is dropped (SPEC should say so). The posture change after `settle_s` may read as motion. https://github.com/sondalab-ai/roboemot/pull/171 | `circadian/runner.py`, `vision/scene_motion.py` | check first |
| Music: on the realtime path the voice-at-face check still runs over music (review M2). A broadband tick reads as music. One very large chunk can turn `stopped` into `started` in the same feed. https://github.com/sondalab-ai/roboemot/pull/183 | `voice/wake.py`, `voice/music.py` | review |
| Music test and doc gaps: no test of rhythmic speech through the octave rule, of the clock rejection against `music.min_band_periodicity`, of `music.stopped` on a reload that turns music off, or of the Vosk path with the real detector. The full list is in the plan's "Deferred minor findings". https://github.com/sondalab-ai/roboemot/pull/181 | `tests/test_music_detector.py`, `docs/plans/2026-11-04-music-dance.md` | review |
| Pin the YuNet and SFace model URLs to commit `f88e9b2bafd21f1cad242fb5af6d78f2bcba16a3` (tag 4.10.0) instead of `raw/main`. The sha256 check still protects the download. | `vision/models.yaml:22,31` | proposed |
| Simplify the configuration (owner TODO). | `TODO.md` | proposed |
| Installer: check that the main-flow pip steps no longer print a timed ✓ before the final "installed" line. Update the README status rows. | installer, `README.md` | check first |

### Robot environment (not code)

- The daemon 1.13.0 can get stuck in the `stopping` state. Only a reboot clears it.
- The robot's Wi-Fi is weak: -66 to -82 dBm on a DFS 5 GHz channel. Move the robot closer to the access point, or use 2.4 GHz.
