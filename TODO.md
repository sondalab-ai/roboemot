# TODO

## Persona wiring (SPEC 8, 11A), with the conversation pipeline (M3)

The persona pieces exist and are tested; these callers wait for the pillars they plug into.

- [x] Build each LLM request with `roboemot.llm.system_prompt` + `build_messages`, using the
      register shaped by `roboemot.persona.apply_to_register` and the summary caps from
      `conversation.*_summary_max_chars`.
- [x] One app-wide `Router` (and daily token budget) shared by the traits worker, the conversation
      pipeline and the web UI route check (SPEC 8): today `start_webui` borrows the traits
      worker's router (`RoboEmotApp.start_webui`), done in M3 slice 5: `RouterHub`, shared by traits, conversation, speech and checks.
- [x] Pass `roboemot.persona.effects(cfg.persona)` to `compose_delivery` in the TTS path (`SpeechOutput.say`, M3 slice 4).
- [x] Apply `motion_amplitude` / `reaction_intensity` from the persona-shaped register in the
      expression path. Done in M2: `RegisterTracker.tick` shapes the register with
      `apply_to_register`, `ExpressionRenderer` scales by it every tick (SPEC 7, Expression map;
      `test_expression.py`, `test_app_expression.py`, and the M2 exit test). The reflex path is
      deliberately left out: reflexes do not read the register or the persona (SPEC 10).
- [ ] Scale `motion.idle.gaze_shift_interval_s` by `idle_period_x` once gaze shifts run (only
      breathing and the scan interval use it today).
- [x] Create the app's `Router` and run `apply_llm_traits` at startup and on every
      `config.changed` (trait source `llm`). Done: `TraitsWorker` (`roboemot.persona.llm_traits`)
      runs it on its own thread and event loop, woken by `persona.*` or `llm_routes` changes
      (coalesced, its own derived-layer change ignored, router rebuilt with its token counts
      kept, shutdown cancels a request in flight). `RoboEmotApp` creates it with the shared
      `NetworkMonitor` (`make_router` is overridable). The `TraitCache` is
      `<data folder>/llm_traits.json` (`paths.traits_cache_path`), removed by uninstall with the
      folder when the installer created it (SPEC 4A, Installer; `test_install_wizard.py`). Tests: `test_traits_worker.py`,
      `test_app_llm_traits.py`. Open: the `NetworkMonitor` still watches the routes it was built
      with, so a route added by hot reload is not probed until restart.
- [x] Per-route history trimming (SPEC 8, Conversation state). Done: `llm_routes[].context_budget_tokens`
      (8192) and `conversation.turn_reserve_tokens` (256); `trim_history` drops whole turns,
      oldest first, and the router applies it per route; load-time rejection names the route
      (`roboemot.llm.prompt`, `test_llm_history.py`). The conversation worker (M3 slice 5) builds the history and applies the `conversation.history_turns` cap.
- [ ] Secret store for `${secret:name}` route keys (SPEC 8: stored encrypted on the robot). The
      app's `make_router` has no resolver, so such routes send an empty key; the app logs a
      warning at startup for each one.
- [ ] Scan soon after a person leaves (SPEC 5A, Scanning): when the person target is lost, start a
      sweep after a new `world.scan.target_lost_delay_s` (N s, jittered, scaled like the idle
      interval) instead of waiting the full `world.scan.idle_interval_s` (120 s). Same rules as
      the idle scan: a person target or speaker hint cancels it, not while resting.
- [ ] Pseudonyms to cloud routes (SPEC 12, Data): `privacy.pseudonyms_to_cloud` is refused by the
      config validator ("not implemented in v1 M3") because requests still carry real names;
      build the name-to-alias map for `build_messages(names=...)`, send aliases to routes that
      are not local, and restore the web UI consent text for it.

## Barge-in and skills (M3 slice 6, SPEC 8 stage 8, 11)

- [ ] Tune `voice.barge_in.*` on the robot (SPEC 15 robot items): `echo_corr_max`, `echo_max_lag_ms`
      (speaker-to-microphone delay of the Wireless), `threshold`, `min_speech_ms`. The unit tests use
      synthetic echo; the real-model test (`pytest -m models -k spoken_stop`) uses `say` clips.
      Measure the onset-to-stop latency from the voice debug section (`bargein.latency`).
- [x] A stop phrase that stopped the robot is not answered afterwards: `conversation.stop_drop_window_s`
      drops an utterance made only of stop phrases heard soon after a barge-in. A stop phrase said
      later, with nobody speaking, is still an ordinary utterance.
- [ ] Robot checks for barge-in (HIL, real playback): the Wireless's capture/playback lag against
      `voice.barge_in.echo_max_lag_ms`; whether `echo_corr_max` separates the robot's voice from a
      person talking over it on the real speaker and room (the unit tests use synthetic and smeared
      echo; a user quieter than the echo is not heard by design); the cm4 CPU of the second Silero
      instance, the FFT per speech frame and the keyword Vosk (`bargein.keyword_seconds` is the
      proxy); the robot saying "stop" itself in its own answer; capture and `clear_player` latency,
      which the audio-clock latency figure leaves out.
- [ ] The pre-roll handed to the listener on a barge-in (`pre_roll_ms`, 64 ms) still holds a little of
      the robot's own voice; with real AEC it could be raised.
- [ ] `look_at` names come from the registry display names of what is in view and are sent to the
      route as tool-schema choices, like the scene summary; route them through the pseudonym map when
      `privacy.pseudonyms_to_cloud` exists.
- [ ] `play` offers every label of the manifest's `expression_map` (affect expressions). Named
      gestures that are not affect labels (nod, shake) need their own map and primitives.
