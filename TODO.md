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
      `test_app_llm_traits.py`. The `NetworkMonitor` follows hot-reloaded routes (`set_routes`, called from
      the app's `config.changed` handler; `test_health.py`).
- [x] Per-route history trimming (SPEC 8, Conversation state). Done: `llm_routes[].context_budget_tokens`
      (8192) and `conversation.turn_reserve_tokens` (256); `trim_history` drops whole turns,
      oldest first, and the router applies it per route; load-time rejection names the route
      (`roboemot.llm.prompt`, `test_llm_history.py`). The conversation worker (M3 slice 5) builds the history and applies the `conversation.history_turns` cap.
- [ ] Secret store for `${secret:name}` route keys (SPEC 8: stored encrypted on the robot). The
      app's `make_router` has no resolver, so such routes send an empty key; the app logs a
      warning at startup for each one.
- [x] Scan soon after a person leaves (SPEC 5A, Scanning). Done: `world.scan.target_lost_delay_s`
      (8 s) with its own `target_lost_jitter_s` (3 s), scaled by `idle_period_x`; the policy uses
      it instead of the idle interval when the person target or speaker hint goes away. A person or
      speaker engaging again cancels it, and it never runs while resting or in quiet hours
      (`ScanPolicy`, `test_scan.py`).
- [x] Pseudonyms to cloud routes (SPEC 8, 12 Data): `privacy.pseudonyms_to_cloud` loads; the router
      gives aliases to routes that are not local (`llm_routes[].locality`, making a route local needs the admin; `auto` is this machine and the home network)
      and restores names in the reply text and in tool-call arguments. Open: the aliases are handed
      out afresh after an app restart; personal details that are not registry names are not replaced;
      registry names refused for aliasing (no letter or digit, a template prefix word, a structural
      word such as "user") are sent unaliased.
- [ ] Pseudonyms for the persona trait rating (SPEC 8, 12 Data): `rate_traits`
      (`persona/llm_traits.py`) streams `persona.prompt` without `pseudonyms=`, so a registry name
      written into the persona prompt reaches a cloud route as typed.
- [ ] Pseudonyms in tool payload keys (SPEC 8, 12 Data): a JSON key in a tool call or result that
      equals a registry name ("target" for a record "Target") is aliased in the history sent to a
      cloud route. Nothing leaks, but the model sees a renamed key; alias JSON values only and let
      the leak check refuse a name in a key.

## Barge-in and skills (M3 slice 6, SPEC 8 stage 8, 11)

- [x] A stop phrase that stopped the robot is not answered afterwards: `conversation.stop_drop_window_s`
      drops an utterance made only of stop phrases heard soon after a barge-in. A stop phrase said
      later, with nobody speaking, is still an ordinary utterance.
- [ ] Barge-in tuning and checks on the robot (real playback, XMOS AEC, `clear_player`, CPU of the
      second Silero and the FFT, `pre_roll_ms`): SPEC 15 "Barge-in on the robot". The real-model
      tests (`pytest -m models -k spoken_stop`, `test_m3_exit.py`) use `say` clips; the onset-to-stop
      latency on the robot is in the voice debug section (`bargein.latency`).
- [x] `look_at` names come from the registry display names of what is in view and are sent to the
      route as tool-schema choices, like the scene summary; with `privacy.pseudonyms_to_cloud` the
      cloud route gets aliases in the enum and the executor gets the real name back.
- [x] `play` offers every label of the manifest's `expression_map` plus the named gestures of its
      `gesture_map` (nod, shake): same renderer, register, persona, minor cap and limiter (SPEC 7, 11).
- [ ] Fast path: a nod on a spoken "yes"/"sì" and a shake on "no" (the `speech.keyword.yes/no` events
      exist; needs an app handler that calls `ExpressionRenderer.request`, and a rule for when the
      robot is the one asking). Nod and shake signs and amplitudes are untested on the hardware.
- [ ] Edit wake phrases and the robot name from the web UI (needs an admin-confirmed settings editor,
      SPEC 11A).
- [ ] Edit the children's built-in lines (`safety.minor_lines`) and the minor blocklist from the web UI
      (same editor). Until then they keep their defaults (owner decision, 2026-10-18).

## M3 exit (SPEC 14, `docs/eval/m3.md`)

- [x] Exit test `packages/roboemot-reachymini/tests/test_m3_exit.py` (`pytest -m models`): Italian and
      English through the real app on a scripted client, the three budgets, a barge-in and the child case.
- [ ] The robot, owner and licence items of the M3 report are in SPEC 15 (exit on the robot, barge-in,
      CM4 load, Piper on the robot and its voice licences, wake phrases, OpenAI speech in Italian,
      children's lines). Not repeated here.
