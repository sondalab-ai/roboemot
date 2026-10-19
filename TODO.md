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
- [x] Secret store for `${secret:name}` route keys (SPEC 8 Secrets, 12 Data). Done: `secrets.json` in the
      data folder (0600, atomic writes, refused when group/other accessible or not owned by the user;
      `roboemot.secret_store`), `roboemot secrets {set,remove,list}` (value from stdin or a no-echo
      prompt), installer `--secret NAME` (recorded by name for uninstall). The router resolves keys at
      build (startup and each config reload, `RouterHub.sync` rebuilds when a secret changed); a route
      whose secret cannot be used is unavailable with one warning, and the CLI check and the web UI
      check refuse it. Keys are in no log, debug report or web UI response (`test_secret_leak.py`).
      Intended (owner decisions 2026-10-18): the web UI never sets or rotates a key (CLI and installer
      only; no TLS until M5); no encryption at rest (file modes only). Verified: a key rotated on disk
      is picked up live, when the next utterance is answered or the traits worker syncs
      (`secrets.json` is stat'ed each time): `RouterHub.sync` rebuilds the router, budgets kept, and
      the next request carries the new key (`test_secret_store.py`).
- [x] Scan soon after a person leaves (SPEC 5A, Scanning). Done: `world.scan.target_lost_delay_s`
      (8 s) with its own `target_lost_jitter_s` (3 s), scaled by `idle_period_x`; the policy uses
      it instead of the idle interval when the person target or speaker hint goes away. A person or
      speaker engaging again cancels it, and it never runs while resting or in quiet hours
      (`ScanPolicy`, `test_scan.py`).
- [x] Pseudonyms to cloud routes (SPEC 8, 12 Data): `privacy.pseudonyms_to_cloud` loads; the router
      gives aliases to routes that are not local (`llm_routes[].locality`, making a route local needs the admin; `auto` is this machine and the home network)
      and restores names in the reply text and in tool-call arguments. Intended (owner decision
      2026-10-18): aliases live in RAM only and are new after an app restart. Open: personal details
      that are not registry names are not replaced; registry names refused for aliasing (no letter or digit, a template prefix word, a structural
      word such as "user") are sent unaliased.
- [x] Pseudonyms residual: `persona/llm_traits.py` (`rate_traits`) streamed `persona.prompt`
      without `pseudonyms=`. `TraitsWorker` now takes a `NameSource` (registry names) and the
      rating goes through the same aliasing, leak check and restore as the conversation; names
      that cannot be read mean no request.
- [x] Pseudonyms residual: `llm/conformance.py` sent through `stream_route`, which bypassed the
      pseudonyms. `stream_route(..., pseudonyms=)` shares the router's guard (a leak raises
      `PseudonymLeak` before sending) and the web UI route check passes the map. Open:
      `roboemot llm check` (CLI) does not open the registry, so it still sends its synthetic
      prompts without pseudonyms.
- [x] Pseudonyms residual: registry names refused for aliasing are sent unaliased and the admin
      saw only their count in the log. The Routes tab now lists them on request
      (`POST /api/pseudonyms/refused`, session, CSRF and the operator PIN re-entered); the log
      still gives only the count. Open: the list has names, not record ids, so renaming is done
      from the Registry tab.
- [x] Pseudonyms residual: a JSON key in a tool call or result that equals a registry name
      ("target" for a record "Target") was aliased in the history sent to a cloud route. Now only
      string values are aliased. The leak check skips only the keys the app writes itself (the
      skills' parameter names and result keys, `SkillRegistry.structural_keys()`), in tool call
      arguments and `tool` messages; any other key that holds a name (a tool result keyed by
      names, the result key of a custom skill) blocks the route instead of being renamed.
- [ ] Pseudonyms follow-up: `NameSource` (traits, route checks) and the conversation's own
      `Pseudonyms` are two maps with different aliases; merge them into one source.
- [x] Pseudonyms residual: a name written with `\uXXXX` escapes in text that is not JSON was
      neither scrubbed nor caught by the leak check. Escapes now count as their character in
      matching (surrogate escapes are not decoded).

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
- [x] Fast path: a nod on a spoken "yes" and a shake on "no" (`speech.keyword.yes/no` through
      `ExpressionRenderer.request`, SPEC 7, 8): final text only, at most
      `voice.fastpath.gesture_max_words` words, exactly one of yes / no, never while the robot speaks,
      silent within `voice.fastpath.answer_window_s` of a question of its own (`SpeechOutput.question_end`). Italian `sì`
      is not a default yes phrase (accents are folded, `si` is a pronoun). Nod and shake signs and
      amplitudes are untested on the hardware.
- [x] Settings API (SPEC 11A, epic settings-secrets slice 4): superseded by the full mirror below.
- [x] Web UI settings mirror (SPEC 11A, 12; epic webui-settings-mirror): every config field is listed and
      editable (`config/catalogue.py`, `webui/settings.py`, `webui/static/settings.js`), core then robot
      section, one PIN per write, safety loosening to the app YAML, entries added and removed (Remove only
      where the server accepts it, `removable`), minors blocklist words removable behind the PIN, language
      keys limited to the voice models, a keyed route may change endpoint after a PIN plus an
      acknowledged host, route URLs with credentials are write-only, private fields behind a PIN reveal,
      Restart RoboEmot (no reboot). Exit tests: `test_webui_exit.py` (volume heard on the next utterance,
      unit level because no simulator harness covers speech; restart flag), `test_secret_leak.py`
      (no secret in any `/api/settings` response), JS tests (restart badge).
- [ ] Nod and shake (yes/no gestures) on the hardware: signs and amplitudes are still untested, see above.
- [ ] Restart RoboEmot verified on the real daemon: `POST /api/apps/restart-current-app` and
      `GET /api/apps/current-app-status` are taken from the Pollen sources (1.11.0, 1.12.0), not yet run
      on a robot; check the zero pose, the WebRTC drop and the kill after 20 s.

## M3 exit (SPEC 14, `docs/eval/m3.md`)

- [x] Exit test `packages/roboemot-reachymini/tests/test_m3_exit.py` (`pytest -m models`): Italian and
      English through the real app on a scripted client, the three budgets, a barge-in and the child case.
- [ ] The robot, owner and licence items of the M3 report are in SPEC 15 (exit on the robot, barge-in,
      CM4 load, Piper on the robot and its voice licences, wake phrases, OpenAI speech in Italian,
      children's lines). Not repeated here.

## CI and tests

- [x] `packages/roboemot-reachymini/tests/test_app_conversation.py::test_the_app_offers_the_default_skills_and_wires_barge_in`
      was timing-flaky under load: the listener thread fed quiet chunks into `BargeIn.on_chunk` between the
      test's loud ones. The test now stops the listener first, so one thread drives the hook.
- [x] `packages/roboemot-core/tests/test_llm_router.py::test_hostile_tool_arguments_do_not_fail_the_route_or_get_parsed`
      failed on Python 3.15 only: its JSON parser no longer raises `RecursionError` on deep input. Tool-call
      arguments are now kept only as an object within `TOOL_ARGS_MAX_DEPTH` (SPEC 8). CI runs 3.11 and 3.12 only.
- [x] JS behaviour tests for the web UI (`ask()`, the PIN dialog, the settings editor): `node:test` + jsdom in
      `packages/roboemot-core/tests/webui_js/` (dev-only, lockfile), CI job `webui-js`.
- [x] `Pseudonyms._aliased_json` / `_aliased` recursed on deep tool-message content: `_nesting` and
      `TOOL_ARGS_MAX_DEPTH` now live in `pseudonyms.py` (the router imports them); JSON nested deeper than the
      limit is aliased as plain text (SPEC 8, 12).
- [ ] `conversation.tool_arg_max_chars` and the router's `TOOL_ARGS_MAX_CHARS` are two caps on the same thing:
      decide one owner (SPEC 8, 11A).
- [ ] Confirm the robot's Python version (`python scripts/robot_ssh.py -- python3 --version`). The Piper hash locks cover
      aarch64 py3.11 and py3.12 until then; drop the unused one (SPEC 4A Installer).
- [ ] Confirm on the robot that the app runs as the same user the installer writes `secrets.json` as over SSH;
      otherwise the uid check makes secret routes unavailable (SPEC 8 Secrets, 12 Data).
- [x] Runtime test that a hostile `base_url` redirect (307 to another host) reaches nothing: `test_redirect_auth.py`.
      The openai SDK client followed redirects (the prompt went to the target, without the key); LLM routes
      now do not follow them, like the speech client (SPEC 8, 12).
