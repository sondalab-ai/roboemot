# TODO

## Persona wiring (SPEC 8, 11A), with the conversation pipeline (M3)

The persona pieces exist and are tested; these callers wait for the pillars they plug into.

- [ ] Build each LLM request with `roboemot.llm.system_prompt` + `build_messages`, using the
      register shaped by `roboemot.persona.apply_to_register` and the summary caps from
      `conversation.*_summary_max_chars`.
- [ ] Pass `roboemot.persona.effects(cfg.persona)` to `compose_delivery` in the TTS path.
- [x] Apply `motion_amplitude` / `reaction_intensity` from the persona-shaped register in the
      expression path. Done in M2: `RegisterTracker.tick` shapes the register with
      `apply_to_register`, `ExpressionRenderer` scales by it every tick (SPEC 7, Expression map;
      `test_expression.py`, `test_app_expression.py`, and the M2 exit test). The reflex path is
      deliberately left out: reflexes do not read the register or the persona (SPEC 10).
- [ ] Scale `motion.idle.gaze_shift_interval_s` by `idle_period_x` once gaze shifts run (only
      breathing and the scan interval use it today).
- [ ] Create the app's `Router` and run `apply_llm_traits` at startup and on every
      `config.changed` (trait source `llm`). Put the `TraitCache` file where the installer's
      uninstall removes it (SPEC 4A, Installer: leave the robot clean).
- [ ] Per-route history trimming (SPEC 8, Conversation state): drop oldest turns only; add a
      per-route context budget and the load-time check that system prompt parts 1 to 4 fit it.
