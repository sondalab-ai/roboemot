# Body prompt (design, 2026-11-16)

## Intent

Every LLM that speaks as the robot (chat, realtime, Pollen, Gemini Live) must know the robot's
physical body as well as its character. The body is per product (Reachy Mini has antennas, other
robots do not). Core stays robot-agnostic. The owner sets the text from the web UI. Defaults stay
short to save tokens.

## Current state

- `persona.prompt` (admin-only, web UI textarea, Settings > Persona) is part 1 of the system
  prompt for every conversation route (`llm/prompt.py::_prompt`).
- No text about the body reaches the LLM.

## Approaches

1. **Chosen: config field with a per-robot default from the manifest.** `persona.body_prompt`;
   the manifest sets it under `behaviour:` (config layer 2); the owner overrides it in the web UI.
   Reuses layering, the admin guard and the web UI Reset.
2. Manifest-only, read-only in the web UI: the owner cannot correct or extend it.
3. One merged persona field seeded from the manifest: the per-robot default is lost after the
   first edit.

## Design

- Schema: `persona.body_prompt: str = ""` (`admin_only`), `persona.body_prompt_max_chars`.
- Manifest allowlist: `behaviour.persona.body_prompt` only.
- Manifests: a short body text per robot, written from the manifest's own pose groups, sensors and
  expressive channels.
- Prompt: part 1b, after `persona.prompt`, inside `_prompt`, so every conversation route and the
  route check get it.
- Not in: the trait rating (rates the persona text, does not speak as the robot) and the voice
  preview (a fixed line).
- Budget: `system_tokens_worst_case` counts the body text at `body_prompt_max_chars`.
- Web UI: Reset restores the manifest text; an empty value sends no body text.
- SPEC: 8 (prompt parts, free-text list), 11A (Persona, Guardrails), 4 (manifest behaviour keys).

## Non-goals

- No change to `DEFAULT_PERSONA_PROMPT`.
- No per-language body text (the reply-language rule covers output language).
