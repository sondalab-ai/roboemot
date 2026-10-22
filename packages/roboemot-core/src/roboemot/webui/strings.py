"""Page strings in Italian and English; the page follows ``conversation.language`` (SPEC 8)."""

from __future__ import annotations

DEFAULT_LANGUAGE = "en"

EN: dict[str, str] = {
    "no_pin": "No operator PIN is set. Set the PIN with the installer or the CLI.",
    "err_busy": "Busy checking another PIN. Try again in a moment.",
    "err_consent_who_guardian": "For a minor, the person consenting is the parent or guardian.",
    "err_guardian_is_subject": "The guardian cannot be the person being enrolled.",
    "err_bad_host": "Host not accepted.",
    "title": "RoboEmot",
    "dialog_cancel": "Cancel",
    "dialog_confirm": "Confirm",
    "dialog_retry_in": "Try again in {n} s.",
    "confirm_merge": "Merge {drop} into {keep}? This cannot be undone.",
    "theme": "Theme",
    "theme_auto": "Auto",
    "theme_light": "Light",
    "theme_dark": "Dark",
    "theme_high_contrast": "Contrast",
    "loading": "Loading...",
    "login_title": "Operator login",
    "pin": "PIN",
    "login": "Log in",
    "logout": "Log out",
    "tab_enroll": "Enrollment",
    "tab_registry": "Registry",
    "tab_catalogue": "Catalogue",
    "tab_debug": "Diagnostics",
    "tab_live": "Live",
    "live_title": "Inner state",
    "live_intro": (
        "What the robot is doing right now. This tab updates while it is open and in "
        "view; nothing is stored."
    ),
    "live_disabled_title": "The Live tab is off",
    "live_disabled_body": (
        "It shows who the robot sees, so it is off by default. To turn it on, open "
        "Settings, find webui.live.enabled and switch it on. It applies at once, with "
        "no restart."
    ),
    "live_mood": "Mood",
    "live_valence": "Valence",
    "live_arousal": "Arousal",
    "live_intensity": "Intensity",
    "live_energy": "Energy",
    "live_circ_awake": "awake",
    "live_circ_drowsy": "drowsy",
    "live_circ_resting": "resting",
    "live_circ_off": "Circadian rhythm is off.",
    "live_modes": "Modes",
    "live_awake": "Awake",
    "live_asleep": "Asleep",
    "live_online": "Online",
    "live_offline": "Offline",
    "live_minor_yes": "Possible minor in view",
    "live_minor_no": "No minor in view",
    "live_paused": "Paused while asleep",
    "live_paused_none": "Nothing paused",
    "live_part_app_tick": "behaviour",
    "live_part_vision": "vision",
    "live_part_light_reader": "light reader",
    "live_part_listener": "listening",
    "live_register": "Register",
    "live_persona": "Personality",
    "live_conversation": "Conversation",
    "live_phase_idle": "Idle",
    "live_phase_listening": "Listening",
    "live_phase_thinking": "Thinking",
    "live_phase_speaking": "Speaking",
    "live_route": "Route",
    "live_route_pending": "connecting",
    "live_elapsed": "Waiting for {n} s",
    "live_attention": "In view",
    "live_people": "People: {n}",
    "live_nobody": "Nobody in view.",
    "live_unknown": "Unknown",
    "live_gaze": "gaze target",
    "live_more": "and {n} more",
    "live_reflexes": "Reflexes",
    "live_reflex_last": "Last: {name}, {n} s ago",
    "live_reflex_none": "None fired yet.",
    "live_motion": "Motion",
    "live_motion_writing": "Writing poses",
    "live_motion_external": "External control",
    "live_motion_motors_off": "Motors off",
    "live_system": "System",
    "live_cpu_temp": "CPU temperature: {n} °C",
    "live_cpu_temp_none": "CPU temperature: not available",
    "live_levels": "Audio levels",
    "live_levels_note": "Levels only: no sound leaves the robot and none is kept.",
    "live_mic": "Microphone",
    "live_out": "Robot voice",
    "live_db": "{n} dBFS",
    "live_floor": "Noise floor: {n} dBFS",
    "live_level_paused": "paused while asleep",
    "live_level_no_data": "no signal",
    "live_level_unavailable": "not available",
    "live_transcript": "Conversation",
    "live_transcript_note": (
        "Kept in the robot's memory only, never saved. Names appear as the conversation has them."
    ),
    "live_transcript_off": (
        "The transcript is off, so nothing is kept. To turn it on, open Settings, find "
        "webui.live.transcript and switch it on. It applies at once; switching it off "
        "forgets what was kept."
    ),
    "live_transcript_empty": "Nothing said yet.",
    "live_transcript_user": "Heard",
    "live_transcript_robot": "Said",
    "live_transcript_cut": "cut off",
    "live_transcript_clear": "Clear",
    "err_live_disabled": "The Live tab is off.",
    "err_live_unavailable": "The Live tab is not available on this robot.",
    "live_video": "Camera",
    "live_video_alt": "Live camera picture",
    "live_video_pause": "Pause",
    "live_video_resume": "Resume",
    "live_video_paused": "Paused.",
    "live_video_note": (
        "A still picture refreshed a few times a second, with face boxes and the names of "
        "people the robot knows. Shown only on this page, never saved. The robot does not "
        "wake or move for it."
    ),
    "live_video_off": (
        "The camera preview is off. To turn it on, open Settings, find webui.live.video and "
        "switch it on. It applies at once. While anyone watches it, every open page shows a "
        "banner."
    ),
    "live_video_banner": "Live video view active",
    "err_video_off": "The camera preview is off.",
    "err_video_asleep": "The robot is asleep, so the camera is off.",
    "err_video_no_vision": "Vision is not running on this robot, so there is no picture.",
    "err_video_no_frame": "Waiting for the next camera picture.",
    "err_video_stale": "The camera has not sent a new picture for a while.",
    "err_cross_site": "The camera picture is only served to this page.",
    "live_video_idle": (
        "The preview stopped after a while without activity on this page. Press Resume."
    ),
    "tab_routes": "Routes",
    "tab_settings": "Settings",
    "settings_intro": (
        "Every configuration field, with its current value and the layer that decides it. Each "
        "change asks for the operator PIN and applies at once. A change that loosens a safety "
        "rule is saved to the app configuration file instead and survives a restart."
    ),
    "settings_core": "Core",
    "settings_robot": "Robot",
    "settings_nav": "Setting groups",
    "settings_search": "Search",
    "settings_search_hint": "By path or description.",
    "settings_changed_only": "Changed only",
    "settings_matches": "{n} settings shown",
    "settings_source_runtime": "Runtime",
    "settings_source_derived": "Derived",
    "settings_source_app": "App file",
    "settings_source_preset": "Preset",
    "settings_source_manifest": "Robot manifest",
    "settings_source_defaults": "Defaults",
    "settings_source_schema": "Built-in",
    "settings_restart": "Restart RoboEmot",
    "settings_restart_hint": 'Settings marked "Restart needed" apply after RoboEmot restarts.',
    "settings_restart_pending": "Saved changes are waiting for a restart.",
    "settings_restart_title": "Restart RoboEmot?",
    "settings_restart_message": (
        "RoboEmot stops and starts again. This page is unavailable for a moment and you will "
        "have to sign in again. What happens:"
    ),
    "settings_restart_confirm": "Restart",
    "settings_restart_started": (
        "Restarting. Wait a few seconds, then reload the page and sign in again."
    ),
    "settings_restart_effect_zero_pose": (
        "The robot goes to its zero pose once the app has stopped."
    ),
    "settings_restart_effect_remote_session_drop": "A remote video or audio session may drop.",
    "settings_restart_effect_kill_on_timeout": (
        "If RoboEmot has not stopped after {timeout_s} s, it is killed."
    ),
    "err_restart_unavailable": "RoboEmot cannot be restarted from here right now.",
    "settings_restart_badge": "Restart needed",
    "settings_restart_goto": "Go to Restart RoboEmot",
    "settings_restart_needed": "Restart RoboEmot for it to take effect.",
    "settings_not_measured": "not measured",
    "settings_show_all": "Show all {n}",
    "settings_private": "Private",
    "settings_secret": "Secret",
    "settings_readonly": "Read-only",
    "settings_json_readonly": "Structured value: edit it in the configuration file.",
    "settings_superset_hint": (
        "Removing a word loosens the child-safe filter: it is saved to the app config and needs "
        "the PIN."
    ),
    "settings_null": "No value (null)",
    "settings_route_move_message": (
        "If this route has an API key, saving sends that key to {host}. Enter the PIN to allow it."
    ),
    "settings_write_only_hint": (
        "The current address holds credentials and is not shown. Type a whole new address to "
        "replace it; it may not hold credentials."
    ),
    "settings_slider": "{path} slider",
    "settings_tuple_item": "Value {n} of {total}",
    "settings_new_item": "New entry",
    "settings_remove_item": "Remove {item}",
    "settings_remove_n": "Remove row {n}",
    "settings_dict_key": "Key",
    "settings_dict_value": "Value",
    "settings_add_entry": "Add entry",
    "settings_entry_name": "Name",
    "settings_remove_entry": "Remove entry",
    "settings_no_entries": "No entries.",
    "settings_reveal_message": (
        "Show the private settings (minors, privacy, location). They stay in this page until you "
        "log out or reload."
    ),
    "settings_reset_layer": "Reset to lower layer",
    "settings_reset_done": "Reset.",
    "settings_entry_added": "Entry added.",
    "settings_entry_removed": "Entry removed.",
    "settings_layer_runtime": "Applied now.",
    "settings_layer_app": (
        "Saved permanently in the app configuration file: a safety setting was loosened, and it "
        "survives a restart."
    ),
    "err_key_duplicate": "Two rows use the same key.",
    "settings_default": "Default",
    "settings_save": "Save",
    "settings_add": "Add entry",
    "settings_remove": "Remove",
    "settings_admin": "Admin",
    "settings_reveal": "Show with PIN",
    "settings_hidden": "Hidden: minors, privacy and location settings need the PIN to be shown.",
    "settings_admin_hint": "Safety-relevant. The operator PIN doubles as the admin PIN.",
    "settings_pin_title": "Confirm the change",
    "settings_not_persisted": "It is live but was not saved to disk: it ends at the next restart.",
    "settings_unchanged": "Nothing changed.",
    "settings_none": "No setting can be changed from the web UI.",
    "err_routes_unavailable": "Routes are not available.",
    "err_route_unknown": "No such route.",
    "err_route_busy": "Another check is running. Try again in a moment.",
    "err_route_budget": "The daily token budget of the route is spent.",
    "err_route_secret": "The route key is not available. Set it on the robot: roboemot secrets set",
    "route_error_secret": "The key of the route is not available.",
    "route_secret_set": "key {name}: set",
    "route_secret_missing": "key {name}: not set",
    "route_error_timeout": "The check took too long and was stopped.",
    "route_error_budget": "The daily token budget ran out during the check.",
    "route_error_failed": "The check could not be completed.",
    "route_running": "Check running...",
    "routes_intro": (
        "Conformance check of each LLM route: streaming, tool call, first-token latency and "
        "Italian. It is advisory: it does not enable or block a route, and it uses tokens from "
        "the daily budget of the route."
    ),
    "routes_none": "No LLM route is configured.",
    "route_add_title": "Add a route",
    "route_add_intro": (
        "A route is an OpenAI-compatible model the robot can talk to. Pick its key from the "
        "stored keys below; a key is never typed here. Changes ask for the operator PIN and apply "
        "at once."
    ),
    "route_name": "Name",
    "route_url": "Base URL",
    "route_url_keep": "Leave empty to keep it",
    "route_url_placeholder": "https://api.example.com/v1",
    "route_url_hint_edit": "The stored address is not shown if it holds credentials.",
    "route_model": "Model",
    "route_locality": "Where the model runs",
    "route_locality_auto": "Automatic",
    "route_locality_local": "Local (home network)",
    "route_locality_cloud": "Cloud",
    "route_locality_hint": (
        "A route on this machine or the home network sees real names. Setting one up needs the "
        "configuration file; from here only cloud routes are accepted."
    ),
    "route_key": "Key",
    "route_key_none": "No key",
    "route_key_literal": "A key written in the configuration file",
    "route_key_hint": "Only stored keys can be chosen. Add one in the API keys card below.",
    "route_key_message": "The key will be sent to {host}. Enter the PIN to allow it.",
    "route_key_insecure": (
        "This address is not encrypted (http): the key will cross the network in clear."
    ),
    "route_add": "Add route",
    "route_save": "Save route",
    "route_edit": "Edit",
    "route_remove": "Remove",
    "route_remove_title": "Remove route {name}?",
    "route_remove_message": "The route is deleted. Its key stays in the store.",
    "route_added": "Route {name} added.",
    "route_saved": "Route {name} saved.",
    "route_removed": "Route {name} removed.",
    "secrets_title": "API keys",
    "secrets_intro": (
        "API keys are stored on the robot and can only be written here, never read back: the "
        "page shows the name of a key and whether a route uses it, not its value. A route "
        "refers to a key by its name."
    ),
    "secrets_none": "No key is stored yet.",
    "secret_name": "Name",
    "secret_name_hint": (
        "Letters, digits, dot, dash or underscore; it starts with a letter or digit."
    ),
    "secret_value": "Key",
    "secret_value_hint": "Pasted once and cleared on save. The page never shows it again.",
    "secret_save": "Save key",
    "secret_saved": "Key {name} saved.",
    "secret_removed": "Key {name} removed.",
    "secret_remove": "Remove",
    "secret_remove_title": "Remove key {name}?",
    "secret_remove_message": "The key is deleted from the robot. This cannot be undone.",
    "secret_used_by": "used by {routes}",
    "secret_unused": "not used by a route",
    "secret_set": "set",
    "secret_unset": "not set",
    "secret_in_use_hint": "Choose another key or no key on those routes first.",
    "err_secrets_unavailable": "API keys are not available.",
    "err_secret_name_invalid": "The key name is not valid.",
    "err_secret_value_invalid": (
        "The key must be visible ASCII characters without spaces, and not too long."
    ),
    "err_secret_in_use": "A route uses this key. Choose another key on that route first.",
    "err_secret_unknown": "No such key.",
    "err_secret_store": "The key store cannot be used. Check its file permissions on the robot.",
    "err_pseudonyms_unavailable": "The list of names is not available.",
    "refused_title": "Names not replaced by aliases",
    "refused_hint": (
        "Registry names that cannot be replaced by an alias (no letter or digit, a word of an "
        "alias template, a word of the request format, a backslash) reach cloud routes as "
        "typed. Rename them to keep them private. Enter the admin PIN to see the list."
    ),
    "refused_show": "Show names",
    "refused_none": "No name is sent as typed.",
    "refused_off": "Pseudonyms for cloud routes are off.",
    "route_check": "Check",
    "route_checking": "Checking...",
    "route_never": "Not checked yet",
    "route_ok": "All checks passed",
    "route_failed": "Some checks failed",
    "route_at": "Checked {at}",
    "check_streaming": "Streaming",
    "check_tool_round_trip": "Tool call round trip",
    "check_first_token_latency": "First-token latency",
    "check_italian": "Italian output",
    "status_pass": "pass",
    "status_fail": "fail",
    "status_skip": "skipped",
    "enroll_intro": "Ask the person to look at the robot. No images are shown or stored here.",
    "enroll_start": "Start enrollment",
    "enroll_cancel": "Cancel",
    "enroll_complete": "Save enrollment",
    "frames": "Frames collected",
    "state": "State",
    "state_idle": "Idle",
    "state_collecting": "Collecting",
    "state_ready": "Ready for consent",
    "state_completed": "Done",
    "state_cancelled": "Cancelled",
    "state_timed_out": "Timed out",
    "state_completing": "Saving",
    "state_unavailable": "Vision is not running",
    "hint_no_face": "No face in view",
    "hint_too_far": "Move closer",
    "hint_too_close": "Move back a little",
    "hint_too_dark": "Too dark",
    "hint_multiple_faces": "Only one person at a time",
    "hint_look_at_robot": "Look at the robot",
    "hint_hold_still": "Hold still",
    "hint_turn_head": "Turn your head slightly left and right",
    "display_name": "Display name",
    "consent_who": "Name of the person who consents",
    "guardian": "Parent or guardian name",
    "guardian_hint": "Required unless the person is declared adult or senior.",
    "age_group": "Age group",
    "gender": "Gender",
    "age_unknown": "Not declared",
    "age_child": "Child",
    "age_teen": "Teenager",
    "age_adult": "Adult",
    "age_senior": "Senior",
    "gender_unknown": "Not declared",
    "gender_female": "Female",
    "gender_male": "Male",
    "scope_title": "Consent covers",
    "scope_face": "Face recognition",
    "scope_voice": "Speaker recognition (voice)",
    "scope_refresh": "Adding new samples over time",
    "agree": "The person (or guardian) has read the text above and agrees.",
    "consent_title": "Consent",
    "consent_biometric": (
        "The robot stores face (and, if chosen, voice) embeddings of this person to recognise "
        "them. These are biometric data, a special category under GDPR Art. 9. Camera images "
        "are discarded after enrollment. For a minor, a parent or guardian must consent. The "
        "person can ask to see or erase their data at any time."
    ),
    "consent_cloud_names": (
        "Names, preferences and conversation text may be sent to the cloud language-model "
        "provider the robot uses, which processes them as personal data. The robot sends real "
        "names to such routes."
    ),
    "consent_cloud_pseudonyms": (
        "Names, preferences and conversation text may be sent to the cloud language-model "
        "provider the robot uses, which processes them as personal data. On such routes the "
        "robot replaces the names in its registry with neutral aliases. Routes on this "
        "machine or the home network receive real names, so a gateway on the network that "
        "forwards to a cloud provider must be marked as cloud by the administrator. A cloud "
        "speech voice still hears the names it speaks. A nickname, a misspelt name or other "
        "personal details said aloud are not replaced, nor is a registry name the robot cannot "
        "alias (one without letters, a word of the alias template or a structural word such "
        'as "user"): it is sent as written.'
    ),
    "registry_title": "Registry",
    "registry_empty": "Nothing registered yet.",
    "kind_person": "Person",
    "kind_animal": "Animal",
    "kind_object": "Object",
    "status_pending": "pending",
    "status_active": "active",
    "last_seen": "Last seen",
    "never": "never",
    "rename": "Rename",
    "merge": "Merge into...",
    "merge_pick": "Merge into this one",
    "forget": "Forget (erase everything)",
    "delete": "Delete",
    "confirm_forget": "Erase all data of {name}? This cannot be undone.",
    "confirm_delete": "Delete {name}? This cannot be undone.",
    "confirm_purge": "Purge this unreadable item? This cannot be undone.",
    "quarantined_title": "Unreadable items",
    "quarantined_none": "None.",
    "purge": "Purge",
    "new_name": "New name",
    "catalogue_intro": (
        "Import XMP sidecars written by your photo tool. People, pets and objects "
        "become pending records; people stay name-only until they go through "
        "enrollment. A photo is optional, is read in memory and is never stored."
    ),
    "catalogue_xmp": "XMP sidecar (.xmp)",
    "catalogue_photo": "Its photo (JPEG or PNG, optional)",
    "catalogue_import": "Import",
    "catalogue_pending": "To review: pending and automatic records",
    "catalogue_none": "Nothing to review.",
    "catalogue_live": "Seen now",
    "catalogue_live_none": "The robot does not see any animal or object right now.",
    "catalogue_pick": "Pick what the robot sees to confirm a record.",
    "catalogue_confirm": "This is {name}",
    "catalogue_activate": "Activate from photo",
    "catalogue_report": "{created} created, {existing} already known.",
    "catalogue_conflicts": "Tags to fix",
    "catalogue_truncated": "The sidecar was longer than the limit; the rest was ignored.",
    "catalogue_clear": "Forget collected sightings",
    "catalogue_clear_confirm": (
        "Forget what the robot has collected about animals and objects it does not know yet?"
    ),
    "catalogue_cleared": "Forgotten: {n}.",
    "tombstones_clear": "Allow deleted imports to come back ({n})",
    "tombstones_clear_confirm": (
        "A new import will create the pending records you deleted before. Continue?"
    ),
    "tombstones_cleared": "Cleared: {n}.",
    "catalogue_read_failed": "The file could not be read.",
    "photo_used": "Photo used.",
    "photo_no_embedder": "Photo skipped: object recognition is not running.",
    "photo_dims_mismatch": "Photo skipped: its shape does not match the sidecar.",
    "origin_auto": "automatic",
    "origin_xmp": "from catalogue",
    "origin_enrolled": "enrolled",
    "has_appearance": "has appearance",
    "no_appearance": "no appearance yet",
    "conflict_category_conflict": "two different categories",
    "conflict_unknown_category": "category not recognised",
    "err_xmp_required": "Choose an XMP file.",
    "err_xmp_too_large": "The XMP file is too large.",
    "err_xmp_not_xml": "The file is not valid XMP.",
    "err_xmp_forbidden_xml": "The file uses XML features that are not accepted.",
    "err_image_format": "Only JPEG and PNG photos are accepted.",
    "err_image_corrupt": "The photo could not be read.",
    "err_image_too_large": "The photo is too large.",
    "err_image_too_many_pixels": "The photo has too many pixels.",
    "err_image_unsupported": "Photos cannot be read on this robot.",
    "err_not_pending": "This record is already active.",
    "err_entry_gone": "The robot no longer sees that. Pick it again.",
    "err_no_appearance": "There is no appearance to use yet. Pick something the robot sees.",
    "err_kind_mismatch": "That is not the same kind of thing.",
    "err_category_mismatch": "That is not the same category.",
    "err_catalogue_unavailable": "The catalogue is not available.",
    "err_import_busy": "Another import is running. Try again in a moment.",
    "err_pending_invalid": "That imported name is not available any more.",
    "photo_no_regions": "Photo not used: the sidecar has no animal or object regions.",
    "photo_partial": "Photo partly used: time ran out. Import the same files again to finish.",
    "pending_link": "Link to an imported name",
    "pending_none": "None: this is a new person",
    "debug_title": "Diagnostics report",
    "debug_refresh": "Refresh",
    "saved": "Saved.",
    "err_generic": "Something went wrong.",
    "err_unauthorized": "Please log in.",
    "err_csrf": "Session check failed, reload the page.",
    "err_bad_pin": "Wrong PIN.",
    "err_setting_not_allowed": "This setting cannot be changed from the web UI.",
    "err_setting_guardrail": "The change was refused: it would loosen a safety or privacy rule.",
    "err_setting_invalid": "The value is not valid for this setting.",
    "err_settings_unavailable": "Settings are not available.",
    "err_value_type": "The value has the wrong type.",
    "err_value_range": "The value is outside the allowed range.",
    "err_value_choice": "The value is not one of the allowed choices.",
    "err_text_invalid": "The text has characters that are not allowed.",
    "err_persist_failed": "The configuration file could not be written; nothing was changed.",
    "err_value_merged": "Entries can be added or changed here, not removed.",
    "err_text_too_long": "The text is too long.",
    "err_key_invalid": "The name can only use letters, digits, dash and underscore.",
    "err_entry_exists": "An entry with this name already exists.",
    "err_entry_invalid": "The entry has a field that cannot be set from the web UI.",
    "err_entry_in_lower_layer": (
        "This entry comes from the configuration file and cannot be removed here."
    ),
    "err_endpoint_ack_required": (
        "This route has an API key. Saving sends that key to {host}. Enter the PIN and press "
        "the button again to allow it."
    ),
    "err_url_credentials": (
        "The address must not hold a user name, password, token or ${secret:...} reference. "
        "Store the key with `roboemot secrets set NAME` and use ${secret:NAME} as the api_key."
    ),
    "err_language_unsupported": "The voice stack has no speech models for this language.",
    "err_not_resettable": (
        "This field belongs to an entry added here and cannot be reset on its own."
    ),
    "err_list_empty": "The list needs at least one entry.",
    "err_list_too_long": "The list has too many entries.",
    "err_phrase_too_long": "An entry is too long.",
    "err_phrase_invalid": "An entry has characters that are not allowed.",
    "err_line_empty": "This line cannot be empty.",
    "err_line_too_long": "The line is too long.",
    "err_line_invalid": "The line has characters or placeholders that are not allowed.",
    "err_line_blocked": "The line contains a word the child-safety word list blocks.",
    "err_name_blocked": "The name is blocked by the child-safety word list.",
    "err_locked": (
        "Too many attempts. Try again later, or restart the app or run set-pin on the robot."
    ),
    "err_pin_digits": "The PIN must be digits only.",
    "err_pin_too_short": "The PIN is too short.",
    "err_pin_not_set": "No PIN is set yet.",
    "err_name_required": "A name is required.",
    "err_name_invalid": "The name has characters that are not allowed.",
    "err_name_too_long": "The name is too long.",
    "err_guardian_required": "A parent or guardian name is required unless the person is adult.",
    "err_consent_not_agreed": "Consent has to be confirmed.",
    "err_scope_invalid": "Pick at least one consent scope.",
    "err_choice_invalid": "Invalid choice.",
    "err_id_invalid": "Invalid identifier.",
    "err_unknown_field": "Unexpected field.",
    "err_confirm_required": "Confirmation required.",
    "err_not_found": "Not found.",
    "err_wrong_kind": "Not possible for this kind of record.",
    "err_consent_refused": "The registry refused the consent record.",
    "err_enrollment_not_ready": "Enrollment is not ready to be saved.",
    "err_forget_incomplete": "Erasure was incomplete. Try again.",
    "err_not_implemented": "Not available yet.",
    "err_body_too_large": "Request too large.",
    "err_bad_request": "Bad request.",
    "err_origin": "Request origin refused.",
    "err_vector_leak": "Response blocked.",
}

IT: dict[str, str] = {
    "no_pin": "Nessun PIN operatore impostato. Impostalo con l'installer o con la CLI.",
    "err_busy": "Sto controllando un altro PIN. Riprova tra un momento.",
    "err_consent_who_guardian": "Per un minore dà il consenso il genitore o tutore.",
    "err_guardian_is_subject": "Il tutore non può essere la persona registrata.",
    "err_bad_host": "Host non accettato.",
    "dialog_cancel": "Annulla",
    "dialog_confirm": "Conferma",
    "dialog_retry_in": "Riprova tra {n} s.",
    "confirm_merge": "Unire {drop} in {keep}? Non si può annullare.",
    "theme": "Tema",
    "theme_auto": "Auto",
    "theme_light": "Chiaro",
    "theme_dark": "Scuro",
    "theme_high_contrast": "Contrasto",
    "title": "RoboEmot",
    "loading": "Caricamento...",
    "login_title": "Accesso operatore",
    "pin": "PIN",
    "login": "Accedi",
    "logout": "Esci",
    "tab_enroll": "Registrazione",
    "tab_registry": "Anagrafica",
    "tab_catalogue": "Catalogo",
    "tab_debug": "Diagnostica",
    "tab_live": "Live",
    "live_title": "Stato interno",
    "live_intro": (
        "Che cosa sta facendo il robot adesso. La scheda si aggiorna finché è aperta e "
        "visibile; non viene salvato nulla."
    ),
    "live_disabled_title": "La scheda Live è spenta",
    "live_disabled_body": (
        "Mostra chi vede il robot, quindi è spenta di default. Per accenderla apri "
        "Impostazioni, cerca webui.live.enabled e attivala. Vale subito, senza riavvio."
    ),
    "live_mood": "Umore",
    "live_valence": "Valenza",
    "live_arousal": "Attivazione",
    "live_intensity": "Intensità",
    "live_energy": "Energia",
    "live_circ_awake": "sveglio",
    "live_circ_drowsy": "assonnato",
    "live_circ_resting": "a riposo",
    "live_circ_off": "Il ritmo circadiano è spento.",
    "live_modes": "Modalità",
    "live_awake": "Sveglio",
    "live_asleep": "Dorme",
    "live_online": "In linea",
    "live_offline": "Fuori linea",
    "live_minor_yes": "Possibile minore in vista",
    "live_minor_no": "Nessun minore in vista",
    "live_paused": "In pausa durante il sonno",
    "live_paused_none": "Niente in pausa",
    "live_part_app_tick": "comportamento",
    "live_part_vision": "visione",
    "live_part_light_reader": "lettura della luce",
    "live_part_listener": "ascolto",
    "live_register": "Registro",
    "live_persona": "Personalità",
    "live_conversation": "Conversazione",
    "live_phase_idle": "A riposo",
    "live_phase_listening": "In ascolto",
    "live_phase_thinking": "Sta pensando",
    "live_phase_speaking": "Sta parlando",
    "live_route": "Rotta",
    "live_route_pending": "in connessione",
    "live_elapsed": "In attesa da {n} s",
    "live_attention": "In vista",
    "live_people": "Persone: {n}",
    "live_nobody": "Nessuno in vista.",
    "live_unknown": "Sconosciuto",
    "live_gaze": "sguardo su di lui",
    "live_more": "e altri {n}",
    "live_reflexes": "Riflessi",
    "live_reflex_last": "Ultimo: {name}, {n} s fa",
    "live_reflex_none": "Nessuno finora.",
    "live_motion": "Movimento",
    "live_motion_writing": "Scrive le pose",
    "live_motion_external": "Controllo esterno",
    "live_motion_motors_off": "Motori spenti",
    "live_system": "Sistema",
    "live_cpu_temp": "Temperatura CPU: {n} °C",
    "live_cpu_temp_none": "Temperatura CPU: non disponibile",
    "live_levels": "Livelli audio",
    "live_levels_note": "Solo livelli: nessun suono lascia il robot e nessuno viene conservato.",
    "live_mic": "Microfono",
    "live_out": "Voce del robot",
    "live_db": "{n} dBFS",
    "live_floor": "Rumore di fondo: {n} dBFS",
    "live_level_paused": "in pausa mentre dorme",
    "live_level_no_data": "nessun segnale",
    "live_level_unavailable": "non disponibile",
    "live_transcript": "Conversazione",
    "live_transcript_note": (
        "Resta solo nella memoria del robot, non viene mai salvata. I nomi compaiono come "
        "li ha la conversazione."
    ),
    "live_transcript_off": (
        "La trascrizione è spenta, quindi non si conserva nulla. Per attivarla apri "
        "Impostazioni, cerca webui.live.transcript e attivala. Vale subito; spegnerla "
        "cancella quanto conservato."
    ),
    "live_transcript_empty": "Ancora nulla di detto.",
    "live_transcript_user": "Sentito",
    "live_transcript_robot": "Detto",
    "live_transcript_cut": "interrotto",
    "live_transcript_clear": "Cancella",
    "err_live_disabled": "La scheda Live è spenta.",
    "err_live_unavailable": "La scheda Live non è disponibile su questo robot.",
    "live_video": "Camera",
    "live_video_alt": "Immagine dal vivo della telecamera",
    "live_video_pause": "Pausa",
    "live_video_resume": "Riprendi",
    "live_video_paused": "In pausa.",
    "live_video_note": (
        "Un'immagine ferma aggiornata alcune volte al secondo, con i riquadri dei volti e i "
        "nomi delle persone che il robot conosce. Si vede solo in questa pagina e non viene "
        "mai salvata. Il robot non si sveglia e non si muove per questo."
    ),
    "live_video_off": (
        "L'anteprima della telecamera è spenta. Per accenderla apri Impostazioni, cerca "
        "webui.live.video e attivala. Vale subito. Finché qualcuno la guarda, ogni pagina "
        "aperta mostra un avviso."
    ),
    "live_video_banner": "Visione video in diretta attiva",
    "err_video_off": "L'anteprima della telecamera è spenta.",
    "err_video_asleep": "Il robot dorme, quindi la telecamera è spenta.",
    "err_video_no_vision": "La visione non è attiva su questo robot, quindi non c'è immagine.",
    "err_video_no_frame": "In attesa della prossima immagine della telecamera.",
    "err_video_stale": "La telecamera non invia una nuova immagine da un po'.",
    "err_cross_site": "L'immagine della telecamera è servita solo a questa pagina.",
    "live_video_idle": (
        "L'anteprima si è fermata dopo un po' senza attività su questa pagina. Premi Riprendi."
    ),
    "tab_routes": "Modelli",
    "tab_settings": "Impostazioni",
    "settings_intro": (
        "Ogni campo di configurazione, con il valore attuale e il livello che lo decide. Ogni "
        "modifica chiede il PIN operatore e vale subito. Una modifica che allenta una regola di "
        "sicurezza viene invece salvata nel file di configurazione dell'app e resta dopo un "
        "riavvio."
    ),
    "settings_core": "Nucleo",
    "settings_robot": "Robot",
    "settings_nav": "Gruppi di impostazioni",
    "settings_search": "Cerca",
    "settings_search_hint": "Per percorso o descrizione.",
    "settings_changed_only": "Solo modificate",
    "settings_matches": "{n} impostazioni mostrate",
    "settings_source_runtime": "Runtime",
    "settings_source_derived": "Derivato",
    "settings_source_app": "File dell'app",
    "settings_source_preset": "Preset",
    "settings_source_manifest": "Manifesto del robot",
    "settings_source_defaults": "Predefiniti",
    "settings_source_schema": "Integrato",
    "settings_restart": "Riavvia RoboEmot",
    "settings_restart_hint": (
        "Le impostazioni con «Riavvio necessario» valgono dopo il riavvio di RoboEmot."
    ),
    "settings_restart_pending": "Le modifiche salvate attendono un riavvio.",
    "settings_restart_title": "Riavviare RoboEmot?",
    "settings_restart_message": (
        "RoboEmot si ferma e riparte. Questa pagina non sarà raggiungibile per un attimo e "
        "dovrai accedere di nuovo. Cosa succede:"
    ),
    "settings_restart_confirm": "Riavvia",
    "settings_restart_started": (
        "Riavvio in corso. Attendi qualche secondo, poi ricarica la pagina e accedi di nuovo."
    ),
    "settings_restart_effect_zero_pose": "Il robot va in posa zero quando l'app si è fermata.",
    "settings_restart_effect_remote_session_drop": "Una sessione video o audio remota può cadere.",
    "settings_restart_effect_kill_on_timeout": (
        "Se RoboEmot non si ferma entro {timeout_s} s, viene terminato."
    ),
    "err_restart_unavailable": "Al momento RoboEmot non può essere riavviato da qui.",
    "settings_restart_badge": "Riavvio necessario",
    "settings_restart_goto": "Vai a Riavvia RoboEmot",
    "settings_restart_needed": "Riavvia RoboEmot perché abbia effetto.",
    "settings_not_measured": "non misurato",
    "settings_show_all": "Mostra tutte e {n}",
    "settings_private": "Privata",
    "settings_secret": "Segreta",
    "settings_readonly": "Sola lettura",
    "settings_json_readonly": "Valore strutturato: si modifica nel file di configurazione.",
    "settings_superset_hint": (
        "Togliere una parola allenta il filtro per i minori: viene salvato nella "
        "configurazione dell'app e richiede il PIN."
    ),
    "settings_null": "Nessun valore (null)",
    "settings_route_move_message": (
        "Se questa route ha una chiave API, salvando la chiave viene inviata a {host}. "
        "Inserisci il PIN per consentirlo."
    ),
    "settings_write_only_hint": (
        "L'indirizzo attuale contiene credenziali e non viene mostrato. Scrivi un nuovo indirizzo "
        "completo per sostituirlo; non può contenere credenziali."
    ),
    "settings_slider": "Cursore di {path}",
    "settings_tuple_item": "Valore {n} di {total}",
    "settings_new_item": "Nuova voce",
    "settings_remove_item": "Rimuovi {item}",
    "settings_remove_n": "Rimuovi la riga {n}",
    "settings_dict_key": "Chiave",
    "settings_dict_value": "Valore",
    "settings_add_entry": "Aggiungi voce",
    "settings_entry_name": "Nome",
    "settings_remove_entry": "Rimuovi voce",
    "settings_no_entries": "Nessuna voce.",
    "settings_reveal_message": (
        "Mostra le impostazioni private (minori, privacy, posizione). Restano in questa pagina "
        "finché non esci o ricarichi."
    ),
    "settings_reset_layer": "Torna al livello inferiore",
    "settings_reset_done": "Ripristinato.",
    "settings_entry_added": "Voce aggiunta.",
    "settings_entry_removed": "Voce rimossa.",
    "settings_layer_runtime": "Applicata subito.",
    "settings_layer_app": (
        "Salvata in modo permanente nel file di configurazione dell'app: una regola di sicurezza "
        "è stata allentata e resta dopo un riavvio."
    ),
    "err_key_duplicate": "Due righe usano la stessa chiave.",
    "settings_default": "Predefinito",
    "settings_save": "Salva",
    "settings_add": "Aggiungi voce",
    "settings_remove": "Rimuovi",
    "settings_admin": "Admin",
    "settings_reveal": "Mostra con il PIN",
    "settings_hidden": (
        "Nascosto: le impostazioni su minori, privacy e posizione si mostrano con il PIN."
    ),
    "settings_admin_hint": "Rilevante per la sicurezza. Il PIN operatore vale come PIN admin.",
    "settings_pin_title": "Conferma la modifica",
    "settings_not_persisted": "È attiva ma non salvata su disco: termina al prossimo riavvio.",
    "settings_unchanged": "Nessuna modifica.",
    "settings_none": "Nessuna impostazione si può modificare dalla web UI.",
    "err_routes_unavailable": "I modelli non sono disponibili.",
    "err_route_unknown": "Modello sconosciuto.",
    "err_route_busy": "Un altro controllo è in corso. Riprova tra poco.",
    "err_route_budget": "Il budget giornaliero di token del modello è esaurito.",
    "err_route_secret": "La chiave non è disponibile. Impostala sul robot: roboemot secrets set.",
    "route_error_secret": "La chiave del modello non è disponibile.",
    "route_secret_set": "chiave {name}: impostata",
    "route_secret_missing": "chiave {name}: non impostata",
    "route_error_timeout": "Il controllo ha richiesto troppo tempo ed è stato fermato.",
    "route_error_budget": "Il budget giornaliero di token è finito durante il controllo.",
    "route_error_failed": "Non è stato possibile completare il controllo.",
    "route_running": "Controllo in corso...",
    "routes_intro": (
        "Controllo di conformità di ogni modello LLM: streaming, chiamata di strumenti, latenza "
        "del primo token e italiano. È solo informativo: non abilita né blocca un modello e usa "
        "token dal budget giornaliero del modello."
    ),
    "routes_none": "Nessun modello LLM configurato.",
    "route_add_title": "Aggiungi un modello",
    "route_add_intro": (
        "Un modello è un servizio compatibile OpenAI con cui il robot può parlare. La chiave si "
        "sceglie tra quelle salvate qui sotto; non si scrive mai in questo modulo. Le modifiche "
        "chiedono il PIN operatore e valgono subito."
    ),
    "route_name": "Nome",
    "route_url": "Indirizzo base",
    "route_url_keep": "Lascia vuoto per non cambiarlo",
    "route_url_placeholder": "https://api.example.com/v1",
    "route_url_hint_edit": "L'indirizzo salvato non viene mostrato se contiene credenziali.",
    "route_model": "Modello",
    "route_locality": "Dove gira il modello",
    "route_locality_auto": "Automatico",
    "route_locality_local": "Locale (rete di casa)",
    "route_locality_cloud": "Cloud",
    "route_locality_hint": (
        "Un modello su questa macchina o sulla rete di casa vede i nomi veri. Configurarne uno "
        "richiede il file di configurazione; da qui si accettano solo modelli cloud."
    ),
    "route_key": "Chiave",
    "route_key_none": "Nessuna chiave",
    "route_key_literal": "Una chiave scritta nel file di configurazione",
    "route_key_hint": (
        "Si possono scegliere solo chiavi salvate. Aggiungila nella scheda Chiavi API."
    ),
    "route_key_message": "La chiave verrà inviata a {host}. Inserisci il PIN per consentirlo.",
    "route_key_insecure": (
        "Questo indirizzo non è cifrato (http): la chiave attraverserà la rete in chiaro."
    ),
    "route_add": "Aggiungi modello",
    "route_save": "Salva modello",
    "route_edit": "Modifica",
    "route_remove": "Rimuovi",
    "route_remove_title": "Rimuovere il modello {name}?",
    "route_remove_message": "Il modello viene eliminato. La sua chiave resta nell'archivio.",
    "route_added": "Modello {name} aggiunto.",
    "route_saved": "Modello {name} salvato.",
    "route_removed": "Modello {name} rimosso.",
    "secrets_title": "Chiavi API",
    "secrets_intro": (
        "Le chiavi API sono salvate sul robot e qui si possono solo scrivere, mai rileggere: la "
        "pagina mostra il nome di una chiave e se un modello la usa, non il suo valore. Un "
        "modello si riferisce a una chiave con il suo nome."
    ),
    "secrets_none": "Nessuna chiave salvata.",
    "secret_name": "Nome",
    "secret_name_hint": (
        "Lettere, cifre, punto, trattino o underscore; inizia con una lettera o una cifra."
    ),
    "secret_value": "Chiave",
    "secret_value_hint": (
        "Si incolla una volta e il campo viene svuotato al "
        "salvataggio. La pagina non la mostra più."
    ),
    "secret_save": "Salva chiave",
    "secret_saved": "Chiave {name} salvata.",
    "secret_removed": "Chiave {name} rimossa.",
    "secret_remove": "Rimuovi",
    "secret_remove_title": "Rimuovere la chiave {name}?",
    "secret_remove_message": "La chiave viene cancellata dal robot. Non si può annullare.",
    "secret_used_by": "usata da {routes}",
    "secret_unused": "non usata da nessun modello",
    "secret_set": "impostata",
    "secret_unset": "non impostata",
    "secret_in_use_hint": "Prima scegli un'altra chiave, o nessuna chiave, su quei modelli.",
    "err_secrets_unavailable": "Le chiavi API non sono disponibili.",
    "err_secret_name_invalid": "Il nome della chiave non è valido.",
    "err_secret_value_invalid": (
        "La chiave deve avere caratteri ASCII visibili, senza spazi, e non essere troppo lunga."
    ),
    "err_secret_in_use": (
        "Un modello usa questa chiave. Prima scegli un'altra chiave su quel modello."
    ),
    "err_secret_unknown": "Chiave sconosciuta.",
    "err_secret_store": (
        "L'archivio delle chiavi non è utilizzabile. Controlla i permessi del file sul robot."
    ),
    "err_pseudonyms_unavailable": "L'elenco dei nomi non è disponibile.",
    "refused_title": "Nomi non sostituiti da alias",
    "refused_hint": (
        "I nomi dell'anagrafica che non possono essere sostituiti da un alias (senza lettere o "
        "cifre, una parola di un modello di alias, una parola del formato della richiesta, una "
        "barra rovesciata) arrivano ai modelli cloud così come sono. Rinominali per tenerli "
        "privati. Inserisci il PIN admin per vedere l'elenco."
    ),
    "refused_show": "Mostra i nomi",
    "refused_none": "Nessun nome viene inviato così com'è.",
    "refused_off": "Gli pseudonimi per i modelli cloud sono disattivati.",
    "route_check": "Controlla",
    "route_checking": "Controllo in corso...",
    "route_never": "Mai controllato",
    "route_ok": "Tutti i controlli superati",
    "route_failed": "Alcuni controlli falliti",
    "route_at": "Controllato {at}",
    "check_streaming": "Streaming",
    "check_tool_round_trip": "Chiamata di strumenti",
    "check_first_token_latency": "Latenza del primo token",
    "check_italian": "Risposta in italiano",
    "status_pass": "ok",
    "status_fail": "errore",
    "status_skip": "saltato",
    "enroll_intro": (
        "Chiedi alla persona di guardare il robot. Qui non vengono mostrate né salvate immagini."
    ),
    "enroll_start": "Avvia registrazione",
    "enroll_cancel": "Annulla",
    "enroll_complete": "Salva registrazione",
    "frames": "Fotogrammi raccolti",
    "state": "Stato",
    "state_idle": "In attesa",
    "state_collecting": "Raccolta in corso",
    "state_ready": "Pronto per il consenso",
    "state_completed": "Completata",
    "state_cancelled": "Annullata",
    "state_timed_out": "Tempo scaduto",
    "state_completing": "Salvataggio",
    "state_unavailable": "La visione non è attiva",
    "hint_no_face": "Nessun volto inquadrato",
    "hint_too_far": "Avvicinati",
    "hint_too_close": "Allontanati un po'",
    "hint_too_dark": "Troppo buio",
    "hint_multiple_faces": "Una persona alla volta",
    "hint_look_at_robot": "Guarda il robot",
    "hint_hold_still": "Resta fermo",
    "hint_turn_head": "Gira leggermente la testa a sinistra e a destra",
    "display_name": "Nome da mostrare",
    "consent_who": "Nome di chi dà il consenso",
    "guardian": "Nome del genitore o tutore",
    "guardian_hint": "Obbligatorio salvo che la persona sia dichiarata adulta o anziana.",
    "age_group": "Fascia d'età",
    "gender": "Genere",
    "age_unknown": "Non dichiarata",
    "age_child": "Bambino",
    "age_teen": "Adolescente",
    "age_adult": "Adulto",
    "age_senior": "Anziano",
    "gender_unknown": "Non dichiarato",
    "gender_female": "Femmina",
    "gender_male": "Maschio",
    "scope_title": "Il consenso copre",
    "scope_face": "Riconoscimento del volto",
    "scope_voice": "Riconoscimento di chi parla (voce)",
    "scope_refresh": "Aggiunta di nuovi campioni nel tempo",
    "agree": "La persona (o il tutore) ha letto il testo sopra ed è d'accordo.",
    "consent_title": "Consenso",
    "consent_biometric": (
        "Il robot conserva i vettori del volto (e, se scelto, della voce) di questa persona per "
        "riconoscerla. Sono dati biometrici, categoria particolare ai sensi dell'art. 9 GDPR. "
        "Le immagini della camera vengono scartate dopo la registrazione. Per un minore serve "
        "il consenso di un genitore o tutore. La persona può chiedere in ogni momento di "
        "vedere o cancellare i propri dati."
    ),
    "consent_cloud_names": (
        "Nomi, preferenze e testo delle conversazioni possono essere inviati al fornitore cloud "
        "del modello linguistico usato dal robot, che li tratta come dati personali. Su queste "
        "rotte il robot invia i nomi reali."
    ),
    "consent_cloud_pseudonyms": (
        "Nomi, preferenze e testo delle conversazioni possono essere inviati al fornitore cloud "
        "del modello linguistico usato dal robot, che li tratta come dati personali. Su queste "
        "rotte il robot sostituisce i nomi presenti nella sua anagrafica con alias neutri. Le "
        "rotte su questa macchina o sulla rete di casa ricevono i nomi reali, quindi un gateway "
        "in rete che inoltra a un fornitore cloud deve essere segnato come cloud "
        "dall'amministratore. Una voce cloud ascolta comunque i nomi che pronuncia. Un "
        "soprannome, un nome storpiato o altri dati personali detti a voce non vengono "
        "sostituiti, né un nome dell'anagrafica che il robot non può sostituire (senza lettere, "
        'una parola del modello di alias o una parola strutturale come "user"): viene '
        "inviato così com'è."
    ),
    "registry_title": "Anagrafica",
    "registry_empty": "Ancora nulla di registrato.",
    "kind_person": "Persona",
    "kind_animal": "Animale",
    "kind_object": "Oggetto",
    "status_pending": "in sospeso",
    "status_active": "attivo",
    "last_seen": "Visto l'ultima volta",
    "never": "mai",
    "rename": "Rinomina",
    "merge": "Unisci in...",
    "merge_pick": "Unisci in questo",
    "forget": "Dimentica (cancella tutto)",
    "delete": "Elimina",
    "confirm_forget": "Cancellare tutti i dati di {name}? Non si può annullare.",
    "confirm_delete": "Eliminare {name}? Non si può annullare.",
    "confirm_purge": "Eliminare questo elemento illeggibile? Non si può annullare.",
    "quarantined_title": "Elementi illeggibili",
    "quarantined_none": "Nessuno.",
    "purge": "Elimina",
    "new_name": "Nuovo nome",
    "catalogue_intro": (
        "Importa i file XMP scritti dal tuo programma di foto. Persone, animali e "
        "oggetti diventano record in attesa; le persone restano solo un nome finché "
        "non fanno la registrazione. La foto è facoltativa, viene letta in memoria e "
        "non viene mai salvata."
    ),
    "catalogue_xmp": "File XMP (.xmp)",
    "catalogue_photo": "La sua foto (JPEG o PNG, facoltativa)",
    "catalogue_import": "Importa",
    "catalogue_pending": "Da rivedere: record in attesa e automatici",
    "catalogue_none": "Niente da rivedere.",
    "catalogue_live": "Visti adesso",
    "catalogue_live_none": "Il robot non vede animali o oggetti in questo momento.",
    "catalogue_pick": "Scegli ciò che il robot vede per confermare un record.",
    "catalogue_confirm": "Questo è {name}",
    "catalogue_activate": "Attiva dalla foto",
    "catalogue_report": "{created} creati, {existing} già noti.",
    "catalogue_conflicts": "Etichette da correggere",
    "catalogue_truncated": "Il file superava il limite; il resto è stato ignorato.",
    "catalogue_clear": "Dimentica gli avvistamenti raccolti",
    "catalogue_clear_confirm": (
        "Dimenticare ciò che il robot ha raccolto su animali e oggetti che non conosce ancora?"
    ),
    "catalogue_cleared": "Dimenticati: {n}.",
    "tombstones_clear": "Permetti il ritorno dei record eliminati ({n})",
    "tombstones_clear_confirm": (
        "Un nuovo import creerà di nuovo i record in attesa che hai eliminato. Continuare?"
    ),
    "tombstones_cleared": "Rimossi: {n}.",
    "catalogue_read_failed": "Impossibile leggere il file.",
    "photo_used": "Foto usata.",
    "photo_no_embedder": "Foto saltata: il riconoscimento di oggetti non è attivo.",
    "photo_dims_mismatch": "Foto saltata: la sua forma non corrisponde al file XMP.",
    "origin_auto": "automatico",
    "origin_xmp": "da catalogo",
    "origin_enrolled": "registrato",
    "has_appearance": "con aspetto",
    "no_appearance": "senza aspetto",
    "conflict_category_conflict": "due categorie diverse",
    "conflict_unknown_category": "categoria non riconosciuta",
    "err_xmp_required": "Scegli un file XMP.",
    "err_xmp_too_large": "Il file XMP è troppo grande.",
    "err_xmp_not_xml": "Il file non è un XMP valido.",
    "err_xmp_forbidden_xml": "Il file usa funzioni XML non accettate.",
    "err_image_format": "Sono accettate solo foto JPEG e PNG.",
    "err_image_corrupt": "Impossibile leggere la foto.",
    "err_image_too_large": "La foto è troppo grande.",
    "err_image_too_many_pixels": "La foto ha troppi pixel.",
    "err_image_unsupported": "Su questo robot non si possono leggere le foto.",
    "err_not_pending": "Questo record è già attivo.",
    "err_entry_gone": "Il robot non lo vede più. Sceglilo di nuovo.",
    "err_no_appearance": "Non c'è ancora un aspetto da usare. Scegli qualcosa che il robot vede.",
    "err_kind_mismatch": "Non è lo stesso tipo di cosa.",
    "err_category_mismatch": "Non è la stessa categoria.",
    "err_catalogue_unavailable": "Il catalogo non è disponibile.",
    "err_import_busy": "Un altro import è in corso. Riprova tra un momento.",
    "err_pending_invalid": "Quel nome importato non è più disponibile.",
    "photo_no_regions": "Foto non usata: il file XMP non ha regioni di animali o oggetti.",
    "photo_partial": (
        "Foto usata in parte: il tempo è finito. Importa di nuovo gli stessi file per finire."
    ),
    "pending_link": "Collega a un nome importato",
    "pending_none": "Nessuno: è una persona nuova",
    "debug_title": "Rapporto diagnostico",
    "debug_refresh": "Aggiorna",
    "saved": "Salvato.",
    "err_generic": "Qualcosa è andato storto.",
    "err_unauthorized": "Accedi per continuare.",
    "err_csrf": "Controllo di sessione fallito, ricarica la pagina.",
    "err_bad_pin": "PIN errato.",
    "err_setting_not_allowed": "Questa impostazione non si può modificare dalla web UI.",
    "err_setting_guardrail": "Modifica rifiutata: allenterebbe una regola di sicurezza o privacy.",
    "err_setting_invalid": "Il valore non è valido per questa impostazione.",
    "err_settings_unavailable": "Le impostazioni non sono disponibili.",
    "err_value_type": "Il valore ha il tipo sbagliato.",
    "err_value_range": "Il valore è fuori dall'intervallo ammesso.",
    "err_value_choice": "Il valore non è tra quelli ammessi.",
    "err_text_invalid": "Il testo contiene caratteri non ammessi.",
    "err_persist_failed": "Impossibile scrivere il file di configurazione; non è cambiato nulla.",
    "err_value_merged": "Qui le voci si possono aggiungere o modificare, non rimuovere.",
    "err_text_too_long": "Il testo è troppo lungo.",
    "err_key_invalid": "Il nome può usare solo lettere, cifre, trattino e trattino basso.",
    "err_entry_exists": "Esiste già una voce con questo nome.",
    "err_entry_invalid": ("La voce ha un campo che non si può impostare dalla web UI."),
    "err_entry_in_lower_layer": (
        "Questa voce viene dal file di configurazione e non si può rimuovere da qui."
    ),
    "err_endpoint_ack_required": (
        "Questa route ha una chiave API. Salvando, la chiave viene inviata a {host}. Inserisci il "
        "PIN e premi di nuovo il pulsante per consentirlo."
    ),
    "err_url_credentials": (
        "L'indirizzo non deve contenere utente, password, token o riferimenti ${secret:...}. "
        "Salva la chiave con `roboemot secrets set NOME` e usa ${secret:NOME} come api_key."
    ),
    "err_language_unsupported": "Per questa lingua non ci sono modelli vocali nell'app.",
    "err_not_resettable": (
        "Questo campo appartiene a una voce aggiunta qui e non si può ripristinare da solo."
    ),
    "err_list_empty": "La lista richiede almeno una voce.",
    "err_list_too_long": "La lista ha troppe voci.",
    "err_phrase_too_long": "Una voce è troppo lunga.",
    "err_phrase_invalid": "Una voce contiene caratteri non ammessi.",
    "err_line_empty": "Questa frase non può essere vuota.",
    "err_line_too_long": "La frase è troppo lunga.",
    "err_line_invalid": "La frase contiene caratteri o segnaposto non ammessi.",
    "err_line_blocked": "La frase contiene una parola bloccata dalla lista per i bambini.",
    "err_name_blocked": "Il nome è bloccato dalla lista di parole per i bambini.",
    "err_locked": (
        "Troppi tentativi. Riprova più tardi, oppure riavvia l'app o esegui set-pin sul robot."
    ),
    "err_pin_digits": "Il PIN deve contenere solo cifre.",
    "err_pin_too_short": "Il PIN è troppo corto.",
    "err_pin_not_set": "Nessun PIN ancora impostato.",
    "err_name_required": "Il nome è obbligatorio.",
    "err_name_invalid": "Il nome contiene caratteri non ammessi.",
    "err_name_too_long": "Il nome è troppo lungo.",
    "err_guardian_required": (
        "Serve il nome di un genitore o tutore, salvo che la persona sia adulta."
    ),
    "err_consent_not_agreed": "Il consenso va confermato.",
    "err_scope_invalid": "Scegli almeno un ambito di consenso.",
    "err_choice_invalid": "Scelta non valida.",
    "err_id_invalid": "Identificativo non valido.",
    "err_unknown_field": "Campo inatteso.",
    "err_confirm_required": "Serve la conferma.",
    "err_not_found": "Non trovato.",
    "err_wrong_kind": "Non possibile per questo tipo di elemento.",
    "err_consent_refused": "L'anagrafica ha rifiutato il consenso.",
    "err_enrollment_not_ready": "La registrazione non è pronta per essere salvata.",
    "err_forget_incomplete": "Cancellazione incompleta. Riprova.",
    "err_not_implemented": "Non ancora disponibile.",
    "err_body_too_large": "Richiesta troppo grande.",
    "err_bad_request": "Richiesta non valida.",
    "err_origin": "Origine della richiesta rifiutata.",
    "err_vector_leak": "Risposta bloccata.",
}

STRINGS: dict[str, dict[str, str]] = {"en": EN, "it": IT}


def language_for(configured: str) -> str:
    """The page language for a ``conversation.language`` value (``it``, ``it-IT``, ...)."""
    short = configured.lower().replace("_", "-").split("-")[0]
    return short if short in STRINGS else DEFAULT_LANGUAGE
