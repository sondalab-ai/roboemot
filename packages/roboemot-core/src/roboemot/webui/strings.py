"""Page strings in Italian and English; the page follows ``conversation.language`` (SPEC 8)."""

from __future__ import annotations

DEFAULT_LANGUAGE = "en"

EN: dict[str, str] = {
    "no_pin": "No operator PIN is set. Set the PIN with the installer or the CLI.",
    "pin_confirm": "PIN (needed to forget or purge)",
    "err_busy": "Busy checking another PIN. Try again in a moment.",
    "err_consent_who_guardian": "For a minor, the person consenting is the parent or guardian.",
    "err_guardian_is_subject": "The guardian cannot be the person being enrolled.",
    "err_bad_host": "Host not accepted.",
    "title": "RoboEmot",
    "loading": "Loading...",
    "login_title": "Operator login",
    "pin": "PIN",
    "login": "Log in",
    "logout": "Log out",
    "tab_enroll": "Enrollment",
    "tab_registry": "Registry",
    "tab_catalogue": "Catalogue",
    "tab_debug": "Diagnostics",
    "enroll_intro": "Ask the person to look at the robot. No images are shown or stored here.",
    "enroll_start": "Start enrollment",
    "enroll_cancel": "Cancel",
    "enroll_complete": "Save enrollment",
    "frames": "Frames collected",
    "state": "State",
    "state_idle": "Idle",
    "state_collecting": "Collecting",
    "state_ready": "Ready for consent",
    "state_done": "Done",
    "state_failed": "Failed",
    "hint_no_face": "No face in view",
    "hint_too_far": "Move closer",
    "hint_too_close": "Move back a little",
    "hint_too_dark": "Too dark",
    "hint_multiple_faces": "Only one person at a time",
    "hint_look_at_robot": "Look at the robot",
    "hint_hold_still": "Hold still",
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
        "provider the robot uses, which processes them as personal data. The robot sends "
        "pseudonyms instead of names to such routes."
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
    "catalogue_later": "Catalogue import is not available yet.",
    "debug_title": "Diagnostics report",
    "debug_refresh": "Refresh",
    "saved": "Saved.",
    "err_generic": "Something went wrong.",
    "err_unauthorized": "Please log in.",
    "err_csrf": "Session check failed, reload the page.",
    "err_bad_pin": "Wrong PIN.",
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
    "pin_confirm": "PIN (serve per dimenticare o eliminare)",
    "err_busy": "Sto controllando un altro PIN. Riprova tra un momento.",
    "err_consent_who_guardian": "Per un minore dà il consenso il genitore o tutore.",
    "err_guardian_is_subject": "Il tutore non può essere la persona registrata.",
    "err_bad_host": "Host non accettato.",
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
    "state_done": "Completata",
    "state_failed": "Fallita",
    "hint_no_face": "Nessun volto inquadrato",
    "hint_too_far": "Avvicinati",
    "hint_too_close": "Allontanati un po'",
    "hint_too_dark": "Troppo buio",
    "hint_multiple_faces": "Una persona alla volta",
    "hint_look_at_robot": "Guarda il robot",
    "hint_hold_still": "Resta fermo",
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
        "rotte il robot invia pseudonimi al posto dei nomi."
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
    "catalogue_later": "L'importazione del catalogo non è ancora disponibile.",
    "debug_title": "Rapporto diagnostico",
    "debug_refresh": "Aggiorna",
    "saved": "Salvato.",
    "err_generic": "Qualcosa è andato storto.",
    "err_unauthorized": "Accedi per continuare.",
    "err_csrf": "Controllo di sessione fallito, ricarica la pagina.",
    "err_bad_pin": "PIN errato.",
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
