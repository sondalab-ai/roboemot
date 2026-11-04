# Spike: a dedicated "Neve" wake-word spotter (SPEC 8, 13)

Date: 2026-11-04. Research and measurement only, no product code changed. Section references are to
`docs/specs/kickoff.md`. The scripts that produced the numbers are in
`docs/spikes/2026-11-04-neve-wake-word/` (they download into the scratch folder named by `WAKEWORD_DIR` or
given on the command line, never into the repo).

## Question

Today the wake word is a by-product of Vosk (SPEC 8 stage 2): "ehi/ciao neve" in the final text, or the
bare name in a final of at most `voice.fastpath.alias_wake_max_words` (4) words. Vosk small recognised a
synthetic "Neve" about one time in four (earlier measurement). "Neve" is also an Italian word (snow).
Wanted: a keyword spotter on the mic stream for "Neve", "ehi Neve", "ciao Neve" (Italian first, English
speakers second), with few false wakes from Italian sentences that contain "neve", from TV and music.

## Short answer

- **Candidate: openWakeWord** (Apache-2.0 code, ONNX Runtime, which the robot already has). microWakeWord
  is second (new TFLite runtime, harder training). sherpa-onnx KWS has no Italian model.
- **Licence: a model we train can ship under Apache-2.0 terms, but not with the recipe's data.** The
  recipe's negative features are CC BY-NC-SA 4.0 (openWakeWord) or CC BY-NC 4.0 (microWakeWord), and the
  pretrained word models are CC BY-NC-SA 4.0. Trained here only on CC BY 4.0, CC0 and generated data, the
  model needs attribution only. The two shared feature files are Apache-2.0 by origin, with an unanswered
  README ambiguity (5 open issues).
- **Trained** with the official `train.py` on 6,400 Piper positives (Italian first, 813 English speakers
  second) and 14 h of permissive speech (plus an augmented copy). Training takes 9 minutes on the Mac; data
  preparation took about 2 hours, 5.2 GB downloaded.
- **Measured (synthetic voices only, no real recordings):** "ehi/ciao Neve" is detected well in quiet
  rooms (Italian voices 87 to 95% at threshold 0.999; the 87% is `serena-medium`, same speaker as a
  training voice), the bare "Neve" is not (11 to 16%). In a room with noise at 10 dB SNR the phrase drops
  to 20 to 35%. False accepts are 1.6 to 1.9 per hour on read speech and **4.1 per hour** on far-field
  dinner-party speech and music, against openWakeWord's own target of 0.5. Sentences with "neve" rarely
  fire (1% of clips, 26 per hour of nonstop snow talk, against about 490 per hour for today's Vosk rule);
  greetings with other names do ("Ehi Eva!", "Ciao Elena!").
- **Against today's Vosk rule** (same clips): Vosk is better on the bare name (62 to 91% clean against
  11 to 16%); the spotter is better on the phrase in white noise (61 to 78% against 14 to 46%). Both
  together reach 97 to 99% on the phrase in quiet rooms and 69 to 84% in noise.
- **CPU:** 25 ms per audio second on the Mac (1 thread, measured), so about **20 to 51% of one CM4 core**
  always on (SPEC 13's 8 to 20x guess). TFLite is no faster.
- **Recommendation: no-go to ship this model; go for one more data slice** on openWakeWord, phrase-only
  ("ehi/ciao Neve"), with the bare name left to Vosk's short-final rule. Details in section 6.

## Setup and limits

| Item | Value |
| --- | --- |
| Host | Apple M3 Pro (12 cores), macOS, Python 3.12.13 (uv), ONNX Runtime 1.31.1, PyTorch 2.8.0 for training; the Mac was shared with other work (a video call) during the runs |
| openWakeWord | 0.6.0 from GitHub `main` (2026-11-04), feature models from release v0.5.1 (sha256 in section 2) |
| Piper | `piper-tts` 1.8.0 (the version the robot uses), voices at `rhasspy/piper-voices` revision `e7229f7d33a10e145e109b886d661984201ac3c4` |
| Vosk baseline | `vosk` 0.3.44 (PyPI), `vosk-model-small-it-0.22` (sha256 matches `voice/models.yaml`), free text only, no grammar pass |
| Robot | Not reached. Every CM4 number is an estimate |
| Real recordings of "Neve" | None. No recordings of people were made or used, so every recall number is on synthetic voices |
| Music | Only through the openWakeWord FP validation set (2 h of MUSDB, reverberated, plus DiPCo dinner-party audio); no separate music set |

Conditions applied to each evaluation clip (`evaluate.py`): `clean` (1 s silence before, 0.5 s after);
`white10` (white noise at 10 dB SNR, no room, the same condition the Vosk baseline used); `reverb`
(simulated room, rooms not used in training); `snr10` and `snr5` (room plus coloured noise or LibriSpeech
test-clean babble at 10 or 5 dB). A clip "fires" when any 80 ms window scores at or above the threshold.
Continuous streams count an accept at the first window at or above the threshold, then nothing for 2 s.

## 1. Candidates

| Candidate | What it is | Runtime on the robot | Italian | Training | Verdict |
| --- | --- | --- | --- | --- | --- |
| [openWakeWord](https://github.com/dscripka/openWakeWord) 0.6.0 | Fixed mel front end + Google `speech_embedding` (frozen) + a small classifier per word (DNN, 16 x 96 input, about 1.28 s of context) | ONNX Runtime, already a dependency (`inference_framework="onnx"`); the package itself is pure Python (`py3-none-any`), but 0.6.0 also pulls `ai-edge-litert` on Linux/macOS (we would vendor the 3 small `.onnx` files and the ~30-line streaming loop, or install with `--no-deps`) | The frozen embedding was trained on speech in general, so a classifier can be trained for any word from synthetic clips | Official recipe (`notebooks/automatic_model_training.ipynb`, `openwakeword/train.py`), PyTorch, CPU is enough | **Chosen and trained here** |
| [microWakeWord](https://github.com/kahrendt/microWakeWord) | MixConv streaming CNN on the TFLite Micro front end (40 features / 10 ms), int8, built for ESP32 | TFLite: `pymicro-wakeword` 2.5.0 (Apache-2.0) has an aarch64 wheel tagged `manylinux_2_35` (needs glibc 2.35+), or `ai-edge-litert` 2.3.0 (`cp312 manylinux_2_27_aarch64`). Either is a new native dependency next to ONNX Runtime | Same synthetic-clip approach | TensorFlow 2.18+ and mmap-ninja feature sets; the notebook calls a usable model "a lot of experimentation" and recommends a GPU | Second choice: cheaper per second, but a new runtime, heavier training, not trained here |
| [sherpa-onnx KWS](https://k2-fsa.github.io/sherpa/onnx/kws/pretrained_models/index.html) (Apache-2.0 code) | Open-vocabulary keyword spotting with a small streaming zipformer transducer; no training, keywords are token strings | ONNX Runtime (own wheel `sherpa-onnx` 2.0.1 has `cp312 manylinux aarch64`) | No: the published KWS models are English (GigaSpeech), Chinese (WenetSpeech) and zh-en only | None needed, but no Italian model and no published model licence on the page | Not usable for Italian today |
| Vosk grammar pass (today) | Second Vosk decode limited to a grammar of keywords | Already in the robot (SPEC 8 stage 3) | Yes | None | The baseline measured below |

Proprietary or key-gated engines (Picovoice Porcupine, Sensory, others) are out of scope by the brief.
Older open engines (Snowboy, Mycroft Precise, Rhasspy Raven) are unmaintained and were not considered.

## 2. Licences

Rule used: the licence of a model **we train** is decided by what goes into its weights: the code is
Apache-2.0 in both projects, but the training data is not. Data used only to *measure* a model does not
end up in the shipped file.

| Item | Licence | Source | Restriction on a model we train and ship (Apache-2.0 product) |
| --- | --- | --- | --- |
| openWakeWord code | Apache-2.0 | [LICENSE](https://github.com/dscripka/openWakeWord/blob/main/LICENSE), [README "License"](https://github.com/dscripka/openWakeWord#license) | None |
| openWakeWord pretrained word models (alexa, hey_jarvis, ...) | CC BY-NC-SA 4.0 "due to the inclusion of datasets with unknown or restrictive licensing" | [README "License"](https://github.com/dscripka/openWakeWord#license) (README line 278 at the commit read) | Not used; we train our own |
| `melspectrogram.onnx` (v0.5.1, sha256 `ba2b0e0f...b176f`) | No separate licence. It is a fixed torchaudio-style mel filterbank exported to ONNX, no learned weights | [README "Model Architecture"](https://github.com/dscripka/openWakeWord#model-architecture) | None expected (no training data). See the open question below |
| `embedding_model.onnx` (v0.5.1, sha256 `70d16429...75c1f`) | Google `speech_embedding`, **Apache 2.0** (Kaggle model metadata `licenseName: "Apache 2.0"`), re-implemented in Keras and converted | [Kaggle model](https://www.kaggle.com/models/google/speech-embedding), [conversion notebook](https://github.com/dscripka/openWakeWord/blob/main/notebooks/converting_google_speech_embedding_model.ipynb), paper [arXiv:2002.01322](https://arxiv.org/abs/2002.01322) | None from Google's terms; keep the Apache notice |
| Open question on the two shared files | The README's NC sentence says "all of the included pre-trained models", which some read as covering these two files. No maintainer answer as of 2026-11-04 | Open issues [#313](https://github.com/dscripka/openWakeWord/issues/313), [#338](https://github.com/dscripka/openWakeWord/issues/338), [#346](https://github.com/dscripka/openWakeWord/issues/346), [#348](https://github.com/dscripka/openWakeWord/issues/348), [#357](https://github.com/dscripka/openWakeWord/issues/357) (the hashes quoted in #348 match the files measured here) | Low risk by reasoning (filterbank + Apache Google model), but not confirmed. Fallback: re-export both from the primary sources ourselves (torchaudio MelSpectrogram with the same parameters; the Kaggle model through the conversion notebook) |
| openWakeWord training negatives: `openwakeword_features_ACAV100M_2000_hrs_16bit.npy` (17.28 GB) | **CC BY-NC-SA 4.0** (dataset card); ACAV100M is YouTube audio | [dataset card](https://huggingface.co/datasets/davidscripka/openwakeword_features) (revision `985bf1b4`) | **Yes**: a model trained on it inherits NC (and arguably SA). Not used here |
| openWakeWord FP validation set `validation_set_features.npy` (185 MB, 11 h: DiPCo, Santa Barbara Corpus, MUSDB) | CC BY-NC-SA 4.0 (same card) | same card | Used here **only to measure** false accepts, not to train or select the model |
| MIT environmental impulse responses (recipe's RIRs) | No licence stated on the page | [IR Survey](https://mcdermottlab.mit.edu/Reverb/IR_Survey.html), [HF mirror](https://huggingface.co/datasets/davidscripka/MIT_environmental_impulse_responses) (card: `license: unknown`) | Unclear, so **not used**: RIRs were simulated with [pyroomacoustics](https://github.com/LCAV/pyroomacoustics) (MIT) instead |
| AudioSet clips (recipe's background noise) | Labels CC BY 4.0; the audio is YouTube content with its own rights | [AudioSet](https://research.google.com/audioset/download.html) | Unclear, **not used** |
| FMA (recipe's music background) | Per-track Creative Commons, some NC/ND; the HF mirror tag `openrail` is not the audio's licence | [FMA](https://github.com/mdeff/fma) | Mixed, **not used** |
| microWakeWord code | Apache-2.0 | [LICENSE](https://github.com/kahrendt/microWakeWord/blob/main/LICENSE) | None |
| microWakeWord negative feature sets (CHiME-6, DiPCo, FMA, FSD50K, WHAM, LibriSpeech, VOiCES) | **CC BY-NC 4.0** (dataset card); its own notebook says models trained with the recipe's data "should be considered as appropriate for **non-commercial** personal use only" | [dataset card](https://huggingface.co/datasets/kahrendt/microwakeword), [basic_training_notebook.ipynb](https://github.com/kahrendt/microWakeWord/blob/main/notebooks/basic_training_notebook.ipynb) | **Yes** (NC) |
| Multilingual LibriSpeech, Italian (used: 9 h subset to train, dev to select, test to measure) | CC BY 4.0 | [HF dataset](https://huggingface.co/datasets/facebook/multilingual_librispeech) (revision `2e83e618`), [OpenSLR 94](https://www.openslr.org/94/) | Attribution only |
| LibriSpeech dev-clean (train) and test-clean (measure) | CC BY 4.0 | [OpenSLR 12](https://www.openslr.org/12/) | Attribution only |
| Piper engine `piper-tts` 1.8.0 | GPL-3.0 | [piper1-gpl](https://github.com/OHF-Voice/piper1-gpl) | None on generated audio; the engine only ran in its own venv, as in the product |
| Piper voices used to train (positives and hard negatives), revision `e7229f7d` of [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) | Per MODEL_CARD: `it_IT-riccardo-x_low` M-AILABS ("See URL"; the primary page did not resolve; permissive per secondary sources, **unverified**); `it_IT-serena-high` CC BY 4.0 ([dataset](https://huggingface.co/datasets/committa/serena-synthetic-it-27h): generated with Qwen3-TTS-1.7B-Base, [Apache-2.0](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base), from Tatoeba sentences, CC BY 2.0 FR); `en_US-libritts_r-medium` CC BY 4.0 ([LibriTTS-R, OpenSLR 141](http://www.openslr.org/141/)); `es_ES-davefx`, `pt_PT-tugão`, `pt_BR-cadu`, `ro_RO-mihai` CC0; `es_AR-daniela-high` CC BY-SA 4.0 | `https://huggingface.co/rhasspy/piper-voices/blob/e7229f7d33a10e145e109b886d661984201ac3c4/<lang>/<locale>/<voice>/<quality>/MODEL_CARD`, for example [it_IT/serena/high](https://huggingface.co/rhasspy/piper-voices/blob/e7229f7d33a10e145e109b886d661984201ac3c4/it/it_IT/serena/high/MODEL_CARD), [en_US/libritts_r/medium](https://huggingface.co/rhasspy/piper-voices/blob/e7229f7d33a10e145e109b886d661984201ac3c4/en/en_US/libritts_r/medium/MODEL_CARD), [es_AR/daniela/high](https://huggingface.co/rhasspy/piper-voices/blob/e7229f7d33a10e145e109b886d661984201ac3c4/es/es_AR/daniela/high/MODEL_CARD) ([OpenSLR 61](https://www.openslr.org/61/)), CC0 voices: [OHF-Voice/voice-datasets](https://github.com/OHF-Voice/voice-datasets) | Attribution for CC BY; `daniela` (SA) and `riccardo` (unverified) should be dropped from a shipped training set, which is cheap (`daniela` is 200 of 6,400 positives; `riccardo` is 1,200 positives and 816 hard negatives) |
| Piper voices used only to measure: `it_IT-paola-medium` (CC0 per its [dataset](https://huggingface.co/datasets/paolapersico1/Voice-Dataset-Italian)), `it_IT-serena-medium` (same dataset as `serena-high`), `es_MX-claude-high` (apache-2.0 per MODEL_CARD, [dataset space](https://huggingface.co/spaces/HirCoir/Piper-TTS-Spanish)), `pt_BR-faber-medium` (CC0) | as listed | MODEL_CARDs at the same revision, same URL pattern | n/a |

**Verdict.** Trained with the official recipe's data (ACAV100M features, or microWakeWord's sets), the
model is **non-commercial**. Trained as done here (MLS Italian, LibriSpeech, Piper voices, simulated rooms
and synthetic noise), every input is CC BY 4.0, CC0 or code-generated, so the shipped model carries
attribution duties only (after dropping the two voices named above). The two shared openWakeWord feature
files are Apache-2.0 by origin, with the README ambiguity noted.

## 3. Training a "Neve" model

Recipe: openWakeWord's official `train.py` (commit read on 2026-11-04, version 0.6.0), steps
`--augment_clips` and `--train_model` unchanged, with the documented config keys (`examples/custom_model.yml`:
DNN, `layer_size` 32, 50,000 steps, `max_negative_weight` 1500, `target_false_positives_per_hour` 0.2).
Only the data sources were replaced, for licence reasons (section 2) and because the recipe's clip generator
is English-only (`piper-sample-generator` with the LibriTTS-R voice):

| Input | What was used | Size |
| --- | --- | --- |
| Positives, train | Piper 1.8.0, random `length_scale` 0.8 to 1.35, `noise_scale` 0.4 to 0.9, `noise_w` 0.5 to 1.1. Italian voices `riccardo-x_low`, `serena-high` with the text "Neve." / "Neve!" / "Ehi Neve!" / "Ciao, Neve." and raw IPA with closed and open e (`nˈeve`, `nˈɛve`, `ˈei nˈeve`, `tʃˈao nˈeve`); five Spanish, Portuguese and Romanian voices speaking the Italian IPA (accent spread); `en_US-libritts_r-medium`, 813 of its 904 speakers (ids not divisible by 10), with "Nevay", "Hey Nevay", "Chow Nevay" and the Italian and English IPA (English speakers saying the name) | 6,400 clips (Italian 2,400, other Romance 1,000, English 3,000) |
| Positives, validation (early stopping) | Same voices, other random draws | 700 |
| Hard negatives (adversarial) | Italian sentences with "neve" in context ("C'è la neve.", "Domani nevica in montagna.": 170 texts, 50 hand-written plus 120 templated), 58 near words and near names, ("deve", "beve", "nave", "ciao Eva", "ehi Nevio", ...), 20 English near words ("never", "Nevada", "hey Eve"). Split by text: 3/5 train, 1/5 validation, 1/5 test only | 1,978 train, 298 validation |
| General negatives | MLS Italian 9 h subset + LibriSpeech dev-clean 5.4 h, as openWakeWord windows (16 x 96, stride 320 ms). Round 2 adds one augmented copy of the same audio (simulated room, noise at 0 to 20 dB SNR, gain) and 900 pure noise clips | 14.4 h clean (round 1); + 14.2 h augmented (round 2) |
| Rooms and noise for augmentation | 300 simulated rooms (pyroomacoustics, RT60 0.15 to 0.8 s, 0.4 to 3.5 m), 300 x 5 s white, pink, brown noise and hum; LibriSpeech dev-clean as speech background | generated, no download |
| FP validation (model selection) | MLS Italian dev (5.2 h), clean in round 1, half augmented in round 2 | 5.2 h |

Two rounds were trained, plus one reference run on the recipe's own negative data:

- **v1 (permissive)**: as above, clean general negatives. Its positives were always augmented (room and
  noise) and its general negatives never were, so it learned a shortcut: "room or noise" became evidence
  for "Neve". On the held-out sets, adding reverb **raised** both recall (paola 80% to 87%) and false fires
  on "neve" sentences (3% to 18%, 42% at 10 dB SNR). This is the reason for v2.
- **v2 (permissive)**: three changes from v1: the augmented speech and pure-noise negatives, an augmented
  FP validation stream for model selection, and a new batch mix (clean speech 512 and augmented speech
  1,024 windows per batch, instead of clean speech 1,024).
- **NC reference**: v1's data plus the first 249 h of the recipe's ACAV100M feature file (700,000 rows
  fetched with an HTTP range request, 2.15 GB of 17.28 GB). This adds both volume and real-world noise at
  once, and it does **not** include v2's augmented negatives, so it is a "recipe data" reference, not a
  like-for-like licence comparison. CC BY-NC-SA: not shippable.

Compute and downloads against the time box (Mac M3 Pro, shared with other work during the run):

| Step | Wall time | Note |
| --- | --- | --- |
| Piper clip generation (11,408 clips, 8 processes) | about 60 min | `serena-high` is about 10x slower than the medium voices; its remaining hard negatives (134 train, all validation and test) were skipped |
| Decode MLS (opus) and LibriSpeech to WAV | about 45 min | single process, competing with the generation |
| `train.py --augment_clips` (features for 9,376 clips) | about 2 min | |
| Feature extraction for about 30 h of negatives | about 10 min | |
| Augmented negatives (round 2) | 16 min | 8 processes |
| `train.py --train_model`, one model | 9 min (539 s, v1) | CPU only; the TFLite export step fails without `onnx_tf` (not needed, ONNX is the target) |
| Downloads | about 5.2 GB | Piper voices 0.8 GB, MLS 0.3 GB (+0.5 GB shard not used), LibriSpeech 0.7 GB, openWakeWord FP validation 0.18 GB, ACAV head 2.15 GB, Vosk small-it 0.05 GB |

Training itself is minutes; data preparation is the cost. The full official recipe needs about 17.3 GB for
the ACAV100M features alone, plus AudioSet, FMA and the MIT RIRs; it was not run in full (licence, and the
20 GB time box).

macOS notes for whoever repeats this: `piper-tts` 1.8.0's macOS wheel ignores an espeak-ng data path longer
than its internal buffer and falls back to a build-time path (use a short symlink); the recipe pins
`scipy` below 1.15 indirectly (`acoustics` imports `sph_harm`), needs `torchaudio` 2.8 (2.9 requires
`torchcodec`) and the `fork` start method (`run_train.py`); `onnx_tf`/TensorFlow 2.8 for the TFLite export do
not install on macOS arm64 with Python 3.12 and were skipped.

## 4. Measurements

### 4.1 CPU cost

Measured with `bench_cpu.py`: 120 s of audio streamed in 80 ms chunks (1,280 samples) through openWakeWord's
streaming front end (`AudioFeatures`, as `Model.predict` uses it) plus one classifier call per chunk,
ONNX Runtime with 1 intra-op thread, 3 runs, machine lightly loaded (a download running).

| Path | Mac, ms of CPU per s of audio (measured) | Per 80 ms chunk, p50 / p99 (measured) | CM4, per s of audio (estimate) |
| --- | ---: | --- | --- |
| Front end + Neve classifier, ONNX Runtime | 25.2 to 25.4 | 2.0 / 2.3 ms | 0.20 to 0.51 s, so **20 to 51% of one A72 core** always on |
| Front end only, ONNX Runtime | 25.1 | 2.0 / 2.2 ms | the classifier adds about 0.2 ms/s on the Mac, under 1% of the total |
| Front end only, TFLite (`ai-edge-litert`) | 23.7 to 24.2 | 1.9 / 2.1 ms | same range: TFLite brings no gain worth a second runtime |

The CM4 column applies SPEC 13's guessed Mac-to-CM4 factor of 8 to 20x (not measured, a guess). A second
anchor: the openWakeWord README says "a single core of a Raspberry Pi 3 can run 15-20 openWakeWord models
simultaneously in real-time" (README "Project Goals"), which fits a front end well under one core on an older chip.
VAD-gated, the cost scales with the share of time someone speaks: at 20 to 40% speech, about 4 to 20% of
one core (estimate). For scale: today the Vosk wake decode under load costs 460 to 710 ms per audio second
(SPEC 8, measured on the robot 2026-11-02), and the CPU is at about 95% with vision on, so even the low end
needs headroom that the governor must find (vision rates first, SPEC 13). Memory: the three ONNX files are
2.6 MB together; the classifier is 205 KB.

Where it runs: ONNX Runtime, already on the robot. TFLite would add `ai-edge-litert` (aarch64 cp312 wheel
exists, `manylinux_2_27`) or `pymicro-wakeword` (`manylinux_2_35`, glibc 2.35+) next to ONNX Runtime, for no
measured gain on this model.

### 4.2 Recall and false fires (all three models)

`evaluate.py` and `operating_points.py`. Clip sets are held out: positives from voices never used in
training (`paola`, 91 LibriTTS-R speakers, `es_MX-claude`, `pt_BR-faber`) and `serena-medium`, which is
**the same speaker** as the training voice `serena-high` (same dataset, other model), so it is not a true
held-out voice. Negatives use test-only texts (section 3), spoken by `paola`, `serena-medium` and the
training voice `riccardo` (136 of the 408 "neve" sentence clips, 48 of the 144 near-word clips). The
thresholds were swept on these same sets (there was no separate tuning set), so every "at 0.999" number is
optimistic by selection. "neve sentences" fire = false wake; "near words" fire = false wake. Streams:
MLS Italian test (5.36 h, read audiobooks, 2 utterances contain a `nev*` word), LibriSpeech test-clean
(5.59 h), openWakeWord FP validation (10.7 h: DiPCo far-field dinner party, Santa Barbara conversation,
reverberated MUSDB music; NC data, measuring only). The FP validation set is the closest to a living room
with a TV and is the headline false-accept number.

**v2 (permissive, the candidate):**

| Threshold | paola clean | paola room+noise 10 dB | serena-medium clean | LibriTTS held-out clean | LibriTTS room+noise 10 dB | "neve" sentences clean | "neve" sentences room+noise 10 dB | near words clean | MLS-it FA/h | LibriSpeech FA/h | FP-validation FA/h |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 86% | 57% | 76% | 78% | 46% | 4% | 10% | 29% | 18.7 | 24.2 | 80.5 |
| 0.99 | 82% | 46% | 73% | 73% | 38% | 2% | 4% | 22% | 7.8 | 6.6 | 26.6 |
| 0.998 | 79% | 39% | 70% | 69% | 31% | 2% | 2% | 15% | 4.7 | 2.7 | 9.9 |
| 0.999 | 71% | 26% | 62% | 64% | 25% | 1% | 1% | 10% | 1.9 | 1.6 | 4.1 |
| 0.9993 | 14% | | | 51% | | | | | 0.9 | 0.9 | 2.0 |
| 0.9995 | 1% | | | 6% | | | | | 0.2 | 0.5 | 0.1 |

The scores saturate just under 1 (the highest stream score was 0.9995), so above 0.999 the operating point
collapses: there is no threshold with both a usable recall and under 0.5 false accepts per hour.

**v1 (permissive, round 1, shortcut)** at 0.999: paola clean 66%, room+noise 48%; FA/h 10.6 (MLS-it),
5.7 (LibriSpeech), **283.6** (FP validation). The noisy far-field set is where the shortcut shows.

**NC reference (recipe data, 249 h of ACAV100M, not shippable):**

| Threshold | paola clean | paola room+noise 10 dB | serena-medium clean | LibriTTS held-out clean | "neve" sentences clean | near words clean | MLS-it FA/h | LibriSpeech FA/h | FP-validation FA/h |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 68% | 50% | 68% | 62% | 2% | 12% | 10.4 | 7.0 | 72.1 |
| 0.99 | 60% | 38% | 63% | 51% | 0% | 6% | 2.1 | 0.7 | 10.4 |
| 0.995 | 52% | 34% | 60% | 42% | 0% | 5% | 1.3 | 0.4 | 4.5 |
| 0.997 | 25% | | | 22% | | | 0.4 | 0.0 | 1.7 |

At equal false accepts on the far-field FP validation set (about 4 per hour), v2 keeps more recall than
the NC reference (paola clean 71% against 52%), while the NC reference has fewer false accepts on clean read
speech at that point (MLS-it 1.3 against 1.9 per hour, LibriSpeech 0.4 against 1.6). Neither dominates.
Neither reaches openWakeWord's own target (README: under 5% false rejects and under 0.5 false accepts per
hour).

**The phrase and the bare name behave very differently** (v2, evaluation voices, share of clips detected):

| Voice | Condition | "ehi/ciao Neve" at 0.5 | at 0.999 | bare "Neve" at 0.5 | at 0.999 |
| --- | --- | ---: | ---: | ---: | ---: |
| paola (n 279 / 121) | clean | 100% | 95% | 52% | 16% |
| paola | room + noise 10 dB | 71% | 35% | 25% | 5% |
| serena-medium (same speaker as a training voice; n 135 / 65) | clean | 99% | 87% | 29% | 11% |
| LibriTTS held-out, English speakers (n 356 / 144) | clean | 96% | 84% | 35% | 13% |
| LibriTTS held-out | room + noise 10 dB | 60% | 35% | 12% | 2% |
| es_MX-claude (n 107 / 43) | clean | 99% | 93% | 63% | 44% |
| pt_BR-faber (n 108 / 42) | clean | 96% | 92% | 7% | 0% |

What fires on negatives (v2, threshold 0.5, clean): "Ehi Eva!" 11 of 13, "Ciao Elena!" 10 of 11, "Ciao
Nello!" 10 of 18, "Deve." 5 of 14, English "Hey Nate." 35 of 41. The model has learned "greeting + a
two-syllable name" more than "Neve". "neve" inside a sentence rarely fires: sentence-final "...la neve."
16 of 116 clean, "neve" elsewhere 2 of 292 ("C'è la neve." alone 8 of 13). Hard negatives with other names
after "ciao"/"ehi" are the most useful addition for the next round.

### 4.3 False accepts per hour, summary

| Detector | MLS-it (5.36 h) | LibriSpeech (5.59 h) | FP validation, far field + music (10.7 h) |
| --- | ---: | ---: | ---: |
| v2 at 0.999 | 1.9 | 1.6 | 4.1 |
| v2 at 0.998 | 4.7 | 2.7 | 9.9 |
| NC reference at 0.995 | 1.3 | 0.4 | 4.5 |
| Vosk text rule (today), free text | 0.0 (0 wakes; "neve" in the text 3 times) | not run | not run (features only, no audio) |

False accepts per hour on Italian speech that is **all** "neve" sentences (the 408 test clips, 0.193 h of
speech without the padding; a dense worst case, as if someone talked only about snow): v2 at 0.999 fires 5
times, **26 per hour**, clean and with room + noise alike (at 0.99: 47 clean, 83 with noise). Today's Vosk
rule wakes 95 times clean (23.3%), **about 490 per hour**, and 73 times in white noise (about 380 per
hour), because short sentences such as "C'è la neve." pass its short-final rule.

### 4.4 Baseline: today's Vosk text rule on the same clips

`vosk_baseline.py` (free text only, no grammar pass; wake = "ehi/ciao/hey neve" in the final, or "neve" in
a final of at most 4 words, `voice.fastpath.alias_wake_max_words`):

| Set | n | Wake, clean | Wake, white noise 10 dB | "neve" in the text, clean |
| --- | ---: | ---: | ---: | ---: |
| Positives, `paola` (held out) | 400 | 78.5% | 47.8% | 78.5% |
| Positives, `serena-medium` | 200 | 91.5% | 18.5% | 91.5% |
| Sentences with "neve" (34 test texts x 3 voices x 4) | 408 | 23.3% (false) | 17.9% (false) | 87.0% |
| Near words and names | 144 | 0.0% | 0.7% | 0.0% |

On the 5.27 h of MLS-it test audio (`vosk_stream_fa.py`, every Vosk final counted as a final text), the
Vosk rule woke **0** times; the word "neve" appeared in its text 3 times, all in long finals.

The brief's earlier figure was "about one time in four" for a synthetic "Neve". Here Vosk heard the word
in 78.5% (paola) and 91.5% (serena-medium) of clean clips. The clips differ: these are short isolated
phrases ("Neve!", "Ciao Neve.") with 1 s of silence, Piper's own Italian voices, random speaking rates;
the earlier measurement's conditions are not recorded in the SPEC. White noise at 10 dB drops Vosk to 18 to
48%. The Vosk rule also wakes on 23% of the "neve" sentences, because many are 3 or 4 words ("C'è la
neve.", "Che bella la neve!") and pass the short-final rule.

**Vosk, spotter (v2 at 0.999) and both together**, per clip, split by phrase and bare name (white noise
10 dB is the same kind of condition for both, with different noise draws):

| Voice | Condition | Kind (n) | Vosk rule | Spotter | Either |
| --- | --- | --- | ---: | ---: | ---: |
| paola | clean | "ehi/ciao Neve" (279) | 86% | 95% | 97% |
| paola | clean | bare "Neve" (121) | 62% | 16% | 64% |
| paola | white noise 10 dB | "ehi/ciao Neve" | 46% | 78% | 84% |
| paola | white noise 10 dB | bare "Neve" | 51% | 16% | 59% |
| serena-medium | clean | "ehi/ciao Neve" (135) | 92% | 87% | 99% |
| serena-medium | clean | bare "Neve" (65) | 91% | 11% | 92% |
| serena-medium | white noise 10 dB | "ehi/ciao Neve" | 14% | 61% | 69% |
| serena-medium | white noise 10 dB | bare "Neve" | 28% | 9% | 29% |

So the spotter adds little for the bare name (Vosk already does better) and a lot for the phrase in noise
(+38 and +55 points over Vosk alone); in quiet rooms the two together reach 97 to 99% on the phrase. This
is the evidence for a phrase-only spotter beside Vosk.

## 5. Integration sketch (no code)

Where it sits. A new `roboemot.voice.spotter` beside the VAD in the listener's chunk path
(`voice/listener.py` `_process`/`_ingest`), fed the same 16 kHz mono samples (the hub's two channels are
identical, so one is enough). It runs on the listener thread: per 80 ms it costs one mel call, one
embedding call and one classifier call (cost in section 4). It needs about 1.3 s of continuous audio
context, so it reads the stream **beside** the VAD, not behind it. Two ways to run it:

- Always on while the robot is awake and not speaking (simplest; the cost below is for this).
- VAD-gated: idle while there is no speech, and on a speech start first fed the listener's pre-roll
  (it must be at least about 1 s) to warm up. Cheaper in a quiet room, but a missed or late VAD onset
  costs the wake word. Start with "always on" and measure on the robot.

It stops where the listener already stops: while the robot speaks (`SpeechActivity`; the robot's own
Piper voice is `paola`, and the v2 spotter fires on 86% of clean `paola` clips at 0.5 (100% of the
"ehi/ciao Neve" ones), so it must never hear the robot), while asleep (`robot.asleep`,
`runtime.pause_when_asleep`) and while paused.
The barge-in path keeps its own detector; the spotter does not run on audio captured over the robot's
voice.

How a detection wakes. A detection is a score at or above `voice.spotter.threshold` on one 80 ms
window, then a refractory time. It is stamped with the audio time of the window and attached to the
utterance whose span covers it (or the one that starts within a short grace after it, because the
detection can come before the VAD start is confirmed). That utterance gets the `wake` category from the
spotter, the same way a grammar-pass rescue adds one today, so `WakeGate` (`voice/wake.py`) decides
`wake_word` and opens the addressed window as now. No new wake path, no change to the window rules.
With the realtime route's audio input, a spotter hit at speech start makes the utterance addressed
**before** the voice-at-face check, so the wake decode of the first 3 s (`wake_decode_max_s`, measured
460 to 710 ms per audio second at load, SPEC 8) is no longer needed for it.

The bare name. One model cannot tell "Neve" from "ehi Neve", and in a sentence "la neve" sounds the
same as the name. The measurement settles most of it: the bare name is detected 11 to 16% of the time at
a usable threshold, the phrase 87 to 95%. Two options:

1. One model (as trained here) and keep the short-final rule for it: a spotter hit counts as a wake
   only when the utterance it lands in is short (`voice.spotter.bare_max_s`, for example 1.5 s of speech,
   from the VAD, no Vosk needed). That trades the false wakes from sentences (measured below) for missed
   wakes when the owner says "Neve, che ore sono?" in one breath.
2. Two models on the shared front end (each extra model costs only its classifier): `neve_phrase` for
   "ehi/ciao Neve", always a wake; `neve_bare` gated by the short-utterance rule. Training both is the
   same recipe with split positives.
3. (Recommended for the next slice.) A phrase-only model; the bare name stays with Vosk's short-final
   rule as today. Simplest, and it drops the part the spotter does worst: on the bare name Vosk alone
   scored 62 to 91% clean against the spotter's 11 to 16% (section 4.4).

What stays with Vosk. Transcription, all other keywords (stop, yes, no, greetings), the grammar pass,
speaking rate, and the text wake phrases as a **fallback** (the spotter is an extra detector, so a Vosk
hit still wakes). The name-from-platform logic (`robot_name()`, `name_wake_templates`,
`vosk_model_find_word`) stays for the text path.

Config fields (all under `voice.spotter`, SPEC 11A, with unit and range):

| Field | Meaning | Start value |
| --- | --- | --- |
| `enabled` | Run the spotter | true on cm4 once measured |
| `model` | Manifest name of the classifier (fetched and sha256-verified like the voice models, `voice/models.yaml`) | `neve-oww` |
| `threshold` | Score for a detection | 0.999 for v2 (scores saturate near 1; to tune on the robot) |
| `refractory_s` | No second detection for this long | 2.0 s |
| `attach_grace_s` | A detection before an utterance start still belongs to it | 0.5 s |
| `bare_max_s` | Option 1 only: longest utterance in which a hit wakes | 1.5 s |
| `gate` | `always` or `vad` (with the pre-roll warm-up) | `always` |
| `intra_op_threads` | ONNX Runtime threads | 1 |

Debug report (counts and timings only, SPEC 12A): detections, detections that woke, detections
dropped by the short-utterance rule, detections with no utterance to attach to, wakes that Vosk alone
would have missed and the reverse, the score summary (min, median, p90, max, last) of detections, and
the cost per audio second (last, worst) and its share of `voice.listener.budget_share`.

Privacy (SPEC 12). The spotter keeps at most its rolling feature buffer (about 2 s of mel frames and
16 embedding frames) in memory; it never writes audio, features, embeddings or scores per frame to logs
or reports. Only the counts and summaries above leave the process.

Renamed robots. The model only knows "Neve". If the owner renames the robot in the Pollen app, the
spotter still wakes on "Neve" and not on the new name; the new name keeps working only through the Vosk
text path (which today requires every word of it to be in the recognizer's vocabulary). Options: turn
the spotter off when the platform name is not "Neve" (`robot_name()` already tells), or ship it
regardless as "Neve is always a wake word". Training a model per name on the robot is not realistic
(section 3 times). This is an owner decision.

## How to reproduce

All paths go under `WAKEWORD_DIR` (outside the repo); every script takes it as an argument or an
environment variable and nothing is written to the repo. Three venvs, as used here: `piper`
(`piper-tts==1.8.0 scipy`), `train` (`onnxruntime numpy "scipy<1.15" pyroomacoustics pyarrow soundfile
onnx`, plus openWakeWord installed `--no-deps` from a clone, and its v0.5.1 `melspectrogram.onnx` /
`embedding_model.onnx` in `openwakeword/resources/models/`), `train2` (the same with `torch==2.8.0
torchaudio==2.8.0 speechbrain acoustics audiomentations torch-audiomentations pronouncing mutagen torchinfo
torchmetrics` for `train.py`), and `vosk` (`vosk`). Run Python with `-I`. Order, from
`docs/spikes/2026-11-04-neve-wake-word/`:

1. `fetch_voices.sh`, `fetch_negatives.sh`, `fetch_round2.sh` (NC reference only); Vosk small-it from
   `voice/models.yaml`'s URL into `$WAKEWORD_DIR/data/vosk/`.
2. `gen_clips.py $WAKEWORD_DIR/out` (piper venv, `ESPEAK_DATA` set to a short symlink), `prep_speech.py`,
   `make_rir_noise.py`, `neg_features.py`, `aug_neg_features.py $WAKEWORD_DIR mls_9h ls_dev`, `fpval_aug.py`,
   `acav_subset.py` (NC reference only), `train_configs.py`.
3. openWakeWord `train.py --training_config neve.yaml --augment_clips` once, then `run_train.py
   <path>/train.py --training_config <config> --train_model` per config.
4. `dyn_batch.py` on each exported model, `evaluate.py`, `operating_points.py`, `textmap.py` then
   `summarize.py`, `vosk_baseline.py`, `vosk_stream_fa.py`,
   `bench_cpu.py`.

## 6. Recommendation

**No-go to ship the model trained here; go for openWakeWord with one more slice, phrase-only.**

Why not ship now: at the threshold where the phrase still works (0.999), v2 gives about 4 false accepts per
hour on far-field speech and music, 8 times openWakeWord's own target; phrase recall in a noisy room is 20
to 35%; the bare name barely works. Added as an extra wake path beside Vosk, it would add those 2 to 4 false
wakes per hour (Vosk's text rule had 0 on 5.3 h of Italian speech). Every recall number is on synthetic
voices, and the thresholds were chosen on the evaluation sets themselves.

Why it is still worth one slice: as a second detector beside Vosk it raises phrase recall in white noise
from 14 to 46% (Vosk alone) to 69 to 84% (either), and to 97 to 99% in quiet rooms (section 4.4); alone it
is 87 to 95% on the phrase in quiet rooms (Italian voices) and 84% for unseen English speakers. "neve"
inside sentences fires on 1% of clips (Vosk's rule: 23%). The residual false accepts are of a kind
training data fixes (other names after "ciao"/"ehi"), and the changes between rounds (augmented
negatives, augmented selection stream, batch mix) cut far-field false accepts from 284 to 4 per hour at
0.999. The runtime fits what the robot has (ONNX Runtime, 2.6 MB), and the licence path is clean.

Expected numbers if the next slice works (estimate, to be confirmed): phrase recall at least 90% clean and
60% at 10 dB SNR, under 0.5 false accepts per hour on the FP validation set, at 20 to 51% of one CM4 core
always on (or 4 to 20% VAD-gated, estimate).

Next slice (data and measurement, still no product code):

1. Phrase-only positives ("ehi Neve", "ciao Neve", "hey Neve"), plus hard negatives "ciao/ehi + other
   two-syllable names" (Eva, Elena, Nello, Nina, Neri, Nate, ...) in Italian and English.
2. More permissive negative volume: the full MLS Italian train set (8 shards, 3.88 GB, CC BY 4.0; shard 0
   holds 30.8 h, so about 247 h in all), with the augmented copy; check whether a CC0 or CC BY music set
   is usable for training (none was found here).
3. Positives with augmentation that also covers the robot's mic (no room-only shortcut: augment negatives
   the same way, as in v2), and drop `daniela` and `riccardo` from the shipped training set.
4. Measure on the CM4 itself: the front end's cost with `scripts/bench/run_cm4.py`-style timing while
   vision runs (the 20 to 51% estimate decides always-on against VAD-gated).
5. Collect a small consented set of real "ehi Neve" / "ciao Neve" recordings on the robot (owner and
   family), kept off the repo, to replace the synthetic recall numbers.

If that slice still misses 0.5 false accepts per hour, keep Vosk's text rule and the grammar pass, and drop
the spotter.
