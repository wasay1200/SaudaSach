# SaudaSach — Urdu synthetic-speech detection

Research model repository for the SaudaSach FYP. **The Android/React Native app is deferred.** This repository contains the model, reproducible training/evaluation tools, deployment export and SRS evidence.

The study compares a from-scratch compact log-mel CNN trained normally against the same architecture trained with environmental noise and actual Opus/GSM transcoding. An MFCC-statistics linear SVM is the classical reference. No pretrained detector weights are used. Performance targets are hypotheses, not guarantees; read the measured results and limitations in `reports/`.

## Installation

Python 3.11 or 3.12, FFmpeg with `libopus` and `libgsm`, and sufficient disk space for the public corpus and derived caches are required. The executed experiment uses CPU PyTorch; no GPU is required. A clean environment is recommended:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pytest -q
```

For exact versions used for the completed experiment, see `requirements-lock.txt` and each checkpoint's metadata. On Windows, activate `.venv\Scripts\activate` and run the individual Python commands below rather than the Bash script.

## Reproduce the research experiment

```bash
python download_dataset.py
python scripts/download_noise.py
saudasach audit
saudasach features
saudasach train
saudasach export
saudasach evaluate --final-test
saudasach benchmark
```

Alternatively run `bash scripts/run_study.sh`. Downloads require internet; scoring with the completed bundle is local. Raw recordings, noise WAVs and large feature caches are excluded from Git. Source metadata, split manifests, recipes, weights, ONNX exports and measurements are retained. Final evaluation refuses a second execution in the same output directory: reuse the saved predictions for reporting. A genuinely new frozen experiment belongs in a separate checkout/output directory; do not delete the guard to tune on the test set.

## Score an audio file

```bash
saudasach predict /absolute/path/to/audio.wav
```

Input limits are 3–60 seconds and 20 MiB. WAV is the tested interchange format; other formats depend on the installed libsndfile decoder and are not claimed as Android import support. The frontend resamples to mono 16 kHz, uses two-second windows with a 0.5-second scoring step, and averages logits. Outputs are “Synthetic speech suspected”, “No strong synthetic evidence”, or “Inconclusive”. The score is an **uncalibrated logit**, not an authenticity probability. The energy-based activity check is not a validated speech detector.

## Layout

| Directory | Contents |
|---|---|
| `configs/` | Frozen study specification |
| `src/saudasach/` | Audio, data audit, models, training, evaluation, ONNX inference |
| `manifests/` | Corpus splits, exclusions and deterministic augmentation recipes |
| `artifacts/` | Trained checkpoints, development-selected thresholds, deployment bundle |
| `reports/` | Final metrics, per-clip predictions, plots, parity and host benchmark |
| `docs/` | SRS source snapshot, requirement traceability and protocol |
| `tests/` | Input, leakage, codec, numerical and bundle checks |

## Claims and limitations

The corpus contains two provided synthetic families. Cross-family SVM tests are narrower than generalization to modern unseen cloning services. Folder labels define split groups; their equivalence to independent identities, transcripts and source recordings is not independently established. Exact decoded duplicates are removed and linked folders are merged before splitting; near duplicates and shared texts may remain.

The controlled primary experiment uses up to the first six seconds of each retained clip to fit the available compute budget, with clip-level metrics. File inference supports longer clips but accuracy over full-length calls is not established. This is a voice-note research model, not a system for intercepting cellular or WhatsApp calls. It does not verify identity, truthfulness, or absence of fraud. Real Android latency, feature parity, genuine code-switched Urdu–English performance, validated VAD, and usability remain future work.

## Data and permissions

Read `DATA_LICENSES.md` before use. This experiment uses public research corpora with noncommercial conditions. No new voice cloning or participant collection is performed, and raw recordings are not redistributed. No additional license grant for trained weights is implied beyond source permissions; resolve any deployment rights before commercial use.
