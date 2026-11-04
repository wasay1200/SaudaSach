# Model interface for the later app

The app is not included in this delivery. The working reference implementation is `saudasach predict`. Treat `artifacts/deployment/metadata.json` and the matching model as one versioned bundle; verify the model SHA-256, schema, label order and feature settings before use. Full precision is the default. INT8 is an evaluated alternative, not a silent replacement.

## Input and frontend

Accept one file of 3–60 seconds, no more than 20 MiB. WAV is the verified interchange format. Additional Android decoders must be tested in the app phase. Decode to float32, downmix channels by arithmetic mean, resample to mono 16,000 Hz with the reference librosa `soxr_hq` resampler, and do not apply peak or loudness normalization. Reject empty/nonfinite samples and decoder errors. The current energy/activity screen is heuristic, not validated VAD.

For local file scoring, take 32,000-sample windows at 8,000-sample steps. Append the last full window if the regular step does not reach the end. Inputs shorter than a window are zero-padded by the research helper, but the file interface requires at least three seconds.

For each window, compute the exact Python-reference frontend:

| Setting | Value |
|---|---|
| STFT FFT length | 512 |
| Window length | 400 samples |
| Window | periodic Hann, centered within the 512-point frame |
| Frame hop | 160 samples |
| Frame centering | false |
| Spectrum | magnitude squared |
| Mel bands | 64 |
| Mel range | 0–8,000 Hz |
| Mel convention | Slaney (`htk=False`), Slaney area normalization |
| Decibels | `10*log10(max(power, 1e-10))`, reference 1.0 |
| Dynamic-range floor | 80 dB below the window's maximum |
| Tensor | float32 `[batch, 64, 197]`, channels then frames |

Pass these **unnormalized log-mel tensors** to ONNX. The graph includes the training mean/std normalization. Do not normalize a second time. `artifacts/frontend_fixtures/fixtures.npz` contains ten artificial canonical-PCM/log-mel/normalized-log-mel fixtures for later mobile parity checks; they are not detection evaluation samples. The SRS mobile tolerance remains untested until the mobile frontend is implemented.

## Scores and outcomes

The input tensor is named `logmel`. The output is `synthetic_logit`, one scalar per window. Average the window logits, then apply the fixed development-selected thresholds from metadata. An aggregate inside the uncertainty interval produces `Inconclusive`. Above the interval, use `Synthetic speech suspected`; below it, use `No strong synthetic evidence`.

A logit is not a calibrated probability. Never present it as “percent authentic.” Every outcome must retain the limitation that this does not verify identity, truthfulness, or absence of fraud. A technical failure must remain an error rather than a completed low-risk result.

## Research versus demo aggregation

Primary controlled experiment results use at most six seconds per clip and a two-second step, with per-clip mean logits. The file-inference reference supports 3–60-second inputs with a 0.5-second step. These are distinct profiles: the saved primary test metrics do not establish identical performance for longer or overlapping-window inputs. Host benchmark evidence does not establish Android latency, energy use, permissions, cleanup, sharing or offline UI behavior.
