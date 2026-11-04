# Completed model experiment

Development-selected deployment model: **cnn_augmented_seed123**. Models were trained from scratch on the pinned public Urdu corpus. This report records the executed experiment, not a claim that the whole SRS app is complete.

## Dataset

Downloaded 6794 recordings; retained 6668; excluded 126 after duration, silence and exact-duplicate auditing. The retained split contains 16 folder groups.

| Partition | Clips | Genuine | Synthetic | Folder groups |
|---|---:|---:|---:|---|
| train | 4277 | 2100 | 2177 | speaker_03, speaker_04, speaker_05, speaker_07, speaker_08, speaker_09, speaker_11, speaker_13, speaker_14, speaker_16 |
| dev | 1200 | 600 | 600 | speaker_10, speaker_12, speaker_17 |
| test | 1191 | 599 | 592 | speaker_01, speaker_02, speaker_06 |

Folder-group-disjoint, not independently verified speaker identity or text/source independence. No near-duplicate guarantee.

## Main CNN comparison

Mean augmented-minus-ordinary balanced-accuracy difference over the eight degraded conditions and three paired seeds: **+2.41 percentage points**. Clean change: **-1.61 points**. The combined SRS improvement target was **missed**. This is a point-estimate target check, not proof of statistical significance; inspect the paired folder-bootstrap intervals.

| Condition | Ordinary mean BA | Augmented mean BA | Difference (points) |
|---|---:|---:|---:|
| clean | 90.95% | 89.34% | -1.61 |
| noise20 | 86.30% | 87.35% | +1.05 |
| noise10 | 82.27% | 87.41% | +5.14 |
| noise5 | 75.79% | 84.30% | +8.51 |
| opus24 | 90.41% | 89.33% | -1.08 |
| gsm13 | 86.46% | 88.48% | +2.01 |
| noise10_opus24 | 83.99% | 87.65% | +3.66 |
| noise10_gsm13 | 87.61% | 88.49% | +0.88 |
| opus12 | 89.79% | 88.93% | -0.86 |

Overall paired folder-bootstrap 95% interval for the mean degraded gain: -0.82 to +6.02 points, conditional on the three trained seeds and fixed conditions. Only 3 held-out groups are available.

## Selected model and SVM references

| Model / condition | Clips | Balanced accuracy | EER | Genuine FPR |
|---|---:|---:|---:|---:|
| cnn_augmented_seed123 / clean | 1191 | 89.38% | 11.15% | 17.53% |
| cnn_augmented_seed123 / noise10 | 1191 | 85.22% | 14.02% | 28.38% |
| cnn_augmented_seed123 / opus24 | 1191 | 89.29% | 11.82% | 16.36% |
| cnn_augmented_seed123 / gsm13 | 1191 | 89.13% | 12.84% | 18.36% |
| svm_reference / clean | 1191 | 96.65% | 3.38% | 4.17% |
| svm_reference / noise10 | 1191 | 69.59% | 20.78% | 56.59% |
| svm_reference / opus24 | 1191 | 96.65% | 3.01% | 4.17% |
| svm_reference / gsm13 | 1191 | 95.38% | 4.39% | 3.84% |
| svm_without_tacotron / clean | 892 | 81.36% | 11.95% | 4.51% |
| svm_without_tacotron / noise10 | 892 | 68.12% | 27.99% | 56.93% |
| svm_without_tacotron / opus24 | 892 | 81.13% | 11.69% | 6.68% |
| svm_without_tacotron / gsm13 | 892 | 78.51% | 9.02% | 1.34% |
| svm_without_tts / clean | 898 | 50.17% | 24.08% | 0.00% |
| svm_without_tts / noise10 | 898 | 50.00% | 33.89% | 0.00% |
| svm_without_tts / opus24 | 898 | 50.00% | 30.38% | 0.00% |
| svm_without_tts / gsm13 | 898 | 50.00% | 18.39% | 0.00% |

The two SVM family-holdout rows use genuine plus the held-out synthetic family only; their sample composition differs from the main comparison. They do not establish unseen modern cloning-tool robustness. Full per-class metrics, confusion matrices, all nine conditions, confidence intervals and frozen per-clip scores are in `test_metrics.json`, `test_predictions.csv.gz` and `comparison.json`.

## Deployment evidence

The selected CNN has 21,393 trainable parameters. PyTorch/ONNX maximum absolute logit error was 1.4305115e-06 on ten development fixtures. INT8 export reduced the ONNX file by 56.21%, with mean EER change -0.01 points; the worst condition changed EER by +0.17 points. The SRS quantization point-estimate target was **achieved**. Full precision remains the default bundle.

CPU-host preprocessing-plus-inference timing across 50 warmed runs: median 0.0182 s, p95 0.0251 s. This is not an Android measurement. See `host_benchmark.json` for the exact profile.

## Limits and remaining work

- Primary evaluation analyzes up to six seconds per clip, using nonoverlapping two-second windows plus a final full window. File inference uses a 0.5-second step; its longer-clip and overlapping-window operating metrics are not established by this experiment.
- Folder groups and exact hashes are disjoint, but people, transcripts, near duplicates and source identities are not independently verified. Few final-test groups make bootstrap intervals coarse.
- Environmental categories are limited. GSM and Opus transcoding do not reproduce a complete cellular or WhatsApp transmission path.
- No independent modern generator dataset or labeled genuine Urdu–English code-switching set was available.
- Energy gating is not validated VAD. The uncertainty band is heuristic; the logit is not a calibrated probability.
- App screens, Android sharing/recording, device tests, mobile feature parity, privacy lifecycle and usability are deferred by request. See `docs/SRS_TRACEABILITY.md`.

## Review evidence

- `data_audit.json`: sample exclusions, exact-duplicate audit and split counts.
- `artifacts/*/metadata.json`: seeds, source revisions, environment versions, split/config hashes and frozen thresholds.
- `artifacts/*/history.json`: CNN learning histories.
- `final_test_access.json`: frozen test-use record and prediction hash.
- `export_parity.json`, `quantization_evaluation.json`, `host_benchmark.json`: export and deployment measurements.
- `test_predictions.csv.gz`: per-clip reproduction of reported test metrics.
