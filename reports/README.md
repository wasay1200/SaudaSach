# Evidence index

Start with [RESULTS.md](RESULTS.md). Full raw score predictions are losslessly compressed as `test_predictions.csv.gz`; no audio is included. Their uncompressed SHA-256 matches `final_test_access.json`. Python can read the archive with `gzip.open`, and pandas can read it with `read_csv` directly.

Run `python scripts/verify_saved_results.py` to recompute all 81 model/condition metric groups from saved predictions, check frozen weight/config/split hashes and validate the deployment bundle. This does not rerun training or tune against test results. `verification.json` records the completed independent arithmetic/integrity check.

- `data_audit.json`: retained/excluded counts and actual folder groups.
- `test_metrics.json`: class recalls, false positives, confusion matrices, balanced accuracy, EER, F1, AUC and bootstrap intervals.
- `comparison.json`: all three paired CNN seed comparisons and aggregate folder-bootstrap interval.
- `final_test_access.json`: immutable-study identities and the completed final-use record.
- `export_parity.json`: ten development-fixture PyTorch versus ONNX comparisons.
- `quantization_evaluation.json`: all nine full-precision/INT8 condition comparisons; the target uses the worst condition's EER change.
- `host_benchmark.json`: CPU-host timing, explicitly not Android evidence.
- `source_revision_map.json`: mapping local training commit IDs to the published matching Python source.
- `demo_predictions.json`: file inference on two real development clips; neither is training/test evidence beyond the declared study.
- `unit_tests.xml`: 14 passing deterministic implementation checks.

The primary improvement target was missed. Some noisy conditions improved, but the overall 95% gain interval crosses zero. False positives, unseen-family failure and the small number of held-out groups prevent a claim of citizen-ready authenticity verification.
