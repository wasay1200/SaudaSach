# SRS implementation traceability

Scope: model/research delivery only. The original SRS remains the requirement baseline; this matrix does not amend or approve it. Python equivalents do not satisfy Android requirements.

| Requirement | Status | Evidence / limitation |
|---|---|---|
| FR-001 | Deferred — app/device phase | docs/SRS_requirements_baseline.txt. App implementation and physical-device verification excluded from this model-only delivery. |
| FR-002 | Deferred — app/device phase | docs/SRS_requirements_baseline.txt. App implementation and physical-device verification excluded from this model-only delivery. |
| FR-003 | Deferred — app/device phase | docs/SRS_requirements_baseline.txt. App implementation and physical-device verification excluded from this model-only delivery. |
| FR-004 | Deferred — app/device phase | docs/SRS_requirements_baseline.txt. App implementation and physical-device verification excluded from this model-only delivery. |
| FR-005 | Deferred — app/device phase | docs/SRS_requirements_baseline.txt. App implementation and physical-device verification excluded from this model-only delivery. |
| FR-006 | Deferred — app/device phase | docs/SRS_requirements_baseline.txt. App implementation and physical-device verification excluded from this model-only delivery. |
| FR-007 | Deferred — app/device phase | docs/SRS_requirements_baseline.txt. App implementation and physical-device verification excluded from this model-only delivery. |
| FR-008 | Deferred — app/device phase | docs/SRS_requirements_baseline.txt. App implementation and physical-device verification excluded from this model-only delivery. |
| FR-009 | Deferred — app/device phase | docs/SRS_requirements_baseline.txt. App implementation and physical-device verification excluded from this model-only delivery. |
| FR-010 | Deferred — app/device phase | docs/SRS_requirements_baseline.txt. App implementation and physical-device verification excluded from this model-only delivery. |
| FR-011 | Deferred — app/device phase | docs/SRS_requirements_baseline.txt. App implementation and physical-device verification excluded from this model-only delivery. |
| FR-012 | Deferred — app/device phase | docs/SRS_requirements_baseline.txt. App implementation and physical-device verification excluded from this model-only delivery. |
| FR-013 | Deferred — app/device phase | docs/SRS_requirements_baseline.txt. App implementation and physical-device verification excluded from this model-only delivery. |
| FR-014 | Deferred — app/device phase | docs/SRS_requirements_baseline.txt. App implementation and physical-device verification excluded from this model-only delivery. |
| FR-015 | Deferred — app/device phase | docs/SRS_requirements_baseline.txt. App implementation and physical-device verification excluded from this model-only delivery. |
| FR-016 | Deferred — app/device phase | docs/SRS_requirements_baseline.txt. App implementation and physical-device verification excluded from this model-only delivery. |
| FR-017 | Implemented with limits | data/source.json; data/noise/source.json; manifests/splits.csv; DATA_LICENSES.md. Public-provider license evidence is retained; original individual consent is not independently audited. |
| FR-018 | Implemented with limits | src/saudasach/common.py; reports/data_audit.json; tests/test_pipeline.py. Rejects folder-group/raw-hash/decoded-hash overlap. Broader related-source identity is not established. |
| FR-019 | Implemented | configs/study.json; manifests/augmentation_*.csv; TrainingData. Same label-independent recipe distribution; deterministic seeds and source IDs. |
| FR-020 | Implemented | artifacts/svm_reference/; reports/test_metrics.json. MFCC statistics, training-only scaling and development-only hyperparameter choice. |
| FR-021 | Implemented | artifacts/cnn_*/; reports/comparison.json. Six from-scratch runs, ordinary/augmented matched architecture and budgets. |
| FR-022 | Deferred (Could) | docs/EXPERIMENT_PROTOCOL.md. Optional TCN not trained. |
| FR-023 | Implemented | artifacts/*/metadata.json; artifacts/selection.json; reports/final_test_access.json. Thresholds and model selection frozen from development before final evaluation. |
| FR-024 | Implemented | reports/test_metrics.json; reports/test_predictions.csv.gz. All mandatory clean/noise/codec conditions and per-class metrics retained. |
| FR-025 | Partial | artifacts/svm_without_*/; reports/test_metrics.json. Directional public-family transfer tests only; code-switching and independent new cloning generators not evaluated. |
| FR-026 | Partial | artifacts/deployment/quantization.json; reports/host_benchmark.json. Model bytes and CPU-host latency measured; reference Android device deferred. |
| FR-027 | Implemented for Python bundle | artifacts/deployment/; reports/export_parity.json. ONNX, preprocessing config, labels, thresholds, aggregation, schema and hash; mobile verification deferred. |
| FR-028 | Implemented | reports/quantization_evaluation.json; artifacts/deployment/model_int8.onnx. Frozen full-precision versus training-calibrated INT8 comparison. |
| FR-029 | Deferred — app/device phase | docs/SRS_requirements_baseline.txt. App implementation and physical-device verification excluded from this model-only delivery. |
| NFR-001 | Deferred — app/device phase | docs/SRS_requirements_baseline.txt. App implementation and physical-device verification excluded from this model-only delivery. |
| NFR-002 | Deferred — app/device phase | docs/SRS_requirements_baseline.txt. App implementation and physical-device verification excluded from this model-only delivery. |
| NFR-003 | Deferred — app/device phase | docs/SRS_requirements_baseline.txt. App implementation and physical-device verification excluded from this model-only delivery. |
| NFR-004 | Deferred — app/device phase | docs/SRS_requirements_baseline.txt. App implementation and physical-device verification excluded from this model-only delivery. |
| NFR-005 | Deferred — app/device phase | docs/SRS_requirements_baseline.txt. App implementation and physical-device verification excluded from this model-only delivery. |
| NFR-006 | Deferred — app/device phase | docs/SRS_requirements_baseline.txt. App implementation and physical-device verification excluded from this model-only delivery. |
| NFR-007 | Deferred — app/device phase | docs/SRS_requirements_baseline.txt. App implementation and physical-device verification excluded from this model-only delivery. |
| NFR-008 | Deferred — app/device phase | docs/SRS_requirements_baseline.txt. App implementation and physical-device verification excluded from this model-only delivery. |
| NFR-009 | Deferred — app/device phase | docs/SRS_requirements_baseline.txt. App implementation and physical-device verification excluded from this model-only delivery. |
| NFR-010 | Deferred — app/device phase | docs/SRS_requirements_baseline.txt. App implementation and physical-device verification excluded from this model-only delivery. |
| NFR-011 | Deferred — app/device phase | docs/SRS_requirements_baseline.txt. App implementation and physical-device verification excluded from this model-only delivery. |
| NFR-012 | Not exercised — no new collection | DATA_LICENSES.md. Only released public corpora used; do not treat their availability as independent participant-consent verification. |
| NFR-013 | Deferred — app/device phase | docs/SRS_requirements_baseline.txt. App implementation and physical-device verification excluded from this model-only delivery. |
| NFR-014 | Deferred — app/device phase | docs/SRS_requirements_baseline.txt. App implementation and physical-device verification excluded from this model-only delivery. |
| NFR-015 | Deferred — app/device phase | docs/SRS_requirements_baseline.txt. App implementation and physical-device verification excluded from this model-only delivery. |
| NFR-016 | Deferred — app/device phase | docs/SRS_requirements_baseline.txt. App implementation and physical-device verification excluded from this model-only delivery. |
| NFR-017 | Deferred — app/device phase | docs/SRS_requirements_baseline.txt. App implementation and physical-device verification excluded from this model-only delivery. |
| NFR-018 | Deferred — app/device phase | docs/SRS_requirements_baseline.txt. App implementation and physical-device verification excluded from this model-only delivery. |
| NFR-019 | Implemented | artifacts/*/metadata.json; requirements-lock.txt. Code revision, source hashes, seed, config/split hashes and package versions retained. |
| NFR-020 | Implemented with limits | reports/data_audit.json; manifests/splits.csv. Zero available group/hash overlaps; no independent guarantee of broader source identity. |
| NFR-021 | Implemented | reports/comparison.json; reports/test_metrics.json. Three paired seeds and 1,000 folder-cluster bootstrap replicates; few groups limit precision. |
| NFR-022 | Target missed | reports/comparison.json. Measured against the declared paired-seed protocol; target not promised. |
| NFR-023 | Deferred — app/device phase | docs/SRS_requirements_baseline.txt. App implementation and physical-device verification excluded from this model-only delivery. |
| NFR-024 | Target achieved | reports/quantization_evaluation.json. Storage and worst-condition EER measured; inspect every condition before deployment. |
