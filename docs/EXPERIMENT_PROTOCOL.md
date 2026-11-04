# Frozen experiment protocol

## Research question

Does environmental-noise and compression augmentation improve the balanced accuracy of a compact Urdu synthetic-speech CNN relative to ordinary training, while limiting clean-audio loss? The declared target is a mean degraded-condition gain of at least five percentage points and a clean loss no greater than two percentage points, averaged across three paired seeds. Targets may fail and will be reported unchanged.

## Dataset and grouping

Use the full pinned public corpus inventory, excluding clips shorter than two seconds, near-silent inputs and exact decoded duplicates. Conflicting-label duplicates are excluded. Link duplicate-connected folder groups before deterministic group splitting. About 70/15/15 percent of the groups are used for train/development/test; exact counts are determined by whole-group assignment and reported. Folder names are a conservative available proxy, not a verified claim of independent people or texts. No corpus-wide fitted statistics enter training.

Use up to the first six seconds per recording for primary feature generation. Window the CNN inputs into two-second segments at two-second training/test-experiment steps, adding the last full segment when needed. Average logits per clip before computing metrics. This experimental aggregation differs from the 0.5-second overlap used by the file-inference demo; no claim of identical operating metrics is made for that later runtime configuration.

## Comparators

1. MFCC mean/std statistics with first/second deltas (240 features), StandardScaler fitted on training only, balanced linear SVM. Select C from 0.01, 0.1, 1, 10 using development balanced accuracy.
2. Compact temporal CNN over 64-channel log-mel features, randomly initialized. Normalize using clean training-window statistics only. Train ordinary and augmented variants with seeds 42, 123, 2024; match architecture, optimizer, batch size, maximum epochs and windows per epoch. Early stopping can yield different actual epoch counts; report them.
3. Two directional SVM family-holdout studies. Exclude one synthetic family from both training and development; evaluate genuine + held-out-family recordings only from the final grouped test. This is a limited public-corpus transfer check, not generalization to arbitrary new generators.

## Augmentation and selection

Augmented training selects clean, noise-only, codec-only or noise+codec with equal probability, independent of label. Environmental noise uses 20 or 10 dB SNR; codecs are actual FFmpeg Opus at 24 kb/s and GSM full rate. Environmental-noise sources are partition-disjoint. Development selection averages clean, noise10, opus24 and gsm13 balanced accuracy at logit zero. Checkpoint selection, operating thresholds, uncertainty interval and deployable-model choice use development only. The uncertainty interval is a documented heuristic, not calibrated confidence.

Nine predeclared tests: clean; environmental noise at 20, 10 and 5 dB; Opus24; GSM; noise10+Opus24; noise10+GSM; Opus12. Noise5 and Opus12 are conditions not present in training. They do not constitute new speech generators. Store the model/config/split hashes before opening final tests. Do not alter the study in response to final results.

## Reporting

Report clip counts, balanced accuracy, ordinary accuracy, synthetic/genuine recall, false-positive rate, F1, ROC-AUC, interpolated EER and confusion matrix for every model/condition. Keep thresholds fixed from development. EER summarizes the test ROC; it is not a deployment threshold and must never be used to retune the released model. Use 1,000 folder-cluster bootstrap replicates for balanced-accuracy intervals and paired CNN differences; very few test groups limit precision. Report seed variation separately; seeds are not independent people.

Export the development-selected CNN to ONNX and compare Python tensor logits to PyTorch. Calibrate INT8 using training windows only, evaluate frozen INT8/full-precision variants with the same development threshold and report size/EER changes. Default full precision remains the selected release pending trade-off review. Host latency is explicitly a CPU-host measurement, not satisfaction of the Android requirement. Do not substitute Python tensor parity for Android feature-extraction parity.
