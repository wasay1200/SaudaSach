"""Build review documents from the saved frozen results; never train or retune."""

import csv, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(p):
    return json.loads((ROOT / p).read_text())


def pct(x):
    return f"{100 * x:.2f}%"


def main():
    audit = read("reports/data_audit.json")
    results = read("reports/test_metrics.json")
    comparison = read("reports/comparison.json")
    selection = read("artifacts/selection.json")
    quant = read("reports/quantization_evaluation.json")
    bench = read("reports/host_benchmark.json")
    bundle = read("artifacts/deployment/metadata.json")
    ci = comparison["mean_degraded_gain_ci95"]
    chosen = selection["selected"]
    meta = read(f"artifacts/{chosen}/metadata.json")
    lines = [
        "# Completed model experiment",
        "",
        f"Development-selected deployment model: **{chosen}**. Models were trained from scratch on the pinned public Urdu corpus. This report records the executed experiment, not a claim that the whole SRS app is complete.",
        "",
        "## Dataset",
        "",
        f"Downloaded {audit['downloaded_files']} recordings; retained {audit['retained']}; excluded {audit['excluded']} after duration, silence and exact-duplicate auditing. The retained split contains {audit['groups']} folder groups.",
        "",
        "| Partition | Clips | Genuine | Synthetic | Folder groups |",
        "|---|---:|---:|---:|---|",
    ]
    for p, v in audit["split_summary"].items():
        lines.append(
            f"| {p} | {v['files']} | {v['genuine']} | {v['synthetic']} | {', '.join(v['groups'])} |"
        )
    lines += [
        "",
        audit["claim"],
        "",
        "## Main CNN comparison",
        "",
        f"Mean augmented-minus-ordinary balanced-accuracy difference over the eight degraded conditions and three paired seeds: **{100 * comparison['mean_degraded_gain']:+.2f} percentage points**. Clean change: **{100 * comparison['mean_clean_change']:+.2f} points**. The combined SRS improvement target was **{'achieved' if comparison['target_achieved'] else 'missed'}**. This is a point-estimate target check, not proof of statistical significance; inspect the paired folder-bootstrap intervals.",
        "",
        "| Condition | Ordinary mean BA | Augmented mean BA | Difference (points) |",
        "|---|---:|---:|---:|",
    ]
    for c in read("configs/study.json")["test_conditions"]:
        import statistics

        a = statistics.mean(
            results[f"cnn_ordinary_seed{s}"][c]["balanced_accuracy"]
            for s in [42, 123, 2024]
        )
        b = statistics.mean(
            results[f"cnn_augmented_seed{s}"][c]["balanced_accuracy"]
            for s in [42, 123, 2024]
        )
        lines.append(f"| {c} | {pct(a)} | {pct(b)} | {100 * (b - a):+.2f} |")
    lines += [
        "",
        f"Overall paired folder-bootstrap 95% interval for the mean degraded gain: {100 * ci['low']:+.2f} to {100 * ci['high']:+.2f} points, conditional on the three trained seeds and fixed conditions. Only {ci['groups']} held-out groups are available.",
    ]
    lines += [
        "",
        "## Selected model and SVM references",
        "",
        "| Model / condition | Clips | Balanced accuracy | EER | Genuine FPR |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in [chosen, "svm_reference", "svm_without_tacotron", "svm_without_tts"]:
        for c in ["clean", "noise10", "opus24", "gsm13"]:
            v = results[model][c]
            lines.append(
                f"| {model} / {c} | {v['n']} | {pct(v['balanced_accuracy'])} | {pct(v['eer'])} | {pct(v['false_positive_rate'])} |"
            )
    lines += [
        "",
        "The two SVM family-holdout rows use genuine plus the held-out synthetic family only; their sample composition differs from the main comparison. They do not establish unseen modern cloning-tool robustness. Full per-class metrics, confusion matrices, all nine conditions, confidence intervals and frozen per-clip scores are in `test_metrics.json`, `test_predictions.csv.gz` and `comparison.json`.",
        "",
        "## Deployment evidence",
        "",
        f"The selected CNN has {meta['parameters']:,} trainable parameters. PyTorch/ONNX maximum absolute logit error was {bundle['onnx_max_abs_error']:.8g} on ten development fixtures. INT8 export reduced the ONNX file by {pct(quant['storage_reduction'])}, with mean EER change {100 * quant['mean_eer_increase']:+.2f} points; the worst condition changed EER by {100 * quant['worst_condition_eer_increase']:+.2f} points. The SRS quantization point-estimate target was **{'achieved' if quant['target_achieved'] else 'missed'}**. Full precision remains the default bundle.",
        "",
        f"CPU-host preprocessing-plus-inference timing across {bench['runs']} warmed runs: median {bench['median_seconds']:.4f} s, p95 {bench['p95_seconds']:.4f} s. This is not an Android measurement. See `host_benchmark.json` for the exact profile.",
        "",
        "## Limits and remaining work",
        "",
        "- Primary evaluation analyzes up to six seconds per clip, using nonoverlapping two-second windows plus a final full window. File inference uses a 0.5-second step; its longer-clip and overlapping-window operating metrics are not established by this experiment.",
        "- Folder groups and exact hashes are disjoint, but people, transcripts, near duplicates and source identities are not independently verified. Few final-test groups make bootstrap intervals coarse.",
        "- Environmental categories are limited. GSM and Opus transcoding do not reproduce a complete cellular or WhatsApp transmission path.",
        "- No independent modern generator dataset or labeled genuine Urdu–English code-switching set was available.",
        "- Energy gating is not validated VAD. The uncertainty band is heuristic; the logit is not a calibrated probability.",
        "- App screens, Android sharing/recording, device tests, mobile feature parity, privacy lifecycle and usability are deferred by request. See `docs/SRS_TRACEABILITY.md`.",
        "",
        "## Review evidence",
        "",
        "- `data_audit.json`: sample exclusions, exact-duplicate audit and split counts.",
        "- `artifacts/*/metadata.json`: seeds, source revisions, environment versions, split/config hashes and frozen thresholds.",
        "- `artifacts/*/history.json`: CNN learning histories.",
        "- `final_test_access.json`: frozen test-use record and prediction hash.",
        "- `export_parity.json`, `quantization_evaluation.json`, `host_benchmark.json`: export and deployment measurements.",
        "- `test_predictions.csv.gz`: per-clip reproduction of reported test metrics.",
        "",
    ]
    (ROOT / "reports/RESULTS.md").write_text("\n".join(lines))
    source = (ROOT / "docs/SRS_requirements_baseline.txt").read_text()
    specs = {
        "FR-017": (
            "Implemented with limits",
            "data/source.json; data/noise/source.json; manifests/splits.csv; DATA_LICENSES.md",
            "Public-provider license evidence is retained; original individual consent is not independently audited.",
        ),
        "FR-018": (
            "Implemented with limits",
            "src/saudasach/common.py; reports/data_audit.json; tests/test_pipeline.py",
            "Rejects folder-group/raw-hash/decoded-hash overlap. Broader related-source identity is not established.",
        ),
        "FR-019": (
            "Implemented",
            "configs/study.json; manifests/augmentation_*.csv; TrainingData",
            "Same label-independent recipe distribution; deterministic seeds and source IDs.",
        ),
        "FR-020": (
            "Implemented",
            "artifacts/svm_reference/; reports/test_metrics.json",
            "MFCC statistics, training-only scaling and development-only hyperparameter choice.",
        ),
        "FR-021": (
            "Implemented",
            "artifacts/cnn_*/; reports/comparison.json",
            "Six from-scratch runs, ordinary/augmented matched architecture and budgets.",
        ),
        "FR-022": (
            "Deferred (Could)",
            "docs/EXPERIMENT_PROTOCOL.md",
            "Optional TCN not trained.",
        ),
        "FR-023": (
            "Implemented",
            "artifacts/*/metadata.json; artifacts/selection.json; reports/final_test_access.json",
            "Thresholds and model selection frozen from development before final evaluation.",
        ),
        "FR-024": (
            "Implemented",
            "reports/test_metrics.json; reports/test_predictions.csv.gz",
            "All mandatory clean/noise/codec conditions and per-class metrics retained.",
        ),
        "FR-025": (
            "Partial",
            "artifacts/svm_without_*/; reports/test_metrics.json",
            "Directional public-family transfer tests only; code-switching and independent new cloning generators not evaluated.",
        ),
        "FR-026": (
            "Partial",
            "artifacts/deployment/quantization.json; reports/host_benchmark.json",
            "Model bytes and CPU-host latency measured; reference Android device deferred.",
        ),
        "FR-027": (
            "Implemented for Python bundle",
            "artifacts/deployment/; reports/export_parity.json",
            "ONNX, preprocessing config, labels, thresholds, aggregation, schema and hash; mobile verification deferred.",
        ),
        "FR-028": (
            "Implemented",
            "reports/quantization_evaluation.json; artifacts/deployment/model_int8.onnx",
            "Frozen full-precision versus training-calibrated INT8 comparison.",
        ),
        "NFR-019": (
            "Implemented",
            "artifacts/*/metadata.json; requirements-lock.txt",
            "Code revision, source hashes, seed, config/split hashes and package versions retained.",
        ),
        "NFR-020": (
            "Implemented with limits",
            "reports/data_audit.json; manifests/splits.csv",
            "Zero available group/hash overlaps; no independent guarantee of broader source identity.",
        ),
        "NFR-021": (
            "Implemented",
            "reports/comparison.json; reports/test_metrics.json",
            "Three paired seeds and 1,000 folder-cluster bootstrap replicates; few groups limit precision.",
        ),
        "NFR-022": (
            "Target " + ("achieved" if comparison["target_achieved"] else "missed"),
            "reports/comparison.json",
            "Measured against the declared paired-seed protocol; target not promised.",
        ),
        "NFR-024": (
            "Target " + ("achieved" if quant["target_achieved"] else "missed"),
            "reports/quantization_evaluation.json",
            "Storage and worst-condition EER measured; inspect every condition before deployment.",
        ),
    }
    records = []
    for prefix, count in [("FR", 29), ("NFR", 24)]:
        for n in range(1, count + 1):
            key = f"{prefix}-{n:03d}"
            needle = "\n" + key + "\n"
            description = source.split(needle, 1)[1].splitlines()[0]
            status, evidence, note = specs.get(
                key,
                (
                    "Deferred — app/device phase",
                    "docs/SRS_requirements_baseline.txt",
                    "App implementation and physical-device verification excluded from this model-only delivery.",
                ),
            )
            if key == "NFR-012":
                status, evidence, note = (
                    "Not exercised — no new collection",
                    "DATA_LICENSES.md",
                    "Only released public corpora used; do not treat their availability as independent participant-consent verification.",
                )
            records.append(
                dict(
                    requirement=key,
                    description=description,
                    status=status,
                    evidence=evidence,
                    limitation=note,
                )
            )
    with open(ROOT / "docs/SRS_TRACEABILITY.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=records[0])
        w.writeheader()
        w.writerows(records)
    text = [
        "# SRS implementation traceability",
        "",
        "Scope: model/research delivery only. The original SRS remains the requirement baseline; this matrix does not amend or approve it. Python equivalents do not satisfy Android requirements.",
        "",
        "| Requirement | Status | Evidence / limitation |",
        "|---|---|---|",
    ]
    for r in records:
        text.append(
            f"| {r['requirement']} | {r['status']} | {r['evidence']}. {r['limitation']} |"
        )
    (ROOT / "docs/SRS_TRACEABILITY.md").write_text("\n".join(text) + "\n")
    print("Wrote RESULTS.md and 53-row SRS traceability.")


if __name__ == "__main__":
    main()
