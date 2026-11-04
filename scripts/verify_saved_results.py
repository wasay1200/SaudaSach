"""Independently recompute exported metrics and verify the frozen artifacts."""

import csv, gzip, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from saudasach.common import ROOT, sha, read_csv, assert_disjoint, write_json
from saudasach.metrics import metrics
from saudasach.bundle import load_bundle

assert_disjoint(read_csv(ROOT / "manifests/splits.csv"))
checksum_file = ROOT / "reports/artifact_checksums.json"
if checksum_file.exists():
    for file, record in json.loads(checksum_file.read_text())["files"].items():
        assert sha(ROOT / file) == record["sha256"], file
selection = json.loads((ROOT / "artifacts/selection.json").read_text())
assert sha(ROOT / "manifests/splits.csv") == selection["manifest_sha256"]
assert sha(ROOT / "configs/study.json") == selection["config_sha256"]
for file, digest in selection["weight_hashes"].items():
    assert sha(ROOT / file) == digest, file
path = ROOT / "reports/test_predictions.csv"
compressed = not path.exists()
with (
    gzip.open(str(path) + ".gz", "rt", newline="")
    if compressed
    else open(path, newline="")
) as f:
    rows = list(csv.DictReader(f))
groups = defaultdict(list)
for r in rows:
    groups[(r["model"], r["condition"])].append(r)
expected = json.loads((ROOT / "reports/test_metrics.json").read_text())
checks = 0
for (model, condition), items in groups.items():
    result = metrics(
        [int(r["label"]) for r in items],
        [float(r["score"]) for r in items],
        float(items[0]["threshold"]),
    )
    for key, value in result.items():
        np.testing.assert_allclose(
            value, expected[model][condition][key], rtol=0, atol=1e-12
        )
        checks += 1
load_bundle(ROOT / "artifacts/deployment")
write_json(
    ROOT / "reports/verification.json",
    {
        "passed": True,
        "frozen_model_hashes_verified": len(selection["weight_hashes"]),
        "metric_groups_recomputed": len(groups),
        "metric_checks": checks,
        "prediction_rows": len(rows),
        "manifest_config_and_available_group_hash_checks": "passed",
        "deployment_bundle_integrity": "passed",
        "note": "Recomputation validates saved arithmetic and integrity, not population-level model quality or Android behavior.",
    },
)
print(
    "Verified frozen models and",
    len(groups),
    "metric groups from",
    len(rows),
    "predictions.",
)
