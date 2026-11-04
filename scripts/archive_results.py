"""Deterministically compress per-clip predictions for GitHub; preserve uncompressed hash."""

import gzip, hashlib, json
from pathlib import Path
from saudasach.common import ROOT, sha, write_json

source = ROOT / "reports/test_predictions.csv"
target = source.with_suffix(".csv.gz")
with open(source, "rb") as inp, open(target, "wb") as output:
    with gzip.GzipFile(
        filename="", mode="wb", fileobj=output, mtime=0, compresslevel=9
    ) as compressor:
        for chunk in iter(lambda: inp.read(1024 * 1024), b""):
            compressor.write(chunk)
with gzip.open(target, "rb") as f:
    decoded = hashlib.sha256(f.read()).hexdigest()
state = json.loads((ROOT / "reports/final_test_access.json").read_text())
assert decoded == state["prediction_sha256"]
state["prediction_archive"] = "test_predictions.csv.gz"
state["prediction_archive_sha256"] = sha(target)
write_json(ROOT / "reports/final_test_access.json", state)
print(
    "Archived prediction bytes",
    source.stat().st_size,
    "->",
    target.stat().st_size,
    "with verified uncompressed SHA-256.",
)
