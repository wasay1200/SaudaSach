"""Record checksums for every delivered model and frontend fixture."""

from saudasach.common import ROOT, sha, write_json

files = {
    str(p.relative_to(ROOT)): {"sha256": sha(p), "bytes": p.stat().st_size}
    for p in sorted((ROOT / "artifacts").rglob("*"))
    if p.is_file()
}
write_json(
    ROOT / "reports/artifact_checksums.json",
    {
        "files": files,
        "scope": "All trained checkpoints, metadata, ONNX variants and artificial frontend fixtures",
    },
)
print("Recorded", len(files), "artifact checksums.")
