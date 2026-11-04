"""Download the public corpus at an immutable revision, excluding raw audio from Git."""

import argparse, json, time
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

p = argparse.ArgumentParser()
p.add_argument("--output", default="data/raw")
a = p.parse_args()
repo = "CSALT/deepfake_detection_dataset_urdu"
config = json.loads(Path("configs/study.json").read_text())
revision = config["revision"]
info = HfApi().dataset_info(repo, revision=revision)
Path("data").mkdir(exist_ok=True)
Path("data/source.json").write_text(
    json.dumps(
        {
            "repository": repo,
            "revision": revision,
            "license": "CC BY-NC 4.0 (dataset README)",
        },
        indent=2,
    )
)
print("Downloading", repo, "revision", revision, flush=True)
snapshot_download(
    repo_id=repo,
    repo_type="dataset",
    revision=revision,
    local_dir=a.output,
    max_workers=20,
)
print("Dataset download complete", flush=True)
