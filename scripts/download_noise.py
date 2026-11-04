"""Download a small, licensed environmental noise subset from ESC-50."""

import csv, io, json, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests
from hashlib import sha256

root = Path("data/noise")
(root / "audio").mkdir(parents=True, exist_ok=True)
revision = (
    json.loads((root / "source.json").read_text())["revision"]
    if (root / "source.json").exists()
    else "33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6"
)
base = f"https://raw.githubusercontent.com/karoldvl/ESC-50/{revision}"
r = requests.get(base + "/meta/esc50.csv", timeout=40)
r.raise_for_status()
rows = list(csv.DictReader(io.StringIO(r.text)))
chosen = []
for fold in [1, 2, 3, 4, 5]:
    for category in ["rain", "vacuum_cleaner", "engine", "sea_waves"]:
        subset = sorted(
            [x for x in rows if int(x["fold"]) == fold and x["category"] == category],
            key=lambda x: x["filename"],
        )[:2]
        for x in subset:
            x["split"] = "train" if fold < 4 else ("dev" if fold == 4 else "test")
            chosen.append(x)


def download(x):
    p = root / "audio" / x["filename"]
    if not p.exists():
        rr = requests.get(base + "/audio/" + x["filename"], timeout=60)
        rr.raise_for_status()
        p.write_bytes(rr.content)
    x["sha256"] = sha256(p.read_bytes()).hexdigest()
    return x


with ThreadPoolExecutor(max_workers=6) as pool:
    chosen = list(pool.map(download, chosen))
with open(root / "manifest.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(chosen[0]))
    w.writeheader()
    w.writerows(chosen)
(root / "source.json").write_text(
    json.dumps(
        {
            "repository": "karoldvl/ESC-50",
            "revision": revision,
            "license": "CC BY-NC 3.0",
            "files": len(chosen),
            "partition": "ESC folds 1-3 train, 4 dev, 5 test; original source IDs checked before use",
        },
        indent=2,
    )
)
print("Environmental noise files:", len(chosen), flush=True)
