"""Optional isolated feature worker; atomically publish only completed condition files."""

import argparse, os, shutil
from pathlib import Path
import numpy as np
from saudasach.common import ROOT, CONFIG
from saudasach import data

p = argparse.ArgumentParser()
p.add_argument("condition", choices=CONFIG["test_conditions"])
a = p.parse_args()
real = ROOT
staging = real / "data" / ("staging_" + a.condition)
if staging.exists():
    raise ValueError("Staging directory exists; inspect an interrupted worker first")
(staging / "data").mkdir(parents=True)
(staging / "manifests").mkdir()
(staging / "configs").symlink_to(real / "configs", target_is_directory=True)
for name in ["raw", "noise"]:
    (staging / "data" / name).symlink_to(real / "data" / name, target_is_directory=True)
(staging / "manifests/splits.csv").symlink_to(real / "manifests/splits.csv")
data.ROOT = staging
data.CONFIG["test_conditions"] = [a.condition]
data.cache_features()
np.testing.assert_array_equal(
    np.load(staging / "data/cache/offsets.npy"),
    np.load(real / "data/cache/offsets.npy"),
)
for suffix in ["_cnn.npy", "_svm.npy"]:
    os.replace(
        staging / "data/cache" / (a.condition + suffix),
        real / "data/cache" / (a.condition + suffix),
    )
os.replace(
    staging / "manifests" / ("augmentation_" + a.condition + ".csv"),
    real / "manifests" / ("augmentation_" + a.condition + ".csv"),
)
os.replace(
    staging / "data/cache" / (a.condition + ".done"),
    real / "data/cache" / (a.condition + ".done"),
)
shutil.rmtree(staging)
print("Atomically published", a.condition, flush=True)
