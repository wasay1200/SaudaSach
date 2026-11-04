"""Ten artificial frontend fixtures for future mobile parity; not speech-evaluation data."""

import json
from pathlib import Path
import numpy as np
import torch
from saudasach.common import ROOT, sha, write_json
from saudasach.audio import logmel, SR
from saudasach.models import CompactCNN

selection = json.loads((ROOT / "artifacts/selection.json").read_text())
model = CompactCNN()
model.load_state_dict(
    torch.load(
        ROOT / "artifacts" / selection["selected"] / "weights.pt",
        weights_only=True,
        map_location="cpu",
    )
)
model.eval()
rng = np.random.default_rng(1078)
t = np.arange(32000) / SR
pcm = []
for i in range(10):
    if i == 0:
        y = np.zeros(32000)
    elif i == 1:
        y = rng.normal(0, 0.02, 32000)
    else:
        y = 0.2 * np.sin(2 * np.pi * (100 + 43 * i) * t) * (
            0.6 + 0.4 * np.sin(2 * np.pi * (1 + i / 4) * t)
        ) + 0.01 * rng.normal(size=32000)
    pcm.append(y.astype(np.float32))
pcm = np.stack(pcm)
features = np.stack([logmel(y) for y in pcm])
normalized = (features - model.mean.numpy()) / model.std.numpy()
out = ROOT / "artifacts/frontend_fixtures"
out.mkdir(exist_ok=True)
np.savez_compressed(
    out / "fixtures.npz",
    canonical_pcm=pcm,
    logmel=features,
    normalized_logmel=normalized,
)
write_json(
    out / "metadata.json",
    {
        "purpose": "Artificial deterministic frontend parity fixtures only; not a deepfake benchmark",
        "count": 10,
        "sample_rate": SR,
        "fixture_sha256": sha(out / "fixtures.npz"),
        "source_model": selection["selected"],
        "target_mobile_max_absolute_normalized_error": 0.001,
        "mobile_test_status": "Not performed",
        "normalization": "(Python logmel - checkpoint mean) / checkpoint std; ONNX performs this internally",
    },
)
print("Ten artificial frontend fixtures saved.")
