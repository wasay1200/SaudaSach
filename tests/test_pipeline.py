"""Small deterministic checks; real corpus results are reported separately."""

import json
import numpy as np
import pytest
import soundfile as sf
import torch
from saudasach.audio import (
    windows,
    logmel,
    mfcc_statistics,
    add_noise,
    codec_roundtrip,
    load_audio,
)
from saudasach.common import assert_disjoint, seed_all
from saudasach.models import CompactCNN
from saudasach.metrics import metrics, threshold_from_dev
from saudasach.bundle import load_bundle


def signal(seconds=2):
    t = np.arange(int(seconds * 16000)) / 16000
    return (
        0.2 * np.sin(2 * np.pi * 210 * t) + 0.1 * np.sin(2 * np.pi * 320 * t)
    ).astype(np.float32)


def test_feature_dimensions_and_finite():
    y = signal()
    assert logmel(y).shape == (64, 197)
    assert mfcc_statistics(y).shape == (240,)
    assert np.isfinite(logmel(y)).all()


def test_rolling_windows_cover_final_audio():
    y = signal(6.3)
    w = windows(y, 0.5)
    assert w.shape == (10, 32000)
    np.testing.assert_array_equal(w[-1], y[-32000:])
    assert windows(signal(0.5)).shape == (1, 32000)


def test_snr_and_determinism():
    y = signal()
    noise = np.random.default_rng(9).normal(size=80000).astype(np.float32)
    a = add_noise(y, noise, 10, "sample")
    b = add_noise(y, noise, 10, "sample")
    np.testing.assert_array_equal(a, b)
    snr = 10 * np.log10(np.mean(y * y) / np.mean((a - y) ** 2))
    assert abs(snr - 10) < 0.01


@pytest.mark.parametrize("codec", ["opus", "gsm"])
def test_real_codec_roundtrip(codec):
    y = signal()
    z = codec_roundtrip(y, codec)
    assert z.shape == y.shape and np.isfinite(z).all()
    assert np.mean((z - y) ** 2) > 1e-8


def test_overlap_is_rejected():
    rows = []
    for split in ["train", "dev", "test"]:
        for label in ["0", "1"]:
            key = split + label
            rows.append(
                dict(split=split, label=label, group=split, sha256=key, pcm_sha256=key)
            )
    assert_disjoint(rows)
    rows[-1]["pcm_sha256"] = rows[0]["pcm_sha256"]
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint(rows)


def test_metrics_orient_synthetic_as_positive():
    y = [0, 0, 1, 1]
    s = [-0.5, -0.2, 0.1, 0.3]
    m = metrics(y, s, 0)
    assert m["balanced_accuracy"] == 1 and m["eer"] == 0
    assert m["confusion_matrix"] == [[2, 0], [0, 2]]
    assert -0.2 < threshold_from_dev(y, s) <= 0.1


def test_stereo_resample(tmp_path):
    y = signal()
    p = tmp_path / "stereo.wav"
    sf.write(p, np.column_stack([y, y]), 8000, subtype="FLOAT")
    z = load_audio(p, max_seconds=1)
    assert len(z) == 16000 and z.ndim == 1


def test_network_training_and_seed():
    seed_all(42)
    a = CompactCNN()
    state = a.head[0].weight.detach().clone()
    seed_all(42)
    b = CompactCNN()
    torch.testing.assert_close(state, b.head[0].weight)
    x = torch.randn(8, 64, 197)
    y = torch.tensor([0.0, 1.0] * 4)
    opt = torch.optim.Adam(a.parameters(), lr=0.003)
    a.train()
    before = a(x).detach().clone()
    for _ in range(3):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(a(x), y)
        loss.backward()
        opt.step()
    assert torch.isfinite(loss) and not torch.equal(before, a(x).detach())


def test_invalid_bundle_rejected(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({"schema_version": 99}))
    with pytest.raises(ValueError, match="schema"):
        load_bundle(tmp_path)


def valid_metadata():
    from saudasach.common import CONFIG

    return {
        "schema_version": 1,
        "feature_shape": [64, 197],
        "labels": {"0": "genuine", "1": "synthetic"},
        "frontend": CONFIG,
        "threshold": 0.0,
        "uncertainty_low": -0.1,
        "uncertainty_high": 0.1,
        "model_file": "model.onnx",
        "sha256": "0" * 64,
    }


def test_tampered_bundle_rejected(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"corrupt")
    (tmp_path / "metadata.json").write_text(json.dumps(valid_metadata()))
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(tmp_path)


def test_mismatched_frontend_rejected(tmp_path):
    m = valid_metadata()
    m["frontend"] = dict(m["frontend"], sample_rate=8000)
    (tmp_path / "metadata.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="frontend"):
        load_bundle(tmp_path)


def test_aggregate_bootstrap_preserves_pairing():
    from saudasach.metrics import aggregate_paired_difference

    y = np.array([0, 1] * 3)
    groups = np.repeat(["a", "b", "c"], 2)
    a = np.array([-0.1, 0.1] * 3)
    b = -a
    ci = aggregate_paired_difference(y, groups, [(a, a, 0.0, 0.0)], n=30)
    assert ci["low"] == ci["high"] == 0
    ci = aggregate_paired_difference(y, groups, [(b, a, 0.0, 0.0)], n=30)
    assert ci["low"] == ci["high"] == 1


def test_svm_resume_rejects_changed_manifest(tmp_path, monkeypatch):
    from saudasach import train

    records = []
    for split in ["train", "dev", "test"]:
        for label in ["0", "1"]:
            key = split + label
            records.append(
                dict(
                    split=split,
                    label=label,
                    generator="genuine" if label == "0" else "tts",
                    group=split,
                    sha256=key,
                    pcm_sha256=key,
                )
            )
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests/splits.csv").write_text("changed manifest")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/study.json").write_text("{}")
    folder = tmp_path / "artifacts/svm_reference"
    folder.mkdir(parents=True)
    (folder / "metadata.json").write_text(
        json.dumps({"manifest_sha256": "old", "config_sha256": "old"})
    )
    monkeypatch.setattr(train, "ROOT", tmp_path)
    monkeypatch.setattr(train, "read_csv", lambda _: records)
    monkeypatch.setattr(
        train, "load_cache", lambda *args: np.zeros((6, 240), dtype=np.float32)
    )
    with pytest.raises(ValueError, match="Stale SVM"):
        train.train_svm()
