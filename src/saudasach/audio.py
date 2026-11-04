"""One deterministic frontend shared by dataset preparation and local inference."""

from functools import lru_cache
from pathlib import Path
import subprocess, tempfile
import librosa
import numpy as np
import soundfile as sf
from .common import CONFIG, stable_seed

SR = CONFIG["sample_rate"]
WINDOW = int(SR * CONFIG["window_seconds"])


def load_audio(path, max_seconds=None):
    with sf.SoundFile(path) as f:
        frames = -1 if max_seconds is None else int(max_seconds * f.samplerate)
        y = f.read(frames, dtype="float32", always_2d=True).mean(axis=1)
        rate = f.samplerate
    if not len(y) or not np.isfinite(y).all():
        raise ValueError("Empty or invalid audio")
    if rate != SR:
        y = librosa.resample(y, orig_sr=rate, target_sr=SR)
    return np.ascontiguousarray(y, dtype=np.float32)


def windows(y, hop_seconds=2.0):
    if len(y) < WINDOW:
        return np.pad(y, (0, WINDOW - len(y)))[None, :]
    hop = int(hop_seconds * SR)
    starts = list(range(0, len(y) - WINDOW + 1, hop))
    if starts[-1] != len(y) - WINDOW:
        starts.append(len(y) - WINDOW)
    return np.stack([y[s : s + WINDOW] for s in starts])


def logmel(y):
    spec = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=CONFIG["n_fft"],
        win_length=CONFIG["win_length"],
        hop_length=CONFIG["hop_length"],
        n_mels=CONFIG["n_mels"],
        center=False,
        power=2.0,
    )
    return librosa.power_to_db(spec, ref=1.0, top_db=80.0).astype(np.float32)


def mfcc_statistics(y):
    m = librosa.feature.mfcc(
        y=y,
        sr=SR,
        n_mfcc=CONFIG["n_mfcc"],
        n_mels=CONFIG["n_mels"],
        n_fft=CONFIG["n_fft"],
        win_length=CONFIG["win_length"],
        hop_length=CONFIG["hop_length"],
        center=False,
    )
    ds = [
        m,
        librosa.feature.delta(m, mode="nearest"),
        librosa.feature.delta(m, order=2, mode="nearest"),
    ]
    return np.concatenate(
        [v for a in ds for v in (a.mean(axis=1), a.std(axis=1))]
    ).astype(np.float32)


def codec_roundtrip(y, codec, bitrate="24k"):
    with tempfile.TemporaryDirectory(prefix="saudasach_codec_") as d:
        d = Path(d)
        sf.write(d / "in.wav", y, SR, subtype="FLOAT")
        if codec == "opus":
            args = ["-ar", "48000", "-c:a", "libopus", "-b:a", bitrate]
            ext = "ogg"
        elif codec == "gsm":
            args = ["-ar", "8000", "-c:a", "libgsm"]
            ext = "gsm"
        else:
            raise ValueError("Unsupported codec")
        a = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(d / "in.wav"),
                "-ac",
                "1",
                *args,
                str(d / ("encoded." + ext)),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if a.returncode:
            raise RuntimeError(a.stderr)
        b = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(d / ("encoded." + ext)),
                "-ar",
                str(SR),
                "-ac",
                "1",
                "-c:a",
                "pcm_f32le",
                str(d / "out.wav"),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if b.returncode:
            raise RuntimeError(b.stderr)
        z, _ = sf.read(d / "out.wav", dtype="float32")
        return np.pad(z, (0, max(0, len(y) - len(z))))[: len(y)]


@lru_cache(maxsize=48)
def noise_audio(path):
    return load_audio(path)


def add_noise(y, noise, snr, key):
    rng = np.random.default_rng(stable_seed(key, snr))
    need = len(y)
    if len(noise) < need:
        noise = np.tile(noise, int(np.ceil(need / len(noise))))
    start = int(rng.integers(0, len(noise) - need + 1))
    z = noise[start : start + need].astype(np.float64)
    z -= z.mean()
    p = float(np.mean(y.astype(np.float64) ** 2))
    q = float(np.mean(z * z))
    if p < 1e-12 or q < 1e-12:
        raise ValueError("Silent speech or noise")
    return (y + z * np.sqrt(p / (q * 10 ** (snr / 10)))).astype(np.float32)


def condition(y, name, noise_path, key):
    z = y
    if name.startswith("noise"):
        snr = int(name.split("_")[0].replace("noise", ""))
        z = add_noise(y, noise_audio(str(noise_path)), snr, key)
    if "opus" in name:
        z = codec_roundtrip(z, "opus", "12k" if name == "opus12" else "24k")
    elif "gsm" in name:
        z = codec_roundtrip(z, "gsm")
    return z
