"""Corpus auditing, deterministic grouping, splitting and feature caches."""

import hashlib, re, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import soundfile as sf
from sklearn.model_selection import GroupShuffleSplit
from .common import (
    ROOT,
    CONFIG,
    sha,
    stable_seed,
    read_csv,
    write_csv,
    write_json,
    assert_disjoint,
)
from .audio import load_audio, windows, logmel, mfcc_statistics, condition, SR


def audit():
    raw = ROOT / "data/raw"
    paths = sorted(raw.rglob("*.wav"))
    rows = []
    excluded = []
    if len(paths) != 6794:
        raise ValueError(
            f"Incomplete corpus: expected 6794 audio files at pinned revision, found {len(paths)}"
        )

    def inspect(p):
        rel = p.relative_to(raw)
        speaker = next(
            (x for x in rel.parts if re.fullmatch("Speaker_[0-9]+", x)), None
        )
        if speaker is None:
            raise ValueError("Unknown group " + str(rel))
        info = sf.info(p)
        y = load_audio(p)
        if len(y) < SR * 2:
            return None, {"path": str(rel), "reason": "less_than_two_seconds"}
        if np.sqrt(np.mean(y.astype(np.float64) ** 2)) < 1e-5:
            return None, {"path": str(rel), "reason": "near_silence"}
        generator = (
            "genuine"
            if rel.parts[0] == "Bonafide"
            else ("tacotron" if rel.parts[0] == "Spoofed_Tacotron" else "tts")
        )
        if rel.parts[0] not in ["Bonafide", "Spoofed_TTS", "Spoofed_Tacotron"]:
            raise ValueError("Unknown label")
        return {
            "path": rel.as_posix(),
            "label": str(int(generator != "genuine")),
            "generator": generator,
            "group": speaker.lower(),
            "sha256": sha(p),
            "pcm_sha256": hashlib.sha256(y.astype("<f4").tobytes()).hexdigest(),
            "seconds": round(len(y) / SR, 6),
            "original_sample_rate": info.samplerate,
            "license": "CC BY-NC 4.0",
            "source_revision": CONFIG["revision"],
        }, None

    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, (row, exc) in enumerate(pool.map(inspect, paths)):
            if row:
                rows.append(row)
            if exc:
                excluded.append(exc)
            if (i + 1) % 500 == 0:
                print("Audited", i + 1, len(paths), flush=True)
    # Merge any speaker folders linked by exact duplicate content before splitting.
    parent = {r["group"]: r["group"] for r in rows}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    buckets = {}
    for r in rows:
        buckets.setdefault(r["pcm_sha256"], []).append(r)
    retained = []
    duplicate_count = 0
    conflicts = 0
    for group in buckets.values():
        if len({r["label"] for r in group}) > 1:
            conflicts += len(group)
            excluded.extend(
                {"path": r["path"], "reason": "identical_audio_conflicting_labels"}
                for r in group
            )
            continue
        for r in group[1:]:
            parent[find(r["group"])] = find(group[0]["group"])
        retained.append(group[0])
        duplicate_count += len(group) - 1
        excluded.extend(
            {"path": r["path"], "reason": "exact_decoded_duplicate"} for r in group[1:]
        )
    rows = sorted(retained, key=lambda r: r["path"])
    for i, r in enumerate(rows):
        r["group"] = find(r["group"])
        r["index"] = i
    g = np.array([r["group"] for r in rows])
    idx = np.arange(len(rows))
    a, b = next(
        GroupShuffleSplit(
            n_splits=1, test_size=0.15, random_state=CONFIG["split_seed"]
        ).split(idx, groups=g)
    )
    c, d = next(
        GroupShuffleSplit(
            n_splits=1, test_size=0.15 / 0.85, random_state=CONFIG["split_seed"] + 1
        ).split(a, groups=g[a])
    )
    for part, indices in [("train", a[c]), ("dev", a[d]), ("test", b)]:
        for i in indices:
            rows[i]["split"] = part
    assert_disjoint(rows)
    write_csv(ROOT / "manifests/splits.csv", rows)
    if excluded:
        write_csv(ROOT / "manifests/excluded.csv", excluded)
    source = ROOT / "data/raw/README.md"
    if source.exists() and "CC BY-NC 4.0" not in source.read_text():
        raise ValueError("Recheck source license")
    report = {
        "downloaded_files": len(paths),
        "retained": len(rows),
        "excluded": len(excluded),
        "exact_duplicates_removed": duplicate_count,
        "conflicting_label_files_removed": conflicts,
        "groups": len(set(g)),
        "split_summary": {
            p: {
                "files": sum(r["split"] == p for r in rows),
                "genuine": sum(r["split"] == p and r["label"] == "0" for r in rows),
                "synthetic": sum(r["split"] == p and r["label"] == "1" for r in rows),
                "groups": sorted({r["group"] for r in rows if r["split"] == p}),
            }
            for p in ["train", "dev", "test"]
        },
        "claim": "Folder-group-disjoint, not independently verified speaker identity or text/source independence. No near-duplicate guarantee.",
        "manifest_sha256": sha(ROOT / "manifests/splits.csv"),
    }
    write_json(ROOT / "reports/data_audit.json", report)
    print(report, flush=True)
    return rows


def cache_features():
    rows = read_csv(ROOT / "manifests/splits.csv")
    assert_disjoint(rows)
    noises = read_csv(ROOT / "data/noise/manifest.csv")
    sources = {}
    for r in noises:
        sources.setdefault(r["src_file"], set()).add(r["split"])
    if any(len(x) > 1 for x in sources.values()):
        raise ValueError("Noise original-source overlap")
    bank = {
        p: [
            ROOT / "data/noise/audio" / r["filename"] for r in noises if r["split"] == p
        ]
        for p in ["train", "dev", "test"]
    }
    cache = ROOT / "data/cache"
    cache.mkdir(parents=True, exist_ok=True)
    # Warm librosa before threading.
    logmel(np.ones(32000, dtype=np.float32) * 0.01)
    mfcc_statistics(np.ones(32000, dtype=np.float32) * 0.01)
    for cond in CONFIG["test_conditions"]:
        cnn = cache / (cond + "_cnn.npy")
        svm = cache / (cond + "_svm.npy")
        fingerprint = (
            sha(ROOT / "manifests/splits.csv") + ":" + sha(ROOT / "configs/study.json")
        )
        if (
            cnn.exists()
            and svm.exists()
            and (cache / (cond + ".done")).exists()
            and (cache / (cond + ".done")).read_text() == fingerprint
        ):
            continue
        # At most 3 training windows per six-second recording; primary evaluation averages these per clip.
        counts = [
            min(3, max(1, int(np.ceil((min(6, float(r["seconds"])) - 2) / 2)) + 1))
            for r in rows
        ]
        offsets = np.r_[0, np.cumsum(counts)].astype(np.int64)
        x = np.lib.format.open_memmap(
            cnn, mode="w+", dtype="float16", shape=(int(offsets[-1]), 64, 197)
        )
        m = np.lib.format.open_memmap(
            svm, mode="w+", dtype="float32", shape=(len(rows), 240)
        )

        def process(item):
            i, r = item
            y = load_audio(ROOT / "data/raw" / r["path"], max_seconds=6)
            paths = bank[r["split"]]
            npth = paths[stable_seed(r["path"], cond) % len(paths)]
            z = condition(y, cond, npth, r["path"] + ":" + cond)
            a = np.stack([logmel(w) for w in windows(z, 2)])
            return i, a, mfcc_statistics(z), npth.name

        records = []
        started = time.time()
        with ThreadPoolExecutor(max_workers=5) as pool:
            for j, (i, a, f, npth) in enumerate(pool.map(process, enumerate(rows))):
                if len(a) != counts[i]:
                    raise ValueError("Window count mismatch")
                x[offsets[i] : offsets[i + 1]] = a
                m[i] = f
                records.append(
                    {
                        "path": rows[i]["path"],
                        "condition": cond,
                        "noise_file": npth if cond.startswith("noise") else "none",
                        "seed": stable_seed(
                            rows[i]["path"] + ":" + cond,
                            int(cond.split("_")[0].replace("noise", "")),
                        )
                        if cond.startswith("noise")
                        else 0,
                    }
                )
                if (j + 1) % 500 == 0:
                    print(
                        "Features",
                        cond,
                        j + 1,
                        len(rows),
                        "seconds",
                        round(time.time() - started),
                        flush=True,
                    )
        x.flush()
        m.flush()
        np.save(cache / "offsets.npy", offsets)
        write_csv(ROOT / "manifests" / ("augmentation_" + cond + ".csv"), records)
        (cache / (cond + ".done")).write_text(fingerprint)
        print("Feature condition complete:", cond, flush=True)


def load_cache(cond, kind="cnn"):
    return np.load(ROOT / "data/cache" / f"{cond}_{kind}.npy", mmap_mode="r")
