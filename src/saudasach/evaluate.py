import json, time
from pathlib import Path
import numpy as np, torch, joblib
from .common import (
    ROOT,
    CONFIG,
    read_csv,
    write_csv,
    write_json,
    sha,
    assert_disjoint,
    seed_all,
)
from .data import load_cache
from .models import CompactCNN
from .train import clip_scores
from .metrics import metrics, bootstrap_ba, paired_difference


def evaluate_all(final_test=False):
    if not final_test:
        raise ValueError("Explicit --final-test required; freeze the study first")
    selection = json.loads((ROOT / "artifacts/selection.json").read_text())
    rows = read_csv(ROOT / "manifests/splits.csv")
    assert_disjoint(rows)
    if selection["manifest_sha256"] != sha(ROOT / "manifests/splits.csv") or selection[
        "config_sha256"
    ] != sha(ROOT / "configs/study.json"):
        raise ValueError("Frozen study changed")
    for file, digest in selection["weight_hashes"].items():
        if sha(ROOT / file) != digest:
            raise ValueError("Frozen weights changed")
    freeze = ROOT / "reports/final_test_access.json"
    if freeze.exists():
        raise ValueError(
            "Final evaluation already exists. Use saved predictions; do not retune on test results."
        )
    write_json(
        freeze,
        {
            "started_utc": __import__("datetime")
            .datetime.now(__import__("datetime").timezone.utc)
            .isoformat(),
            "selection_sha256": sha(ROOT / "artifacts/selection.json"),
            "manifest_sha256": sha(ROOT / "manifests/splits.csv"),
            "purpose": "Frozen final evaluation; not a model selection loop",
            "completed": False,
        },
    )
    seed_all(42)
    offsets = np.load(ROOT / "data/cache/offsets.npy")
    ids = np.array([i for i, r in enumerate(rows) if r["split"] == "test"])
    labels = np.array([int(r["label"]) for r in rows])
    groups = np.array([r["group"] for r in rows])
    summary = {}
    predictions = []
    scoremap = {}
    metadata = {}
    for out in sorted((ROOT / "artifacts").glob("*")):
        if (
            not out.is_dir()
            or not out.name.startswith(("cnn_", "svm_"))
            or not (out / "metadata.json").exists()
        ):
            continue
        meta = json.loads((out / "metadata.json").read_text())
        metadata[out.name] = meta
        if meta["manifest_sha256"] != sha(ROOT / "manifests/splits.csv"):
            raise ValueError("Split mismatch")
        current = ids
        if out.name.startswith("svm_without_"):
            held = meta["excluded_generator"]
            current = np.array(
                [i for i in ids if rows[i]["generator"] in ["genuine", held]]
            )
        if out.name.startswith("cnn_"):
            model = CompactCNN()
            model.load_state_dict(
                torch.load(out / "weights.pt", weights_only=True, map_location="cpu")
            )
        else:
            model = joblib.load(out / "model.joblib")
        summary[out.name] = {}
        for cond in CONFIG["test_conditions"]:
            if out.name.startswith("cnn_"):
                sc = clip_scores(model, load_cache(cond), current, offsets)
            else:
                sc = model.decision_function(
                    np.asarray(load_cache(cond, "svm")[current])
                )
            yy = labels[current]
            gg = groups[current]
            report = metrics(yy, sc, meta["threshold"])
            report["balanced_accuracy_ci95"] = bootstrap_ba(
                yy, sc, gg, meta["threshold"], n=CONFIG["bootstrap_replicates"]
            )
            if out.name.startswith("cnn_"):
                decided = (sc < meta["uncertainty_low"]) | (
                    sc > meta["uncertainty_high"]
                )
                report["decision_coverage"] = float(decided.mean())
                report["error_among_decided"] = (
                    float(((sc[decided] >= meta["threshold"]) != yy[decided]).mean())
                    if decided.any()
                    else None
                )
            summary[out.name][cond] = report
            scoremap[(out.name, cond)] = (sc, current)
            for i, score in zip(current, sc):
                predictions.append(
                    {
                        "model": out.name,
                        "condition": cond,
                        "path": rows[i]["path"],
                        "group": rows[i]["group"],
                        "generator": rows[i]["generator"],
                        "label": int(labels[i]),
                        "score": float(score),
                        "prediction": int(score >= meta["threshold"]),
                        "threshold": meta["threshold"],
                    }
                )
        print(
            "Evaluated",
            out.name,
            "clean",
            round(summary[out.name]["clean"]["balanced_accuracy"], 4),
            flush=True,
        )
        write_json(ROOT / "reports/test_metrics.json", summary)
    write_csv(ROOT / "reports/test_predictions.csv", predictions)
    differences = {}
    degraded = [c for c in CONFIG["test_conditions"] if c != "clean"]
    for seed in CONFIG["seeds"]:
        a = f"cnn_ordinary_seed{seed}"
        b = f"cnn_augmented_seed{seed}"
        differences[str(seed)] = {}
        for cond in CONFIG["test_conditions"]:
            av, ix = scoremap[(a, cond)]
            bv, _ = scoremap[(b, cond)]
            differences[str(seed)][cond] = {
                "difference": summary[b][cond]["balanced_accuracy"]
                - summary[a][cond]["balanced_accuracy"],
                "ci95": paired_difference(
                    labels[ix],
                    av,
                    bv,
                    groups[ix],
                    metadata[a]["threshold"],
                    metadata[b]["threshold"],
                ),
            }
    seed_diffs = [
        np.mean([differences[str(s)][c]["difference"] for c in degraded])
        for s in CONFIG["seeds"]
    ]
    clean_diffs = [differences[str(s)]["clean"]["difference"] for s in CONFIG["seeds"]]
    comparison = {
        "seed_paired_differences": differences,
        "mean_degraded_gain": float(np.mean(seed_diffs)),
        "std_degraded_gain": float(np.std(seed_diffs, ddof=1)),
        "mean_clean_change": float(np.mean(clean_diffs)),
        "target_gain": 0.05,
        "allowed_clean_loss": 0.02,
        "target_achieved": bool(
            np.mean(seed_diffs) >= 0.05 and np.mean(clean_diffs) >= -0.02
        ),
        "degraded_conditions": degraded,
        "units": "Fractions; multiply by 100 for percentage points.",
        "limitations": [
            "Folder groups are not independently verified speaker/source identities.",
            "Only three held-out groups; intervals are coarse.",
            "Different noise sources but limited environmental categories.",
            "Primary controlled results cover at most six seconds per clip.",
            "No genuine Urdu-English code-switching labels or independently collected clone generators.",
        ],
    }
    write_json(ROOT / "reports/comparison.json", comparison)
    # Confusion matrix and compact robustness chart for the development-selected model.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    chosen = selection["selected"]
    ConfusionMatrixDisplay(
        np.array(summary[chosen]["clean"]["confusion_matrix"]),
        display_labels=["genuine", "synthetic"],
    ).plot(cmap="Blues", colorbar=False)
    plt.title(chosen + " / clean test")
    plt.tight_layout()
    plt.savefig(ROOT / "reports/confusion_clean.png", dpi=160)
    plt.close()
    cs = CONFIG["test_conditions"]
    xx = np.arange(len(cs))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for var in ["ordinary", "augmented"]:
        means = [
            np.mean(
                [
                    summary[f"cnn_{var}_seed{s}"][c]["balanced_accuracy"]
                    for s in CONFIG["seeds"]
                ]
            )
            for c in cs
        ]
        ax.plot(xx, np.array(means) * 100, marker="o", label=var)
    ax.set_xticks(xx, cs, rotation=35, ha="right")
    ax.set_ylabel("Clip balanced accuracy percent")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(ROOT / "reports/robustness.png", dpi=160)
    plt.close(fig)
    state = json.loads(freeze.read_text())
    state["completed"] = True
    state["prediction_sha256"] = sha(ROOT / "reports/test_predictions.csv")
    write_json(freeze, state)
    print(
        "Frozen final evaluation complete. Target achieved:",
        comparison["target_achieved"],
        flush=True,
    )
