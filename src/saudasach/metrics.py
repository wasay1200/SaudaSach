"""Clip-level metrics and speaker-folder cluster bootstrap, never window-level accuracy."""

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    balanced_accuracy_score,
    f1_score,
)


def threshold_from_dev(y, scores):
    f, t, th = roc_curve(y, scores, drop_intermediate=False)
    valid = np.isfinite(th)
    return float(th[np.where(valid)[0][np.argmax((t - f)[valid])]])


def metrics(y, scores, threshold):
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pred = (scores >= threshold).astype(int)
    c = confusion_matrix(y, pred, labels=[0, 1])
    f, t, _ = roc_curve(y, scores, drop_intermediate=False)
    d = f + t - 1
    j = np.flatnonzero(d >= 0)[0]
    if j == 0:
        eer = float(f[0])
    else:
        a = -d[j - 1] / (d[j] - d[j - 1])
        eer = float(f[j - 1] + a * (f[j] - f[j - 1]))
    return {
        "n": len(y),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "accuracy": float((y == pred).mean()),
        "fake_recall": float(c[1, 1] / c[1].sum()),
        "genuine_recall": float(c[0, 0] / c[0].sum()),
        "false_positive_rate": float(c[0, 1] / c[0].sum()),
        "fake_f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "eer": eer,
        "confusion_matrix": c.tolist(),
        "threshold": threshold,
    }


def bootstrap_ba(y, scores, groups, threshold, seed=42, n=1000):
    y = np.asarray(y)
    scores = np.asarray(scores)
    groups = np.asarray(groups)
    names = np.unique(groups)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        ids = np.concatenate(
            [
                np.where(groups == g)[0]
                for g in rng.choice(names, len(names), replace=True)
            ]
        )
        if len(np.unique(y[ids])) == 2:
            values.append(
                float(balanced_accuracy_score(y[ids], scores[ids] >= threshold))
            )
    return {
        "low": float(np.quantile(values, 0.025)),
        "high": float(np.quantile(values, 0.975)),
        "resampling_unit": "speaker-folder group",
        "independent_groups": len(names),
        "replicates": len(values),
        "warning": "Only a few test groups; uncertainty estimates are coarse and do not establish population generalization.",
    }


def paired_difference(y, a, b, groups, ta, tb, n=1000):
    y = np.asarray(y)
    groups = np.asarray(groups)
    names = np.unique(groups)
    rng = np.random.default_rng(43)
    values = []
    for _ in range(n):
        ids = np.concatenate(
            [
                np.where(groups == g)[0]
                for g in rng.choice(names, len(names), replace=True)
            ]
        )
        if len(np.unique(y[ids])) == 2:
            values.append(
                float(
                    balanced_accuracy_score(y[ids], b[ids] >= tb)
                    - balanced_accuracy_score(y[ids], a[ids] >= ta)
                )
            )
    return {
        "low": float(np.quantile(values, 0.025)),
        "high": float(np.quantile(values, 0.975)),
        "units": "fractional balanced-accuracy difference augmented minus ordinary",
        "groups": len(names),
    }


def aggregate_paired_difference(y, groups, comparisons, n=1000, seed=44):
    """Paired cluster bootstrap of mean BA difference over fixed model/condition pairs."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    names = np.unique(groups)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(names), size=(n, len(names)))
    totals = np.array(
        [[np.sum((groups == g) & (y == label)) for label in [0, 1]] for g in names]
    )
    denominators = totals[draws].sum(axis=1)
    valid = (denominators > 0).all(axis=1)
    differences = []
    for ordinary, augmented, ordinary_threshold, augmented_threshold in comparisons:
        accuracies = []
        for scores, threshold in [
            (ordinary, ordinary_threshold),
            (augmented, augmented_threshold),
        ]:
            correct = np.asarray(scores >= threshold) == y
            counts = np.array(
                [
                    [np.sum((groups == g) & (y == label) & correct) for label in [0, 1]]
                    for g in names
                ]
            )
            accuracies.append(
                (counts[draws].sum(axis=1)[valid] / denominators[valid]).mean(axis=1)
            )
        differences.append(accuracies[1] - accuracies[0])
    values = np.mean(differences, axis=0)
    return {
        "low": float(np.quantile(values, 0.025)),
        "high": float(np.quantile(values, 0.975)),
        "resampling_unit": "folder group; same sampled groups across all fixed seed/condition pairs",
        "groups": len(names),
        "replicates": int(valid.sum()),
        "comparisons": len(comparisons),
        "limitation": "Conditional on these trained seeds and conditions; few groups limit population inference.",
    }
