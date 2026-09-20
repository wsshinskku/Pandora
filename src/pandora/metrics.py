"""Episode metrics and paired seed bootstrap, without inventing manuscript results."""

import numpy as np
from scipy.stats import rankdata


def tail_mean(values, highest=False, fraction=0.1):
    x = np.sort(np.asarray(values, dtype=float))
    if len(x) == 0:
        raise ValueError("Empty sample")
    count = max(1, int(np.ceil(len(x) * fraction)))
    return float(x[-count:].mean() if highest else x[:count].mean())


def episode_metrics(kpis, epsilon):
    y = np.asarray(kpis)
    return {
        "throughput_mbps": float(y[:, 0].mean()),
        "throughput_top10_mbps": tail_mean(y[:, 0], True),
        "throughput_bottom10_mbps": tail_mean(y[:, 0]),
        "delay_ms": float(y[:, 1].mean()),
        "delay_top10_ms": tail_mean(y[:, 1]),
        "delay_bottom10_ms": tail_mean(y[:, 1], True),
        "loss_ratio": float(y[:, 2].mean()),
        "violation_ratio": float(y[:, 3].mean()),
        "fairness": float(y[:, 4].mean()),
        "exceedance_probability": float((y[:, 3] > epsilon).mean()),
    }


def prediction_metrics(actual, predicted, probability, epsilon, bins=10):
    labels = (np.asarray(actual)[:, 3] > epsilon).astype(int)
    p = np.asarray(probability)
    positive, negative = int(labels.sum()), int((1 - labels).sum())
    auc = None
    if positive and negative:
        ranks = rankdata(p)
        auc = float(
            (ranks[labels == 1].sum() - positive * (positive + 1) / 2) / (positive * negative)
        )
    ece = 0.0
    bucket = np.minimum((np.clip(p, 0, 1) * bins).astype(int), bins - 1)
    for index in range(bins):
        mask = bucket == index
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - labels[mask].mean())
    return {
        "kpi_mae": np.abs(np.asarray(actual) - predicted).mean(axis=0).tolist(),
        "risk_auroc": auc,
        "risk_brier": float(np.mean((p - labels) ** 2)),
        "risk_ece": float(ece),
    }


def bootstrap_mean(values, rng, resamples=10000):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Bootstrap requires finite seed-level observations")
    if len(values) == 1:
        return {"mean": float(values[0]), "ci95": None, "seeds": 1}
    sampled = values[rng.integers(len(values), size=(resamples, len(values)))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": np.quantile(sampled, [0.025, 0.975]).tolist(),
        "seeds": len(values),
    }


def paired_difference(reference, variant, rng, resamples=10000):
    if set(reference) != set(variant):
        raise ValueError("Paired comparison requires identical seed IDs")
    seeds = sorted(reference)
    return bootstrap_mean([variant[s] - reference[s] for s in seeds], rng, resamples)
