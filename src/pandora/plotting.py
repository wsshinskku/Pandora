"""Generate throughput, delay, and intervention plots from run telemetry."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_run(directory):
    directory = Path(directory)
    with (directory / "telemetry.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("No telemetry to plot")
    methods = list(dict.fromkeys(row["method"] for row in rows))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        throughput = np.sort([float(row["throughput_mbps"]) for row in selected])
        delay = np.sort([float(row["delay_ms"]) for row in selected])
        p = np.arange(1, len(throughput) + 1) / len(throughput)
        axes[0].plot(throughput, p, label=method)
        axes[1].plot(delay, 1 - p, label=method)
    interventions = [
        np.mean([row["intervention"] == "True" for row in rows if row["method"] == m])
        for m in methods
    ]
    axes[2].barh(methods, interventions, color="#167d9a")
    axes[0].set(xlabel="Site throughput (Mbps)", ylabel="CDF", title="Throughput distribution")
    axes[1].set(xlabel="Mean UE delay (ms)", ylabel="CCDF", title="Delay distribution")
    axes[2].set(xlabel="Fraction of slots", xlim=(0, 1), title="Intervention ratio")
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle("Pandora | RAN simulation results", fontsize=13)
    destination = directory / "performance.png"
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    return destination
