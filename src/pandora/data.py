"""Site-local transitions with explicit, episode-level train/calibration/test ownership."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Episode:
    episode_id: int
    split: str
    observations: np.ndarray
    proposals: np.ndarray
    projected: np.ndarray
    executed: np.ndarray
    kpis: np.ndarray

    def __post_init__(self):
        if self.split not in {"train", "calibration", "test"}:
            raise ValueError("Unknown episode split")
        arrays = [self.observations, self.proposals, self.projected, self.executed, self.kpis]
        if any(x.ndim != 2 or len(x) != len(arrays[0]) or not np.isfinite(x).all() for x in arrays):
            raise ValueError("Transition arrays must be finite matrices with equal row counts")
        if len(self.kpis) == 0 or self.kpis.shape[1] != 5:
            raise ValueError(
                "Expected nonempty KPI rows [throughput, delay, loss, violation, fairness]"
            )
        if not self.proposals.shape == self.projected.shape == self.executed.shape:
            raise ValueError("All three action arrays must have the same shape")

    @property
    def features(self):
        # Applied, not proposed or projected actions, are supervised inputs (Eq. 21).
        return np.concatenate([self.observations, self.executed], axis=1)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            episode_id=self.episode_id,
            split=self.split,
            observations=self.observations,
            proposals=self.proposals,
            projected=self.projected,
            executed=self.executed,
            kpis=self.kpis,
        )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(
                int(data["episode_id"]),
                str(data["split"]),
                **{
                    k: data[k].copy()
                    for k in ("observations", "proposals", "projected", "executed", "kpis")
                },
            )


def partition_episodes(episodes):
    ids = [e.episode_id for e in episodes]
    if len(ids) != len(set(ids)):
        raise ValueError("An episode ID appears more than once: possible split leakage")
    return {
        split: [e for e in episodes if e.split == split]
        for split in ("train", "calibration", "test")
    }
