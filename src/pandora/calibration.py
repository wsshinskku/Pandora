"""Local standardized nearest support and empirical one-sided residual margin, Eqs. 26-30."""

import numpy as np
from scipy.spatial import cKDTree


class RiskCalibrator:
    def __init__(self, config, enabled=True):
        self.config, self.enabled = config, enabled
        self.fitted = False

    def fit(self, model, episode):
        if episode.split != "calibration":
            raise ValueError("Calibration must use a disjoint calibration episode")
        x = episode.features
        if len(x) < 2:
            raise ValueError("Leave-one-out support requires at least two calibration samples")
        self.mean = x.mean(axis=0)
        self.scale = np.maximum(x.std(axis=0), 1e-3)
        self.tree = cKDTree((x - self.mean) / self.scale)
        # k=2 excludes each point's own zero-distance neighbor.
        distances = self.tree.query((x - self.mean) / self.scale, k=2)[0][:, 1]
        self.threshold = float(
            np.quantile(distances, self.config.support_quantile, method="higher")
        )
        _, p = model.predict(x)
        labels = (episode.kpis[:, 3] > self.config.epsilon).astype(float)
        self.margin = (
            max(0.0, float(np.quantile(labels - p, 1 - self.config.delta_cal, method="higher")))
            if self.enabled
            else 0.0
        )
        self.fitted = True
        return self

    def distance(self, features):
        if not self.fitted:
            raise RuntimeError("Calibrator has not been fitted")
        x = np.asarray(features)
        if x.ndim != 2 or x.shape[1] != len(self.mean) or not np.isfinite(x).all():
            raise ValueError("Invalid support-query features")
        return self.tree.query((x - self.mean) / self.scale, k=1)[0]

    def supported(self, features):
        return self.distance(features) <= self.threshold + 1e-10

    def upper_risk(self, probabilities):
        if not self.fitted:
            raise RuntimeError("Calibrator has not been fitted")
        return np.clip(np.asarray(probabilities) + self.margin, 0, 1)
