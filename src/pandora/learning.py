"""MLP action-impact model, private affine adapter and sample-weighted FedAvg."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .data import partition_episodes


class ImpactModel(nn.Module):
    def __init__(self, input_dim, personalize=True):
        super().__init__()
        self.input_dim = input_dim
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.adapter_scale = nn.Parameter(torch.zeros(64), requires_grad=personalize)
        self.adapter_bias = nn.Parameter(torch.zeros(64), requires_grad=personalize)
        self.kpi_head = nn.Linear(64, 5)
        self.risk_head = nn.Linear(64, 1)
        # Fixed physical-unit scaling shared by every site (not fitted to test data).
        self.register_buffer("kpi_scale", torch.tensor([100.0, 100.0, 1.0, 1.0, 1.0]))

    def forward(self, x):
        h = self.trunk(x)
        h = h * (1 + self.adapter_scale) + self.adapter_bias
        return self.kpi_head(h), self.risk_head(h).squeeze(-1)

    @torch.no_grad()
    def predict(self, features):
        self.eval()
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.input_dim or not np.isfinite(values).all():
            raise ValueError("Invalid model feature matrix")
        kpi, risk = self(torch.from_numpy(values))
        return (kpi * self.kpi_scale).cpu().numpy(), risk.sigmoid().cpu().numpy()

    def shared_state(self):
        return {
            k: v.detach().cpu().clone()
            for k, v in self.state_dict().items()
            if not k.startswith("adapter_")
        }

    def load_shared(self, state):
        if set(state) != set(self.shared_state()):
            raise ValueError("Shared-state keys mismatch; adapters must remain local")
        full = self.state_dict()
        full.update(state)
        self.load_state_dict(full)


@dataclass(frozen=True)
class ModelUpdate:
    """The entire server-side interface: shared tensors and training sample count only."""

    shared: dict[str, torch.Tensor]
    samples: int


def federated_average(updates):
    if not updates or any(u.samples <= 0 for u in updates):
        raise ValueError("FedAvg requires nonempty positive-count updates")
    keys = set(updates[0].shared)
    if any(set(u.shared) != keys for u in updates) or any(k.startswith("adapter_") for k in keys):
        raise ValueError("Malformed shared model update")
    total = sum(u.samples for u in updates)
    result = {}
    for key in keys:
        tensors = [u.shared[key] for u in updates]
        if any(t.shape != tensors[0].shape or not torch.isfinite(t).all() for t in tensors):
            raise ValueError("Inconsistent or nonfinite shared parameter")
        result[key] = sum(t * (u.samples / total) for t, u in zip(tensors, updates))
    return result


class LocalLearner:
    def __init__(self, model, config, epsilon, seed):
        self.model, self.config, self.epsilon = model, config, epsilon
        self.generator = torch.Generator().manual_seed(seed)

    def fit(self, episodes):
        parts = partition_episodes(episodes)
        if parts["calibration"] or parts["test"] or not parts["train"]:
            raise ValueError("Training accepts training episodes only")
        x = np.concatenate([e.features for e in episodes]).astype(np.float32)
        y = np.concatenate([e.kpis for e in episodes]).astype(np.float32)
        features, targets = torch.from_numpy(x), torch.from_numpy(y)
        labels = (targets[:, 3] > self.epsilon).float()
        c = self.config
        optimizer = torch.optim.Adam(
            (p for p in self.model.parameters() if p.requires_grad), lr=c.learning_rate
        )
        self.model.train()
        losses = []
        for _ in range(c.epochs):
            order = torch.randperm(len(x), generator=self.generator)
            for indices in order.split(c.batch_size):
                pred, logits = self.model(features[indices])
                # Sum across KPI coordinates, then average over samples, as in Eq. 24.
                mse = ((pred - targets[indices] / self.model.kpi_scale) ** 2).sum(1).mean()
                risk = nn.functional.binary_cross_entropy_with_logits(logits, labels[indices])
                reg = (
                    self.model.adapter_scale.square().sum() + self.model.adapter_bias.square().sum()
                )
                loss = mse + c.risk_weight * risk + c.adapter_penalty * reg
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                optimizer.step()
                losses.append(float(loss.detach()))
        return ModelUpdate(self.model.shared_state(), len(x)), float(np.mean(losses))
