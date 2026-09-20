"""Deterministic analytical RAN sandbox. This is NOT ns-3 or QuaDRiGa."""

from dataclasses import dataclass

import numpy as np

from .actions import ActionSpace


@dataclass
class Telemetry:
    demand: np.ndarray
    capacity: np.ndarray
    slices: np.ndarray
    queue: np.ndarray
    throughput: np.ndarray
    delay: np.ndarray
    cell_load: np.ndarray

    def vector(self, buffer_mbit):
        """Stable physical-unit scaling; never fit to held-out test observations."""
        return np.concatenate(
            [
                self.demand / 10,
                self.capacity.ravel() / 200,
                np.eye(3)[self.slices].ravel(),
                self.queue / buffer_mbit,
                self.throughput / 10,
                self.delay / 1000,
                self.cell_load,
            ]
        )


def jain(values):
    values = np.asarray(values, dtype=float)
    denominator = len(values) * np.square(values).sum()
    return float(values.sum() ** 2 / denominator) if denominator > 0 else 0.0


class AnalyticalRAN:
    backend_name = "analytical-sandbox"

    def __init__(self, config, seed, site, episode, channel_trace=None):
        self.config, self.site, self.episode = config, site, episode
        self.space = ActionSpace(config.users, config.cells)
        # Slice assignment is stable across episodes; channels/traffic are episode-specific.
        topology_rng = np.random.default_rng(np.random.SeedSequence([seed, site, 991]))
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, site, episode, 817]))
        profile = config.profiles[site % len(config.profiles)]
        proportions = {
            "embb": [0.75, 0.15, 0.10],
            "latency": [0.2, 0.65, 0.15],
            "mixed": [0.4, 0.3, 0.3],
            "bursty": [0.5, 0.25, 0.25],
        }[profile]
        self.slices = topology_rng.choice(3, config.users, p=proportions)
        self.profile = profile
        n, g, t = config.users, config.cells, config.slots
        base_sinr = topology_rng.uniform(7, 20, (n, g))
        innovations = self.rng.normal(0, 1, (t, n, g))
        fading = np.empty_like(innovations)
        fading[0] = innovations[0]
        for step in range(1, t):
            fading[step] = 0.95 * fading[step - 1] + 0.31 * innovations[step]
        sinr_db = np.clip(base_sinr + 3 * fading, -10, 35)
        self.capacities = config.bandwidth_mhz * np.log2(1 + 10 ** (sinr_db / 10))
        if channel_trace is not None:
            if (
                channel_trace.shape != (t, n, g)
                or not np.isfinite(channel_trace).all()
                or np.any(channel_trace < 0)
            ):
                raise ValueError("Channel trace must contain finite nonnegative capacities [T,U,G]")
            self.capacities = channel_trace.copy()
        rates = np.array([3.2, 0.55, 1.0])[self.slices]
        self.arrivals = self.rng.gamma(6, 1 / 6, (t, n)) * rates
        # The burst mask is generated up front; every method sees identical exogenous traces.
        burst = self.rng.random(t) < (0.14 if profile == "bursty" else 0.02)
        self.arrivals[burst] *= 3.0
        self.queue = np.zeros(n)
        self.throughput = np.zeros(n)
        self.delay = np.full(n, 5.0)
        self.cell_load = np.zeros(g)
        self.slot = 0

    def observe(self):
        if self.slot >= self.config.slots:
            raise RuntimeError("Episode completed")
        return Telemetry(
            self.arrivals[self.slot].copy(),
            self.capacities[self.slot].copy(),
            self.slices.copy(),
            self.queue.copy(),
            self.throughput.copy(),
            self.delay.copy(),
            self.cell_load.copy(),
        )

    def step(self, action):
        if self.slot >= self.config.slots:
            raise RuntimeError("Episode completed")
        if not self.space.valid(action, 1e-5):
            raise ValueError("Simulator received action outside the hard domain")
        rho, nu, omega = self.space.split(action)
        dt = self.config.slot_seconds
        demand = self.arrivals[self.slot]
        # Capacity sharing is a plant response, not a change to the applied action.
        allocation = rho[:, None] * nu * (0.5 + omega[:, None])
        self.cell_load = allocation.sum(axis=0)
        effective = allocation / np.maximum(1.0, self.cell_load)[None]
        service = (effective * self.capacities[self.slot]).sum(axis=1)
        offered = self.queue + demand * dt
        delivered = np.minimum(offered, np.maximum(service, 0) * dt)
        remaining = np.maximum(offered - delivered, 0)
        dropped = np.maximum(remaining - self.config.buffer_mbit, 0)
        self.queue = np.minimum(remaining, self.config.buffer_mbit)
        self.throughput = delivered / dt
        self.delay = 5 + 1000 * self.queue / np.maximum(service, 1e-6)
        violations = (self.throughput < np.array(self.config.min_rate_mbps)[self.slices]) | (
            self.delay > np.array(self.config.max_delay_ms)[self.slices]
        )
        loss = float(dropped.sum() / max(offered.sum(), 1e-12))
        kpis = np.array(
            [
                self.throughput.sum(),
                self.delay.mean(),
                loss,
                violations.mean(),
                jain(self.throughput),
            ]
        )
        self.slot += 1
        return kpis
