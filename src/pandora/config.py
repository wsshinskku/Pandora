"""Validated experiment configuration; paper values and implementation choices are documented."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml


@dataclass
class ContractConfig:
    period: int = 30
    epsilon: float = 0.10
    kappa: float = 0.10
    delta_cal: float = 0.05
    support_quantile: float = 0.95
    lower_quantile: float = 0.10
    upper_quantile: float = 0.90
    coupling_quantile: float = 0.90
    minimum_safe: int = 24
    verification_samples: int = 64
    shrink_factor: float = 0.20
    max_shrinks: int = 3
    perturbation_std: float = 0.05
    weights: list[float] = field(default_factory=lambda: [1.0, 1.5, 2.0])
    solver_tolerance: float = 1e-5


@dataclass
class LearningConfig:
    fl_period: int = 600
    epochs: int = 3
    batch_size: int = 256
    learning_rate: float = 1e-3
    risk_weight: float = 1.0
    adapter_penalty: float = 1e-4
    personalize: bool = True
    federate: bool = True
    calibrate: bool = True
    coupling: bool = True


@dataclass
class SimulationConfig:
    sites: int = 4
    users: int = 20
    cells: int = 2
    slots: int = 3600
    train_episodes: int = 4
    slot_seconds: float = 1.0
    bandwidth_mhz: float = 20.0
    buffer_mbit: float = 5.0
    min_rate_mbps: list[float] = field(default_factory=lambda: [1.0, 0.15, 0.02])
    max_delay_ms: list[float] = field(default_factory=lambda: [150.0, 50.0, 500.0])
    profiles: list[str] = field(default_factory=lambda: ["embb", "latency", "mixed", "bursty"])
    aggressiveness: float = 1.0


@dataclass
class Config:
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    contract: ContractConfig = field(default_factory=ContractConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    seeds: list[int] = field(default_factory=lambda: list(range(20)))
    bootstrap_samples: int = 10000
    threads: int = 1

    def validate(self):
        s, c, learning = self.simulation, self.contract, self.learning
        for name, value in {
            "sites": s.sites,
            "users": s.users,
            "cells": s.cells,
            "slots": s.slots,
            "train_episodes": s.train_episodes,
            "period": c.period,
            "fl_period": learning.fl_period,
            "epochs": learning.epochs,
            "batch_size": learning.batch_size,
            "minimum_safe": c.minimum_safe,
            "verification_samples": c.verification_samples,
            "bootstrap_samples": self.bootstrap_samples,
            "threads": self.threads,
        }.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if s.cells > s.users or s.slots < 2:
            raise ValueError("Require users >= cells and at least two slots")
        for name in (
            "epsilon",
            "kappa",
            "delta_cal",
            "support_quantile",
            "lower_quantile",
            "upper_quantile",
            "coupling_quantile",
            "shrink_factor",
        ):
            if not 0 < getattr(c, name) < 1:
                raise ValueError(f"contract.{name} must lie strictly between 0 and 1")
        if c.lower_quantile >= c.upper_quantile or not 1 <= c.minimum_safe <= 128:
            raise ValueError("Invalid contract quantiles or minimum_safe")
        if not isinstance(c.max_shrinks, int) or c.max_shrinks < 0:
            raise ValueError("max_shrinks must be a nonnegative integer")
        positive = [
            *c.weights,
            c.solver_tolerance,
            c.perturbation_std,
            s.slot_seconds,
            s.bandwidth_mhz,
            s.buffer_mbit,
            *s.min_rate_mbps,
            *s.max_delay_ms,
            learning.learning_rate,
            learning.risk_weight,
        ]
        if not np.all(np.isfinite(positive)) or min(positive) <= 0:
            raise ValueError(
                "Scales, thresholds, learning rate and weights must be finite and positive"
            )
        if len(c.weights) != 3 or len(s.min_rate_mbps) != 3 or len(s.max_delay_ms) != 3:
            raise ValueError("Exactly three action weights and three slice SLAs are required")
        if (
            not np.isfinite(s.aggressiveness)
            or s.aggressiveness < 0
            or learning.adapter_penalty < 0
        ):
            raise ValueError("Invalid aggressiveness or adapter penalty")
        if not s.profiles or set(s.profiles) - {"embb", "latency", "mixed", "bursty"}:
            raise ValueError("Unknown traffic profile")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Seeds must be nonempty and unique")
        if any(not isinstance(x, int) or x < 0 for x in self.seeds):
            raise ValueError("Seeds must be nonnegative integers")
        return self

    def to_dict(self):
        return asdict(self)


def load_config(path: str | Path | None = None) -> Config:
    if path is None:
        raw = {}
    else:
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        raw = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a mapping")
    raw = dict(raw)
    return Config(
        simulation=SimulationConfig(**raw.pop("simulation", {})),
        contract=ContractConfig(**raw.pop("contract", {})),
        learning=LearningConfig(**raw.pop("learning", {})),
        **raw,
    ).validate()
