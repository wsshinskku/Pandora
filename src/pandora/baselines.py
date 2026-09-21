"""Executable xApp coordination comparison methods."""

import numpy as np

from .contracts import Contract, ProjectionError, base_contract


class BaselineController:
    def __init__(self, method, space, config, fallback):
        if method not in {
            "independent",
            "static",
            "adaptive",
            "scheduler-reference",
            "qos-reference",
        }:
            raise ValueError(f"Unknown reference baseline {method}")
        self.method, self.space, self.config, self.fallback = method, space, config, fallback
        self.active = base_contract(space)
        self.compliant = []

    def decide(self, proposal, slot, telemetry):
        if self.method == "independent":
            return proposal.copy()
        if self.method == "static":
            return self.fallback.project(proposal, self.space.weights(self.config.weights))[0]
        if self.method == "adaptive":
            if slot % self.config.period == 0 and len(self.compliant) >= 2:
                data = np.asarray(self.compliant[-600:])
                candidate = Contract(
                    self.space,
                    np.clip(np.quantile(data, 0.1, axis=0), 0, 1),
                    np.clip(np.quantile(data, 0.9, axis=0), 0, 1),
                    np.empty((0, self.space.dim)),
                    np.empty(0),
                )
                if candidate.feasible_point() is not None:
                    self.active = candidate
            try:
                return self.active.project(proposal, self.space.weights(self.config.weights))[0]
            except ProjectionError:
                return self.fallback.project(proposal, self.space.weights(self.config.weights))[0]
        if self.method == "scheduler-reference":
            # Execute one owner's block per slot and suppress other blocks to neutral.
            action = self.space.neutral()
            block = self.space.blocks[slot % 3]
            action[block] = proposal[block]
            return self.space.normalize(action)
        # A heuristic QoS comparator that boosts resource/priority for queued UEs.
        rho, nu, omega = (x.copy() for x in self.space.split(proposal))
        urgency = telemetry.queue / max(float(telemetry.queue.max()), 1e-6)
        rho += 0.1 * self.space.cells / self.space.users * urgency
        omega = np.maximum(omega, urgency)
        return self.space.normalize(self.space.join(rho, nu, omega))

    def feedback(self, action, kpis):
        if kpis[3] <= self.config.epsilon:
            self.compliant.append(action.copy())
