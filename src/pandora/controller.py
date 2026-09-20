"""Runtime slot loop: periodic contracts, per-slot support guard and QP enforcement."""

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

from .contracts import ProjectionError


@dataclass
class Decision:
    action: np.ndarray
    distance: float
    normalized_distance: float
    intervention: bool
    fallback: bool
    supported: bool
    reason: str
    synthesis: dict | None = None


class PandoraController:
    def __init__(self, synthesizer):
        self.synthesizer = synthesizer
        self.active = synthesizer.fallback
        self.recent = deque(maxlen=15)
        self.weights = synthesizer.space.weights(synthesizer.config.weights)

    def decide(self, observation, proposal, slot):
        if slot < 0:
            raise ValueError("Slots are zero-based and nonnegative")
        s, diagnostics = self.synthesizer, None
        if slot % s.config.period == 0:
            self.active, diagnostics = s.synthesize(observation, proposal, list(self.recent))
        covered = bool(s.calibrator.supported(np.r_[observation, proposal][None])[0])
        runtime = self.active if covered else s.fallback
        reason = runtime.reason if covered else "unsupported-proposal"
        try:
            action, distance = runtime.project(proposal, self.weights, s.config.solver_tolerance)
        except ProjectionError:
            runtime, reason = s.fallback, "projection-solver-failure"
            # If even fallback solving fails, execute its checked feasible neutral action.
            try:
                action, distance = runtime.project(
                    proposal, self.weights, s.config.solver_tolerance
                )
            except ProjectionError:
                action = s.space.neutral()
                if not runtime.contains(action):
                    raise ProjectionError("No verified feasible emergency action") from None
                distance = float(np.sqrt(np.dot(self.weights, (action - proposal) ** 2)))
                reason = "neutral-emergency-action"
        self.recent.append(proposal.copy())
        return Decision(
            action,
            distance,
            distance / np.sqrt(self.weights.sum()),
            bool(distance > s.config.solver_tolerance or runtime.fallback),
            runtime.fallback,
            covered,
            reason,
            asdict(diagnostics) if diagnostics else None,
        )
