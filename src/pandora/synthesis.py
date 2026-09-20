"""Algorithm 1: candidate generation, support/risk screening, fitting, verification and shrink."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist

from .contracts import Contract, ProjectionError


@dataclass
class SynthesisDiagnostics:
    candidates: int = 128
    supported: int = 0
    safe: int = 0
    verification_failures: int = 0
    shrinks: int = 0
    fallback: bool = False
    reason: str = "accepted"


def candidate_actions(space, proposal, recent, rng, perturbation_std):
    # Early periods repeat available history, preserving the paper's 1+15+64+48 count.
    history = [np.array(a, copy=True) for a in recent[-15:]] or [proposal.copy()]
    history = [history[j % len(history)] for j in range(15)]
    perturbed = [
        space.normalize(history[rng.integers(15)] + rng.normal(0, perturbation_std, space.dim))
        for _ in range(64)
    ]
    return np.array([space.normalize(proposal), *history, *perturbed, *space.sample(rng, 48)])


class ContractSynthesizer:
    def __init__(self, space, config, model, calibrator, H, limits, fallback, seed):
        self.space, self.config, self.model, self.calibrator = space, config, model, calibrator
        self.H, self.limits, self.fallback = H, limits, fallback
        self.rng = np.random.default_rng(seed)

    def features(self, observation, actions):
        return np.column_stack([np.repeat(observation[None], len(actions), axis=0), actions])

    def screen(self, observation, actions):
        x = self.features(observation, actions)
        support = self.calibrator.supported(x)
        _, probability = self.model.predict(x)
        return support, support & (self.calibrator.upper_risk(probability) <= self.config.kappa)

    def synthesize(self, observation, proposal, recent):
        c = self.config
        diagnostics = SynthesisDiagnostics()

        def fail(reason):
            diagnostics.fallback, diagnostics.reason = True, reason
            return self.fallback, diagnostics

        candidates = candidate_actions(self.space, proposal, recent, self.rng, c.perturbation_std)
        support, safe_mask = self.screen(observation, candidates)
        diagnostics.supported, diagnostics.safe = int(support.sum()), int(safe_mask.sum())
        safe = candidates[safe_mask]
        if len(safe) < c.minimum_safe:
            return fail("insufficient-safe-candidates")
        lower = np.maximum(0, np.quantile(safe, c.lower_quantile, axis=0))
        upper = np.minimum(1, np.quantile(safe, c.upper_quantile, axis=0))
        h = np.minimum(self.limits, np.quantile(safe @ self.H.T, c.coupling_quantile, axis=0))
        contract = Contract(self.space, lower, upper, self.H, h)
        # Paper leaves medoid metric unspecified: use raw Euclidean action distance.
        medoid = safe[np.argmin(cdist(safe, safe).sum(axis=1))]
        if not contract.contains(medoid):
            feasible_safe = safe[[contract.contains(a) for a in safe]]
            if len(feasible_safe):
                medoid = feasible_safe[np.argmin(cdist(feasible_safe, safe).sum(axis=1))]
            else:
                # No unscreened projection is promoted to a safe medoid.
                return fail("no-feasible-screened-anchor")
        for shrink in range(c.max_shrinks + 1):
            try:
                samples = contract.sample(c.verification_samples, self.rng, start=medoid)
            except ProjectionError:
                return fail("empty-or-numerically-invalid-region")
            if self.screen(observation, samples)[1].all():
                diagnostics.shrinks = shrink
                return contract, diagnostics
            diagnostics.verification_failures += 1
            if shrink < c.max_shrinks:
                contract = contract.shrink(medoid, c.shrink_factor)
        diagnostics.shrinks = c.max_shrinks
        return fail("verification-failed")
