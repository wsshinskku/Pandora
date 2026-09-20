import numpy as np
import pytest
from scipy.optimize import minimize

from pandora.actions import ActionSpace
from pandora.config import ContractConfig
from pandora.contracts import Contract, base_contract, fallback_contract
from pandora.synthesis import ContractSynthesizer, candidate_actions


def test_weighted_projection_has_known_solution():
    space = ActionSpace(1, 1)
    contract = Contract(
        space, np.zeros(3), np.ones(3), np.array([[1.0, 0.0, 1.0]]), np.array([0.8])
    )
    actual, distance = contract.project(np.array([0.8, 1.0, 0.8]), np.array([1.0, 1.0, 4.0]))
    np.testing.assert_allclose(actual, [0.16, 1.0, 0.64], atol=1e-5)
    assert distance > 0 and contract.contains(actual)


def test_qp_agrees_with_independent_slsqp_oracle():
    rng = np.random.default_rng(31)
    space = ActionSpace(3, 2)
    H, limits = space.coupling_template(np.array([0, 1, 2]))
    contract = fallback_contract(space, space.sample(rng, 80), H, limits)
    G, b, E, e = contract.inequalities()
    weights = space.weights([1, 1.5, 2])
    for proposal in space.sample(rng, 4):
        actual, _ = contract.project(proposal, weights)
        oracle = minimize(
            lambda a: np.dot(weights, (a - proposal) ** 2),
            space.neutral(),
            jac=lambda a: 2 * weights * (a - proposal),
            method="SLSQP",
            constraints=[
                {"type": "ineq", "fun": lambda a: b - G @ a},
                {"type": "eq", "fun": lambda a: E @ a - e},
            ],
            options={"ftol": 1e-11, "maxiter": 500},
        )
        assert oracle.success
        np.testing.assert_allclose(actual, oracle.x, atol=3e-5)


def test_identity_and_sampling_preserve_simplex_and_budget():
    rng = np.random.default_rng(4)
    space = ActionSpace(4, 2)
    contract = base_contract(space)
    proposal = space.neutral()
    projected, distance = contract.project(proposal, np.ones(space.dim))
    np.testing.assert_array_equal(projected, proposal)
    assert distance == 0
    samples = contract.sample(100, rng)
    assert all(contract.contains(a) for a in samples)
    assert np.std(samples[:, 0]) > 0.01


def test_fallback_includes_neutral_and_respects_hard_limits():
    rng = np.random.default_rng(2)
    space = ActionSpace(6, 2)
    H, limits = space.coupling_template(np.arange(6) % 3)
    fallback = fallback_contract(space, space.sample(rng, 30), H, limits)
    assert fallback.contains(space.neutral())
    assert np.all(fallback.h <= limits)
    singleton = fallback_contract(space, [], H, limits)
    np.testing.assert_allclose(singleton.sample(5, rng), np.tile(space.neutral(), (5, 1)))


def test_candidate_count_domain_and_current_proposal():
    rng = np.random.default_rng(1)
    space = ActionSpace(4, 2)
    proposal = space.neutral()
    candidates = candidate_actions(space, proposal, [], rng, 0.05)
    assert candidates.shape == (128, space.dim)
    np.testing.assert_allclose(candidates[0], proposal)
    assert all(space.valid(a) for a in candidates)


def test_invalid_contract_weights_and_nonfinite_inputs():
    space = ActionSpace(2, 1)
    contract = base_contract(space)
    with pytest.raises(ValueError):
        contract.project(space.neutral(), np.zeros(space.dim))
    with pytest.raises(ValueError):
        space.normalize(np.full(space.dim, np.nan))
    with pytest.raises(ValueError):
        Contract(
            space, np.ones(space.dim), np.zeros(space.dim), np.empty((0, space.dim)), np.empty(0)
        )


class AlwaysSupported:
    def supported(self, features):
        return np.ones(len(features), dtype=bool)

    def upper_risk(self, probabilities):
        return probabilities


class FixedRisk:
    def __init__(self, risk):
        self.risk = risk

    def predict(self, features):
        return np.zeros((len(features), 5)), np.full(len(features), self.risk)


def test_synthesis_accepts_verified_region_and_rejects_high_risk():
    space = ActionSpace(2, 1)
    H, limits = np.empty((0, space.dim)), np.empty(0)
    fallback = fallback_contract(space, [], H, limits)
    synthesis = ContractSynthesizer(
        space, ContractConfig(), FixedRisk(0), AlwaysSupported(), H, limits, fallback, 3
    )
    contract, diagnostics = synthesis.synthesize(np.zeros(2), space.neutral(), [])
    assert not diagnostics.fallback
    assert contract.feasible_point() is not None
    synthesis.model = FixedRisk(1)
    contract, diagnostics = synthesis.synthesize(np.zeros(2), space.neutral(), [])
    assert contract is fallback and diagnostics.reason == "insufficient-safe-candidates"


def test_verification_failure_shrinks_then_falls_back():
    class RejectVerification(FixedRisk):
        def predict(self, features):
            # Candidates pass; every 64-action verification batch fails.
            risk = 1 if len(features) == 64 else 0
            return np.zeros((len(features), 5)), np.full(len(features), risk)

    space = ActionSpace(2, 1)
    H, limits = np.empty((0, space.dim)), np.empty(0)
    fallback = fallback_contract(space, [], H, limits)
    synthesis = ContractSynthesizer(
        space, ContractConfig(), RejectVerification(0), AlwaysSupported(), H, limits, fallback, 3
    )
    contract, diagnostics = synthesis.synthesize(np.zeros(2), space.neutral(), [])
    assert contract is fallback
    assert diagnostics.verification_failures == 4 and diagnostics.shrinks == 3


def test_shrink_is_subset_when_anchor_is_feasible():
    space = ActionSpace(3, 2)
    contract = base_contract(space)
    shrunk = contract.shrink(space.neutral(), 0.2)
    assert all(contract.contains(a) for a in shrunk.sample(20, np.random.default_rng(3)))
