"""Linear contracts, nonempty fallback, exact weighted QP and interior verification sampling."""

from dataclasses import dataclass, field

import numpy as np
import osqp
from scipy import sparse
from scipy.linalg import null_space
from scipy.optimize import linprog

from .actions import ActionSpace


class ProjectionError(RuntimeError):
    pass


@dataclass
class Contract:
    space: ActionSpace
    lower: np.ndarray
    upper: np.ndarray
    H: np.ndarray
    h: np.ndarray
    fallback: bool = False
    reason: str = "learned"
    _solvers: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self.lower, self.upper, self.H, self.h = (
            np.asarray(x, dtype=float).copy() for x in (self.lower, self.upper, self.H, self.h)
        )
        d = self.space.dim
        if self.lower.shape != (d,) or self.upper.shape != (d,):
            raise ValueError("Invalid contract bounds shape")
        if self.H.ndim != 2 or self.H.shape[1] != d or self.h.shape != (len(self.H),):
            raise ValueError("Invalid coupling shape")
        if any(not np.isfinite(x).all() for x in (self.lower, self.upper, self.H, self.h)):
            raise ValueError("Contract values must be finite")
        if np.any(self.lower > self.upper) or np.any(self.lower < 0) or np.any(self.upper > 1):
            raise ValueError("Contract bounds must satisfy 0 <= lower <= upper <= 1")

    def contains(self, action, tol=1e-5):
        a = np.asarray(action)
        return (
            self.space.valid(a, tol)
            and np.all(a >= self.lower - tol)
            and np.all(a <= self.upper + tol)
            and np.all(self.H @ a <= self.h + tol)
        )

    def inequalities(self):
        domain, lo, hi = self.space.linear_domain()
        # Last row is the resource budget. Simplexes are separate equalities.
        G = np.vstack(
            [np.eye(self.space.dim), -np.eye(self.space.dim), self.H, domain[-1:].toarray()]
        )
        b = np.r_[self.upper, -self.lower, self.h, hi[-1]]
        return G, b, domain[:-1].toarray(), lo[:-1]

    def feasible_point(self):
        G, b, E, e = self.inequalities()
        result = linprog(
            np.zeros(self.space.dim),
            A_ub=G,
            b_ub=b,
            A_eq=E,
            b_eq=e,
            bounds=[(None, None)] * self.space.dim,
            method="highs",
        )
        if not result.success or not self.contains(result.x, 1e-6):
            return None
        return result.x

    def project(self, proposal, weights, tol=1e-5):
        a = np.asarray(proposal, dtype=float)
        self.space.split(a)
        w = np.asarray(weights, dtype=float)
        if w.shape != a.shape or not np.isfinite(w).all() or np.any(w <= 0):
            raise ValueError("Projection weights must be finite and strictly positive")
        if self.contains(a, tol):
            return a.copy(), 0.0
        key = (tuple(w), tol)
        if key not in self._solvers:
            domain, dl, du = self.space.linear_domain()
            A = sparse.vstack(
                [sparse.eye(self.space.dim), sparse.csc_matrix(self.H), domain], format="csc"
            )
            lo = np.r_[self.lower, np.full(len(self.h), -np.inf), dl]
            hi = np.r_[self.upper, self.h, du]
            solver = osqp.OSQP()
            solver.setup(
                P=sparse.diags(2 * w, format="csc"),
                q=-2 * w * a,
                A=A,
                l=lo,
                u=hi,
                eps_abs=tol / 10,
                eps_rel=tol / 10,
                max_iter=20000,
                polishing=True,
                verbose=False,
            )
            self._solvers[key] = solver
        solver = self._solvers[key]
        solver.update(q=-2 * w * a)
        result = solver.solve(raise_error=False)
        if result.info.status_val not in (1, 2) or result.x is None:
            raise ProjectionError(f"OSQP failed: {result.info.status}")
        if not self.contains(result.x, tol):
            raise ProjectionError("OSQP result violates contract beyond configured tolerance")
        # Do not clip or normalize here: post-QP changes can violate coupled rows.
        distance = float(np.sqrt(np.dot(w, (result.x - a) ** 2)))
        return result.x.copy(), distance

    def sample(self, count, rng, start=None):
        """Hit-and-run in the affine hull, including degenerate/fixed-coordinate contracts.

        Samples are correlated numerical checks, not a proof of safety. A Chebyshev
        center avoids getting stuck at a linear-program vertex.
        """
        G, b, E, e = self.inequalities()
        fixed = np.flatnonzero(self.upper - self.lower < 1e-10)
        if len(fixed):
            E = np.vstack([E, np.eye(self.space.dim)[fixed]])
            e = np.r_[e, self.lower[fixed]]
        basis = null_space(E)
        point = self.feasible_point() if start is None else np.array(start, copy=True)
        if point is None or not self.contains(point, 1e-6):
            raise ProjectionError("Cannot sample an empty contract")
        if basis.shape[1] == 0:
            return np.repeat(point[None], count, axis=0)
        norms = np.linalg.norm(G @ basis, axis=1)
        center = linprog(
            np.r_[np.zeros(self.space.dim), -1.0],
            A_ub=np.column_stack([G, norms]),
            b_ub=b,
            A_eq=np.column_stack([E, np.zeros(len(E))]),
            b_eq=e,
            bounds=[(None, None)] * self.space.dim + [(0, None)],
            method="highs",
        )
        if center.success:
            point = center.x[:-1]
        samples = []
        for step in range(32 + count * 3):
            direction = basis @ rng.normal(size=basis.shape[1])
            length = np.linalg.norm(direction)
            if length < 1e-12:
                continue
            direction /= length
            slopes, slack = G @ direction, b - G @ point
            pos, neg = slopes > 1e-10, slopes < -1e-10
            upper = np.min(slack[pos] / slopes[pos], initial=np.inf)
            lower = np.max(slack[neg] / slopes[neg], initial=-np.inf)
            if np.isfinite(lower + upper) and lower <= upper:
                point = point + rng.uniform(lower, upper) * direction
            if step >= 32 and (step - 32) % 3 == 0:
                samples.append(point.copy())
        result = np.array(samples)
        if len(result) != count or not all(self.contains(x, 1e-6) for x in result):
            raise ProjectionError("Numerical contract sampling failure")
        return result

    def shrink(self, medoid, gamma):
        return Contract(
            self.space,
            (1 - gamma) * self.lower + gamma * medoid,
            (1 - gamma) * self.upper + gamma * medoid,
            self.H,
            (1 - gamma) * self.h + gamma * (self.H @ medoid),
        )


def base_contract(space):
    return Contract(
        space,
        np.zeros(space.dim),
        np.ones(space.dim),
        np.empty((0, space.dim)),
        np.empty(0),
        reason="base-domain",
    )


def fallback_contract(space, actions, H, limits):
    """20/80 box and 80th-percentile coupling, expanded only enough to contain a0."""
    neutral = space.neutral()
    if not space.valid(neutral) or np.any(H @ neutral > limits + 1e-8):
        raise ValueError("Neutral action violates the operator's hard limits")
    data = np.asarray(actions)
    if data.size == 0:
        data = neutral[None]
    if data.ndim != 2 or data.shape[1] != space.dim or not all(space.valid(a) for a in data):
        raise ValueError("Fallback requires valid calibration actions")
    lower = np.minimum(np.quantile(data, 0.2, axis=0), neutral)
    upper = np.maximum(np.quantile(data, 0.8, axis=0), neutral)
    h = np.minimum(limits, np.maximum(np.quantile(data @ H.T, 0.8, axis=0), H @ neutral))
    contract = Contract(
        space,
        np.clip(lower, 0, 1),
        np.clip(upper, 0, 1),
        H,
        h,
        fallback=True,
        reason="conservative-fallback",
    )
    if not contract.contains(neutral):
        raise ValueError("Fallback construction did not preserve the neutral action")
    return contract
