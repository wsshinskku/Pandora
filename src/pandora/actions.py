"""Joint action [rho_U, nu_UxG, omega_U], hard domain and sparse coupling template."""

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass
class ActionSpace:
    users: int
    cells: int

    @property
    def dim(self):
        return self.users * (self.cells + 2)

    @property
    def blocks(self):
        n, g = self.users, self.cells
        return slice(0, n), slice(n, n + n * g), slice(n + n * g, self.dim)

    def split(self, action):
        a = np.asarray(action, dtype=float)
        if a.shape != (self.dim,) or not np.isfinite(a).all():
            raise ValueError(f"Action must be a finite vector of length {self.dim}")
        r, v, w = self.blocks
        return a[r], a[v].reshape(self.users, self.cells), a[w]

    def join(self, rho, nu, omega):
        return np.concatenate([np.ravel(rho), np.ravel(nu), np.ravel(omega)])

    def neutral(self):
        return self.join(
            np.full(self.users, 0.8 * self.cells / self.users),
            np.full((self.users, self.cells), 1 / self.cells),
            np.full(self.users, 0.5),
        )

    def normalize(self, action):
        rho, nu, omega = (x.copy() for x in self.split(action))
        rho = np.clip(rho, 0, 1)
        rho *= min(1, self.cells / max(rho.sum(), 1e-12))
        nu = np.maximum(nu, 0)
        sums = nu.sum(axis=1, keepdims=True)
        nu = np.divide(nu, sums, out=np.full_like(nu, 1 / self.cells), where=sums > 0)
        return self.join(rho, nu, np.clip(omega, 0, 1))

    def sample(self, rng, count):
        samples = []
        for _ in range(count):
            rho = rng.dirichlet(np.ones(self.users)) * self.cells * rng.uniform(0.5, 1)
            nu = rng.dirichlet(np.ones(self.cells), size=self.users)
            samples.append(self.normalize(self.join(rho, nu, rng.uniform(0, 1, self.users))))
        return np.array(samples)

    def weights(self, blocks):
        return np.concatenate(
            [
                np.full(self.users, blocks[0]),
                np.full(self.users * self.cells, blocks[1]),
                np.full(self.users, blocks[2]),
            ]
        )

    def linear_domain(self):
        """Simplex equalities and total resource budget, retained even in H=0 ablation."""
        rows = np.zeros((self.users + 1, self.dim))
        for u in range(self.users):
            rows[u, self.users + u * self.cells : self.users + (u + 1) * self.cells] = 1
        rows[-1, : self.users] = 1
        lower = np.r_[np.ones(self.users), -np.inf]
        upper = np.r_[np.ones(self.users), self.cells]
        return sparse.csc_matrix(rows), lower, upper

    def valid(self, action, tol=1e-5):
        a = np.asarray(action)
        if a.shape != (self.dim,) or not np.isfinite(a).all():
            return False
        rho, nu, _ = self.split(a)
        return bool(
            np.all(a >= -tol)
            and np.all(a <= 1 + tol)
            and rho.sum() <= self.cells + tol
            and np.all(np.abs(nu.sum(axis=1) - 1) <= tol)
        )

    def coupling_template(self, slices):
        """Reference operator template; exact H and b were not specified in the PDF."""
        rows, limits = [], []
        n, g = self.users, self.cells
        for u in range(n):
            for cell in range(g):
                row = np.zeros(self.dim)
                row[n + u * g + cell] = 1 / g
                row[u] = -n / g
                rows.append(row)
                limits.append(0.20)
            row = np.zeros(self.dim)
            row[n + n * g + u], row[u] = 1, -n / g
            rows.append(row)
            limits.append(0.35)
        for slice_id in np.unique(slices):
            row = np.zeros(self.dim)
            members = np.flatnonzero(slices == slice_id)
            row[members] = -1
            rows.append(row)
            limits.append(-0.10 * g * len(members) / n)
        return np.array(rows), np.array(limits)
