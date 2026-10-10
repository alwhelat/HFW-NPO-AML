"""
Binary Nomadic People Optimizer (NPO) with clan-family hierarchy, adaptive
Levy flight, anti-stagnation restarts and hill-climbing post-processing.

Copied from the published HFW-NPO implementation (Alwhelat & Abiyev,
Electronics 2026, 15(13), 2970). The search itself is unchanged. The only
edits are: an optional random seed, a `verbose` switch for quiet runs, and
an evaluation counter. PD-1/PD-L1 anchoring is applied by the fitness
function (see fitness.py), so every optimiser in the study is anchored the
same way.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gamma as sp_gamma


STAG_PATIENCE      = 8
LEVY_BASE          = 0.01
LEVY_MAX           = 0.40
RESTART_FRAC       = 0.20
PERTURB_FRAC       = 0.10
LOCAL_SEARCH_ITERS = 50


def _levy_step(dim: int, beta: float = 1.5) -> np.ndarray:
    num = sp_gamma(1.0 + beta) * np.sin(np.pi * beta / 2.0)
    den = sp_gamma((1.0 + beta) / 2.0) * beta * 2.0 ** ((beta - 1.0) / 2.0)
    sigma_u = (num / den) ** (1.0 / beta)
    u = np.random.normal(0.0, sigma_u, dim)
    v = np.random.normal(0.0, 1.0, dim)
    return u / (np.abs(v) ** (1.0 / beta))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


def _stochastic_binarize(pos: np.ndarray) -> np.ndarray:
    prob = _sigmoid(pos)
    return (np.random.uniform(0.0, 1.0, len(pos)) < prob).astype(np.float64)


def _deterministic_binarize(pos: np.ndarray) -> np.ndarray:
    return (_sigmoid(pos) > 0.5).astype(np.float64)


class NPOOptimizer:
    """
    Binary NPO with clan-family hierarchy, adaptive Levy flight,
    anti-stagnation restarts, and hill-climbing post-processing.
    """

    def __init__(
        self,
        obj_func,
        dim:        int,
        lb:         float = -10.0,
        ub:         float =  10.0,
        n_clans:    int   = 10,
        n_families: int   = 20,
        max_iter:   int   = 300,
        seed:       int | None = None,
        verbose:    bool  = False,
        log_fn=None,
    ) -> None:
        if seed is not None:
            np.random.seed(seed)
        self.obj_func   = obj_func
        self.dim        = dim
        self.lb         = lb
        self.ub         = ub
        self.n_clans    = n_clans
        self.n_families = n_families
        self.pop_size   = n_clans * n_families
        self.max_iter   = max_iter
        self.verbose    = verbose
        self.log_fn     = log_fn
        self.n_evals    = 0
        self.history: list[float] = []

        self.population = np.random.uniform(lb, ub, (self.pop_size, dim))
        self.fitness    = np.full(self.pop_size, np.inf)

        self.global_best_pos = np.zeros(dim)
        self.global_best_fit = np.inf
        self.clan_best_pos   = np.zeros((n_clans, dim))
        self.clan_best_fit   = np.full(n_clans, np.inf)

        self.best_binary: np.ndarray | None = None

    def _log(self, msg: str) -> None:
        if self.log_fn is not None:
            self.log_fn(msg.strip())
        elif self.verbose:
            print(msg)

    def _clan_of(self, idx: int) -> int:
        return idx // self.n_families

    def _evaluate(self, pos: np.ndarray) -> float:
        self.n_evals += 1
        return float(self.obj_func(_stochastic_binarize(pos)))

    def _evaluate_binary(self, binary: np.ndarray) -> float:
        self.n_evals += 1
        return float(self.obj_func(binary))

    def _update_global_best(self, fit: float, pos: np.ndarray) -> bool:
        if fit < self.global_best_fit:
            self.global_best_fit = fit
            self.global_best_pos = pos.copy()
            return True
        return False

    def _update_clan_best(self, c: int, fit: float, pos: np.ndarray) -> bool:
        if fit < self.clan_best_fit[c]:
            self.clan_best_fit[c] = fit
            self.clan_best_pos[c] = pos.copy()
            return True
        return False

    def _migrate(self) -> None:
        best_c = int(np.argmin(self.clan_best_fit))
        for c in range(self.n_clans):
            if c == best_c:
                continue
            s = c * self.n_families
            e = s + self.n_families
            worst = s + int(np.argmax(self.fitness[s:e]))
            self.population[worst] = np.clip(
                self.population[worst]
                + 0.5 * (self.clan_best_pos[best_c] - self.population[worst]),
                self.lb, self.ub,
            )
            nf = self._evaluate(self.population[worst])
            self.fitness[worst] = nf
            self._update_clan_best(c, nf, self.population[worst])
            self._update_global_best(nf, self.population[worst])

    def _restart_worst_agents(self, frac: float) -> None:
        n_restart = max(1, int(frac * self.pop_size))
        for idx in np.argsort(self.fitness)[::-1][:n_restart]:
            if np.random.rand() < 0.5:
                self.population[idx] = np.random.uniform(self.lb, self.ub, self.dim)
            else:
                perturbed = self.global_best_pos.copy()
                n_flip = max(1, int(PERTURB_FRAC * self.dim))
                perturbed[np.random.choice(self.dim, n_flip, replace=False)] = \
                    np.random.uniform(self.lb, self.ub, n_flip)
                self.population[idx] = np.clip(perturbed, self.lb, self.ub)
            nf = self._evaluate(self.population[idx])
            self.fitness[idx] = nf
            c = self._clan_of(idx)
            self._update_clan_best(c, nf, self.population[idx])
            self._update_global_best(nf, self.population[idx])

    def _local_search(self, binary: np.ndarray, best_fit: float) -> tuple[np.ndarray, float]:
        """Flip-one-bit hill climbing to refine the best solution."""
        current     = binary.copy()
        current_fit = best_fit
        improved    = 0

        for _ in range(LOCAL_SEARCH_ITERS):
            idx = np.random.randint(0, self.dim)
            candidate = current.copy()
            candidate[idx] = 1.0 - candidate[idx]
            if candidate.sum() == 0:
                continue
            cfit = self._evaluate_binary(candidate)
            if cfit < current_fit:
                current     = candidate
                current_fit = cfit
                improved   += 1

        self._log(f"   local search: improvements={improved}, "
                  f"genes={int(current.sum())}, fit={current_fit:.6f}")
        return current, current_fit

    def optimize(self) -> tuple:
        self._log(f"   NPO: {self.n_clans}x{self.n_families}={self.pop_size} agents, "
                  f"{self.max_iter} iters, dim={self.dim}")

        for i in range(self.pop_size):
            fit = self._evaluate(self.population[i])
            c   = self._clan_of(i)
            self.fitness[i] = fit
            self._update_clan_best(c, fit, self.population[i])
            self._update_global_best(fit, self.population[i])

        stag_counter = 0
        prev_best    = self.global_best_fit

        for t in range(self.max_iter):
            progress = t / max(self.max_iter - 1, 1)
            alpha_t  = 0.50 * (1.0 - progress)
            beta_t   = 0.80 * progress

            levy_scale = LEVY_BASE * (
                1.0 + (LEVY_MAX / LEVY_BASE - 1.0)
                * min(stag_counter / STAG_PATIENCE, 1.0)
            )

            for i in range(self.pop_size):
                c          = self._clan_of(i)
                vec_intra  = self.clan_best_pos[c] - self.population[i]
                vec_global = self.global_best_pos  - self.population[i]
                levy_noise = levy_scale * _levy_step(self.dim)

                new_pos = np.clip(
                    self.population[i]
                    + alpha_t * vec_intra
                    + beta_t  * vec_global
                    + levy_noise,
                    self.lb, self.ub,
                )
                new_fit = self._evaluate(new_pos)

                if new_fit < self.fitness[i]:
                    self.fitness[i]    = new_fit
                    self.population[i] = new_pos
                    self._update_clan_best(c, new_fit, new_pos)
                    self._update_global_best(new_fit, new_pos)

            if t > 0 and t % 5 == 0:
                self._migrate()

            if self.global_best_fit < prev_best - 1e-9:
                stag_counter = 0
                prev_best    = self.global_best_fit
            else:
                stag_counter += 1

            if stag_counter >= STAG_PATIENCE:
                self._restart_worst_agents(RESTART_FRAC)
                stag_counter = 0
                prev_best    = self.global_best_fit

            self.history.append(float(self.global_best_fit))

            if (t + 1) % 50 == 0 or t == 0:
                self._log(f"   iter {t+1}/{self.max_iter}: best={self.global_best_fit:.6f}")

        best_binary  = _deterministic_binarize(self.global_best_pos)
        best_fit_det = self._evaluate_binary(best_binary)
        refined, refined_fit = self._local_search(best_binary, best_fit_det)

        if refined_fit < self.global_best_fit:
            self.global_best_fit = refined_fit
            self.best_binary     = refined
        else:
            self.best_binary = best_binary

        return self.global_best_pos, self.global_best_fit

    def get_selected_indices(self) -> np.ndarray:
        if self.best_binary is not None:
            return np.sort(np.where(self.best_binary > 0.5)[0])
        binary = _deterministic_binarize(self.global_best_pos)
        return np.sort(np.where(binary > 0.5)[0])
