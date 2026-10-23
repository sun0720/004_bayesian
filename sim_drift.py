#!/usr/bin/env python3
"""Triangular (drifting-sensitivity) arm for h001-gptsol (review 011, P1-8).

Design: scalar Gaussian location Y = gamma + eps, truth 0, flat prior on gamma.
Dictionary (K=5): b1(eps)=sin(1.5 eps)/||.||, a triangular direction
b_d(eps;n) = cos(theta_n) u0(eps) + sin(theta_n) u1(eps), and three even
directions orthogonalised against b1 and b_d.  u0, u1 are orthonormal odd
directions obtained from sin(2 eps), sin(3 eps).  theta_n is chosen so that the
sensitivity of b_d is exactly lambda_n/2:

    cos(theta_n) kappa_0 + sin(theta_n) kappa_1 = lambda_n / 2.

Thus the screening condition holds with equality up to the factor 1/2, the
direction is omitted by the threshold selector, and the omitted sensitivity
drifts with n at the threshold rate.  The perturbation is a local tilt of size
A=1.5 along b_d, so A2 (span) holds and the omitted scaled bias is
A * (lambda_n/2) / J_{1}, approaching zero at the threshold rate.

This is the non-oracle regime the fixed-dictionary theorems do not cover; the
arm is descriptive evidence, not a test of P10 (whose K is fixed).

Outputs: results_drift.csv and summary_drift.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

R = 500
SEED = 20261023
NS = (300, 1000, 3000, 10000)
A_TILT = 1.5
K = 5
PRIOR_SD_DELTA = 0.1

_XQ, _WQ = np.polynomial.hermite.hermgauss(80)
_EQ = math.sqrt(2.0) * _XQ
_WQ = _WQ / math.sqrt(math.pi)

_SX = np.linspace(-9.0, 9.0, 120001)
_LOGPHI = -0.5 * _SX * _SX
_PHI = np.exp(_LOGPHI)
_ZPHI = float(np.trapezoid(_PHI, _SX))


def _ip(f, g):
    return float(np.trapezoid(f * g * _PHI, _SX) / _ZPHI)


def _norm(f):
    return f / math.sqrt(_ip(f, f))


def _orth(f, basis):
    g = f.copy()
    for b in basis:
        g = g - _ip(b, g) * b
    return _norm(g)


def _mean_eps(f):
    return float(np.trapezoid(_SX * f * _PHI, _SX) / _ZPHI)


# fixed basis
B1 = _norm(np.sin(1.5 * _SX))
U0 = _orth(np.sin(2.0 * _SX), [B1])
U1 = _orth(np.sin(3.0 * _SX), [B1, U0])
C1 = _mean_eps(B1)
K0 = _mean_eps(U0)
K1 = _mean_eps(U1)
# even directions (zero sensitivity), orthogonalised against b1 and the odd block
EVEN = []
for _k in (1.0, 2.0, 3.0):
    EVEN.append(_orth(np.cos(_k * _SX), [B1, U0, U1] + EVEN))


def _lam(n: int, k: int = K) -> float:
    return (math.log(k) ** 0.25) * (n ** (-1.0 / 6.0))


def drifting_direction(n: int):
    """Return (bd, cd, theta) with cd = lam/2.

    c(theta) = cos(theta) K0 + sin(theta) K1 is not monotone on [0, 2pi);
    scan a fine grid, collect sign changes, and take the root with the smallest
    |theta| (closest to the fixed dictionary).
    """
    target = _lam(n) / 2.0
    grid = np.linspace(-math.pi, math.pi, 20001)
    vals = np.cos(grid) * K0 + np.sin(grid) * K1 - target
    roots = []
    for i in range(len(grid) - 1):
        if vals[i] == 0.0:
            roots.append(grid[i])
        elif vals[i] * vals[i + 1] < 0:
            lo, hi = grid[i], grid[i + 1]
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if (math.cos(lo) * K0 + math.sin(lo) * K1 - target) * (
                        math.cos(mid) * K0 + math.sin(mid) * K1 - target) <= 0:
                    hi = mid
                else:
                    lo = mid
            roots.append(0.5 * (lo + hi))
    if not roots:
        raise RuntimeError(f"no theta for n={n}, target={target}")
    th = min(roots, key=abs)
    bd = math.cos(th) * U0 + math.sin(th) * U1
    cd = math.cos(th) * K0 + math.sin(th) * K1
    return bd, cd, th


def dictionary(n: int):
    bd, cd, th = drifting_direction(n)
    cols = [B1, bd, EVEN[0], EVEN[1], EVEN[2]]
    C = np.array([C1, cd, 0.0, 0.0, 0.0])
    return np.column_stack(cols), C, th


def _grid_interp(f):
    """Evaluate a function stored on _SX at arbitrary eps."""
    return lambda e: np.interp(e, _SX, f)


def k0(z):
    z = np.clip(np.asarray(z, dtype=float), -30.0, 30.0)
    return 2.0 / (1.0 + np.exp(-2.0 * z))


def gen_data(n: int, bd, rng) -> np.ndarray:
    t = A_TILT / math.sqrt(n)
    logw = _LOGPHI + t * bd
    w = np.exp(logw - logw.max())
    cdf = np.cumsum(w)
    cdf = cdf / cdf[-1]
    u = rng.random(n)
    return np.interp(u, cdf, _SX)


def fit_augmented(y, S, C, dictf, N):
    e0 = y - y.mean()
    x0 = np.array([float(y.mean()), 0.0])

    def r_of(e):
        out = np.zeros_like(e)
        for j in S:
            if C[j] != 0.0:
                out = out + C[j] * dictf[j](e)
        return out

    r_nodes = np.zeros_like(_EQ)
    for j in S:
        if C[j] != 0.0:
            r_nodes = r_nodes + C[j] * dictf[j](_EQ)

    def negll(params):
        gamma, delta = params
        e = y - gamma
        if len(S) == 0:
            return 0.5 * np.sum(e * e) + 0.5 * N * math.log(2 * math.pi) + 0.5 * (delta / PRIOR_SD_DELTA) ** 2
        r = r_of(e)
        kz = k0(delta * r)
        if not np.isfinite(kz).all() or np.any(kz <= 0):
            return 1e12
        Ez = float((_WQ * k0(delta * r_nodes)).sum())
        if not np.isfinite(Ez) or Ez <= 0:
            return 1e12
        return -( -0.5 * np.sum(e * e) - 0.5 * N * math.log(2 * math.pi)
                  + np.sum(np.log(kz)) - N * math.log(Ez)
                  - 0.5 * (delta / PRIOR_SD_DELTA) ** 2 )

    res = minimize(negll, x0, method="L-BFGS-B",
                   bounds=[(-np.inf, np.inf), (-6.0, 6.0)],
                   options={"maxiter": 150, "ftol": 1e-10})
    if not res.success or not np.isfinite(res.fun):
        return float(y.mean()), 0.0, True
    return float(res.x[0]), float(res.x[1]), False


def target_se(y, gamma_hat, S, C, dictf):
    N = len(y)
    e = y - gamma_hat
    if len(S) == 0:
        return float(e.std(ddof=1) / math.sqrt(N))
    r = np.zeros_like(e)
    for j in S:
        r = r + C[j] * dictf[j](e)
    Js = float(np.mean((e - r) ** 2))
    return math.sqrt(1.0 / (Js * N)) if Js > 1e-10 else float(e.std(ddof=1) / math.sqrt(N))


def one_rep(n, rng):
    H, C, th = dictionary(n)
    dictf = [_grid_interp(H[:, j]) for j in range(H.shape[1])]
    bd = H[:, 1]
    y = gen_data(n, bd, rng)
    m = max(10, min(int(round(n ** (2.0 / 3.0))), max(10, n // 2)))
    lam = _lam(n)

    ep = y[:m] - y[:m].mean()
    Hp = np.column_stack([f(ep) for f in dictf])
    c_hat = (ep[:, None] * Hp).mean(axis=0)
    J_hat = float(np.mean(ep * ep))
    kappa_hat = np.abs(c_hat) / max(J_hat, 1e-10)
    S = np.where(kappa_hat > lam)[0]

    y_main = y[m:]
    N = len(y_main)
    naive = float(y_main.mean())
    naive_se = float(y_main.std(ddof=1) / math.sqrt(N))

    if len(S) and (1.0 - float(np.sum(C[S] ** 2))) > 1e-8:
        gamma, delta_hat, fb = fit_augmented(y_main, S, C, dictf, N)
        se = target_se(y_main, gamma, S, C, dictf)
    else:
        gamma, delta_hat, se, fb = naive, 0.0, naive_se, False

    # reference bound for the fixed set S={1} (deterministic; not an upper
    # bound for a triangular dictionary -- reported as a reference only)
    J_ref = 1.0 - float(C[0] ** 2)
    exact = A_TILT * float(C[1]) / max(J_ref, 1e-12)
    omitted = max([abs(C[j]) for j in range(K) if j not in set(S.tolist())] + [0.0])
    cover = int(abs(gamma) <= 1.96 * se)
    cover_naive = int(abs(naive) <= 1.96 * naive_se)
    return dict(n=n, lam=lam, theta=th, c_d=float(C[1]), nS=len(S),
                sel1=int(0 in set(S.tolist())), seld=int(1 in set(S.tolist())),
                gamma=gamma, se=se, naive=naive, naive_se=naive_se,
                cover=cover, cover_naive=cover_naive,
                exact=exact, fallback=int(fb), omitted=omitted,
                delta_abs=abs(delta_hat))


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for n in NS:
        reps = [one_rep(n, rng) for _ in range(R)]
        g = np.array([r["gamma"] for r in reps])
        se = np.array([r["se"] for r in reps])
        nv = np.array([r["naive"] for r in reps])
        nse = np.array([r["naive_se"] for r in reps])
        cov = np.array([r["cover"] for r in reps])
        covn = np.array([r["cover_naive"] for r in reps])
        sc = math.sqrt(n) * g
        rows.append(dict(
            n=n, R=R, lam=float(reps[0]["lam"]), c_d=float(reps[0]["c_d"]),
            exact_bound=float(reps[0]["exact"]),
            scaled_bias=float(np.mean(sc)),
            scaled_bias_se=float(np.std(sc, ddof=1) / math.sqrt(R)),
            coverage=float(np.mean(cov)),
            coverage_se=float(math.sqrt(np.mean(cov) * (1 - np.mean(cov)) / R)),
            naive_coverage=float(np.mean(covn)),
            naive_coverage_se=float(math.sqrt(np.mean(covn) * (1 - np.mean(covn)) / R)),
            sel1=float(np.mean([r["sel1"] for r in reps])),
            seld=float(np.mean([r["seld"] for r in reps])),
            mean_nS=float(np.mean([r["nS"] for r in reps])),
            mean_abs_delta=float(np.mean([r["delta_abs"] for r in reps])),
            frac_delta_tiny=float(np.mean([r["delta_abs"] < 1e-3 for r in reps])),
            fallbacks=int(np.sum([r["fallback"] for r in reps])),
        ))
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(parents=True, exist_ok=True)
    import csv
    with (out / "results_drift.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / "summary_drift.json").write_text(json.dumps({
        "R": R, "seed": SEED, "n": list(NS), "A": A_TILT,
        "model": "Y = gamma + eps, eps tilted by A*b_d(eps;n)/sqrt(n)",
        "dictionary": "b1 fixed; b_d triangular with sensitivity lambda_n/2; 3 even directions",
        "note": "descriptive non-oracle arm; not a test of P10",
    }, indent=2))
    for r in rows:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
