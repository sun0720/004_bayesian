#!/usr/bin/env python3
"""One-nuisance arm for h001-gptsol (review 010, P1-4).

Design: Y | (gamma, sigma) ~ N(gamma, sigma^2), target gamma, nuisance sigma.
Dictionary: b_j(Y) = sin(jY), j = 1,2,3, centred and orthogonalised against the
nuisance score and each other at theta0 = (0,1), then whitened.  In this Gaussian
location-scale model I_{gamma sigma}=0, so the efficient score equals the
gamma-score and the projection onto the nuisance score is vacuous.  The arm
exercises the theta-dependence of C_S(theta) and the P8 plug-in term, not the
projection.

Perturbation: dP_{n,a}/dP_{theta0} = exp(a b_1(Y)/sqrt(n)) / Z_n(a).

Outputs: results_nuisance.csv (bias, RMSE, coverage by n and estimator),
summary_nuisance.json.  Fixed seed; deterministic given the environment.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

R = 500
SEED = 20261022
NS = (300, 1000, 3000)
A_VALUES = (0.0, 1.5)
K = 3
PRIOR_SD_DELTA = 0.1
ZFREQ = 1.0  # the tilted direction is b_1

_XQ, _WQ = np.polynomial.hermite.hermgauss(60)
_ZQ = math.sqrt(2.0) * _XQ
_WQ = _WQ / math.sqrt(math.pi)


def k0(z):
    z = np.clip(np.asarray(z, dtype=float), -30.0, 30.0)
    return 2.0 / (1.0 + np.exp(-2.0 * z))


# ---- dictionary: sin(jY), orthogonalised at theta0=(0,1), whitened ----------
def _raw(y):
    y = np.asarray(y, dtype=float)
    return np.column_stack([np.sin(j * y) for j in range(1, K + 1)])


def _gram(y, w=None):
    b = _raw(y)
    if w is None:
        w = np.ones(len(y)) / len(y)
    return (b * w[:, None]).T @ b


def build_dictionary():
    """Gram-Schmidt under N(0,1); return coefficients on sin(j y).

    Each orthonormalised direction is a finite linear combination of the raw
    functions sin(j y), so it can be evaluated analytically at any y.
    """
    A = np.eye(K)
    w = _WQ
    nu = (1.0 / math.sqrt(2.0)) * (_ZQ ** 2 - 1.0)
    for j in range(K):
        v = A[:, j].copy()
        # remove the nuisance-score component: score_sigma = (z^2-1)/sqrt(2)
        raw = _raw(_ZQ)
        coef_nu = np.sum(w * (raw @ v) * nu) / np.sum(w * nu * nu)
        # express nu in the raw basis is not needed: subtract it from the values,
        # then re-fit the coefficients numerically.
        vals = raw @ v - coef_nu * nu
        # solve least squares for the coefficients on sin(j y)
        v = np.linalg.lstsq(raw, vals, rcond=None)[0]
        for u in range(j):
            vals = vals - np.sum(w * vals * (raw @ A[:, u])) * (raw @ A[:, u])
            v = np.linalg.lstsq(raw, vals, rcond=None)[0]
        norm = math.sqrt(np.sum(w * (raw @ v) ** 2))
        A[:, j] = v / norm
    return A


A_DICT = build_dictionary()


def b_eval(y):
    """Dictionary evaluated analytically: b_j(y) = sum_k A[k,j] sin(k y)."""
    y = np.asarray(y, dtype=float)
    return _raw(y) @ A_DICT


# ---- model quantities -------------------------------------------------------
def moments(theta):
    """Return J, C (K,), Omega (K,K) at theta=(gamma,sigma)."""
    gamma, sigma = theta
    y = gamma + sigma * _ZQ
    s = (y - gamma) / sigma ** 2
    b = b_eval(y)
    C = (b * (_WQ * s)[:, None]).sum(axis=0)
    Omega = (b * _WQ[:, None]).T @ b
    J = float((_WQ * s * s).sum())
    return J, C, Omega


def r_of(y, theta, S):
    y = np.asarray(y, dtype=float)
    if len(S) == 0:
        return np.zeros_like(y)
    J, C, Om = moments(theta)
    b = b_eval(y)
    coef = np.linalg.solve(Om[np.ix_(S, S)], C[S])
    return b[:, S] @ coef


def z_norm(theta, delta, S):
    if len(S) == 0 or delta == 0.0:
        return 1.0
    r = r_of(theta[0] + theta[1] * _ZQ, theta, S)
    return float((_WQ * k0(delta * r)).sum())


def neg_loglik(params, y, S):
    gamma, log_sigma, delta = params
    sigma = math.exp(log_sigma)
    if not (0.2 < sigma < 5.0):
        return 1e12
    e = y - gamma
    n = len(y)
    ll = -0.5 * np.sum((e / sigma) ** 2) - n * math.log(sigma) - 0.5 * n * math.log(2 * math.pi)
    if len(S) > 0 and abs(delta) > 0:
        r = r_of(y, (gamma, sigma), S)
        kz = k0(delta * r)
        if not np.isfinite(kz).all() or np.any(kz <= 0):
            return 1e12
        Z = z_norm((gamma, sigma), delta, S)
        if not np.isfinite(Z) or Z <= 0:
            return 1e12
        ll += float(np.sum(np.log(kz))) - n * math.log(Z)
    ll -= 0.5 * (delta / PRIOR_SD_DELTA) ** 2
    return -ll


def fit(y, S, x0=None):
    if x0 is None:
        x0 = (float(np.mean(y)), math.log(float(np.std(y, ddof=1))), 0.0)
    res = minimize(neg_loglik, np.array(x0, dtype=float), args=(y, S),
                   method="L-BFGS-B",
                   bounds=[(-np.inf, np.inf), (math.log(0.2), math.log(5.0)), (-6.0, 6.0)],
                   options={"maxiter": 200, "ftol": 1e-10})
    if not res.success or not np.isfinite(res.fun):
        return float(np.mean(y)), float(np.std(y, ddof=1)), 0.0
    return float(res.x[0]), math.exp(float(res.x[1])), float(res.x[2])


def se_feasible(y, theta_hat, S):
    gamma, sigma = theta_hat
    n = len(y)
    if len(S) == 0:
        return float(np.std(y, ddof=1) / math.sqrt(n))
    J, C, Om = moments((gamma, sigma))
    b = b_eval(y)
    s = (y - gamma) / sigma ** 2
    tilde = s - b[:, S] @ np.linalg.solve(Om[np.ix_(S, S)], C[S])
    V = float(np.mean(tilde * tilde))
    return math.sqrt(max(1.0 / (V * n), 0.0))


# ---- data generation --------------------------------------------------------
def tilt_const(a, n):
    return a / math.sqrt(n)


def tilted_mean(a, n):
    t = tilt_const(a, n)
    b1 = b_eval(_ZQ)[:, 0]
    w = _WQ * np.exp(t * b1)
    return float((w * _ZQ).sum() / w.sum())


def draw(n, rng, a):
    if a == 0.0:
        return rng.normal(size=n)
    t = tilt_const(a, n)
    b1 = b_eval(_ZQ)[:, 0]
    logm = float(np.max(t * b1))
    M = math.exp(logm)
    out = np.empty(n)
    filled = 0
    while filled < n:
        cand = rng.normal(size=max(64, n - filled))
        acc = np.exp(t * b_eval(cand)[:, 0] - logm)
        ok = cand[rng.random(len(cand)) < acc]
        take = min(len(ok), n - filled)
        out[filled:filled + take] = ok[:take]
        filled += take
    return out


def pilot_select(y_p, lam):
    g = float(np.mean(y_p))
    s = float(np.std(y_p, ddof=1))
    J, C, Om = moments((g, s))
    kappa = np.abs(C) / J
    return np.array([j for j in range(K) if kappa[j] > lam], dtype=int)


# ---- one replication --------------------------------------------------------
def one_rep(n, a, rng):
    m = max(10, int(round(n ** (2.0 / 3.0))))
    m = min(m, n // 3)
    y_all = draw(n, rng, a)
    y_p, y_m = y_all[:m], y_all[m:]
    lam = (math.log(K) ** 0.25) * n ** (-1.0 / 6.0)
    S = pilot_select(y_p, lam)

    g_aug, s_aug, d_aug = fit(y_m, S)
    se_aug = se_feasible(y_m, (g_aug, s_aug), S)

    g_nv = float(np.mean(y_m))
    se_nv = float(np.std(y_m, ddof=1) / math.sqrt(len(y_m)))

    oracle_shift = tilted_mean(a, n)
    g_or = g_nv - oracle_shift
    se_or = se_nv

    z = 1.959963984540054
    J0, C0, Om0 = moments((0.0, 1.0))
    J1, C1v, Om1 = moments((g_aug, s_aug))
    plugin = float(np.mean(np.abs(C1v[S] - C0[S]))) if len(S) > 0 else 0.0
    return {
        "S_size": int(len(S)),
        "plugin_abs": plugin,
        "selected_1": int(0 in S.tolist()),
        "aug_bias": g_aug,
        "aug_rmse": g_aug ** 2,
        "aug_cov": float(abs(g_aug) <= z * se_aug),
        "naive_bias": g_nv,
        "naive_rmse": g_nv ** 2,
        "naive_cov": float(abs(g_nv) <= z * se_nv),
        "oracle_bias": g_or,
        "oracle_rmse": g_or ** 2,
        "oracle_cov": float(abs(g_or) <= z * se_or),
    }


def main() -> int:
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows = []
    t0 = time.time()
    for n in NS:
        for a in A_VALUES:
            reps = [one_rep(n, a, rng) for _ in range(R)]
            for est in ("aug", "naive", "oracle"):
                bias = np.array([r[f"{est}_bias"] for r in reps])
                cov = np.array([r[f"{est}_cov"] for r in reps])
                plugin = np.mean([r["plugin_abs"] for r in reps]) if est == "aug" else 0.0
                rows.append({
                    "n": n, "a": a, "estimator": est, "R": R,
                    "plugin_abs": plugin,
                    "bias": float(bias.mean()),
                    "bias_se": float(bias.std(ddof=1) / math.sqrt(R)),
                    "rmse": float(math.sqrt(np.mean(bias ** 2))),
                    "coverage": float(cov.mean()),
                    "coverage_se": float(math.sqrt(cov.mean() * (1 - cov.mean()) / R)),
                })
            print(f"n={n} a={a}: aug bias={np.mean([r['aug_bias'] for r in reps]):+.4f} "
                  f"cov={np.mean([r['aug_cov'] for r in reps]):.3f} "
                  f"S={np.mean([r['S_size'] for r in reps]):.2f} "
                  f"sel1={np.mean([r['selected_1'] for r in reps]):.3f}", flush=True)
    import csv
    csv_path = out_dir / "results_nuisance.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {
        "R": R, "seed": SEED, "n": list(NS), "a": list(A_VALUES),
        "dictionary": "sin(jY), j=1..3, orthogonalised at theta0",
        "model": "Y ~ N(gamma, sigma^2), target gamma, nuisance sigma",
        "runtime_s": round(time.time() - t0, 1),
    }
    (out_dir / "summary_nuisance.json").write_text(json.dumps(summary, indent=2))
    print(f"written {csv_path} ({len(rows)} rows, {summary['runtime_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
