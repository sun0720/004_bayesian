#!/usr/bin/env python3
"""sim.py — h001 自适应局部稳健贝叶斯估计量的 Monte Carlo。

复现命令：python3 sim.py

设计（与 sim/plan.md ③ 一致）：
  基线：Y = gamma + eps, eps ~ N(0,1)，gamma 是目标参数（k=1, q=0）。
  字典：5 个**光滑有界**函数——h1 = sin(1.5 eps)/N1（相关方向，E[eps h1]=c1>0），
        h2..h5 由 cos(k eps) 中心化并 Gram-Schmidt 正交归一（无关方向，E[eps h_j]=0）。
        有界字典使 b_j 与乘积 s*b_j 均被 |eps| 控制，满足冻结 A1 的 sub-Gaussian/MGF 细化。
  局部错设：真实密度为 phi(eps) exp(t h1(eps)) / Z，t = rho / sqrt(n)（沿 h1 的加性对数似然扰动）。
  负例（A2 破）：真实密度为 phi(eps) exp(t (h1 + g_perp)) / Z，g_perp = tanh(0.8 eps)/norm
        不在字典张成内；t 仍为 rho/sqrt(n)，故 A1 成立、只破 A2。
  估计量：naive（样本均值）· target（试样本选 S，主样本增广后验中心，随机 Ŝ）·
        fixed（S = S* 固定，P7/P9 的证明对象）· oracle（已知真实局部平移的偏差校正）。

只用 numpy / scipy / matplotlib；数据自生成，固定种子与独立流。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from pathlib import Path

import numpy as np
from numpy.random import SeedSequence, default_rng
from scipy.optimize import minimize
from scipy.special import logsumexp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── 字典：光滑有界基（满足 A1 的 sub-Gaussian/MGF 细化）─────────────────────
GX = np.linspace(-10.0, 10.0, 60001)
GW = np.exp(-0.5 * GX * GX) / math.sqrt(2.0 * math.pi)
GW = GW / np.trapezoid(GW, GX)
DX = GX[1] - GX[0]


def ip(f, g):
    """在 N(0,1) 下的内积（数值积分）。"""
    return float(np.sum(GW * f * g) * DX)


A_FREQ = 1.5
_h1_raw = np.sin(A_FREQ * GX)
N1 = math.sqrt(ip(_h1_raw, _h1_raw))
H1_GRID = _h1_raw / N1
C1 = ip(GX, H1_GRID)                      # E[eps h1]
JS_TRUE = 1.0 - C1 * C1                   # J_{S*}，S*={1}

# 无关方向：cos(k eps) 中心化 + Gram-Schmidt
BS = []
for _k in (1, 2, 3, 4):
    _f = np.cos(_k * GX)
    _g = _f - ip(np.ones_like(_f), _f) * np.ones_like(_f)
    for _b in BS:
        _g = _g - ip(_b, _g) * _b
    BS.append(_g / math.sqrt(ip(_g, _g)))

# 负例方向：tanh(0.8 eps)，不在 B 的张成内（奇函数、非正弦形）
_gp_raw = np.tanh(0.8 * GX)
GP_GRID = _gp_raw / math.sqrt(ip(_gp_raw, _gp_raw))

# 阈值边界臂的第六方向：sin(0.18 eps) 对 h1 正交化；敏感度 c6 落在 (0, lambda_n]
# 使固定 S*={1} 时被省略的方向有非零但低于阈值的敏感度——这正是 P4(b) 的有效区域。
A6_FREQ = 3.0
_h6_raw = np.sin(A6_FREQ * GX)
_h6 = _h6_raw - ip(_h6_raw, H1_GRID) * H1_GRID
H6_GRID = _h6 / math.sqrt(ip(_h6, _h6))
C6 = ip(GX, H6_GRID)                      # E[eps h6]
K_BOUNDARY = 6

# 近奇异单元：选中的两方向 h1 与 h7（c7 接近 1-|c1| 的上界），扰动沿 h8（与两者正交）。
A7_FREQ = 1.2
_h7_raw = np.sin(A7_FREQ * GX)
_h7 = _h7_raw - ip(_h7_raw, H1_GRID) * H1_GRID
H7_GRID = _h7 / math.sqrt(ip(_h7, _h7))
C7 = ip(GX, H7_GRID)
_h8_raw = np.sin(3.0 * GX)
_h8 = _h8_raw - ip(_h8_raw, H1_GRID) * H1_GRID - ip(_h8_raw, H7_GRID) * H7_GRID
H8_GRID = _h8 / math.sqrt(ip(_h8, _h8))
C8 = ip(GX, H8_GRID)
J_NS = 1.0 - C1 * C1 - C7 * C7
K_NEAR = 3

C_TRUE = np.array([C1, 0.0, 0.0, 0.0, 0.0])   # E[eps b_j]
S_STAR = np.array([0])                         # 真实相关方向 S*={1}
K = 5
N_GRID = [100, 300, 1000, 3000, 10000]
RHO_GRID = [0.0, 0.75, 1.5]       # 局部错设强度 a_1；0 = 关掉污染
R_DEFAULT = 1000
SEED_DEFAULT = 20261020
PRIOR_SD_DELTA = 0.1              # delta 的先验标准差（A4：正连续、两阶矩）
FALLBACKS = 0                     # L-BFGS-B 失败、退回样本均值的次数（可复现性记录）

def N_eff(n: int) -> int:
    m = max(10, min(int(round(n ** (2.0 / 3.0))), max(10, n // 2)))
    return n - m


def dict_matrix(e: np.ndarray) -> np.ndarray:
    """光滑有界字典在 eps 处的取值；eps 截到 [-10,10]（截断概率可忽略）。"""
    e = np.clip(np.asarray(e, dtype=float), -10.0, 10.0)
    cols = [np.sin(A_FREQ * e) / N1] + [np.interp(e, GX, b) for b in BS]
    return np.column_stack(cols)


_XQ, _WQ = np.polynomial.hermite.hermgauss(80)
_EQ = math.sqrt(2.0) * _XQ
_WQ = _WQ / math.sqrt(math.pi)


def k0(z: np.ndarray | float):
    z = np.clip(np.asarray(z, dtype=float), -30.0, 30.0)
    return 2.0 / (1.0 + np.exp(-2.0 * z))


def normal_expect_k0(delta: float, S: np.ndarray) -> float:
    """E_{eps~N(0,1)}[k0(delta * r_S(eps))]，r_S = sum_{j in S} c_j b_j。"""
    if len(S) == 0:
        return 1.0
    r = r_of(_EQ, S)
    return float((_WQ * k0(delta * r)).sum())


def r_of(e: np.ndarray, S: np.ndarray) -> np.ndarray:
    """r_S(eps) = sum_{j in S} c_j b_j(eps)，保留输入形状；只算 S 里的方向。"""
    e = np.asarray(e, dtype=float)
    out = np.zeros_like(e)
    for j in np.asarray(S).ravel():
        cj = float(C_TRUE[j])
        if cj == 0.0:
            continue
        if j == 0:
            out = out + cj * (np.sin(A_FREQ * e) / N1)
        else:
            out = out + cj * np.interp(e, GX, BS[j - 1])
    return out


def neg_loglik(params, y, S, N):
    gamma, delta = params
    e = y - gamma
    if len(S) == 0:
        return 0.5 * np.sum(e * e) + 0.5 * N * math.log(2.0 * math.pi)
    r = r_of(e, S)
    kz = k0(delta * r)
    if not np.isfinite(kz).all() or np.any(kz <= 0.0):
        return 1e12
    Ez = normal_expect_k0(delta, S)
    if not np.isfinite(Ez) or Ez <= 0.0:
        return 1e12
    ll = (-0.5 * np.sum(e * e) - 0.5 * N * math.log(2.0 * math.pi)
          + np.sum(np.log(kz)) - N * math.log(Ez)
          - 0.5 * (delta / PRIOR_SD_DELTA) ** 2)
    return -ll


def fit_augmented(y: np.ndarray, S: np.ndarray):
    """增广后验中心（增广 MLE）；失败时退回样本均值。"""
    global FALLBACKS
    N = len(y)
    x0 = np.array([float(y.mean()), 0.0])
    res = minimize(neg_loglik, x0, args=(y, S, N), method="L-BFGS-B",
                   bounds=[(-np.inf, np.inf), (-6.0, 6.0)],
                   options={"maxiter": 150, "ftol": 1e-10})
    if not res.success or not np.isfinite(res.fun):
        FALLBACKS += 1
        return float(y.mean()), 0.0
    return float(res.x[0]), float(res.x[1])


def target_se(y: np.ndarray, gamma_hat: float, S: np.ndarray) -> float:
    """理论 ⑩ 的 \\hat V：\\hat J_S = N^{-1} sum \\tilde ell^2，SE = sqrt(\\hat J_S^{-1}/N)。"""
    N = len(y)
    e = y - gamma_hat
    if len(S) == 0:
        return float(e.std(ddof=1) / math.sqrt(N))
    H = dict_matrix(e)
    tilde = e - H[:, S] @ C_TRUE[S]
    Js = float(np.mean(tilde * tilde))
    if Js <= 1e-10:
        return float(e.std(ddof=1) / math.sqrt(N))
    return math.sqrt(1.0 / (Js * N))


BS8 = list(BS)
for _k in (5, 6, 7):
    _f = np.cos(_k * GX)
    _g = _f - ip(np.ones_like(_f), _f) * np.ones_like(_f)
    for _b in BS8:
        _g = _g - ip(_b, _g) * _b
    BS8.append(_g / math.sqrt(ip(_g, _g)))


def dict_matrix8(e: np.ndarray) -> np.ndarray:
    e = np.clip(np.asarray(e, dtype=float), -10.0, 10.0)
    cols = [np.sin(A_FREQ * e) / N1] + [np.interp(e, GX, b) for b in BS8]
    return np.column_stack(cols)


BS12 = list(BS8)
for _k in (8, 9, 10, 11):
    _f = np.cos(_k * GX)
    _g = _f - ip(np.ones_like(_f), _f) * np.ones_like(_f)
    for _b in BS12:
        _g = _g - ip(_b, _g) * _b
    BS12.append(_g / math.sqrt(ip(_g, _g)))


def dict_matrixK(e: np.ndarray, k: int) -> np.ndarray:
    e = np.clip(np.asarray(e, dtype=float), -10.0, 10.0)
    cols = [np.sin(A_FREQ * e) / N1] + [np.interp(e, GX, b) for b in BS12[: k - 1]]
    return np.column_stack(cols)


def pilot_selectK(y: np.ndarray, n: int, k: int):
    """Growing-dictionary pilot: K_n directions, threshold (log K_n)^{1/4} n^{-1/6}."""
    m = int(round(n ** (2.0 / 3.0)))
    m = max(10, min(m, max(10, n // 2)))
    e = y[:m] - y[:m].mean()
    H = dict_matrixK(e, k)
    c_hat = (e[:, None] * H).mean(axis=0)
    J_hat = float(np.mean(e * e))
    kappa = np.abs(c_hat) / max(J_hat, 1e-10)
    lam = (math.log(k) ** 0.25) * (n ** (-1.0 / 6.0))
    S = np.where(kappa > lam)[0]
    return S, kappa, lam, m


def one_rep_K(n: int, rho: float, rng, k: int) -> dict:
    y = gen_data(n, rng, rho)
    SK, kap, lam, m = pilot_selectK(y, n, k)
    y_main = y[m:]
    N = len(y_main)
    naive = float(y_main.mean())
    naive_se = float(y_main.std(ddof=1) / math.sqrt(N))
    shift = true_mean(rho / math.sqrt(n))
    oracle = naive - shift
    S_fit = np.array([0]) if 0 in set(SK.tolist()) else np.array([], dtype=int)
    if len(S_fit) > 0:
        fixed, _d = fit_augmented(y_main, S_fit)
        fixed_se = target_se(y_main, fixed, S_fit)
    else:
        fixed, fixed_se = naive, naive_se
    return dict(fixed=fixed, fixed_se=fixed_se, naive=naive, naive_se=naive_se,
                oracle=oracle, oracle_se=naive_se, n=n, lam=lam, nS=len(SK),
                rel=int(0 in set(SK.tolist())))


def pilot_select8(y: np.ndarray, n: int):
    """试点筛选，字典有 8 个方向（K_n=8）：检验 sqrt(log K_n / m_n) 这一环节。"""
    m = int(round(n ** (2.0 / 3.0)))
    m = max(10, min(m, max(10, n // 2)))
    e = y[:m] - y[:m].mean()
    H = dict_matrix8(e)
    c_hat = (e[:, None] * H).mean(axis=0)
    J_hat = float(np.mean(e * e))
    kappa = np.abs(c_hat) / max(J_hat, 1e-10)
    lam = (math.log(8) ** 0.25) * (n ** (-1.0 / 6.0))
    S = np.where(kappa > lam)[0]
    return S, kappa, lam, m


def pilot_select(y: np.ndarray, n: int):
    """消失试样本：m=n^{2/3}；返回 S、\\hat kappa、阈值 lambda_n。"""
    m = int(round(n ** (2.0 / 3.0)))
    m = max(10, min(m, max(10, n // 2)))
    e = y[:m] - y[:m].mean()
    H = dict_matrix(e)
    c_hat = (e[:, None] * H).mean(axis=0)
    J_hat = float(np.mean(e * e))
    kappa = np.abs(c_hat) / max(J_hat, 1e-10)
    lam = (math.log(K) ** 0.25) * (n ** (-1.0 / 6.0))
    S = np.where(kappa > lam)[0]
    return S, kappa, lam, m


# ── DGP：phi(eps) exp(t h1) / Z（A2 负例再加 exp(t g_perp)）─────────────────
_SX = np.linspace(-9.0, 9.0, 120001)
_SLOG = -0.5 * _SX * _SX
_H1S = np.sin(A_FREQ * _SX) / N1
_GPS = np.interp(_SX, GX, GP_GRID)
_H6S = np.interp(_SX, GX, H6_GRID)
_H8S = np.interp(_SX, GX, H8_GRID)
_SAMPLERS: dict = {}
_TRUE_MU: dict = {}


def _sampler(t: float, kind: str):
    key = (round(float(t), 12), kind)
    if key not in _SAMPLERS:
        if kind == "boundary":
            logw = _SLOG + t * _H6S
        elif kind == "nearsing":
            logw = _SLOG + t * _H8S
        else:
            logw = _SLOG + t * _H1S + (t * _GPS if kind == "a2" else 0.0)
        w = np.exp(logw - logw.max())
        cdf = np.cumsum(w)
        cdf = cdf / cdf[-1]
        _SAMPLERS[key] = cdf
        mean = float(np.trapezoid(w * _SX, _SX) / np.trapezoid(w, _SX))
        _TRUE_MU[key] = mean
    return _SAMPLERS[key], _TRUE_MU[key]


def gen_data(n: int, rng, rho: float, kind: str = "local") -> np.ndarray:
    t = rho / math.sqrt(n)
    cdf, _mu = _sampler(t, kind)
    u = rng.random(n)
    return np.interp(u, cdf, _SX)


def true_mean(t: float, kind: str = "local") -> float:
    return _sampler(t, kind)[1]


def one_rep(n: int, rho: float, rng, kind: str = "local") -> dict:
    y = gen_data(n, rng, rho, kind)
    S, kappa, lam, m = pilot_select(y, n)
    y_main = y[m:]
    N = len(y_main)

    naive = float(y_main.mean())
    naive_se = float(y_main.std(ddof=1) / math.sqrt(N))

    # oracle：已知真实局部平移（数值真值，非一阶近似）
    shift = true_mean(rho / math.sqrt(n), kind)
    oracle = naive - shift
    oracle_se = naive_se

    if len(S) > 0 and (1.0 - float(np.sum(C_TRUE[S] ** 2))) > 1e-8:
        target, _delta = fit_augmented(y_main, S)
        tse = target_se(y_main, target, S)
    else:
        target, tse = naive, naive_se

    # 固定-S 臂：S = S*。P7/P9 只对固定 S 成立，这一条才是被证明对象
    # （已知 S*，不做选择；与 target 共用同一批抽签，成对可比）。
    fixed, _fdelta = fit_augmented(y_main, S_STAR)
    fixed_se = target_se(y_main, fixed, S_STAR)

    # D4：总体 kappa（真值）；关联两方向之外全为 0
    omitted_true = max([float(C_TRUE[j]) for j in range(K) if j not in set(S.tolist())] + [0.0])
    omitted_hat = max([float(kappa[j]) for j in range(K) if j not in set(S.tolist())] + [0.0])
    return dict(naive=naive, naive_se=naive_se, target=target, target_se=tse,
                fixed=fixed, fixed_se=fixed_se,
                oracle=oracle, oracle_se=oracle_se, S=S, nS=len(S),
                rel_selected=int(0 in set(S.tolist())),
                omitted_kappa=omitted_hat, omitted_kappa_true=omitted_true,
                lam=lam, n=n, rho=rho, kind=kind)


def one_rep_boundary(n: int, a6: float, rng, k_dict: int = K_BOUNDARY) -> dict:
    """阈值边界臂：扰动沿第六方向 h6，固定 S*={1} 把它省略。

    理论：scaled bias = a6 * J_S^{-1} c6（精确值）；可计算界 = a6 * ||J_S^{-1}J||op * lambda_n，
    在标量设计里两者与 M||J_S^{-1}J||op lambda_n 同形。筛选条件 c6 <= lambda_n 在每个格点成立。
    """
    y = gen_data(n, rng, a6, kind="boundary")
    m = int(round(n ** (2.0 / 3.0)))
    m = max(10, min(m, max(10, n // 2)))
    y_main = y[m:]
    N = len(y_main)
    naive = float(y_main.mean())
    naive_se = float(y_main.std(ddof=1) / math.sqrt(N))
    shift = true_mean(a6 / math.sqrt(n), kind="boundary")
    oracle = naive - shift
    fixed, _d = fit_augmented(y_main, S_STAR)
    fixed_se = target_se(y_main, fixed, S_STAR)
    lam = (math.log(k_dict) ** 0.25) * (n ** (-1.0 / 6.0))
    return dict(fixed=fixed, fixed_se=fixed_se, naive=naive, naive_se=naive_se,
                oracle=oracle, oracle_se=naive_se, n=n, a6=a6, lam=lam)


def _r_ns(e: np.ndarray) -> np.ndarray:
    e = np.asarray(e, dtype=float)
    return C1 * np.interp(e, GX, H1_GRID) + C7 * np.interp(e, GX, H7_GRID)


def _negll_ns(params, y, N):
    gamma, delta = params
    e = y - gamma
    r = _r_ns(e)
    kz = k0(delta * r)
    if not np.isfinite(kz).all() or np.any(kz <= 0.0):
        return 1e12
    Ez = float((_WQ * k0(delta * _r_ns(_EQ))).sum())
    if not np.isfinite(Ez) or Ez <= 0.0:
        return 1e12
    return -( -0.5 * np.sum(e * e) - 0.5 * N * math.log(2.0 * math.pi)
              + np.sum(np.log(kz)) - N * math.log(Ez)
              - 0.5 * (delta / PRIOR_SD_DELTA) ** 2 )


def one_rep_nearsing(n: int, a8: float, rng, k_dict: int = K_NEAR) -> dict:
    """Near-singular cell: S={1,7} with J_S small; perturbation along h8 (orthogonal to S)."""
    global FALLBACKS
    y = gen_data(n, rng, a8, kind="nearsing")
    m = int(round(n ** (2.0 / 3.0)))
    m = max(10, min(m, max(10, n // 2)))
    y_main = y[m:]
    N = len(y_main)
    naive = float(y_main.mean())
    naive_se = float(y_main.std(ddof=1) / math.sqrt(N))
    shift = true_mean(a8 / math.sqrt(n), kind="nearsing")
    oracle = naive - shift
    res = minimize(_negll_ns, np.array([naive, 0.0]), args=(y_main, N), method="L-BFGS-B",
                   bounds=[(-np.inf, np.inf), (-6.0, 6.0)],
                   options={"maxiter": 150, "ftol": 1e-10})
    if not res.success or not np.isfinite(res.fun):
        FALLBACKS += 1
        fixed = naive
    else:
        fixed = float(res.x[0])
    tilde = (y_main - fixed) - _r_ns(y_main - fixed)
    Jhat = float(np.mean(tilde * tilde))
    fixed_se = math.sqrt(1.0 / (max(Jhat, 1e-12) * N))
    lam = (math.log(k_dict) ** 0.25) * (n ** (-1.0 / 6.0))
    return dict(fixed=fixed, fixed_se=fixed_se, naive=naive, naive_se=naive_se,
                oracle=oracle, oracle_se=naive_se, n=n, a6=a8, lam=lam)


def one_rep_k8(n: int, rho: float, rng) -> dict:
    """K_n=8 arm: the pilot selects on an eight-direction dictionary, then the
    augmented fit runs on the selected relevant direction (the extra directions
    have zero sensitivity by construction)."""
    y = gen_data(n, rng, rho)
    S8, kap, lam, m = pilot_select8(y, n)
    y_main = y[m:]
    N = len(y_main)
    naive = float(y_main.mean())
    naive_se = float(y_main.std(ddof=1) / math.sqrt(N))
    shift = true_mean(rho / math.sqrt(n))
    oracle = naive - shift
    S_fit = np.array([0]) if 0 in set(S8.tolist()) else np.array([], dtype=int)
    if len(S_fit) > 0:
        fixed, _d = fit_augmented(y_main, S_fit)
        fixed_se = target_se(y_main, fixed, S_fit)
    else:
        fixed, fixed_se = naive, naive_se
    return dict(fixed=fixed, fixed_se=fixed_se, naive=naive, naive_se=naive_se,
                oracle=oracle, oracle_se=naive_se, n=n, lam=lam, nS=len(S8),
                rel=int(0 in set(S8.tolist())))


def summarize(vals: np.ndarray, ses: np.ndarray, R: int) -> dict:
    bias = float(vals.mean())
    bias_se = float(vals.std(ddof=1) / math.sqrt(R))
    rmse = float(math.sqrt(bias ** 2 + vals.var(ddof=1)))
    cover = float(np.mean(np.abs(vals) < 1.96 * ses))
    cover_se = float(math.sqrt(cover * (1.0 - cover) / R))
    z = vals / np.maximum(ses, 1e-12)
    sd = max(float(z.std(ddof=1)), 1e-12)
    zz = (z - z.mean()) / sd
    return dict(bias=bias, bias_se=bias_se, rmse=rmse, coverage=cover,
                coverage_se=cover_se,
                coverage_upper95=(3.0 / R if cover == 0.0 else float("nan")),
                z_mean=float(z.mean()), z_sd=sd,
                z_skew=float(np.mean(zz ** 3)), z_kurt=float(np.mean(zz ** 4)))


def slope(xs, ys):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    ok = np.isfinite(xs) & np.isfinite(ys) & (ys > 0)
    if ok.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(xs[ok]), np.log(ys[ok]), 1)[0])


def posterior_moments(y: np.ndarray, S: np.ndarray, n_gamma: int = 61, n_delta: int = 81):
    """数值后验：在 (gamma, delta) 网格上直接积分，检查 BvM 的后验形状（D2）。

    返回后验均值、后验标准差、偏度，并与**总体**渐近基准 sqrt(J_S^{-1}/N) 对照
    （同时报与经验 se 的比值，便于分辨「估计误差」与「后验收缩」）。
    """
    N = len(y)
    gh, dh = fit_augmented(y, S)
    se = target_se(y, gh, S)
    J_S = 1.0 - float(np.sum(C_TRUE[S] ** 2))     # k=1、J=1、Omega=I 时的有效信息
    se_pop = math.sqrt(1.0 / (J_S * N)) if J_S > 0 else se
    ggrid = gh + np.linspace(-4.0, 4.0, n_gamma) * se
    dgrid = np.linspace(-0.6, 0.6, n_delta)
    logpost = np.empty((n_delta, n_gamma))
    for i, d in enumerate(dgrid):
        e = y[None, :] - ggrid[:, None]                       # (n_gamma, N)
        r = r_of(e, S)                                        # (n_gamma, N)
        kz = k0(d * r)
        logq = (-0.5 * np.sum(e * e, axis=1) - 0.5 * N * math.log(2.0 * math.pi)
                + np.sum(np.log(kz), axis=1) - N * math.log(normal_expect_k0(d, S))
                - 0.5 * (d / PRIOR_SD_DELTA) ** 2)
        logpost[i, :] = logq
    # 对 delta 积分（均匀网格权重）→ gamma 的边际
    lp = logsumexp(logpost, axis=0)
    w = np.exp(lp - lp.max())
    w = w / w.sum()
    mean = float(np.sum(w * ggrid))
    sd = float(math.sqrt(np.sum(w * (ggrid - mean) ** 2)))
    skew = float(np.sum(w * ((ggrid - mean) / sd) ** 3))
    return dict(post_mean=mean, post_sd=sd, post_skew=skew,
                mle=gh, se_emp=se, se_pop=se_pop,
                ratio_sd=sd / se_pop, ratio_sd_emp=sd / se,
                diff_mean=mean - gh)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="小网格自测（默认写临时目录，不碰正式结果）")
    ap.add_argument("--R", type=int, default=R_DEFAULT)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    global FALLBACKS
    FALLBACKS = 0

    n_grid = [100, 1000] if args.quick else N_GRID
    rho_grid = [0.0, 1.5] if args.quick else RHO_GRID
    R = 20 if args.quick else args.R
    seed = args.seed

    if args.out is not None:
        out = Path(args.out)
    elif args.quick:
        out = Path(tempfile.mkdtemp(prefix="sim-quick-"))
    else:
        out = Path("results")
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    z_diag: dict = {}
    sel_diag: dict = {}
    post_diag: dict = {}

    for rho in rho_grid:
        streams = SeedSequence(seed).spawn(R)
        for n in n_grid:
            reps = [one_rep(n, rho, default_rng(s)) for s in streams]
            for name in ("naive", "target", "fixed", "oracle"):
                vals = np.array([r[name] for r in reps], dtype=float)
                ses = np.array([r[f"{name}_se"] for r in reps], dtype=float)
                st = summarize(vals, ses, R)
                rows.append(dict(n=n, rho=rho, estimator=name,
                                 bias=st["bias"], bias_se=st["bias_se"],
                                 rmse=st["rmse"], coverage=st["coverage"],
                                 coverage_se=st["coverage_se"],
                                 scaled_bias=math.sqrt(n) * st["bias"]))
            nS = np.array([r["nS"] for r in reps], dtype=float)
            rel = np.array([r["rel_selected"] for r in reps], dtype=float)
            om_t = np.array([r["omitted_kappa_true"] for r in reps], dtype=float)
            sel_diag[f"{rho}_{n}"] = dict(mean_nS=float(nS.mean()),
                                           select_freq=float(rel.mean()),
                                           omitted_kappa_true_mean=float(om_t.mean()),
                                           omitted_kappa_true_violation=float(np.mean(om_t > reps[0]["lam"])),
                                           lam=float(reps[0]["lam"]))
            zt = np.array([r["fixed"] for r in reps], dtype=float)
            zts = np.array([r["fixed_se"] for r in reps], dtype=float)
            zf = zt / np.maximum(zts, 1e-12)
            zn = np.array([r["naive"] / max(r["naive_se"], 1e-12) for r in reps])
            sdf = max(float(zf.std(ddof=1)), 1e-12)
            z_diag[f"{rho}_{n}"] = dict(
                fixed_mean=float(zf.mean()), fixed_sd=sdf,
                fixed_skew=float(np.mean(((zf - zf.mean()) / sdf) ** 3)),
                fixed_kurt=float(np.mean(((zf - zf.mean()) / sdf) ** 4)),
                naive_mean=float(zn.mean()), naive_sd=float(zn.std(ddof=1)))

    # D2 的后验形状检查：n=1000 取 60 个复制（完整后验数值积分，不跑 MCMC）
    post_n = 1000
    post_reps = 20 if args.quick else 60

    def _post_rep(ss):
        yy = gen_data(post_n, default_rng(ss), rho_grid[-1])
        mm = max(10, min(int(round(post_n ** (2.0 / 3.0))), post_n // 2))
        return posterior_moments(yy[mm:], S_STAR)          # 与估计量同一主样本

    post_streams = SeedSequence(seed).spawn(post_reps)
    pv = [_post_rep(s) for s in post_streams]
    post_diag[f"{rho_grid[-1]}_{post_n}"] = dict(
        R=post_reps,
        ratio_sd_mean=float(np.mean([p["ratio_sd"] for p in pv])),
        ratio_sd_se=float(np.std([p["ratio_sd"] for p in pv], ddof=1) / math.sqrt(post_reps)),
        ratio_sd_emp_mean=float(np.mean([p["ratio_sd_emp"] for p in pv])),
        diff_mean_mean=float(np.mean([p["diff_mean"] for p in pv])),
        post_skew_mean=float(np.mean([p["post_skew"] for p in pv])))

    rho_last = rho_grid[-1]
    rates = {}
    for name in ("naive", "target", "fixed", "oracle"):
        sub = [r for r in rows if r["rho"] == rho_last and r["estimator"] == name]
        sb = [abs(r["scaled_bias"]) for r in sub]
        rates[name] = dict(
            rmse_slope=slope([r["n"] for r in sub], [r["rmse"] for r in sub]),
            scaled_bias_slope=slope([r["n"] for r in sub], sb),
            scaled_bias_slope_drop_last=slope([r["n"] for r in sub[:-1]], sb[:-1]))

    # 负例：A2 破（g_perp 不在字典张成内），t 仍为 rho/sqrt(n)（A1 成立）
    neg = {}
    for n in (300, 1000, 10000):
        neg_streams = SeedSequence(seed + 1).spawn(R)
        reps = [one_rep(n, rho_last, default_rng(s), kind="a2") for s in neg_streams]
        for name in ("fixed", "naive", "oracle"):
            vals = np.array([r[name] for r in reps], dtype=float)
            ses = np.array([r[f"{name}_se"] for r in reps], dtype=float)
            neg[f"{n}_{name}"] = summarize(vals, ses, R)

    # 阈值边界臂：被省略方向有非零但低于阈值的敏感度（P4(b) 的有效区域）
    boundary = {}
    b_rows = []
    a6 = rho_last
    b_streams = SeedSequence(seed + 2).spawn(R)
    for n in (n_grid if not args.quick else [100, 1000]):
        reps = [one_rep_boundary(n, a6, default_rng(s)) for s in b_streams]
        lam = reps[0]["lam"]
        exact = a6 * abs(C6) / JS_TRUE
        comp = a6 * (1.0 / JS_TRUE) * lam
        for name in ("fixed", "naive", "oracle"):
            vals = np.array([r[name] for r in reps], dtype=float)
            ses = np.array([r[f"{name}_se"] for r in reps], dtype=float)
            st = summarize(vals, ses, R)
            b_rows.append(dict(n=n, arm="boundary", estimator=name,
                               bias=st["bias"], bias_se=st["bias_se"], rmse=st["rmse"],
                               coverage=st["coverage"], coverage_se=st["coverage_se"],
                               scaled_bias=math.sqrt(N_eff(n)) * st["bias"],
                               scaled_bias_se=math.sqrt(N_eff(n)) * st["bias_se"],
                               lam=lam, exact_scaled_bound=exact,
                               computable_scaled_bound=comp))
        boundary[f"{n}"] = dict(
            a6=a6, kappa6=abs(C6), lam=lam, exact_scaled_bound=exact,
            computable_scaled_bound=comp, screening_holds=bool(abs(C6) <= lam),
            fixed_bias=b_rows[-3]["bias"], fixed_bias_se=b_rows[-3]["bias_se"],
            fixed_scaled_bias=b_rows[-3]["scaled_bias"],
            fixed_coverage=b_rows[-3]["coverage"], fixed_coverage_se=b_rows[-3]["coverage_se"],
            naive_coverage=b_rows[-2]["coverage"],
            oracle_coverage=b_rows[-1]["coverage"])

    # 稳健性单元：增长 K_n（同一设计，阈值用 K=8）与近奇异（S={1,7}，J_S small）
    robust = {}
    def _add_arm(rows_out, reps, arm, exact, comp, extra):
        for name in ("fixed", "naive", "oracle"):
            vals = np.array([r[name] for r in reps], dtype=float)
            ses = np.array([r[f"{name}_se"] for r in reps], dtype=float)
            st = summarize(vals, ses, R)
            rows_out.append(dict(n=reps[0]["n"], arm=arm, estimator=name,
                                 bias=st["bias"], bias_se=st["bias_se"], rmse=st["rmse"],
                                 coverage=st["coverage"], coverage_se=st["coverage_se"],
                                 scaled_bias=math.sqrt(N_eff(reps[0]["n"])) * st["bias"],
                                 scaled_bias_se=math.sqrt(N_eff(reps[0]["n"])) * st["bias_se"],
                                 lam=reps[0]["lam"], exact_scaled_bound=exact,
                                 computable_scaled_bound=comp))
        fx = [r for r in rows_out if r["arm"] == arm and r["estimator"] == "fixed"][-1]
        robust[f"{arm}_{reps[0]['n']}"] = dict(
            lam=reps[0]["lam"], exact_scaled_bound=exact,
            computable_scaled_bound=comp, screening_holds=bool(abs(extra) <= reps[0]["lam"]),
            fixed_scaled_bias=fx["scaled_bias"], fixed_scaled_bias_se=fx["scaled_bias_se"],
            fixed_coverage=fx["coverage"], fixed_coverage_se=fx["coverage_se"])

    for n in (n_grid if not args.quick else [100, 1000]):
        reps = [one_rep_boundary(n, a6, default_rng(s), k_dict=8) for s in b_streams]
        lam = reps[0]["lam"]
        _add_arm(b_rows, reps, "boundary_k8", a6 * abs(C6) / JS_TRUE,
                 a6 * (1.0 / JS_TRUE) * lam, abs(C6))
        reps = [one_rep_nearsing(n, a6, default_rng(s)) for s in b_streams]
        lam = reps[0]["lam"]
        _add_arm(b_rows, reps, "nearsing", a6 * abs(C8) / J_NS,
                 a6 * (1.0 / J_NS) * lam, abs(C8))

    # 扩大字典臂（K_n=8）：八方向字典上做试点筛选并拟合（额外方向敏感度为零）
    k8 = {}
    for n in (n_grid if not args.quick else [100, 1000]):
        streams8 = SeedSequence(seed + 3).spawn(R)
        reps8 = [one_rep_k8(n, rho_last, default_rng(s)) for s in streams8]
        vals = np.array([r["fixed"] for r in reps8]); ses = np.array([r["fixed_se"] for r in reps8])
        st = summarize(vals, ses, R)
        nS = float(np.mean([r["nS"] for r in reps8])); rel = float(np.mean([r["rel"] for r in reps8]))
        k8[f"{n}"] = dict(lam=float(reps8[0]["lam"]), mean_nS=nS, select_freq=rel,
                          bias=st["bias"], bias_se=st["bias_se"], rmse=st["rmse"],
                          coverage=st["coverage"], coverage_se=st["coverage_se"],
                          scaled_bias_N=math.sqrt(N_eff(n)) * st["bias"], R=R)
    # 随 n 增长的字典臂：K_n = max(5, round(n^{1/4})), capped at 12
    growing = {}
    for n in (n_grid if not args.quick else [100, 1000]):
        kk = max(5, min(12, int(round(n ** 0.25))))
        streamsK = SeedSequence(seed + 4).spawn(R)
        repsK = [one_rep_K(n, rho_last, default_rng(s), kk) for s in streamsK]
        vals = np.array([r["fixed"] for r in repsK]); ses = np.array([r["fixed_se"] for r in repsK])
        st = summarize(vals, ses, R)
        growing[f"{n}"] = dict(k=kk, lam=float(repsK[0]["lam"]),
                               mean_nS=float(np.mean([r["nS"] for r in repsK])),
                               select_freq=float(np.mean([r["rel"] for r in repsK])),
                               bias=st["bias"], bias_se=st["bias_se"], coverage=st["coverage"],
                               coverage_se=st["coverage_se"], R=R)
    csv_path = out / "results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = dict(rho_last=rho_last, rates=rates, selection=sel_diag, z=z_diag,
                   posterior=post_diag, negative=neg, boundary=boundary,
                   boundary_robustness=robust, k8_pilot=k8, growing_k=growing,
                   meta=dict(R=R, seed=seed, n_grid=n_grid, rho_grid=rho_grid,
                             a_freq=A_FREQ, c1=C1, js=JS_TRUE,
                             s_star=[int(j) + 1 for j in S_STAR],
                             a6_freq=A6_FREQ, c6=abs(C6), K_boundary=K_BOUNDARY,
                             fallback_count=FALLBACKS, c7=abs(C7), c8=abs(C8), js_nearsing=J_NS,
                             crn_note="同一 rho 下所有 n 共用同一组 seed 流（共同随机数）；"
                                      "复制间独立，跨 n 不独立"))
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    b_csv = out / "results_boundary.csv"
    with b_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(b_rows[0].keys()))
        w.writeheader()
        w.writerows(b_rows)

    sub = {name: [r for r in rows if r["rho"] == rho_last and r["estimator"] == name]
           for name in ("naive", "target", "fixed", "oracle")}
    rho_zero = 0.0 if 0.0 in rho_grid else rho_grid[0]
    sub0 = {name: [r for r in rows if r["rho"] == rho_zero and r["estimator"] == name]
            for name in ("naive", "target", "fixed", "oracle")}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name in ("naive", "target", "fixed", "oracle"):
        ns = [r["n"] for r in sub[name]]
        rms = [r["rmse"] for r in sub[name]]
        cov = [r["coverage"] for r in sub[name]]
        cse = [r["coverage_se"] for r in sub[name]]
        axes[0].loglog(ns, rms, "-o", label=name)
        axes[1].errorbar(ns, cov, yerr=1.96 * np.array(cse), fmt="-o", capsize=3,
                         label=f"{name}, rho={rho_last:g}")
        ns0 = [r["n"] for r in sub0[name]]
        cov0 = [r["coverage"] for r in sub0[name]]
        cse0 = [r["coverage_se"] for r in sub0[name]]
        axes[1].errorbar(ns0, cov0, yerr=1.96 * np.array(cse0), fmt="--s", capsize=3,
                         alpha=0.55, label=f"{name}, rho={rho_zero:g}")
    axes[0].set_xlabel("n"); axes[0].set_ylabel("RMSE"); axes[0].legend()
    axes[1].axhline(0.95, color="grey", ls="--", lw=1)
    axes[1].set_xscale("log")
    axes[1].set_ylim(top=1.01)
    axes[1].set_xlabel("n"); axes[1].set_ylabel("Coverage ± 1.96·MC SE")
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / f"rate_coverage_rho{rho_last:g}.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ns = sorted(set(r["n"] for r in rows))
    mean_ns = [sel_diag[f"{rho_last}_{n}"]["mean_nS"] for n in ns]
    freq = [sel_diag[f"{rho_last}_{n}"]["select_freq"] for n in ns]
    ax.plot(ns, mean_ns, "-o", label="mean |S|")
    ax.set_xscale("log"); ax.set_xlabel("n"); ax.set_ylabel("mean |S|"); ax.legend(loc="upper left")
    ax2 = ax.twinx()
    ax2.plot(ns, freq, "-s", color="tab:orange", label="P(relevant selected)")
    ax2.set_ylabel("P(relevant selected)")
    ax2.set_ylim(max(0.0, min(freq) - 0.04), 1.005)
    ax2.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / f"selection_rho{rho_last:g}.png", dpi=120)
    plt.close(fig)

    print(f"R={R} seed={seed} a_freq={A_FREQ} c1={C1:.4f} J_S={JS_TRUE:.4f} fallbacks={FALLBACKS}")
    for r in rows:
        print(f"rho={r['rho']:<5g} n={r['n']:<6d} {r['estimator']:6s} "
              f"bias={r['bias']:+.4f}±{r['bias_se']:.4f} rmse={r['rmse']:.4f} "
              f"cov={r['coverage']:.3f}±{r['coverage_se']:.3f} "
              f"scaled_bias={r['scaled_bias']:+.3f}")
    print("rates:", json.dumps(rates))
    print("selection:", json.dumps({k: sel_diag[k] for k in sel_diag if k.startswith(str(rho_last))}))
    print("z:", json.dumps({k: z_diag[k] for k in z_diag if k.startswith(str(rho_last))}))
    print("posterior:", json.dumps(post_diag))
    print("negative(A2):", json.dumps(neg))
    print("boundary:", json.dumps(boundary))
    print("robustness:", json.dumps(robust))
    print("k8_pilot:", json.dumps(k8))
    print("growing_k:", json.dumps(growing))
    print(f"表：{csv_path}")
    print(f"图：{out / f'rate_coverage_rho{rho_last:g}.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
