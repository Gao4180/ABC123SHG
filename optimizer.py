# -*- coding: utf-8 -*-
"""
组合优化模块：马科维茨均值-方差优化（基于 scipy.optimize）。
约定：mu 为年化收益向量，cov 为年化协方差矩阵，rf 为无风险利率。
另含：预期收益假设覆盖 + 简化版 Black-Litterman 观点融合。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252


def estimate_params(prices: pd.DataFrame, shrinkage: bool = True):
    """
    由日收益估计年化收益向量 mu 与协方差矩阵 cov。
    shrinkage=True 时用 Ledoit-Wolf 收缩估计协方差：
    样本协方差在资产多/数据短时噪声大，收缩后优化结果更稳、更少极端权重。
    """
    rets = prices.pct_change().dropna()
    mu = rets.mean() * TRADING_DAYS
    if shrinkage:
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf().fit(rets.values)
            cov = pd.DataFrame(lw.covariance_ * TRADING_DAYS,
                               index=rets.columns, columns=rets.columns)
        except Exception:                    # sklearn 不可用时退回样本协方差
            cov = rets.cov() * TRADING_DAYS
    else:
        cov = rets.cov() * TRADING_DAYS
    return mu.values, cov.values, rets


def apply_return_overrides(mu, tickers, overrides: dict):
    """
    用前瞻性收益假设覆盖历史均值（对标机构 CMA 资本市场假设）。
    overrides: {ticker: 年化收益}，只覆盖非空项。
    """
    mu = np.asarray(mu, dtype=float).copy()
    for i, t in enumerate(tickers):
        v = overrides.get(t)
        if v is not None and np.isfinite(v):
            mu[i] = v
    return mu


def black_litterman(mu, cov, tickers, view_down: set[str], rf: float = 0.02,
                    tau: float = 0.05, view_shrink: float = 0.5):
    """
    简化版 Black-Litterman：以历史 mu 为先验，择时信号为绝对观点。
    view_down：跌破均线/动量转负的资产集合，观点 = 其预期收益向 rf 收缩
    view_shrink 比例（机构做法：观点强度 = 信心水平的函数）。
    返回融合观点后的后验收益向量。
    """
    mu = np.asarray(mu, dtype=float)
    idx = [i for i, t in enumerate(tickers) if t in view_down]
    if not idx:
        return mu
    n = len(mu)
    k = len(idx)
    P = np.zeros((k, n))
    for r, i in enumerate(idx):
        P[r, i] = 1.0
    Q = np.array([rf + (mu[i] - rf) * (1 - view_shrink) for i in idx])
    ts = tau * cov
    # 观点不确定性：与先验同量级（信心中等）
    omega = np.diag(np.diag(P @ ts @ P.T))
    omega[omega <= 1e-12] = 1e-6
    A = np.linalg.inv(ts)
    W = np.linalg.inv(omega)
    posterior = np.linalg.inv(A + P.T @ W @ P) @ (A @ mu + P.T @ W @ Q)
    return posterior


def _vol(w, cov):
    return float(np.sqrt(w @ cov @ w))


def _solve(obj, n, constraints, bounds, x0=None):
    """统一的 SLSQP 求解入口。"""
    if x0 is None:
        x0 = np.full(n, 1.0 / n)
    res = minimize(obj, x0, method="SLSQP", bounds=bounds,
                   constraints=constraints, options={"maxiter": 500})
    if not res.success:
        # 优化失败时退回等权，保证页面不崩溃
        return x0
    w = np.clip(res.x, 0, None)
    s = w.sum()
    return w / s if s > 0 else x0


def min_variance(mu, cov, cap=0.4):
    """保守档：全局最小方差组合。"""
    n = len(mu)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0.0, cap)] * n
    return _solve(lambda w: w @ cov @ w, n, cons, bounds)


def max_sharpe(mu, cov, rf=0.02, cap=0.4, extra_constraints=None):
    """平衡档 / 方案D：最大夏普组合，可附加自定义约束。"""
    n = len(mu)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if extra_constraints:
        cons += extra_constraints
    bounds = [(0.0, cap)] * n

    def neg_sharpe(w):
        vol = _vol(w, cov)
        if vol <= 1e-10:
            return 1e6
        return -(w @ mu - rf) / vol

    return _solve(neg_sharpe, n, cons, bounds)


def max_return_capped_vol(mu, cov, vol_limit=0.25, cap=0.4):
    """激进档：波动率不超过 vol_limit 的前提下最大化预期收益。"""
    n = len(mu)
    cons = [
        {"type": "eq", "fun": lambda w: w.sum() - 1},
        {"type": "ineq", "fun": lambda w: vol_limit ** 2 - w @ cov @ w},
    ]
    bounds = [(0.0, cap)] * n
    return _solve(lambda w: -(w @ mu), n, cons, bounds)


def risk_parity(cov, cap=1.0):
    """风险平价：使各资产对组合总波动的风险贡献尽量相等（全天候思路）。"""
    n = cov.shape[0]
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(1e-6, cap)] * n

    def obj(w):
        vol = _vol(w, cov)
        if vol <= 1e-10:
            return 1e6
        rc = w * (cov @ w) / vol          # 各资产风险贡献
        target = vol / n                   # 目标：每份贡献 vol/n
        return float(((rc - target) ** 2).sum())

    return _solve(obj, n, cons, bounds)


def hrp(cov, cap=1.0):
    """
    层级风险平价（Hierarchical Risk Parity，López de Prado 2016）。
    与马科维茨不同：不需要预期收益、不做协方差矩阵求逆，
    对参数估计误差不敏感，样本外表现通常更稳。三步：
    1) 相关矩阵 → 距离矩阵 → 层次聚类（树状图）
    2) 准对角化：按聚类叶序重排资产，相似的资产相邻
    3) 递归二分：簇间按"簇方差的反比"分配权重，簇内继续二分
    """
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    if n == 1:
        return np.array([1.0])
    vols = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    corr = np.clip(cov / np.outer(vols, vols), -1.0, 1.0)
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(dist, 0.0)
    order = list(leaves_list(linkage(squareform(dist, checks=False),
                                     method="single")))

    def cluster_var(idx):
        """簇内用逆方差权重求簇方差。"""
        sub = cov[np.ix_(idx, idx)]
        ivp = 1.0 / np.clip(np.diag(sub), 1e-12, None)
        ivp = ivp / ivp.sum()
        return float(ivp @ sub @ ivp)

    w = np.ones(n)
    clusters = [order]
    while clusters:
        cluster = clusters.pop(0)
        if len(cluster) < 2:
            continue
        mid = len(cluster) // 2
        left, right = cluster[:mid], cluster[mid:]
        var_l, var_r = cluster_var(left), cluster_var(right)
        alpha = 1.0 - var_l / max(var_l + var_r, 1e-18)
        w[left] *= alpha
        w[right] *= 1.0 - alpha
        clusters.extend([left, right])
    w = w / w.sum()
    # 迭代"灌水式"施加上限：超上限的资产钉在 cap，余额按比例分配给未超限资产
    for _ in range(20):
        over = w > cap
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        under = ~over
        if not under.any() or w[under].sum() <= 0:
            break
        w[under] += excess * w[under] / w[under].sum()
    s = w.sum()
    return w / s if s > 0 else np.full(n, 1.0 / n)


def optimize_plan_A(mu, cov, defensive_idx, vol_limit=0.05, cap=0.4):
    """
    方案A「稳健保本型」：在波动 ≤ vol_limit 的约束下，最大化债券+现金类资产占比。
    defensive_idx：债券/现金类资产在池中的下标列表。
    """
    n = len(mu)
    mask = np.zeros(n)
    mask[defensive_idx] = 1.0
    cons = [
        {"type": "eq", "fun": lambda w: w.sum() - 1},
        {"type": "ineq", "fun": lambda w: vol_limit ** 2 - w @ cov @ w},
    ]
    bounds = [(0.0, cap)] * n

    def obj(w):
        # 最大化防御资产占比；轻微惩罚波动，使解更稳
        return -(w @ mask) + 0.01 * (w @ cov @ w)

    w = _solve(obj, n, cons, bounds)
    # 若 5% 波动约束不可行（极少见），退化为全局最小方差
    if _vol(w, cov) > vol_limit * 1.05:
        w = min_variance(mu, cov, cap=cap)
    return w


def optimize_plan_B(mu, cov, eq_idx, bond_idx, rf=0.02, cap=0.4):
    """
    方案B「经典60/40」：股票合计 60%、债券合计 40%，其余资产置 0，
    在该约束下求最大夏普。
    """
    n = len(mu)
    eq_mask = np.zeros(n); eq_mask[eq_idx] = 1.0
    bd_mask = np.zeros(n); bd_mask[bond_idx] = 1.0
    cons = [
        {"type": "eq", "fun": lambda w: w.sum() - 1},
        {"type": "eq", "fun": lambda w: w @ eq_mask - 0.6},
        {"type": "eq", "fun": lambda w: w @ bd_mask - 0.4},
    ]
    bounds = []
    for i in range(n):
        if i in eq_idx or i in bond_idx:
            bounds.append((0.0, cap))
        else:
            bounds.append((0.0, 0.0))   # 非股非债资产固定为 0

    def neg_sharpe(w):
        vol = _vol(w, cov)
        if vol <= 1e-10:
            return 1e6
        return -(w @ mu - rf) / vol

    x0 = np.zeros(n)
    x0[eq_idx] = 0.6 / len(eq_idx)
    x0[bond_idx] = 0.4 / len(bond_idx)
    return _solve(neg_sharpe, n, cons, bounds, x0=x0)


def optimize_plan_D(mu, cov, eq_idx, rf=0.02, cap=0.4, eq_floor=0.5):
    """方案D「积极增长型」：股票占比不低于 eq_floor 的最大夏普组合。"""
    n = len(mu)
    eq_mask = np.zeros(n); eq_mask[eq_idx] = 1.0
    cons = [{"type": "ineq", "fun": lambda w: w @ eq_mask - eq_floor}]
    return max_sharpe(mu, cov, rf=rf, cap=cap, extra_constraints=cons)
