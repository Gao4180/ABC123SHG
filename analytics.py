# -*- coding: utf-8 -*-
"""
分析增强模块：蒙特卡洛前景模拟、历史情景压力测试、数据质量检查。
灵感来自机构级方案，做了适合个人投资者的简化。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------- 蒙特卡洛前景模拟 ----------

def monte_carlo_forecast(ann_ret: float, ann_vol: float,
                         years: float = 1.0, paths: int = 2000,
                         seed: int = 42, hist_rets=None,
                         method: str = "bootstrap"):
    """
    模拟组合未来走势。
    method="bootstrap"（默认）：对历史日收益有放回抽样（自助法），
      保留真实分布的肥尾与偏度，比正态假设更诚实；
    method="gbm"：几何布朗运动（正态假设），hist_rets 缺失时的回退方案。
    返回 (正收益概率, 分位数表DataFrame 含 p5/p25/p50/p75/p95)。
    """
    rng = np.random.default_rng(seed)
    days = int(TRADING_DAYS * years)
    if method == "bootstrap" and hist_rets is not None and len(hist_rets) > 60:
        pool = np.asarray(hist_rets, dtype=float)
        pool = pool[np.isfinite(pool)]
        daily = rng.choice(pool, size=(days, paths), replace=True)
        nav_paths = np.cumprod(1 + daily, axis=0)
    else:
        mu_d = ann_ret / TRADING_DAYS
        sig_d = ann_vol / np.sqrt(TRADING_DAYS)
        shocks = rng.normal((mu_d - 0.5 * sig_d ** 2), sig_d, size=(days, paths))
        nav_paths = np.exp(np.cumsum(shocks, axis=0))
    nav_paths = np.vstack([np.ones(paths), nav_paths])   # 第 0 天净值为 1
    terminal = nav_paths[-1]
    prob_positive = float((terminal > 1).mean())
    qs = np.percentile(nav_paths, [5, 25, 50, 75, 95], axis=1)
    fan = pd.DataFrame({
        "交易日": np.arange(days + 1),
        "p5": qs[0], "p25": qs[1], "p50": qs[2], "p75": qs[3], "p95": qs[4],
    })
    return prob_positive, fan


def goal_simulation(hist_rets, start_amount: float, monthly_contrib: float,
                    years: float, target: float, paths: int = 2000,
                    seed: int = 42):
    """
    多目标规划用：期初金额 + 每月定投，自助法模拟，估计达到目标金额的概率。
    返回 (达成概率, 分位数表DataFrame, 终值中位数)。
    """
    rng = np.random.default_rng(seed)
    days = int(TRADING_DAYS * years)
    pool = np.asarray(hist_rets, dtype=float)
    pool = pool[np.isfinite(pool)]
    if pool.size == 0:
        raise ValueError("历史收益数据为空，无法模拟，请检查数据源后重试")
    daily = rng.choice(pool, size=(days, paths), replace=True)
    nav = np.full(paths, float(start_amount))
    curve = np.empty((days + 1, paths))
    curve[0] = nav
    contrib_every = 21                       # 约一个自然月的交易日数
    for d in range(days):
        nav = nav * (1 + daily[d])
        if (d + 1) % contrib_every == 0:
            nav = nav + monthly_contrib
        curve[d + 1] = nav
    terminal = curve[-1]
    prob = float((terminal >= target).mean())
    qs = np.percentile(curve, [5, 25, 50, 75, 95], axis=1)
    fan = pd.DataFrame({
        "交易日": np.arange(days + 1),
        "p5": qs[0], "p25": qs[1], "p50": qs[2], "p75": qs[3], "p95": qs[4],
    })
    return prob, fan, float(np.median(terminal))


def mc_summary_text(prob: float, label: str = "该组合") -> str:
    """把蒙特卡洛结果转成一句人话。"""
    return (f"按近5年收益与波动外推，{label}未来 1 年取得正收益的概率约为 "
            f"**{prob:.0%}**（蒙特卡洛 2000 条路径模拟，仅为统计估计）。")


# ---------- 历史情景压力测试 ----------

def stress_test(prices: pd.DataFrame, weights: np.ndarray,
                scenarios: list[dict]) -> pd.DataFrame:
    """
    用真实历史区间重放极端行情：把组合权重代入该区间，算累计收益。
    返回 DataFrame：[情景, 区间, 组合表现]；数据未覆盖的情景自动跳过。
    """
    rets = prices.pct_change().dropna()
    port = (rets * np.asarray(weights)).sum(axis=1)
    rows = []
    for sc in scenarios:
        seg = port.loc[sc["start"]:sc["end"]]
        if len(seg) < 3:                     # 数据未覆盖该区间
            continue
        cum = float((1 + seg).prod() - 1)
        rows.append({
            "情景": sc["name"],
            "区间": f"{sc['start']} ~ {sc['end']}",
            "组合区间表现": f"{cum:+.1%}",
        })
    return pd.DataFrame(rows)


# ---------- 数据质量检查 ----------

def data_quality_report(raw: pd.DataFrame, aligned: pd.DataFrame,
                        jump_sigma: float = 5.0,
                        missing_warn_ratio: float = 0.05) -> list[str]:
    """
    检查行情数据质量，返回需要提示的文本列表：
    - 缺失交易日占比过高
    - 单日涨跌超过 jump_sigma 倍历史波动（价格跳跃，可能是数据错误）
    """
    warns = []
    miss = raw.isna().mean()
    for t, r in miss.items():
        if r > missing_warn_ratio:
            warns.append(f"{t} 有 {r:.0%} 的交易日缺失数据（已用前值补齐），结果可能略有偏差")
    rets = aligned.pct_change().dropna()
    for t in rets.columns:
        r = rets[t]
        vol = r.std()
        if vol > 0:
            jumps = int((r.abs() > jump_sigma * vol).sum())
            if jumps > 0:
                warns.append(f"{t} 检出 {jumps} 次超过 {jump_sigma:.0f}σ 的单日剧烈波动，"
                             "可能是真实极端行情，也可能是数据源脏数据")
    return warns


# ---------- 风险贡献分解（风险归因） ----------

def risk_contribution(cov, weights, tickers):
    """
    把组合总波动分解到每个资产：
    边际风险贡献 MCTR_i = (Σw)_i / σp，风险贡献 CTR_i = w_i · MCTR_i，
    占比 CTR_i / σp（合计 = 100%）。
    专业机构看的是风险预算而非资金比例——60/40 股债组合约九成风险来自股票。
    返回 (DataFrame[资产/资金占比/风险贡献占比/边际风险贡献], 组合年化波动)。
    """
    w = np.asarray(weights, dtype=float)
    cov = np.asarray(cov, dtype=float)
    if w.sum() <= 0:
        return pd.DataFrame(), 0.0
    w = w / w.sum()
    vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
    if vol <= 1e-12:
        return pd.DataFrame(), vol
    mctr = (cov @ w) / vol
    pct = w * mctr / vol
    df = pd.DataFrame({"资产": list(tickers), "资金占比": w,
                       "风险贡献占比": pct, "边际风险贡献": mctr})
    return (df.sort_values("风险贡献占比", ascending=False)
              .reset_index(drop=True), vol)


# ---------- 风格/因子暴露（可投资代理回归） ----------

# 用可投资的指数/ETF 代表风格因子（Fama-French 思路的可投资化近似）：
# 因子数据本身难以免费获取，代理回归是投顾行业的常见折中做法。
FACTOR_PROXIES = {
    "美股": [("SPY", "大盘市场"), ("IWM", "小盘股"), ("QQQ", "科技成长"),
             ("TLT", "长久期债券"), ("GLD", "黄金")],
    "A股": [("000300.SS", "大盘蓝筹"), ("000905.SS", "中小盘"),
            ("399006.SZ", "创业板成长"), ("511010.SS", "利率债")],
}


def factor_exposure(port_rets: pd.Series, factor_rets: pd.DataFrame):
    """
    组合日收益对因子代理日收益的多元线性回归（OLS，含截距）。
    返回 (DataFrame[因子/暴露β], R²)。
    β 解读：该因子涨 1%，组合平均跟随变动 β%（β<0 为反向暴露）。
    """
    if factor_rets is None or factor_rets.shape[1] == 0:
        return pd.DataFrame(), np.nan
    df = pd.concat([port_rets.rename("y"), factor_rets], axis=1).dropna()
    if len(df) < 60:
        return pd.DataFrame(), np.nan
    y = df["y"].values
    X = df.drop(columns="y").values
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    yhat = X1 @ beta
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-18 else np.nan
    out = pd.DataFrame({"因子": list(factor_rets.columns), "暴露β": beta[1:]})
    return out, float(r2) if np.isfinite(r2) else np.nan
