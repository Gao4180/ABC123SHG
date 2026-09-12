# -*- coding: utf-8 -*-
"""
择时与动态配置模块（战术资产配置 TAA）

学术依据：
- Faber (2007) "A Quantitative Approach to Tactical Asset Allocation"：
  月末价格 > 10 个月均线（约等于 200 日均线）则持有，跌破则撤到现金。
  核心价值是降低波动与最大回撤（"股票式收益、债券式回撤"）。
- Moskowitz, Ooi & Pedersen (2012) "Time Series Momentum"：
  过去 12 个月收益为正则做多、为负则回避，跨 58 个市场统计显著。
本模块把这两条规则应用到个人组合：信号表 + 月末调仓的趋势过滤回测。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MA_WINDOW = 200          # Faber 10 个月均线 ≈ 200 日均线
MOM_DAYS = 252           # TSMOM 12 个月动量
MIN_WINDOW = 60          # 数据不足 200 天时的最小均线窗口


# ---------- 当前择时信号 ----------

def trend_signals(prices: pd.DataFrame, ma_window: int = MA_WINDOW,
                  mom_days: int = MOM_DAYS) -> pd.DataFrame:
    """
    对每个资产计算趋势信号，返回 DataFrame：
    [代码, 最新价, 均线, 偏离均线, 近12个月动量, 信号, 上次切换]
    信号规则（Faber + TSMOM 结合）：价格在均线上方且动量为正 → 可买入/持有。
    """
    rows = []
    for t in prices.columns:
        s = prices[t].dropna()
        if len(s) < MIN_WINDOW:
            continue
        win = min(ma_window, max(MIN_WINDOW, len(s) // 2))
        ma = s.rolling(win).mean()
        last, last_ma = float(s.iloc[-1]), float(ma.iloc[-1])
        lb = min(mom_days, len(s) - 1)
        mom = float(s.iloc[-1] / s.iloc[-lb] - 1)
        above = last > last_ma
        signal = "✅ 持有/可买入" if (above and mom > 0) else \
                 "⚠️ 观望/减仓" if (not above and mom < 0) else "🔶 信号矛盾，谨慎"
        # 最近一次金叉/死叉日期
        sign = (s > ma).astype(float)
        sign[ma.isna()] = np.nan
        cross = sign.diff()
        switches = cross[cross.abs() > 0].dropna()
        if len(switches):
            last_sw = switches.index[-1]
            last_sw_txt = f"{last_sw.strftime('%Y-%m-%d')} " \
                          f"{'金叉↑' if switches.iloc[-1] > 0 else '死叉↓'}"
        else:
            last_sw_txt = "近5年未切换"
        rows.append({
            "代码": t,
            "最新价": round(last, 2),
            "均线": round(last_ma, 2),
            "偏离均线": f"{last / last_ma - 1:+.1%}",
            "近12个月动量": f"{mom:+.1%}",
            "当前信号": signal,
            "上次切换": last_sw_txt,
        })
    return pd.DataFrame(rows)


# ---------- 趋势过滤动态配置 ----------

def tactical_weights(base_weights, prices: pd.DataFrame,
                     ma_window: int = MA_WINDOW,
                     cash_ticker: str | None = None) -> np.ndarray:
    """
    按当前信号把基础权重做一次战术调整：
    跌破均线（或动量转负）的资产权重清零，释放的权重转给现金类资产
    （cash_ticker 指定，通常是短债 ETF；没有则该部分视为持有现金，收益≈0）。
    """
    w = np.asarray(base_weights, dtype=float).copy()
    tickers = list(prices.columns)
    sig = trend_signals(prices, ma_window)
    off = {r["代码"] for _, r in sig.iterrows() if "观望" in r["当前信号"]}
    cash_i = tickers.index(cash_ticker) if cash_ticker in tickers else None
    freed = 0.0
    for i, t in enumerate(tickers):
        if t in off and i != cash_i:
            freed += w[i]
            w[i] = 0.0
    if cash_i is not None:
        w[cash_i] += freed
    return w


# ---------- 月末调仓回测：静态 vs 趋势过滤 ----------

def tactical_backtest(prices: pd.DataFrame, base_weights,
                      ma_window: int = MA_WINDOW,
                      cash_ticker: str | None = None,
                      cost_bps: float = 5.0,
                      fees: dict | None = None):
    """
    月末检查一次信号（Faber 原版规则），跌破均线的资产下月权重转给现金类资产。
    计入现实摩擦：
      - cost_bps：单边交易成本（基点），按调仓换手从净值扣除
      - fees：{ticker: 年管理费率}，按日计提拖累净值（静态与动态都扣，公平对比）
    返回 (趋势过滤净值Series, 静态净值Series, 最近调仓记录DataFrame)。
    净值从均线预热期之后开始（起点=1）。
    """
    w = np.asarray(base_weights, dtype=float)
    tickers = list(prices.columns)
    fee_vec = np.array([fees.get(t, 0.0) if fees else 0.0 for t in tickers])
    ma = prices.rolling(ma_window, min_periods=max(MIN_WINDOW, ma_window // 2)).mean()
    above = prices > ma
    rets = prices.pct_change().fillna(0.0)
    dates = prices.index
    # 每月最后一个交易日作为调仓检查点（Faber 原版：只在月末看信号）
    month_ends = set(dates.to_series().groupby([dates.year, dates.month]).max())
    cash_i = tickers.index(cash_ticker) if cash_ticker in tickers else None

    start = ma_window                     # 预热期
    nav_t, nav_s = 1.0, 1.0
    navs_t, navs_s, idx_out = [], [], []
    cur_w = w.copy()
    logs = []
    prev_on = np.ones(len(tickers), dtype=bool)

    for i, d in enumerate(dates):
        if i > start:
            r = rets.iloc[i].values
            nav_t *= 1 + float((r * cur_w).sum())
            nav_s *= 1 + float((r * w).sum())
            # 管理费率按日计提
            nav_t *= 1 - float((fee_vec * cur_w).sum()) / TRADING_DAYS
            nav_s *= 1 - float((fee_vec * w).sum()) / TRADING_DAYS
            navs_t.append(nav_t)
            navs_s.append(nav_s)
            idx_out.append(d)
            if d in month_ends:
                on = above.iloc[i].fillna(True).values
                new_w = np.where(on, w, 0.0)
                if cash_i is not None:
                    new_w[cash_i] = w[cash_i] + (w.sum() - new_w.sum())
                # 单边交易成本：按换手额从净值扣除
                turnover = float(np.abs(new_w - cur_w).sum())
                if turnover > 1e-9:
                    nav_t *= 1 - turnover * cost_bps / 1e4
                    navs_t[-1] = nav_t
                for j, t in enumerate(tickers):
                    if on[j] != prev_on[j] and w[j] > 0.005:
                        logs.append({
                            "日期": d.strftime("%Y-%m-%d"),
                            "资产": t,
                            "动作": "买入/恢复配置" if on[j] else "撤出→转现金类",
                        })
                prev_on = on
                cur_w = new_w

    nav_t_s = pd.Series(navs_t, index=idx_out, name="趋势过滤")
    nav_s_s = pd.Series(navs_s, index=idx_out, name="静态持有")
    log_df = pd.DataFrame(logs)
    if not log_df.empty:
        log_df = log_df.sort_values("日期").tail(10).reset_index(drop=True)
    return nav_t_s, nav_s_s, log_df


def nav_stats(nav: pd.Series, rf: float = 0.02) -> dict:
    """由净值序列直接算年化收益/波动/夏普/最大回撤（用于回测对比）。"""
    r = nav.pct_change().dropna()
    n = len(r)
    if n == 0:
        return {"年化收益": "-", "年化波动": "-", "夏普比率": "-", "最大回撤": "-"}
    ann_ret = float(nav.iloc[-1] ** (TRADING_DAYS / n) - 1)
    ann_vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan
    mdd = float((nav / nav.cummax() - 1).min())
    return {"年化收益": f"{ann_ret:.1%}", "年化波动": f"{ann_vol:.1%}",
            "夏普比率": f"{sharpe:.2f}", "最大回撤": f"{mdd:.1%}"}
