# -*- coding: utf-8 -*-
"""
展示模块：表格、plotly 图表、自动文字分析。两个模式页面共用。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_loader import (
    TRADING_DAYS, annualized_stats, drawdown_periods, load_config,
    max_drawdown, portfolio_nav,
)

RF = load_config()["risk_free_rate"]   # 无风险利率，来自 config.yaml


# ---------- 带进度的数据加载 ----------

def load_with_progress(tickers, label="正在拉取数据"):
    """
    带进度提示的数据加载；限流/数据质量问题给出明确提示。
    返回值与 data_loader.load_prices 相同。
    """
    from data_loader import load_prices   # 延迟导入，避免循环依赖
    bar = st.progress(0.0, text=f"{label}……")

    def _cb(t, i, n):
        bar.progress(i / n, text=f"{label}：{t}")

    result = load_prices(tickers, progress=_cb)
    bar.empty()
    limited = result[5]
    if limited:
        st.warning(
            "以下资产所有数据源（Yahoo / 新浪 / 腾讯 / Alpha Vantage）均未取到："
            + "、".join(limited) +
            "。这通常是数据源限流（几十分钟后自动解除）或代码有误，请稍后刷新重试。"
        )
    # 数据来源标注
    from data_loader import last_sources
    srcs = last_sources()
    ok_list = result[1]
    by_src = {}
    for t in ok_list:
        by_src.setdefault(srcs.get(t, "未知"), []).append(t)
    if by_src:
        st.caption("数据来源：" + "；".join(f"{k}（{len(v)} 个）" for k, v in by_src.items()))
    for q in result[6]:                     # 数据质量提示
        st.caption("⚠️ 数据质量：" + q)
    return result


# ---------- 指标表 ----------

def asset_metrics_table(prices: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    """各资产：配置比例、预期年化收益/波动、夏普、近5年最大回撤。"""
    rets = prices.pct_change().dropna()
    rows = []
    for i, t in enumerate(prices.columns):
        r = rets[t]
        ann_ret = float(r.mean() * TRADING_DAYS)
        ann_vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
        sharpe = (ann_ret - RF) / ann_vol if ann_vol > 0 else np.nan
        nav = (1 + r).cumprod()
        rows.append({
            "资产": t,
            "配置比例": f"{weights[i]:.1%}",
            "预期年化收益": f"{ann_ret:.1%}",
            "预期年化波动": f"{ann_vol:.1%}",
            "夏普比率": f"{sharpe:.2f}" if np.isfinite(sharpe) else "-",
            "近5年最大回撤": f"{max_drawdown(nav):.1%}",
        })
    return pd.DataFrame(rows)


def amount_table(prices: pd.DataFrame, weights: np.ndarray,
                 total_amount: float, currency: str,
                 fees: dict | None = None) -> pd.DataFrame:
    """把配置比例换算成金额，按当前价格折算建议买入股数（取整），并估算年管理费。"""
    last = prices.iloc[-1]
    rows = []
    for i, t in enumerate(prices.columns):
        amt = weights[i] * total_amount
        price = float(last[t])
        shares = int(amt // price) if price > 0 else 0
        fee_rate = (fees or {}).get(t, 0.0)
        rows.append({
            "资产": t,
            "配置金额": f"{amt:,.0f} {currency}",
            "最新收盘价": f"{price:,.2f}",
            "建议买入数量(股)": shares,
            "实际占用金额": f"{shares * price:,.0f} {currency}",
            "年管理费(约)": f"{amt * fee_rate:,.0f} {currency}" if fee_rate else "-",
        })
    return pd.DataFrame(rows)


# ---------- 图表 ----------

def pie_chart(tickers, weights):
    fig = go.Figure(go.Pie(
        labels=list(tickers), values=list(np.round(weights, 4)),
        hole=0.45, hovertemplate="%{label}: %{percent}<extra></extra>",
    ))
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=360,
                      title="配置比例")
    return fig


def nav_chart(navs: dict[str, pd.Series], title="累计净值走势（近5年）"):
    """多条净值曲线对比，hover 可见每日数值。"""
    fig = go.Figure()
    for name, nav in navs.items():
        fig.add_trace(go.Scatter(
            x=nav.index, y=nav.values, name=name, mode="lines",
            hovertemplate="%{x|%Y-%m-%d}<br>净值 %{y:.3f}<extra>%{fullData.name}</extra>",
        ))
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=420,
                      title=title, hovermode="x unified",
                      yaxis_title="累计净值（起点=1）",
                      legend=dict(orientation="h", y=-0.15))
    return fig


def mc_fan_chart(fan: pd.DataFrame, title="蒙特卡洛模拟：未来 1 年净值分布",
                 hline: float = 1.0, y_title: str = "模拟净值"):
    """蒙特卡洛扇形图：中位数线 + 25~75 / 5~95 两条置信带。"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fan["交易日"], y=fan["p95"], mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fan["交易日"], y=fan["p5"], mode="lines",
                             fill="tonexty", fillcolor="rgba(68,114,196,0.15)",
                             line=dict(width=0), name="5%~95% 区间",
                             hovertemplate="第%{x}天<extra></extra>"))
    fig.add_trace(go.Scatter(x=fan["交易日"], y=fan["p75"], mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fan["交易日"], y=fan["p25"], mode="lines",
                             fill="tonexty", fillcolor="rgba(68,114,196,0.30)",
                             line=dict(width=0), name="25%~75% 区间", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fan["交易日"], y=fan["p50"], mode="lines",
                             line=dict(color="#4472C4", width=2), name="中位数路径",
                             hovertemplate="第%{x}天 净值%{y:.3f}<extra>中位数</extra>"))
    fig.add_hline(y=hline, line_dash="dot", line_color="gray")
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=380,
                      title=title, xaxis_title="未来交易日", yaxis_title=y_title,
                      legend=dict(orientation="h", y=-0.18))
    return fig


# ---------- 自动文字分析 ----------

def auto_analysis(tickers, weights, prices, mu, cov, label="该组合"):
    """生成一段组合分析文字：特点、集中度、被剔除资产及原因、大回撤区间。"""
    ann_ret, ann_vol, sharpe, mdd = annualized_stats(prices, weights, rf=RF)
    nav = portfolio_nav(prices, weights)
    w = np.asarray(weights)

    parts = []
    style = ("波动较低、防御属性较强" if ann_vol < 0.10 else
             "风险收益较为均衡" if ann_vol < 0.18 else
             "进攻性较强、波动偏高")
    parts.append(
        f"**{label}**：预期年化收益约 {ann_ret:.1%}，年化波动约 {ann_vol:.1%}，"
        f"夏普比率 {sharpe:.2f}，近5年最大回撤约 {mdd:.1%}，整体{style}。"
    )

    # 集中度（始终输出，权重过高时加警示）
    top = int(np.argmax(w))
    hhi = float((w ** 2).sum())
    eff_n = 1 / hhi if hhi > 0 else len(tickers)
    warn = "，占比较高，需留意单一资产波动对整体的影响" if w[top] >= 0.35 else ""
    parts.append(
        f"最大持仓为 {tickers[top]}（{w[top]:.0%}）{warn}，"
        f"组合等效持仓数约 {eff_n:.1f} 只，分散度{'较好' if eff_n >= 4 else '一般'}。"
    )

    # 被剔除（权重≈0）的资产及原因
    dropped = [i for i in range(len(tickers)) if w[i] < 0.005]
    if dropped:
        reasons = []
        corr = prices.pct_change().dropna().corr()
        for i in dropped:
            reason = "预期收益偏低"
            held = [j for j in range(len(tickers)) if w[j] >= 0.005 and j != i]
            if held:
                cmax = max(abs(corr.iloc[i, j]) for j in held)
                if cmax >= 0.85 and mu[i] < max(mu[j] for j in held):
                    reason = "与持仓资产高度相关且收益不占优"
            reasons.append(f"{tickers[i]}（{reason}）")
        parts.append("优化器自动剔除了：" + "、".join(reasons) + "。")

    # 大回撤区间
    periods = drawdown_periods(nav, top_n=2)
    if periods:
        seg = "；".join(
            f"{p.strftime('%Y-%m')} 至 {t.strftime('%Y-%m')} 回撤 {d:.0%}"
            for p, t, d in periods if d < -0.05
        )
        if seg:
            parts.append(f"近5年主要回撤区间：{seg}。")
    return " ".join(parts)


def portfolio_summary_row(name, prices, weights):
    """汇总一套方案的关键指标，用于模式二的横向对比表。"""
    ann_ret, ann_vol, sharpe, mdd = annualized_stats(prices, weights, rf=RF)
    rets = prices.pct_change().dropna()
    port = (rets * weights).sum(axis=1)
    yearly = (1 + port).groupby(port.index.year).prod() - 1
    worst_year = f"{yearly.idxmin()}年 {yearly.min():.0%}" if len(yearly) else "-"
    return {
        "方案": name,
        "预期年化收益": f"{ann_ret:.1%}",
        "年化波动": f"{ann_vol:.1%}",
        "夏普比率": f"{sharpe:.2f}",
        "最大回撤": f"{mdd:.1%}",
        "最差年度": worst_year,
    }


# ---------- 择时与动态配置（趋势跟踪） ----------

def timing_section(prices, weights, label, names=None, cash_ticker="SHV",
                   key_prefix="timing", fees=None, cost_bps=5.0):
    """
    择时区块：当前信号表 + 战术调整权重 + 月末调仓回测（静态 vs 趋势过滤）。
    回测计入单边交易成本与年管理费率。
    学术依据：Faber (2007) GTAA 200日均线择时；Moskowitz 等 (2012) 12个月时间序列动量。
    """
    import timing

    st.subheader("择时信号：现在该买什么、避什么")
    sig = timing.trend_signals(prices)
    if names:
        sig.insert(1, "具体产品", sig["代码"].map(lambda t: names.get(t, t)))
    st.dataframe(sig, width="stretch", hide_index=True)
    st.caption("规则：价格在 200 日均线上方且近 12 个月动量为正 → 可买入/持有；"
               "跌破均线且动量转负 → 观望/减仓。两者矛盾时谨慎。"
               "依据：Faber (2007) 均线择时模型 + Moskowitz 等 (2012) 时间序列动量。")

    # 战术调整后的当前权重
    w_adj = timing.tactical_weights(weights, prices, cash_ticker=cash_ticker)
    changed = np.abs(w_adj - np.asarray(weights, dtype=float)) > 0.005
    if changed.any():
        tickers = list(prices.columns)
        st.write("**按当前信号战术调整后的权重**（跌破均线的资产转给现金类）：")
        df_adj = pd.DataFrame({
            "资产": [names.get(t, t) if names else t for t in tickers],
            "原比例": [f"{x:.1%}" for x in weights],
            "战术调整比例": [f"{x:.1%}" for x in w_adj],
        })
        df_adj = df_adj[(w_adj >= 0.005) | changed].reset_index(drop=True)
        st.dataframe(df_adj, width="stretch", hide_index=True)
    else:
        st.success("当前所有持仓资产均在均线上方，无需战术调整。")

    # 月末调仓回测（含交易成本与管理费）
    st.subheader("动态配置回测：静态持有 vs 趋势过滤（月末调仓）")
    nav_t, nav_s, logs = timing.tactical_backtest(
        prices, weights, cash_ticker=cash_ticker, cost_bps=cost_bps, fees=fees)
    st.plotly_chart(
        nav_chart({"静态持有（不择时）": nav_s, "趋势过滤（月末调仓）": nav_t},
                  title=f"{label} · 静态 vs 趋势过滤净值"),
        width="stretch", key=f"{key_prefix}_bt")
    cmp_df = pd.DataFrame([
        {"方式": "静态持有（不择时）", **timing.nav_stats(nav_s, rf=RF)},
        {"方式": "趋势过滤（月末调仓）", **timing.nav_stats(nav_t, rf=RF)},
    ])
    st.dataframe(cmp_df, width="stretch", hide_index=True)
    st.caption(f"回测规则：每月最后一个交易日检查一次信号，跌破均线的资产下月转持现金类"
               f"（{cash_ticker or '现金'}），回到均线上方再恢复。"
               f"已计入单边 {cost_bps:.0f}‱ 交易成本与各基金管理费率。历史回测不代表未来。")
    if not logs.empty:
        with st.expander("查看最近几次调仓记录"):
            st.dataframe(logs, width="stretch", hide_index=True)
