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


def nav_chart(navs: dict[str, pd.Series], title="累计净值走势（近5年）",
              benchmark: tuple[str, pd.Series] | None = None):
    """多条净值曲线对比，hover 可见每日数值；可叠加市场基准（灰色虚线）。"""
    fig = go.Figure()
    for name, nav in navs.items():
        fig.add_trace(go.Scatter(
            x=nav.index, y=nav.values, name=name, mode="lines",
            hovertemplate="%{x|%Y-%m-%d}<br>净值 %{y:.3f}<extra>%{fullData.name}</extra>",
        ))
    if benchmark is not None:
        bname, bnav = benchmark
        fig.add_trace(go.Scatter(
            x=bnav.index, y=bnav.values, name=f"基准：{bname}", mode="lines",
            line=dict(color="gray", dash="dash", width=1.5),
            hovertemplate="%{x|%Y-%m-%d}<br>净值 %{y:.3f}<extra>%{fullData.name}</extra>",
        ))
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=420,
                      title=title, hovermode="x unified",
                      yaxis_title="累计净值（起点=1）",
                      legend=dict(orientation="h", y=-0.15))
    return fig


def benchmark_nav(dates, market: str):
    """
    拉取市场基准净值（A股→沪深300，美股→标普500 ETF SPY），
    与组合日期对齐并归一到起点 1。失败返回 None（静默降级，不影响主流程）。
    """
    from data_loader import fetch_close_batch   # 延迟导入：测试打桩可生效
    ticker, label = ("000300.SS", "沪深300") if market == "A股" \
        else ("SPY", "标普500")
    try:
        s = fetch_close_batch((ticker,)).get(ticker)
        if s is None or s.empty:
            return None
        s = s.reindex(dates).ffill().dropna()
        if len(s) < 30:
            return None
        return (f"{label}（{ticker}）", s / s.iloc[0])
    except Exception:
        return None


def risk_contribution_section(tickers, weights, cov, names=None, key_prefix="rc"):
    """风险归因区块：资金占比 vs 风险贡献占比 对比柱状图 + 一句人话解读。"""
    import analytics
    df, _vol = analytics.risk_contribution(cov, weights, list(tickers))
    if df.empty:
        return
    st.markdown("**风险归因：每个资产贡献了多少风险**")
    labels = [f"{names.get(t, t)}（{t}）" if names else t for t in df["资产"]]
    fig = go.Figure()
    fig.add_bar(name="资金占比", x=labels, y=df["资金占比"],
                marker_color="#9DB2D5",
                hovertemplate="%{x}<br>资金占比 %{y:.1%}<extra></extra>")
    fig.add_bar(name="风险贡献占比", x=labels, y=df["风险贡献占比"],
                marker_color="#C00000",
                hovertemplate="%{x}<br>风险贡献 %{y:.1%}<extra></extra>")
    fig.update_layout(barmode="group", height=320,
                      margin=dict(l=10, r=10, t=20, b=10),
                      yaxis_tickformat=".0%",
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_rc")
    top = df.iloc[0]
    tname = names.get(top["资产"], top["资产"]) if names else top["资产"]
    extra = "，是组合最主要的风险来源" if top["风险贡献占比"] - top["资金占比"] > 0.1 else ""
    st.caption(
        f"💡 {tname} 占资金 {top['资金占比']:.0%}，却贡献了组合 {top['风险贡献占比']:.0%} "
        f"的波动{extra}。专业配置看的是风险预算而不仅是资金比例："
        "典型的 60/40 股债组合，约九成风险其实来自股票。")


def factor_section(prices, weights, market, key_prefix="fx"):
    """风格/因子暴露区块：组合日收益对可投资风格代理的回归 β + R²。"""
    import analytics
    from data_loader import fetch_close_batch   # 延迟导入：测试打桩可生效
    proxies = analytics.FACTOR_PROXIES.get(
        market, analytics.FACTOR_PROXIES["美股"])
    fname = dict(proxies)
    try:
        data = fetch_close_batch(tuple(fname.keys()))
    except Exception:
        data = {}
    series = {fname[t]: s for t, s in data.items()
              if s is not None and not s.empty}
    if len(series) < 2:
        st.caption("因子代理数据暂不可用，跳过风格暴露分析。")
        return
    st.markdown("**风格暴露：组合收益由哪些风格驱动**（因子代理回归）")
    # 对齐方式：把因子净值重索引到组合交易日再前向填充（而非取交集），
    # 避免某个因子缺数据时重叠期被掏空导致"重叠区间太短"
    fprices = pd.concat(series, axis=1).sort_index()
    fprices = fprices.reindex(prices.index).ffill()
    frets = fprices.pct_change()
    port_rets = (prices.pct_change().dropna()
                 * np.asarray(weights, dtype=float)).sum(axis=1)
    overlap = pd.concat([port_rets.rename("y"), frets], axis=1).dropna().shape[0]
    df, r2 = analytics.factor_exposure(port_rets, frets)
    if df.empty:
        st.caption(f"因子数据与组合的重叠区间太短（{overlap} 个交易日 < 60），"
                   "无法回归，跳过。这通常是因子指数拉取不完整，刷新重试即可。")
        return
    st.plotly_chart(go.Figure(go.Bar(
        x=df["因子"], y=df["暴露β"],
        marker_color=["#C00000" if b < 0 else "#4472C4" for b in df["暴露β"]],
        hovertemplate="%{x} β=%{y:.2f}<extra></extra>",
    )).update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10),
                     yaxis_title="暴露 β"),
        width="stretch", key=f"{key_prefix}_fx")
    top = df.loc[df["暴露β"].abs().idxmax()]
    r2_txt = f"{r2:.0%}" if np.isfinite(r2) else "未知"
    st.caption(
        f"解读：组合波动约 {r2_txt} 可由上述风格因子解释；"
        f"暴露最高的是「{top['因子']}」（β={top['暴露β']:.2f}）——该因子涨 1%，"
        f"组合平均跟{'涨' if top['暴露β'] >= 0 else '跌'} "
        f"{abs(top['暴露β']):.2f}%。"
        "方法：组合日收益对可投资指数/ETF 的多元回归（Fama-French 思路的可投资化近似），"
        "反映风格暴露而非真实持仓穿透。")


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


def market_views_section(cfg=None):
    """
    主流机构观点快照（人工整理，存于 config.yaml 的 market_views）。
    投行研报没有免费实时数据源，这里用"定期人工更新的快照"折中：
    价值在于给风险归因一个参照系——你的风险集中在哪，机构对该资产的分歧是什么。
    """
    if cfg is None:
        cfg = load_config()
    views = cfg.get("market_views") or {}
    items = views.get("items") or []
    if not items:
        return
    with st.expander("🌐 主流机构观点快照（对照你的风险归因看）"):
        st.caption(f"整理时间：{views.get('updated', '未知')}。"
                   "来源为公开报道的各投行展望，仅供学习参考，非实时数据、不构成投资建议。")
        for it in items:
            st.markdown(f"- {it}")
