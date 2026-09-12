# -*- coding: utf-8 -*-
"""
模式三：多目标规划（Goal-Based Investing，对标机构理财规划流程）
每个目标（养老/教育/买房…）独立一个子组合：
  - 下滑轨道（glide path）：期限越长股票占比越高，临近目标自动转防御
  - KYC 等级限制股票占比上限（适当性）
  - 自助法蒙特卡洛模拟「期初金额 + 每月定投」达到目标金额的概率
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import analytics
import kyc
import optimizer as opt
from data_loader import annualized_stats, load_config, pool_meta, portfolio_nav
from report import RF, load_with_progress, mc_fan_chart, nav_chart

GOAL_PRESETS = ["退休养老", "子女教育", "购房首付", "旅游基金", "自定义"]

# KYC 等级 → 股票占比上限（适当性约束 glide path）
EQUITY_CAP_BY_LEVEL = {1: 0.0, 2: 0.10, 3: 0.30, 4: 0.60, 5: 1.0}

# 目标类型 → 风险预算：eq_bias 乘在标准下滑轨道上（用钱刚性越强越保守）
GOAL_PROFILES = {
    "退休养老": {"eq_bias": 1.0,
                 "note": "期限长、可承受波动，股票占比按标准下滑轨道"},
    "子女教育": {"eq_bias": 0.8,
                 "note": "用钱时间刚性，临近截止不能亏，轨道比养老更保守"},
    "购房首付": {"eq_bias": 0.5,
                 "note": "有明确截止日、本金安全优先，股票占比约为标准轨道的一半"},
    "旅游基金": {"eq_bias": 0.4,
                 "note": "短期享乐目标，以稳健为主"},
    "自定义":   {"eq_bias": 0.9,
                 "note": "按略低于标准轨道的折中处理"},
}


def glide_equity_share(years: float) -> float:
    """下滑轨道：距目标 T 年时的目标股票占比。T=2→20%，T=10→60%，T≥15→85% 封顶。"""
    return float(min(0.85, max(0.10, 0.10 + 0.05 * years)))


def _goal_weights(mu, cov, tickers, meta, eq_share, cap):
    """股票袖珍组合(最大夏普) + 防御袖珍组合(最小方差)，按 eq_share 混合。"""
    idx = {t: i for i, t in enumerate(tickers)}
    eq = [idx[t] for t in tickers if meta.get(t, {}).get("role") == "equity"]
    dfn = [idx[t] for t in tickers
           if meta.get(t, {}).get("role") in ("bond", "cash", "gold")]
    n = len(tickers)
    w = np.zeros(n)
    if eq and eq_share > 0.005:
        cons = [{"type": "ineq", "fun": lambda x: sum(x[i] for i in eq) - 0.99}]
        w_eq = opt.max_sharpe(mu, cov, rf=RF, cap=cap, extra_constraints=cons)
        # 只保留股票部分并归一
        w_eq_only = np.array([w_eq[i] if i in eq else 0.0 for i in range(n)])
        s = w_eq_only.sum()
        if s > 0:
            w += eq_share * w_eq_only / s
    if dfn and eq_share < 0.995:
        w_df = opt.min_variance(mu, cov, cap=cap)
        w_df_only = np.array([w_df[i] if i in dfn else 0.0 for i in range(n)])
        s = w_df_only.sum()
        if s > 0:
            w += (1 - eq_share) * w_df_only / s
    s = w.sum()
    return w / s if s > 0 else np.full(n, 1.0 / n)


def render():
    cfg = load_config()
    meta_all = pool_meta(cfg)
    kyc_result = st.session_state.get("kyc")

    st.header("模式三 · 多目标规划")
    st.caption("每个目标独立子组合：期限决定股票占比（下滑轨道），"
               "KYC 等级决定风险上限，自助法模拟目标达成概率。")

    if not kyc_result:
        st.warning("建议先在左侧边栏完成「风险测评（KYC）」；未测评时按 C3 平衡型处理。")
    level = kyc_result["level"] if kyc_result else 3
    eq_cap = EQUITY_CAP_BY_LEVEL[level]
    max_risk = kyc.LEVEL_RULES[level]["max_risk"]

    # ---------- 适当性提示：KYC 只警示不剔除，资产池内产品全部参与规划 ----------
    pool = list(cfg["asset_pool_cn"])
    over_risk = [p for p in pool if p.get("risk", 5) > max_risk]
    if over_risk:
        st.warning(
            f"⚠️ 适当性提示：按你的测评等级（{kyc.LEVEL_RULES[level]['label']}），"
            f"以下产品风险等级超过 R{max_risk}："
            + "、".join(f"{p['ticker']}（{p.get('name', '')}，R{p.get('risk', '?')}）"
                        for p in over_risk)
            + "。已按你的要求保留并继续参与规划，请知悉其波动可能超出你的风险承受能力。")
    if len(pool) < 3:
        st.error("资产池内可用资产不足 3 个。")
        return
    names = {p["ticker"]: p.get("name", p["ticker"]) for p in pool}
    tickers = [p["ticker"] for p in pool]
    cap = min(st.session_state.get("weight_cap", cfg["weight_cap"]),
              kyc.LEVEL_RULES[level]["cap"])

    # ---------- 目标清单 ----------
    st.subheader("第一步：添加你的目标")
    st.session_state.setdefault("goals", [])
    with st.form("goal_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            gname_preset = st.selectbox("目标类型", GOAL_PRESETS)
            # 输入框 key 跟随类型：切换类型时自动换成对应默认名，避免"永远是退休养老"
            gname = st.text_input(
                "目标名称（可改）",
                value=gname_preset if gname_preset != "自定义" else "我的目标",
                key=f"goal_name_{gname_preset}")
            g_years = st.number_input("距目标期限（年）", 0.5, 30.0, 10.0, 0.5)
        with c2:
            g_target = st.number_input("目标金额（元）", 10000.0, 1e8, 500000.0, 10000.0)
            g_start = st.number_input("目前已投入（元）", 0.0, 1e8, 50000.0, 10000.0)
            g_monthly = st.number_input("每月定投（元）", 0.0, 1e6, 3000.0, 500.0)
        st.caption("目标类型决定风险预算：" +
                   "；".join(f"{k}：{v['note']}" for k, v in GOAL_PROFILES.items()))
        if st.form_submit_button("添加目标", type="primary"):
            gname = gname.strip() or gname_preset
            existing = {g["name"] for g in st.session_state["goals"]}
            base, i = gname, 2
            while gname in existing:         # 重名自动加序号，避免图表 key 冲突
                gname = f"{base} #{i}"
                i += 1
            st.session_state["goals"].append({
                "name": gname, "type": gname_preset, "years": g_years,
                "target": g_target, "start": g_start, "monthly": g_monthly})
            st.rerun()

    goals = st.session_state["goals"]
    if not goals:
        st.info("添加至少一个目标后，下方会为每个目标生成子组合和达成概率。")
        return

    if st.button("清空全部目标"):
        st.session_state["goals"] = []
        st.rerun()

    # ---------- 数据 ----------
    prices, ok, bad, short, _m, _l, _q = load_with_progress(
        tickers, label="正在拉取国内资产池近 5 年数据")
    if len(ok) < 3:
        st.error("可用资产太少，请稍后重试。")
        return
    mu, cov, _ = opt.estimate_params(prices, shrinkage=cfg["shrinkage"])
    cfg_ov = {t: meta_all[t]["exp_return"] for t in prices.columns
              if meta_all.get(t, {}).get("exp_return") is not None}
    if cfg_ov:
        mu = opt.apply_return_overrides(mu, list(prices.columns), cfg_ov)
    tickers_ok = list(prices.columns)

    # ---------- 逐目标输出 ----------
    st.subheader("第二步：每个目标的子组合与达成概率")
    summary = []
    for gi, g in enumerate(goals):
        profile = GOAL_PROFILES.get(g.get("type", "自定义"), GOAL_PROFILES["自定义"])
        # 类型决定风险预算：标准下滑轨道 × 类型系数，再受 KYC 股票上限约束
        eq_share = min(glide_equity_share(g["years"]) * profile["eq_bias"], eq_cap)
        w = _goal_weights(mu, cov, tickers_ok, meta_all, eq_share, cap)
        ann_ret, ann_vol, sharpe, mdd = annualized_stats(prices, w, rf=RF)
        port_rets = (prices.pct_change().dropna() * w).sum(axis=1)
        prob, fan, median_final = analytics.goal_simulation(
            port_rets, g["start"], g["monthly"], g["years"], g["target"],
            paths=cfg["monte_carlo_paths"])

        with st.container(border=True):
            st.subheader(f"🎯 {g['name']}")
            st.caption(f"目标类型「{g.get('type', '自定义')}」：{profile['note']}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("目标达成概率", f"{prob:.0%}")
            c2.metric("终值中位数", f"{median_final:,.0f} 元",
                      delta=f"vs 目标 {g['target']:,.0f}")
            c3.metric("股票占比（下滑轨道×类型）", f"{eq_share:.0%}")
            c4.metric("组合年化波动", f"{ann_vol:.1%}")

            df_w = pd.DataFrame({
                "具体产品": [names.get(t, t) for t in tickers_ok],
                "配置比例": [f"{x:.1%}" for x in w],
            })
            df_w = df_w[w >= 0.005].reset_index(drop=True)
            col_l, col_r = st.columns([1, 2])
            with col_l:
                st.dataframe(df_w, width="stretch", hide_index=True)
                st.caption(f"预期年化收益 {ann_ret:.1%} · 夏普 {sharpe:.2f} · "
                           f"近5年最大回撤 {mdd:.1%}")
            with col_r:
                st.plotly_chart(
                    mc_fan_chart(fan, title=f"{g['name']} · 账户金额模拟分布",
                                 hline=g["target"], y_title="账户金额（元）"),
                    width="stretch", key=f"goal_fan_{gi}")
            if prob < 0.6:
                st.warning("达成概率偏低：可考虑延长期限、提高每月定投、或调低目标金额。")
            elif prob >= 0.85:
                st.success("达成概率很高：也可以考虑降低股票占比，用更稳的方式到达。")
        summary.append({"目标": g["name"], "类型": g.get("type", "-"),
                        "期限(年)": g["years"],
                        "目标金额": f"{g['target']:,.0f}", "达成概率": f"{prob:.0%}",
                        "股票占比": f"{eq_share:.0%}", "年化波动": f"{ann_vol:.1%}"})

    st.subheader("目标总览")
    st.dataframe(pd.DataFrame(summary), width="stretch", hide_index=True)
    st.caption("规则：下滑轨道 = 期限越长股票越多（10 年约 60%），临近目标自动转防御；"
               "KYC 等级进一步压低股票占比上限；达成概率用历史收益自助法模拟 "
               f"{cfg['monte_carlo_paths']} 条路径，含每月定投。")
