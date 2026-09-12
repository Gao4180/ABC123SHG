# -*- coding: utf-8 -*-
"""
模式二：财富方案推荐
基于 config.yaml 的资产池（美股 ETF / 国内 ETF·QDII 可选，含产品风险等级），
自动生成 4 套经典方案并横向对比。KYC 定级后做适当性提示（只警示不剔除）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import analytics
import kyc
import optimizer as opt
from data_loader import annualized_stats, load_config, pool_meta, portfolio_nav
from report import (
    RF, auto_analysis, benchmark_nav, factor_section, load_with_progress,
    market_views_section, mc_fan_chart, nav_chart, pie_chart,
    portfolio_summary_row, risk_contribution_section, timing_section,
)

PLAN_META = {
    "方案A · 稳健保本型": {
        "人群": "适合人群：保守型投资者，本金安全优先，可接受收益接近或略超通胀。",
        "风险": "主要风险：利率上行导致债券价格下跌；通胀超预期侵蚀实际购买力。",
    },
    "方案B · 经典60/40": {
        "人群": "适合人群：追求长期稳健增值、能承受中等波动的投资者。",
        "风险": "主要风险：股债双杀年份（如2022）股债相关性转正，分散失效。",
    },
    "方案C · 全天候组合": {
        "人群": "适合人群：希望在各种经济环境下都保持平稳、厌恶大幅回撤的投资者。",
        "风险": "主要风险：风险平价依赖历史协方差，极端行情下各资产相关性骤升。",
    },
    "方案D · 积极增长型": {
        "人群": "适合人群：风险承受能力高、投资期限长、追求资本增值的激进投资者。",
        "风险": "主要风险：股票占比高，熊市可能出现 30% 以上的回撤，需长期持有消化。",
    },
    "方案E · HRP层级风险平价": {
        "人群": "适合人群：不信任收益预测、希望配置结果对参数误差不敏感的稳健型投资者。",
        "风险": "主要风险：完全不参考预期收益，强趋势行情中可能跑输均值-方差方案。",
    },
}


def _build_plans(mu, cov, tickers, meta, cap=0.4):
    """按角色分组构造 4 套方案：股/债/防御组由 config.yaml 的 role 字段决定。"""
    idx = {t: i for i, t in enumerate(tickers)}
    eq = [idx[t] for t in tickers if meta.get(t, {}).get("role") == "equity"]
    bd = [idx[t] for t in tickers if meta.get(t, {}).get("role") == "bond"]
    defensive = [idx[t] for t in tickers
                 if meta.get(t, {}).get("role") in ("bond", "cash")]

    plans = {}
    if defensive:
        plans["方案A · 稳健保本型"] = opt.optimize_plan_A(
            mu, cov, defensive, vol_limit=0.05, cap=cap)
    if eq and bd:
        plans["方案B · 经典60/40"] = opt.optimize_plan_B(
            mu, cov, eq, bd, rf=RF, cap=cap)
    plans["方案C · 全天候组合"] = opt.risk_parity(cov, cap=cap)
    if eq:
        plans["方案D · 积极增长型"] = opt.optimize_plan_D(
            mu, cov, eq, rf=RF, cap=cap, eq_floor=0.5)
    plans["方案E · HRP层级风险平价"] = opt.hrp(cov, cap=cap)
    return plans


def render():
    cfg = load_config()
    meta_all = pool_meta(cfg)
    kyc_result = st.session_state.get("kyc")
    if kyc_result:
        kyc_result = dict(kyc_result,
                          rules=kyc.LEVEL_RULES[kyc_result["level"]])

    st.header("模式二 · 财富方案推荐")
    st.caption("自动生成 5 套方案并对比（含 HRP 层级风险平价）。资产池、风险等级、费率均可在 config.yaml 中修改。")

    pool_choice = st.radio("选择资产池", ["国内 ETF/QDII 池（人民币可买）",
                                        "美股 ETF 池（需美元账户）"],
                           horizontal=True, key="m2_pool")
    pool = cfg["asset_pool_cn"] if "国内" in pool_choice else cfg["asset_pool"]
    if not pool:
        st.error("config.yaml 中对应资产池为空，无法生成方案。")
        return
    cash_ticker = next((p["ticker"] for p in pool if p.get("role") == "cash"), None)

    with st.expander("查看资产池（具体产品 + 风险等级 + 费率 + 说明）"):
        st.table(pd.DataFrame(
            {"代码": [p["ticker"] for p in pool],
             "具体产品": [p.get("name", p["ticker"]) for p in pool],
             "类别": [p.get("category", "-") for p in pool],
             "风险等级": [f"R{p.get('risk', '?')}" for p in pool],
             "年费率": [f"{p.get('fee', 0):.2%}" for p in pool],
             "说明": [p.get("desc", "") for p in pool]}
        ))

    total_amount = st.number_input("可投入总金额（人民币）", min_value=0.0,
                                   value=100000.0, step=10000.0,
                                   key="m2_amount")
    st.caption("说明：总金额只影响每套方案的「配置金额」换算，不会改变配置比例——"
               "比例由资产间的风险收益结构和你的 KYC 等级决定。")

    # ---------- 适当性提示：KYC 只警示不剔除，资产池内产品全部参与方案生成 ----------
    max_risk = kyc_result["rules"]["max_risk"] if kyc_result else 5
    over_risk = [p for p in pool if p.get("risk", 5) > max_risk]
    if kyc_result and over_risk:
        st.warning(
            f"⚠️ 适当性提示：按你的测评结果（{kyc_result['rules']['label']}），"
            f"以下产品风险等级超过 R{max_risk}："
            + "、".join(f"{p['ticker']}（{p.get('name', '')}，R{p.get('risk', '?')}）"
                        for p in over_risk)
            + "。已按你的要求保留并继续参与方案生成，请知悉其波动可能超出你的风险承受能力。")
    elif kyc_result:
        st.info(f"你的风险测评为 {kyc_result['rules']['label']}，"
                f"当前资产池产品均在 R1~R{max_risk} 适当性范围内。")
    elif over_risk:
        st.caption("完成左侧「风险测评（KYC）」后，超出你风险等级的产品会显示适当性提示（不会剔除）。")
    pool_ok = pool

    if len(pool_ok) < 3:
        st.error("资产池内可用资产不足 3 个，无法生成方案。")
        return

    names = {p["ticker"]: p.get("name", p["ticker"]) for p in pool}
    fees = {p["ticker"]: p.get("fee", 0.0) for p in pool}
    tickers_all = [p["ticker"] for p in pool_ok]
    prices, ok, bad, short, _markets, _limited, _quality = load_with_progress(
        tickers_all, label="正在拉取资产池近 5 年数据")
    for t in bad:
        st.warning(f"找不到 {t}（{names.get(t, t)}）的数据，已从池中剔除。")
    for t in short:
        st.info(f"{t}（{names.get(t, t)}）数据较短，结果参考价值有限。")
    if len(ok) < 3:
        st.error("可用资产太少，无法生成方案，请稍后重试或检查网络。")
        return

    cap = min(st.session_state.get("weight_cap", cfg["weight_cap"]),
              kyc_result["rules"]["cap"] if kyc_result else cfg["weight_cap"])
    mu, cov, _ = opt.estimate_params(prices, shrinkage=cfg["shrinkage"])
    tickers = list(prices.columns)
    # config.yaml 中的前瞻收益假设优先于历史均值
    cfg_ov = {t: meta_all[t]["exp_return"] for t in tickers
              if meta_all.get(t, {}).get("exp_return") is not None}
    if cfg_ov:
        mu = opt.apply_return_overrides(mu, tickers, cfg_ov)
    plans = _build_plans(mu, cov, tickers, meta_all, cap=cap)

    # ---------- 逐套方案展示 ----------
    market = "A股" if "国内" in pool_choice else "美股"
    bench = benchmark_nav(prices.index, market)
    navs = {}
    summary_rows = []
    stress_frames = []
    for plan_name, w in plans.items():
        with st.container(border=True):
            st.subheader(plan_name)
            df_w = pd.DataFrame({
                "具体产品": [names.get(t, t) for t in tickers],
                "类别": [meta_all.get(t, {}).get("category", "-") for t in tickers],
                "风险等级": [f"R{meta_all.get(t, {}).get('risk', '?')}" for t in tickers],
                "配置比例": [f"{x:.1%}" for x in w],
                "配置金额(元)": [f"{x * total_amount:,.0f}" for x in w],
            })
            df_w = df_w[w >= 0.005].reset_index(drop=True)
            st.dataframe(df_w, width="stretch", hide_index=True)

            c1, c2 = st.columns([1, 2])
            with c1:
                st.plotly_chart(pie_chart(tickers, w), width="stretch",
                                key=f"pie_{plan_name}")
            with c2:
                nav = portfolio_nav(prices, w)
                navs[plan_name] = nav
                st.plotly_chart(nav_chart({plan_name: nav},
                                          title=f"{plan_name} 净值 vs 基准（近5年）",
                                          benchmark=bench),
                                width="stretch", key=f"nav_{plan_name}")

            # 蒙特卡洛前景（自助法，含肥尾）
            ann_ret, ann_vol, _, _ = annualized_stats(prices, w, rf=RF)
            port_rets = (prices.pct_change().dropna() * w).sum(axis=1)
            prob, _fan = analytics.monte_carlo_forecast(
                ann_ret, ann_vol, paths=cfg["monte_carlo_paths"],
                hist_rets=port_rets, method=cfg["mc_method"])
            st.metric("未来 1 年正收益概率（蒙特卡洛模拟）", f"{prob:.0%}")

            st.write(auto_analysis(tickers, w, prices, mu, cov, label=plan_name))
            used_hedge = [t for t in tickers
                          if meta_all.get(t, {}).get("role") == "hedge"
                          and w[tickers.index(t)] >= 0.005]
            if used_hedge:
                st.caption("注：" + "、".join(names.get(t, t) for t in used_hedge) +
                           " 为对冲基金复制ETF，与真实对冲基金存在跟踪差异。")
            st.caption(PLAN_META[plan_name]["人群"])
            st.caption(PLAN_META[plan_name]["风险"])

            with st.expander("📊 这套方案的风险从哪来（风险归因 + 风格暴露 + 机构观点）"):
                risk_contribution_section(tickers, w, cov, names=names,
                                          key_prefix=f"m2_{plan_name}")
                factor_section(prices, w, market, key_prefix=f"m2_{plan_name}")
                market_views_section(cfg)

        row = portfolio_summary_row(plan_name, prices, w)
        row["未来1年正收益概率"] = f"{prob:.0%}"
        summary_rows.append(row)

        st_df = analytics.stress_test(prices, w, cfg["stress_scenarios"])
        if not st_df.empty:
            st_df.insert(0, "方案", plan_name)
            stress_frames.append(st_df)

    # ---------- 五套方案横向对比 ----------
    st.subheader("五套方案净值对比（近5年）")
    st.plotly_chart(nav_chart(navs, title="方案 A / B / C / D / E 累计净值对比",
                              benchmark=bench),
                    width="stretch")

    st.subheader("五套方案指标横向对比")
    st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

    # ---------- 压力测试对比 ----------
    st.subheader("压力测试：如果历史极端行情重演，五套方案分别跌多少")
    if stress_frames:
        all_stress = pd.concat(stress_frames, ignore_index=True)
        st.dataframe(all_stress, width="stretch", hide_index=True)
        st.caption("用真实历史区间逐日重放：把每套方案放回那段极端行情，看区间累计表现。")
    else:
        st.caption("当前数据未覆盖压力测试区间。")

    # ---------- 任选一套看蒙特卡洛扇形图 ----------
    with st.expander("🔮 查看某套方案未来 1 年的蒙特卡洛模拟扇形图"):
        pick = st.selectbox("选择方案", list(plans.keys()), key="m2_mc_pick")
        w = plans[pick]
        ann_ret, ann_vol, _, _ = annualized_stats(prices, w, rf=RF)
        port_rets = (prices.pct_change().dropna() * w).sum(axis=1)
        prob, fan = analytics.monte_carlo_forecast(
            ann_ret, ann_vol, paths=cfg["monte_carlo_paths"],
            hist_rets=port_rets, method=cfg["mc_method"])
        st.plotly_chart(mc_fan_chart(fan, title=f"{pick} · 未来 1 年净值分布"),
                        width="stretch", key="m2_mc_fan")
        st.caption(analytics.mc_summary_text(prob, pick))

    # ---------- 择时与动态配置（含交易成本与费率） ----------
    pick_t = st.selectbox("选择要做择时分析的方案", list(plans.keys()),
                          key="m2_timing_pick")
    timing_section(prices, plans[pick_t], pick_t, names=names,
                   cash_ticker=cash_ticker, key_prefix="m2_timing",
                   fees=fees, cost_bps=cfg["trade_cost_bps"])
