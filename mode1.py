# -*- coding: utf-8 -*-
"""
模式一：我的资产优化
用户输入自有资产代码 + 总金额 + 投资期限，KYC 定级自动约束风险档位与权重上限，
输出马科维茨优化结果，支持：预期收益假设编辑、Black-Litterman 择时观点融合、
权重微调、肥尾蒙特卡洛前景模拟、历史情景压力测试、再平衡偏离提醒。

防"闪退"设计：Streamlit 里任何控件交互都会整页重跑，表单提交标志只在
提交那一轮为 True。因此提交后把输入存入 st.session_state["m1_ctx"]，
后续滑块/勾选/改数字触发重跑时按存档继续渲染，页面不再跳回初始状态。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import analytics
import kyc
import optimizer as opt
import timing
from data_loader import load_config, pool_meta, portfolio_nav, annualized_stats
from report import (
    RF, amount_table, asset_metrics_table, auto_analysis,
    load_with_progress, mc_fan_chart, nav_chart, pie_chart, timing_section,
)

RISK_MODES = {
    "保守（最小方差）": "min_var",
    "平衡（最大夏普）": "max_sharpe",
    "激进（波动≤25%下收益最大）": "max_ret",
}


def _metrics_row(prices, weights, prefix=""):
    """四个核心指标卡。"""
    ann_ret, ann_vol, sharpe, mdd = annualized_stats(prices, weights, rf=RF)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("预期年化收益", f"{ann_ret:.1%}")
    c2.metric("年化波动", f"{ann_vol:.1%}")
    c3.metric("夏普比率", f"{sharpe:.2f}")
    c4.metric("近5年最大回撤", f"{mdd:.1%}")
    return ann_ret, ann_vol, sharpe, mdd


def _mc_and_stress(prices, weights, cfg, label, key_prefix):
    """蒙特卡洛前景 + 历史情景压力测试（通用区块）。"""
    ann_ret, ann_vol, _, _ = annualized_stats(prices, weights, rf=RF)
    rets = prices.pct_change().dropna()
    port_rets = (rets * np.asarray(weights)).sum(axis=1)

    st.subheader("未来 1 年前景模拟（蒙特卡洛）")
    method_label = st.selectbox(
        "模拟方法", ["自助法（历史残差重抽样，含肥尾）", "正态假设（几何布朗运动）"],
        key=f"{key_prefix}_method")
    method = "bootstrap" if "自助法" in method_label else "gbm"
    prob, fan = analytics.monte_carlo_forecast(
        ann_ret, ann_vol, paths=cfg["monte_carlo_paths"],
        hist_rets=port_rets, method=method)
    col_txt, col_fan = st.columns([1, 2])
    with col_txt:
        st.metric("未来 1 年正收益概率", f"{prob:.0%}")
        st.caption(analytics.mc_summary_text(prob, label) +
                   ("自助法保留了真实历史分布的肥尾，极端亏损概率通常比正态假设更高。"
                    if method == "bootstrap" else ""))
    with col_fan:
        st.plotly_chart(mc_fan_chart(fan), width="stretch",
                        key=f"{key_prefix}_fan")

    st.subheader("压力测试：如果历史极端行情重演")
    df = analytics.stress_test(prices, weights, cfg["stress_scenarios"])
    if df.empty:
        st.caption("当前数据未覆盖压力测试区间。")
    else:
        st.dataframe(df, width="stretch", hide_index=True)
        st.caption("用真实历史区间逐日重放：把你的组合放回那段极端行情，看会跌多少。")


def render():
    cfg = load_config()
    meta = pool_meta(cfg)
    fees = {t: m.get("fee", 0.0) for t, m in meta.items()}
    kyc_result = st.session_state.get("kyc")
    if kyc_result:
        kyc_result = dict(kyc_result,
                          rules=kyc.LEVEL_RULES[kyc_result["level"]])

    st.header("模式一 · 我的资产优化")
    st.caption("输入你持有或想买的资产（A 股直接输 6 位代码即可，自动识别沪深），"
               "用马科维茨均值-方差模型算出最优配置比例。"
               "提交后随意拖动滑块、改假设，页面不会再跳回初始状态。")

    # ---------- 输入区（表单） ----------
    with st.form("optimize_form"):
        tickers_raw = st.text_input(
            "资产代码或名称（逗号分隔混输：苹果、贵州茅台、沪深300ETF、SPY、600519 都可以）",
            value="SPY, GLD, TLT, QQQ",
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            total_amount = st.number_input("可投入总金额", min_value=0.0,
                                           value=100000.0, step=10000.0)
        with col2:
            currency = st.selectbox("币种", ["人民币", "美元"])
        with col3:
            horizon = st.number_input("投资期限（年）", min_value=0.5,
                                      max_value=30.0, value=5.0, step=0.5)

        # KYC 定级约束风险档位
        if kyc_result:
            level, note = kyc.horizon_rule(horizon, kyc_result["level"])
            if note:
                st.warning("期限规则：" + note)
            allowed = kyc.allowed_modes(level)
            options = [l for l, m in RISK_MODES.items() if m in allowed]
            rules = kyc.LEVEL_RULES[level]
            risk_label = st.radio(
                f"风险档位（{rules['label']}，可选范围已按测评结果收窄）", options)
        else:
            st.info("建议先在左侧边栏完成「风险测评（KYC）」，系统将自动约束可选风险档位。")
            risk_label = st.radio("风险档位", list(RISK_MODES.keys()))
        submitted = st.form_submit_button("开始计算", type="primary")

    if submitted:
        st.session_state["m1_ctx"] = {
            "tickers_raw": tickers_raw, "total_amount": total_amount,
            "currency": currency, "horizon": horizon, "risk_label": risk_label,
        }
    ctx = st.session_state.get("m1_ctx")
    if ctx is None:
        st.info("填写上方信息后点击「开始计算」。"
                "提示：总金额只影响金额换算，不改变配置比例；比例上限由风险测评等级决定。")
        return

    raw_tokens = [t.strip() for t in ctx["tickers_raw"].replace("，", ",").split(",")
                  if t.strip()]
    if len(raw_tokens) < 2:
        st.warning("请至少输入 2 个有效资产，然后重新点击「开始计算」。")
        return

    # 名字/代码混输：名字经腾讯联想解析成代码
    from data_loader import resolve_tokens
    tickers, mapping, unresolved = resolve_tokens(raw_tokens)
    for raw, code, name in mapping:
        st.caption(f"🔎 名字识别：{raw} → **{code}**（{name}）")
    for u in unresolved:
        st.warning(f"无法识别「{u}」，已跳过。建议改用代码输入（如 600519 或 SPY）。")
    if len(tickers) < 2:
        st.warning("识别后有效资产不足 2 个，请补充后重新点击「开始计算」。")
        return

    # ---------- 数据拉取与校验 ----------
    prices, ok, bad, short, markets, _limited, _quality = load_with_progress(
        tickers, label="正在拉取近 5 年日线数据")

    for t in bad:
        st.warning(f"找不到 {t} 的数据，已跳过。")
    for t in short:
        st.info(f"{t} 历史数据较短（不足3年），结果参考价值有限。")
    mset = set(markets.get(t) for t in ok)
    if "A股" in mset and "美股" in mset:
        st.info("提示：你混合输入了 A 股与美元资产，两者币种、交易制度不同，"
                "下表金额换算未做汇率与税费处理，请自行折算。")

    if len(ok) < 2:
        st.error("有效资产不足 2 个，无法优化。请检查代码后重新点击「开始计算」。")
        return

    # ---------- 适当性匹配：KYC 等级限制可买产品 ----------
    if kyc_result:
        max_risk = kyc.LEVEL_RULES[kyc_result["level"]]["max_risk"]
        blocked = [t for t in prices.columns
                   if meta.get(t, {}).get("risk", 5) > max_risk]
        if blocked:
            prices = prices.drop(columns=blocked)
            st.warning(f"适当性匹配：{kyc.LEVEL_RULES[kyc_result['level']]['label']} "
                       f"客户不应持有 R{max_risk} 以上产品，已剔除："
                       + "、".join(f"{t}（R{meta.get(t, {}).get('risk', '?')}）"
                                   for t in blocked))
        if len(prices.columns) < 2:
            st.error("剔除不合规产品后剩余资产不足 2 个，请调整输入或重新测评。")
            return

    # ---------- 参数估计 ----------
    cap = min(st.session_state.get("weight_cap", cfg["weight_cap"]),
              kyc_result["rules"]["cap"] if kyc_result else cfg["weight_cap"])
    mu, cov, _ = opt.estimate_params(prices, shrinkage=cfg["shrinkage"])
    tickers_ok = list(prices.columns)
    # config.yaml 中的前瞻收益假设优先于历史均值
    cfg_ov = {t: meta[t]["exp_return"] for t in tickers_ok
              if meta.get(t, {}).get("exp_return") is not None}
    if cfg_ov:
        mu = opt.apply_return_overrides(mu, tickers_ok, cfg_ov)

    # ---------- 预期收益假设：BL 观点 + 可编辑输入网格 ----------
    with st.expander("📈 预期年化收益假设（默认=历史均值，直接改数字即可，改动即时生效）"):
        use_bl = st.checkbox("融合择时信号观点（Black-Litterman）", value=False,
                             key="m1_use_bl",
                             help="跌破均线/动量转负的资产，预期收益向无风险利率收缩")
        mu_base = mu
        if use_bl:
            sig_df = timing.trend_signals(prices)
            sig_off = {r["代码"] for _, r in sig_df.iterrows()
                       if "观望" in r["当前信号"]}
            if sig_off:
                st.caption("当前处于观望信号的资产：" + "、".join(sorted(sig_off)))
            mu_base = opt.black_litterman(mu, cov, tickers_ok, sig_off, rf=RF,
                                          tau=cfg["bl_tau"],
                                          view_shrink=cfg["bl_view_shrink"])
        cols = st.columns(min(len(tickers_ok), 4))
        overrides = {}
        for i, t in enumerate(tickers_ok):
            cum5y = float(prices[t].iloc[-1] / prices[t].iloc[0] - 1)
            with cols[i % len(cols)]:
                overrides[t] = st.number_input(
                    f"{t}（%）", value=round(float(mu_base[i]) * 100, 2),
                    step=0.5, format="%.2f", key=f"mu_in_{t}",
                    help=f"参考：历史年化均值 {mu[i]:.1%}，近 5 年累计涨幅 {cum5y:+.0%}。"
                         "未来收益由你判断，改数字即重新优化") / 100.0
        mu = opt.apply_return_overrides(mu_base, tickers_ok, overrides)

    # ---------- 优化计算 ----------
    mode = RISK_MODES[ctx["risk_label"]]
    if mode == "min_var":
        weights = opt.min_variance(mu, cov, cap=cap)
    elif mode == "max_sharpe":
        weights = opt.max_sharpe(mu, cov, rf=RF, cap=cap)
    else:
        weights = opt.max_return_capped_vol(mu, cov, vol_limit=0.25, cap=cap)

    # ---------- 输出区 ----------
    _metrics_row(prices, weights)

    st.subheader("配置比例与单资产指标")
    st.dataframe(asset_metrics_table(prices, weights), width="stretch",
                 hide_index=True)

    st.subheader("金额换算（按最新收盘价取整股，含年管理费估算）")
    st.dataframe(amount_table(prices, weights, ctx["total_amount"],
                              ctx["currency"], fees=fees),
                 width="stretch", hide_index=True)

    col_pie, col_nav = st.columns([1, 2])
    with col_pie:
        st.plotly_chart(pie_chart(prices.columns, weights), width="stretch")
    with col_nav:
        nav_opt = portfolio_nav(prices, weights)
        n_assets = len(prices.columns)
        nav_eq = portfolio_nav(prices, np.full(n_assets, 1 / n_assets))
        st.plotly_chart(
            nav_chart({"优化组合": nav_opt, "等权组合": nav_eq},
                      title="组合累计净值 vs 等权组合（近5年）"),
            width="stretch")

    st.subheader("自动分析")
    st.write(auto_analysis(list(prices.columns), weights, prices, mu, cov,
                           label=f"{ctx['risk_label'].split('（')[0]}档组合"))

    # ---------- 再平衡提醒（机构纪律：偏离 >5% 触发） ----------
    with st.expander("⚖️ 再平衡检查：输入你当前的实际持仓比例"):
        cur = []
        cols = st.columns(min(len(tickers_ok), 3))
        for i, t in enumerate(tickers_ok):
            with cols[i % len(cols)]:
                cur.append(st.number_input(f"{t} 当前占比", 0.0, 1.0, 0.0, 0.01,
                                           key=f"rebal_{t}"))
        s = sum(cur)
        if s > 0:
            cur_w = np.array(cur) / s
            dev = cur_w - weights
            rows = [{"资产": t, "目标比例": f"{weights[i]:.1%}",
                     "当前比例": f"{cur_w[i]:.1%}", "偏离": f"{dev[i]:+.1%}",
                     "操作": "🔴 需要调仓" if abs(dev[i]) > 0.05 else "🟢 无需操作"}
                    for i, t in enumerate(tickers_ok)]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            n_adj = int((np.abs(dev) > 0.05).sum())
            if n_adj:
                st.warning(f"{n_adj} 个资产偏离目标超过 5%，建议再平衡回目标比例。")
            else:
                st.success("持仓未显著偏离目标，无需调仓。")
        else:
            st.caption("全部填 0 表示暂不检查。")

    # ---------- 权重微调 ----------
    with st.expander("🔧 手动微调权重（拖动滑块，实时看指标变化）"):
        tickers_ = list(prices.columns)
        raw_w = []
        cols = st.columns(min(len(tickers_), 3))
        for i, t in enumerate(tickers_):
            with cols[i % len(cols)]:
                raw_w.append(st.slider(
                    t, 0.0, 1.0, float(round(weights[i], 2)), 0.01,
                    key=f"adj_{t}"))
        total = sum(raw_w)
        if total <= 0:
            st.warning("权重之和为 0，请至少给一个资产非零权重。")
        else:
            adj = np.array(raw_w) / total       # 自动归一化
            st.caption(f"滑块之和 {total:.0%}，已自动归一化为 100%。")
            _metrics_row(prices, adj)
            _mc_and_stress(prices, adj, cfg, "微调后的组合", "m1_adj")

    # ---------- 优化组合的前景与压力测试 ----------
    _mc_and_stress(prices, weights, cfg,
                   f"{ctx['risk_label'].split('（')[0]}档组合", "m1_opt")

    # ---------- 择时与动态配置（含交易成本与费率） ----------
    cash_t = "SHV" if "SHV" in prices.columns else \
             "511880.SS" if "511880.SS" in prices.columns else None
    timing_section(prices, weights, "你的优化组合", cash_ticker=cash_t,
                   key_prefix="m1_timing", fees=fees,
                   cost_bps=cfg["trade_cost_bps"])
