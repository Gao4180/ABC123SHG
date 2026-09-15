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
    mc_fan_chart, nav_chart, pie_chart, portfolio_summary_row,
    risk_contribution_section, timing_section,
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

            with st.expander("📊 这套方案的风险从哪来（风险归因 + 风格暴露）"):
                risk_contribution_section(tickers, w, cov, names=names,
                                          key_prefix=f"m2_{plan_name}")
                factor_section(prices, w, market, key_prefix=f"m2_{plan_name}")

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

    # ================= 选定方案执行（三件套） =================
    import datetime as _dt
    import execution
    import timing as _timing

    st.divider()
    st.header("📌 选定方案执行")
    sel = st.selectbox("选择你要执行的方案", list(plans.keys()), key="m2x_plan")
    w_sel = plans[sel]

    # ---------- 买入操作清单（最新价 + 当日趋势信号实时计算） ----------
    with st.expander("📋 买入操作清单（按最新价与今日趋势信号实时计算）",
                     expanded=True):
        plan_df = execution.buy_plan(prices, w_sel, total_amount, "元",
                                     meta_all, fees=fees)
        try:
            sig_df = _timing.trend_signals(prices)
            sig_map = dict(zip(sig_df["代码"], sig_df["当前信号"]))

            def _add_sig(row):
                s = str(sig_map.get(row["资产"], ""))
                extra = ""
                if "观望" in s:
                    extra = "今日趋势信号「观望/减仓」，建议分 2~3 批或延后买入"
                elif "矛盾" in s:
                    extra = "今日趋势信号矛盾，建议分批买入降低择时风险"
                return "；".join(x for x in [row["备注"], extra] if x)

            if not plan_df.empty:
                plan_df["备注"] = plan_df.apply(_add_sig, axis=1)
        except Exception:
            pass                                # 信号失败不影响清单
        st.dataframe(plan_df, width="stretch", hide_index=True)
        st.caption("顺序逻辑：先债后股、同类金额大的先买；A 股已按 100 股一手取整。"
                   "备注中的趋势信号每日随最新行情更新。")

    # ---------- PDF 分析报告 ----------
    import report_pdf
    ann_ret, ann_vol, sharpe, mdd = annualized_stats(prices, w_sel, rf=RF)
    _rc, _ = analytics.risk_contribution(cov, w_sel, tickers)
    w_rows = [[t, names.get(t, t), f"{w_sel[i]:.1%}",
               f"{w_sel[i] * total_amount:,.0f} 元"]
              for i, t in enumerate(tickers) if w_sel[i] >= 0.005]
    pdf_bytes = report_pdf.build_portfolio_pdf(
        title=f"财富配置方案报告 · {sel}",
        subtitle=f"总金额 {total_amount:,.0f} 元 · 资产池：{pool_choice.split('（')[0]}",
        metrics={"预期年化收益": f"{ann_ret:.1%}", "年化波动": f"{ann_vol:.1%}",
                 "夏普比率": f"{sharpe:.2f}", "近5年最大回撤": f"{mdd:.1%}"},
        weight_rows=w_rows,
        nav=portfolio_nav(prices, w_sel),
        benchmark=bench,
        rc_df=_rc,
        analysis_text=auto_analysis(tickers, w_sel, prices, mu, cov, label=sel),
        extra_notes=[PLAN_META[sel]["人群"], PLAN_META[sel]["风险"],
                     "蒙特卡洛模拟、压力测试明细见网页版"],
    )
    st.download_button("📄 下载该方案的 PDF 分析报告", data=pdf_bytes,
                       file_name=f"{sel.split(' ')[0]}_方案报告_{_dt.date.today()}.pdf",
                       mime="application/pdf", key="m2_pdf")

    # ================= 我的策略档案（6 位代码 · 云端保存） =================
    import db

    st.divider()
    st.header("📁 我的策略档案（6 位代码）")
    if not db.available():
        st.info("策略档案需要云端数据库支持，当前尚未配置（SUPABASE_URL / "
                "SUPABASE_KEY），本功能暂不可用；页面其余功能不受影响。")
        return

    code_in = st.session_state.get("m2pin_code")
    profile = db.get_profile(code_in) if code_in else None
    if code_in and profile is None:
        st.session_state.pop("m2pin_code", None)
        code_in = None

    if not code_in:
        t_in, t_new = st.tabs(["🔑 输入代码进入", "✨ 创建新档案"])
        with t_new:
            st.caption("选定一套方案后创建档案，系统会分配一个 6 位数字代码，"
                       "下次凭代码（及口令，如设置）随时回来继续。")
            new_plan = st.selectbox("绑定方案", list(plans.keys()),
                                    key="m2pin_newplan")
            new_pass = st.text_input("设置口令（可选，留空则只凭 6 位代码进入）",
                                     type="password", key="m2pin_newpass")
            if st.button("创建档案并分配代码", key="m2pin_create"):
                code = db.create_profile(new_plan, pool_choice, total_amount,
                                         new_pass)
                if code:
                    st.success(f"✅ 档案创建成功！你的专属代码是：**{code}**")
                    st.warning("请立刻记下这 6 位数字（及口令）。"
                               "为安全起见系统不会再次完整展示，丢失无法找回。")
                    st.session_state["m2pin_code"] = code
                    st.rerun()
                else:
                    st.error("创建失败：云端数据库暂时不可用，请稍后重试。")
        with t_in:
            login_code = st.text_input("6 位数字代码", max_chars=6,
                                       key="m2pin_login")
            login_pass = st.text_input("口令（创建时若未设置则留空）",
                                       type="password", key="m2pin_loginpass")
            if st.button("进入我的档案", key="m2pin_enter"):
                prof = db.get_profile(login_code.strip())
                if prof and db.check_pass(prof, login_pass):
                    st.session_state["m2pin_code"] = login_code.strip()
                    st.rerun()
                elif prof:
                    st.error("口令不正确。")
                else:
                    st.error("代码不存在，请核对后重试。")
        return

    # ---------------- 已登录：档案视图 ----------------
    st.success(f"已进入档案 **{code_in}** · 绑定策略：{profile['plan_name']} · "
               f"计划总资金 {float(profile['total_amount']):,.0f} 元")
    c_out, c_plan = st.columns(2)
    with c_out:
        if st.button("退出档案", key="m2pin_logout"):
            st.session_state.pop("m2pin_code", None)
            st.rerun()
    with c_plan:
        with st.popover("更换绑定策略"):
            chg = st.selectbox("换成", list(plans.keys()), key="m2pin_chg")
            if st.button("确认更换", key="m2pin_chg_btn"):
                if db.update_plan(code_in, chg):
                    st.success("已更换绑定策略。")
                    st.rerun()

    plan_name = profile["plan_name"]
    w_prof = plans.get(plan_name)
    if w_prof is None:
        st.warning(f"该档案绑定的「{plan_name}」属于另一个资产池，"
                   "请切换页面上方的资产池选择后再查看体检。")
        w_prof = None
    prof_tickers = ([t for i, t in enumerate(tickers) if w_prof[i] >= 0.005]
                    if w_prof is not None else [])

    # ---------- 记录买入 ----------
    st.subheader("记录一笔买入")
    if not prof_tickers:
        st.caption("当前资产池下无法解析绑定策略，暂不能记账。")
    else:
        b1, b2, b3 = st.columns([2, 1, 1])
        with b1:
            buy_t = st.selectbox(
                "买入产品", prof_tickers,
                format_func=lambda t: f"{t} · {names.get(t, t)}",
                key="m2pin_buy_t")
        with b2:
            buy_amt = st.number_input("买入金额（元）", 0.0, 1e9, 0.0, 1000.0,
                                      key="m2pin_buy_amt")
        with b3:
            buy_date = st.date_input("买入日期", _dt.date.today(),
                                     key="m2pin_buy_date")
        total_plan = float(profile["total_amount"])
        if total_plan > 0 and buy_amt > 0:
            st.caption(f"该笔占计划总资金的 {buy_amt / total_plan:.1%}")
        if st.button("保存这笔买入", key="m2pin_buy_save"):
            if buy_amt <= 0:
                st.error("金额需大于 0。")
            elif db.add_buy(code_in, buy_t, names.get(buy_t, buy_t),
                            buy_amt, str(buy_date)):
                st.success("已保存。")
                st.rerun()
            else:
                st.error("保存失败：云端数据库暂时不可用。")

    buys = db.list_buys(code_in)
    if buys:
        df_buys = pd.DataFrame(
            {"日期": [b["buy_date"] for b in buys],
             "产品": [f"{b['ticker']} · {b.get('name') or b['ticker']}"
                      for b in buys],
             "金额(元)": [f"{float(b['amount']):,.0f}" for b in buys],
             "占总资金": [f"{float(b['amount']) / total_plan:.1%}"
                          if total_plan > 0 else "-" for b in buys]})
        st.dataframe(df_buys, width="stretch", hide_index=True)
        with st.popover("删除某笔记录"):
            del_id = st.selectbox(
                "选择记录", [b["id"] for b in buys],
                format_func=lambda i: next(
                    f"{b['buy_date']} {b['ticker']} {float(b['amount']):,.0f}元"
                    for b in buys if b["id"] == i),
                key="m2pin_del")
            if st.button("确认删除", key="m2pin_del_btn"):
                if db.delete_buy(del_id):
                    st.success("已删除。")
                    st.rerun()
    else:
        st.caption("还没有买入记录。买入后记得回来记一笔，体检才有意义。")

    # ---------- 投后体检（基于真实买入实时计算） ----------
    if buys and w_prof is not None:
        st.subheader("投后体检（实时）")
        last_px = prices.iloc[-1]
        pos_vals, invested_by_t = {}, {}
        for b in buys:
            t = b["ticker"]
            if t not in prices.columns:
                continue
            s = prices[t].dropna()
            d = pd.Timestamp(b["buy_date"])
            hist = s.loc[:d]
            px0 = float(hist.iloc[-1]) if len(hist) else float(s.iloc[0])
            amt = float(b["amount"])
            shares = amt / px0 if px0 > 0 else 0.0
            pos_vals[t] = pos_vals.get(t, 0.0) + shares * float(last_px[t])
            invested_by_t[t] = invested_by_t.get(t, 0.0) + amt
        tot_val = sum(pos_vals.values())
        tot_inv = sum(invested_by_t.values())

        w_map = {t: float(w_prof[i]) for i, t in enumerate(tickers)}
        rows, alerts = [], []
        for t in sorted(set(prof_tickers) | set(pos_vals)):
            cur_w = pos_vals.get(t, 0.0) / tot_val if tot_val > 0 else 0.0
            tgt_w = w_map.get(t, 0.0)
            drift = cur_w - tgt_w
            pnl = (pos_vals.get(t, 0.0) - invested_by_t.get(t, 0.0))
            rows.append({"产品": f"{t} · {names.get(t, t)}",
                         "已投入(元)": f"{invested_by_t.get(t, 0.0):,.0f}",
                         "当前市值(元)": f"{pos_vals.get(t, 0.0):,.0f}",
                         "浮动盈亏(元)": f"{pnl:+,.0f}",
                         "目标占比": f"{tgt_w:.1%}", "当前占比": f"{cur_w:.1%}",
                         "偏离": f"{drift:+.1%}"})
            if tot_inv > 0 and abs(drift) > 0.05:
                alerts.append(f"{names.get(t, t)} 偏离目标 {drift:+.1%}（超 5%）")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        nav_p = portfolio_nav(prices, w_prof)
        cur_dd = float(nav_p.iloc[-1] / nav_p.cummax().iloc[-1] - 1)
        if cur_dd < -0.10:
            alerts.append(f"策略当前回撤 {cur_dd:.1%}，超过 10% 警戒线")
        try:
            sig2 = _timing.trend_signals(prices)
            for _, r in sig2.iterrows():
                if r["代码"] in pos_vals and "观望" in str(r["当前信号"]):
                    alerts.append(f"{r['代码']} 趋势信号「观望/减仓」，"
                                  "下次加仓可暂缓该资产")
        except Exception:
            pass

        m1, m2c, m3c = st.columns(3)
        m1.metric("持仓总市值", f"{tot_val:,.0f} 元")
        m2c.metric("累计盈亏", f"{tot_val - tot_inv:+,.0f} 元")
        m3c.metric("策略当前回撤", f"{cur_dd:.1%}")
        if alerts:
            for a in alerts:
                st.warning("⚠️ " + a)
        else:
            st.success("持仓健康：偏离均在阈值内，无风险预警。")

        # ---------- 新钱智能再平衡（只买不卖，基于真实持仓） ----------
        st.markdown("**💰 新钱怎么加**")
        new_money = st.number_input("本次要新增投入（元）", 0.0, 1e9, 0.0,
                                    5000.0, key="m2pin_newmoney")
        if new_money > 0 and tot_val > 0:
            tick_all = [t for t in tickers
                        if w_map.get(t, 0) > 0 or t in pos_vals]
            tgt_w = np.array([w_map.get(t, 0.0) for t in tick_all])
            cur_w = np.array([pos_vals.get(t, 0.0) / tot_val
                              for t in tick_all])
            alloc_df, leftover = execution.allocate_new_money(
                tick_all, tgt_w, cur_w, tot_val, new_money)
            if alloc_df.empty:
                st.info("当前偏离很小，这笔新钱可暂留现金或按比例均匀加仓。")
            else:
                st.dataframe(alloc_df, width="stretch", hide_index=True)
                if leftover > 1:
                    st.caption(f"买入后剩余 {leftover:,.0f} 元留作现金。")
            st.caption("逻辑：新钱优先补最低配的资产，尽量不靠卖出再平衡"
                       "（Betterment 式 smart rebalancing）。")

