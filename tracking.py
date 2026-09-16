# -*- coding: utf-8 -*-
"""
档案通用组件（三模块复用）：
- sidebar_login：侧栏登录/创建/退出（可选，游客跳过）
- positions_from_buys：买入记录 → 当前持仓市值
- render_archive：模块级「记账 + 持仓记录 + 实时体检 + 新钱再平衡」
策略的保存/恢复由各模块自行调用 db.save_strategy / db.get_strategy（payload 结构不同）。
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st

import db
import execution
import timing
from data_loader import portfolio_nav


# ---------------- 侧栏登录 ----------------

def sidebar_login() -> str | None:
    """侧栏档案区（可选登录）。返回已登录的 6 位代码或 None。"""
    logged = st.session_state.get("pin_code")
    with st.sidebar.expander("📁 我的策略档案（可选登录）", expanded=True):
        if not db.available():
            st.caption("档案云端服务未配置，当前为游客模式，"
                       "全部计算功能可正常使用。")
            return None
        if logged:
            st.success(f"档案代码：**{logged}**")
            st.caption("请记下这 6 位数字，下次凭它（及口令）登录。"
                       "三个模块会各自记住你的策略与买入记录。")
            if st.button("退出档案", key="pin_logout"):
                st.session_state.pop("pin_code", None)
                st.rerun()
            return logged
        st.caption("登录后三个模块各自保存你的策略、买入记录并做投后体检；"
                   "不登录也能使用全部计算功能。")
        t_in, t_new = st.tabs(["🔑 进入", "✨ 创建"])
        with t_in:
            c = st.text_input("6 位数字代码", max_chars=6, key="pin_login_code")
            p = st.text_input("口令（未设置则留空）", type="password",
                              key="pin_login_pass")
            if st.button("进入我的档案", key="pin_enter"):
                try:
                    prof = db.get_profile(c.strip())
                except Exception as e:  # noqa: BLE001
                    st.error(f"云端访问异常：{type(e).__name__}: {e}")
                    prof = None
                if prof and db.check_pass(prof, p):
                    st.session_state["pin_code"] = c.strip()
                    st.rerun()
                elif prof:
                    st.error("口令不正确。")
                elif prof is None and c.strip():
                    st.error("代码不存在，请核对后重试。")
        with t_new:
            p2 = st.text_input("设置口令（可选，留空则只凭代码进入）",
                               type="password", key="pin_new_pass")
            if st.button("创建档案并分配代码", key="pin_create"):
                try:
                    new_code = db.create_profile(p2)
                except Exception as e:  # noqa: BLE001
                    new_code = None
                    st.error(f"创建异常：{type(e).__name__}: {e}")
                if new_code:
                    st.session_state["pin_code"] = new_code
                    st.rerun()
                else:
                    st.error("创建失败：云端暂时不可用，请稍后重试。")
    return st.session_state.get("pin_code")


# ---------------- 持仓折算 ----------------

def positions_from_buys(buys: list[dict], prices: pd.DataFrame):
    """每笔买入按当日收盘价折成股数，股数 × 最新价 = 当前市值。
    返回 (pos_vals, invested_by_t)：{代码: 当前市值} 与 {代码: 累计投入}。"""
    if prices is None or prices.empty:
        return {}, {}
    last_px = prices.iloc[-1]
    pos_vals, invested = {}, {}
    for b in buys:
        t = b["ticker"]
        if t not in prices.columns:
            continue
        s = prices[t].dropna()
        if s.empty:
            continue
        hist = s.loc[:pd.Timestamp(b["buy_date"])]
        px0 = float(hist.iloc[-1]) if len(hist) else float(s.iloc[0])
        amt = float(b["amount"])
        shares = amt / px0 if px0 > 0 else 0.0
        pos_vals[t] = pos_vals.get(t, 0.0) + shares * float(last_px[t])
        invested[t] = invested.get(t, 0.0) + amt
    return pos_vals, invested


# ---------------- 模块级档案区 ----------------

def render_archive(code: str | None, mode: int, target_map: dict,
                   prices: pd.DataFrame, names: dict, product_tickers: list,
                   total_plan: float = 0.0, key_prefix: str = "arch"):
    """
    记账 + 买入记录 + 实时体检（偏离/回撤/信号）+ 新钱再平衡。
    target_map：{代码: 目标权重}；product_tickers：允许记账的产品列表。
    """
    st.subheader("📁 档案：买入记账与投后体检")
    if not code:
        st.info("在左侧边栏「我的策略档案」登录或创建代码后，这里会记住你的买入记录，"
                "并基于真实持仓实时体检。游客模式不影响上方任何计算功能。")
        return

    # ---------- 记录一笔买入 ----------
    if product_tickers:
        b1, b2, b3 = st.columns([2, 1, 1])
        with b1:
            buy_t = st.selectbox(
                "买入产品", product_tickers,
                format_func=lambda t: f"{t} · {names.get(t, t)}",
                key=f"{key_prefix}_buy_t")
        with b2:
            buy_amt = st.number_input("买入金额（元）", 0.0, 1e9, 0.0, 1000.0,
                                      key=f"{key_prefix}_buy_amt")
        with b3:
            buy_date = st.date_input("买入日期", dt.date.today(),
                                     key=f"{key_prefix}_buy_date")
        if total_plan > 0 and buy_amt > 0:
            st.caption(f"该笔占计划总资金的 {buy_amt / total_plan:.1%}")
        if st.button("保存这笔买入", key=f"{key_prefix}_buy_save"):
            if buy_amt <= 0:
                st.error("金额需大于 0。")
            elif db.add_buy(code, mode, buy_t, names.get(buy_t, buy_t),
                            buy_amt, str(buy_date)):
                st.success("已保存。")
                st.rerun()
            else:
                st.error("保存失败：云端暂时不可用。")

    buys = db.list_buys(code, mode)
    if not buys:
        st.caption("还没有买入记录。实际买入后记得回来记一笔，体检才有意义。")
        return

    df_buys = pd.DataFrame(
        {"日期": [b["buy_date"] for b in buys],
         "产品": [f"{b['ticker']} · {b.get('name') or b['ticker']}" for b in buys],
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
            key=f"{key_prefix}_del")
        if st.button("确认删除", key=f"{key_prefix}_del_btn"):
            if db.delete_buy(del_id):
                st.success("已删除。")
                st.rerun()

    # ---------- 实时体检 ----------
    pos_vals, invested = positions_from_buys(buys, prices)
    if not pos_vals:
        st.warning("买入记录中的资产本次未能取到行情，体检暂不可用，请稍后刷新。")
        return
    tot_val = sum(pos_vals.values())
    tot_inv = sum(invested.values())

    rows, alerts = [], []
    for t in sorted(set(product_tickers) | set(pos_vals)):
        cur_w = pos_vals.get(t, 0.0) / tot_val if tot_val > 0 else 0.0
        tgt_w = float(target_map.get(t, 0.0))
        drift = cur_w - tgt_w
        pnl = pos_vals.get(t, 0.0) - invested.get(t, 0.0)
        rows.append({"产品": f"{t} · {names.get(t, t)}",
                     "已投入(元)": f"{invested.get(t, 0.0):,.0f}",
                     "当前市值(元)": f"{pos_vals.get(t, 0.0):,.0f}",
                     "浮动盈亏(元)": f"{pnl:+,.0f}",
                     "目标占比": f"{tgt_w:.1%}", "当前占比": f"{cur_w:.1%}",
                     "偏离": f"{drift:+.1%}"})
        if tot_inv > 0 and abs(drift) > 0.05:
            alerts.append(f"{names.get(t, t)} 偏离目标 {drift:+.1%}（超 5% 阈值）")
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # 策略当前回撤（按目标权重组合净值）
    cur_dd = None
    tw = np.array([target_map.get(c, 0.0) for c in prices.columns], dtype=float)
    if tw.sum() > 0 and prices.shape[1] >= 2:
        nav_p = portfolio_nav(prices, tw / tw.sum())
        cur_dd = float(nav_p.iloc[-1] / nav_p.cummax().iloc[-1] - 1)
        if cur_dd < -0.10:
            alerts.append(f"策略当前回撤 {cur_dd:.1%}，超过 10% 警戒线")
    try:
        sig = timing.trend_signals(prices)
        for _, r in sig.iterrows():
            if r["代码"] in pos_vals and "观望" in str(r["当前信号"]):
                alerts.append(f"{r['代码']} 趋势信号「观望/减仓」，下次加仓可暂缓该资产")
    except Exception:
        pass

    m1, m2, m3 = st.columns(3)
    m1.metric("持仓总市值", f"{tot_val:,.0f} 元")
    m2.metric("累计盈亏", f"{tot_val - tot_inv:+,.0f} 元")
    m3.metric("策略当前回撤", f"{cur_dd:.1%}" if cur_dd is not None else "-")
    if alerts:
        for a in alerts:
            st.warning("⚠️ " + a)
    else:
        st.success("持仓健康：偏离均在阈值内，无风险预警。")

    # ---------- 新钱智能再平衡（只买不卖） ----------
    st.markdown("**💰 新钱怎么加**")
    new_money = st.number_input("本次要新增投入（元）", 0.0, 1e9, 0.0, 5000.0,
                                key=f"{key_prefix}_newmoney")
    if new_money > 0 and tot_val > 0:
        tick_all = [t for t in sorted(set(product_tickers) | set(pos_vals))]
        tgt_w = np.array([target_map.get(t, 0.0) for t in tick_all])
        cur_w = np.array([pos_vals.get(t, 0.0) / tot_val for t in tick_all])
        if tgt_w.sum() > 0:
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
