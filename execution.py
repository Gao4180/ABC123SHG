# -*- coding: utf-8 -*-
"""
执行辅助模块（P2）：把"配置比例"变成"能照着下单的动作"。
- buy_plan：买入操作清单（先债后股、先大额后小额，含 QDII 溢价等提示）
- allocate_new_money：新钱智能再平衡（Betterment 式，只买不卖）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 买入顺序：先防御后进攻。同类中金额大的先买（大头先就位，择时影响小）
_ROLE_ORDER = {"cash": 0, "bond": 1, "gold": 2, "commodity": 3,
               "hedge": 4, "equity": 5}


def buy_plan(prices: pd.DataFrame, weights, total_amount: float,
             currency: str, meta: dict, fees: dict | None = None) -> pd.DataFrame:
    """
    生成买入操作清单。
    列：顺序 / 资产 / 具体产品 / 类别 / 配置金额 / 最新价 / 建议买入 / 预估占用 / 备注
    """
    last = prices.iloc[-1]
    rows = []
    for i, t in enumerate(prices.columns):
        w = float(weights[i])
        if w < 0.005:
            continue
        amt = w * total_amount
        price = float(last[t])
        shares = int(amt // price) if price > 0 else 0
        is_ashare = t.upper().endswith((".SS", ".SZ"))
        if is_ashare and shares >= 100:
            shares = (shares // 100) * 100     # A股按 100 股一手取整
        elif is_ashare and 0 < shares < 100:
            shares = 0
        m = meta.get(t, {})
        cat = str(m.get("category", "-"))
        notes = []
        if "QDII" in cat.upper():
            notes.append("QDII 可能溢价，买入前对比 IOPV/净值，溢价>2% 建议等回落")
        if is_ashare:
            notes.append("A股按 100 股一手，已向下取整")
        if shares == 0:
            notes.append("金额不足 1 手/股，可累积几期再买")
        if m.get("risk", 0) >= 4:
            notes.append("波动较大，可分 2 批间隔买入")
        fee_rate = (fees or {}).get(t, 0.0)
        rows.append({
            "_role": _ROLE_ORDER.get(m.get("role", "equity"), 9),
            "_amt": amt,
            "资产": t,
            "具体产品": m.get("name", t),
            "类别": cat,
            "配置金额": f"{amt:,.0f} {currency}",
            "最新价": f"{price:,.2f}",
            "建议买入": f"{shares} 股",
            "预估占用": f"{shares * price:,.0f} {currency}",
            "年管理费约": f"{amt * fee_rate:,.0f}" if fee_rate else "-",
            "备注": "；".join(notes),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = (df.sort_values(["_role", "_amt"], ascending=[True, False])
            .drop(columns=["_role", "_amt"]).reset_index(drop=True))
    df.insert(0, "顺序", df.index + 1)
    return df


def allocate_new_money(tickers, weights, current_weights,
                       portfolio_value: float, new_money: float) -> pd.DataFrame:
    """
    新钱智能再平衡（Betterment 的核心机制）：只买不卖。
    目标市值_i = 目标权重_i × (当前总市值 + 新钱)
    买入_i = max(0, 目标市值_i − 当前市值_i)，若合计超过新钱则等比缩放，
    不足则剩余留作现金（说明偏离已经很小）。
    返回 DataFrame：资产 / 当前市值 / 目标市值 / 本次买入 / 买入后占比
    """
    w = np.asarray(weights, dtype=float)
    cw = np.asarray(current_weights, dtype=float)
    w = w / w.sum()
    cw = cw / cw.sum()
    total_after = portfolio_value + new_money
    target_vals = w * total_after
    cur_vals = cw * portfolio_value
    buys = np.maximum(0.0, target_vals - cur_vals)
    s = buys.sum()
    if s > new_money and s > 0:
        buys = buys * (new_money / s)
    rows = []
    for i, t in enumerate(tickers):
        rows.append({
            "资产": t,
            "当前占比": f"{cw[i]:.1%}",
            "目标占比": f"{w[i]:.1%}",
            "本次买入": f"{buys[i]:,.0f} 元",
            "买入后占比": f"{(cur_vals[i] + buys[i]) / total_after:.1%}"
            if total_after > 0 else "-",
        })
    df = pd.DataFrame(rows)
    return df[df["本次买入"] != "0 元"].reset_index(drop=True), float(new_money - buys.sum())
