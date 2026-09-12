# -*- coding: utf-8 -*-
"""
KYC 风险测评模块（对标券商基金投顾第一步）
8 道问卷 → 得分 → C1~C5 风险等级 → 自动锁定：
  - 允许的最高风险档位（min_var / max_sharpe / max_ret）
  - 单资产权重上限
  - 可购买的产品风险等级上限（R1~R5，适当性匹配）
另有投资期限规则：期限越短，强制越保守。
"""
from __future__ import annotations

import streamlit as st

# (题干, [(选项, 得分)]...)  得分 1~5，总分映射 C1~C5
QUESTIONS = [
    ("您的年龄段是？",
     [("60 岁以上", 1), ("46-60 岁", 2), ("31-45 岁", 4), ("18-30 岁", 5)]),
    ("您的主要收入来源稳定性如何？",
     [("无固定收入", 1), ("收入不稳定（自由职业/生意波动大）", 2),
      ("收入稳定（固定工资）", 4), ("收入稳定且显著高于支出", 5)]),
    ("您的投资经验？",
     [("几乎为零", 1), ("只买过存款/余额宝/银行理财", 2),
      ("买过债券基金/固收+", 3), ("买过股票/股票基金，3 年以内", 4),
      ("5 年以上股票/基金/衍生品投资经验", 5)]),
    ("这笔钱一年内出现多大亏损，您会觉得无法接受？",
     [("不能承受任何本金损失", 1), ("5% 以内", 2), ("10% 以内", 3),
      ("20% 以内", 4), ("30% 以上也能接受", 5)]),
    ("这笔投资的钱大概什么时候要用？",
     [("1 年内要用", 1), ("1-3 年", 2), ("3-5 年", 3),
      ("5-10 年", 4), ("10 年以上", 5)]),
    ("您对股票、债券、基金等投资知识的了解程度？",
     [("完全不了解", 1), ("听说过但不清楚区别", 2),
      ("了解基本概念", 3), ("能看懂基金季报和产品说明书", 4),
      ("熟悉估值、久期、波动率等指标", 5)]),
    ("您的家庭负担情况？",
     [("上有老下有小，且无应急存款", 1), ("家庭负担重，应急存款不足半年开支", 2),
      ("有负担，应急存款够 6-12 个月", 3),
      ("负担较轻，应急存款充足", 4), ("单身或无负担，另有充足保障", 5)]),
    ("如果您的持仓一个月跌了 15%，您会怎么做？",
     [("立即全部卖出，不再碰", 1), ("卖出一部分止损", 2),
      ("不动，等回本", 3), ("按计划再平衡", 4), ("加仓，认为是机会", 5)]),
]

# C 级 → 规则：最高风险档位 / 权重上限 / 可买产品风险等级 / 描述
LEVEL_RULES = {
    1: {"mode": "min_var",    "cap": 0.20, "max_risk": 1,
        "label": "C1 保守型", "desc": "本金安全第一，只适合现金管理类产品"},
    2: {"mode": "min_var",    "cap": 0.30, "max_risk": 2,
        "label": "C2 稳健型", "desc": "可配债券类，组合波动压到最低"},
    3: {"mode": "max_sharpe", "cap": 0.40, "max_risk": 3,
        "label": "C3 平衡型", "desc": "股债均衡，可配黄金/对冲类，追求风险调整收益"},
    4: {"mode": "max_sharpe", "cap": 0.50, "max_risk": 4,
        "label": "C4 进取型", "desc": "可配大盘股票与商品，波动承受力较强"},
    5: {"mode": "max_ret",    "cap": 0.60, "max_risk": 5,
        "label": "C5 激进型", "desc": "可配高波动成长资产，追求长期资本增值"},
}

_MODE_ORDER = {"min_var": 0, "max_sharpe": 1, "max_ret": 2}
MODE_LABELS = {"min_var": "保守（最小方差）", "max_sharpe": "平衡（最大夏普）",
               "max_ret": "激进（波动≤25%下收益最大）"}


def score_to_level(score: int) -> int:
    """8 题总分 8~40 映射到 C1~C5。"""
    if score <= 14:
        return 1
    if score <= 20:
        return 2
    if score <= 26:
        return 3
    if score <= 33:
        return 4
    return 5


def horizon_rule(years: float, level: int) -> tuple[int, str | None]:
    """
    期限规则：期限越短强制越保守（机构标准做法）。
    返回 (调整后的等级, 说明)；无调整时说明为 None。
    """
    if years < 2 and level > 2:
        return 2, f"投资期限不足 2 年（{years:g} 年），风险等级强制降为 C2 稳健型"
    if years < 5 and level > 3:
        return 3, f"投资期限不足 5 年（{years:g} 年），风险等级最高为 C3 平衡型"
    return level, None


def allowed_modes(level: int) -> list[str]:
    """该等级允许选择的风险档位（不超过定级档位）。"""
    top = _MODE_ORDER[LEVEL_RULES[level]["mode"]]
    return [m for m, o in _MODE_ORDER.items() if o <= top]


def kyc_sidebar_widget(key_prefix="kyc") -> dict | None:
    """
    在 sidebar 渲染 KYC 问卷（expander 内），完成过则显示等级徽章。
    返回 {score, level, rules} 或 None（未做测评）。
    """
    st.session_state.setdefault("kyc", None)

    with st.sidebar.expander("📋 风险测评（KYC）", expanded=st.session_state["kyc"] is None):
        if st.session_state["kyc"]:
            r = st.session_state["kyc"]
            st.success(f"{LEVEL_RULES[r['level']]['label']}（{r['score']} 分）")
            if st.button("重新测评", key=f"{key_prefix}_redo"):
                st.session_state["kyc"] = None
                st.rerun()
        else:
            answers = []
            for i, (q, opts) in enumerate(QUESTIONS):
                choice = st.radio(q, [o for o, _ in opts], key=f"{key_prefix}_q{i}")
                answers.append(dict(opts)[choice])
            if st.button("提交测评", type="primary", key=f"{key_prefix}_submit"):
                score = sum(answers)
                level = score_to_level(score)
                st.session_state["kyc"] = {"score": score, "level": level}
                st.rerun()

    k = st.session_state["kyc"]
    if k:
        return {"score": k["score"], "level": k["level"],
                "rules": LEVEL_RULES[k["level"]]}
    return None
