# -*- coding: utf-8 -*-
"""
个人财富管理配置器 · 入口
左侧 sidebar：KYC 风险测评 + 全局参数；三个模式页面：
我的资产优化 / 财富方案推荐 / 多目标规划。
"""
import streamlit as st

import kyc
import mode1
import mode2
import mode3
from data_loader import load_config

st.set_page_config(page_title="个人财富管理配置器", page_icon="💼", layout="wide")

_cfg = load_config()

# ---------- 左侧：KYC 风险测评 ----------
kyc_result = kyc.kyc_sidebar_widget()

# ---------- 左侧：全局导航 ----------
st.sidebar.title("💼 财富管理配置器")
# 支持 ?page=plan 直接进入「财富方案推荐」
_default = 1 if st.query_params.get("page") == "plan" else 0
page = st.sidebar.radio("功能导航",
                        ["我的资产优化", "财富方案推荐", "多目标规划"],
                        index=_default)

st.sidebar.divider()
# 全局参数：单资产权重上限（KYC 定级后会自动收紧）
_cap_default = float(_cfg["weight_cap"])
if kyc_result:
    _cap_default = min(_cap_default, kyc_result["rules"]["cap"])
st.session_state["weight_cap"] = st.sidebar.slider(
    "单资产权重上限", min_value=0.2, max_value=1.0,
    value=_cap_default, step=0.05,
    help="优化时单个资产允许的最大配置比例；KYC 等级越低上限越紧")
st.sidebar.caption(
    f"数据窗口：近 5 年日线 · 无风险利率 {_cfg['risk_free_rate']:.0%}"
    "（config.yaml 可调）· 数据缓存在本地，12 小时内不重复请求")

if page == "我的资产优化":
    mode1.render()
elif page == "财富方案推荐":
    mode2.render()
else:
    mode3.render()

# ---------- 免责声明 ----------
st.divider()
st.caption("免责声明：本工具基于历史数据统计，不构成投资建议。")
