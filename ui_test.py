# -*- coding: utf-8 -*-
"""界面级验证：用 Streamlit AppTest 无头运行 app.py，模拟三个模式的完整操作。"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

import data_loader

# 合成数据替代网络拉取（覆盖美股池 + 国内池 + 测试用错误代码）
rng = np.random.default_rng(3)
dates = pd.bdate_range("2021-09-01", "2026-09-10")
params = {"GLD": (.08, .15), "BND": (.02, .05), "TLT": (-.01, .13), "SPY": (.12, .17),
          "QQQ": (.16, .24), "DBC": (.05, .18), "QAI": (.04, .06), "HDG": (.03, .07),
          "SHV": (.02, .01), "AAPL": (.2, .28),
          "511880.SS": (.015, .005), "511010.SS": (.02, .02), "518880.SS": (.09, .14),
          "510880.SS": (.07, .15), "510300.SS": (.06, .18), "513500.SS": (.11, .17),
          "159915.SZ": (.10, .26), "513100.SS": (.15, .25)}
cols = {t: pd.Series(100 * np.cumprod(1 + m / 252 + v / np.sqrt(252) * rng.standard_normal(len(dates))),
                     index=dates, name=t) for t, (m, v) in params.items()}
data_loader.fetch_close_batch = lambda ts: {t.strip().upper(): cols[t.strip().upper()] for t in ts if t.strip().upper() in cols}


def _click_submit(at, label):
    btn = next(b for b in at.button if label in (b.label or ""))
    btn.click().run()


print("== 页面初始渲染 ==")
at = AppTest.from_file("app.py", default_timeout=120)
at.run()
assert not at.exception, at.exception
assert any("我的资产优化" in (h.value or "") for h in at.header)
assert any("风险测评" in (e.label or "") for e in at.expander)  # KYC 在侧边栏

print("\n== 模式一：填表并点击「开始计算」 ==")
ti = next(t for t in at.text_input if "资产代码" in (t.label or ""))
ti.set_value("SPY, GLD, TLT, QQQ, BADCODE123")
risk_radio = next(r for r in at.radio if "平衡（最大夏普）" in list(r.options))
risk_radio.set_value("平衡（最大夏普）")
_click_submit(at, "开始计算")
assert not at.exception, at.exception
warns = [w.value for w in at.warning]
print("警告:", warns)
assert any("BADCODE123" in w and "已跳过" in w for w in warns)
metrics = [m.label for m in at.metric]
assert "预期年化收益" in metrics and "近5年最大回撤" in metrics
print("数据表数量:", len(at.dataframe))
assert len(at.dataframe) >= 2
subheaders = [s.value for s in at.subheader]
print("小节:", subheaders)
assert any("自动分析" in s for s in subheaders)
assert any("蒙特卡洛" in s for s in subheaders) and any("压力测试" in s for s in subheaders)
assert any("择时信号" in s for s in subheaders) and any("动态配置回测" in s for s in subheaders)

print("\n== 回归：交互后不闪退（滑块/勾选/改假设触发重跑，结果仍在） ==")
# 拖动微调滑块 → 重跑 → 优化结果区块必须还在
sl = next(s for s in at.slider if s.label == "SPY")
sl.set_value(0.9)
at.run()
assert not at.exception, at.exception
subs2 = [s.value for s in at.subheader]
assert any("自动分析" in s for s in subs2), "拖动滑块后结果区块丢失（闪退回归）"
assert "预期年化收益" in [m.label for m in at.metric]
# 勾选 BL 观点 → 重跑 → 仍在
cb = next(c for c in at.checkbox if "Black-Litterman" in (c.label or ""))
cb.set_value(True)
at.run()
assert not at.exception, at.exception
assert any("自动分析" in (s.value or "") for s in at.subheader)
# 修改预期收益输入 → 重跑 → 仍在
mu_in = next(n for n in at.number_input if "SPY（%）" in (n.label or ""))
mu_in.set_value(5.0)
at.run()
assert not at.exception, at.exception
assert any("自动分析" in (s.value or "") for s in at.subheader)
print("交互重跑三次，结果区块均保持 ✔")

print("\n== 模式一：少于2个有效资产 ==")
ti.set_value("SPY")
_click_submit(at, "开始计算")
assert not at.exception, at.exception

print("\n== 模式二：财富方案推荐（国内 ETF 池） ==")
at2 = AppTest.from_file("app.py", default_timeout=180)
at2.run()
nav = next(r for r in at2.sidebar.radio if r.label == "功能导航")
nav.set_value("财富方案推荐")
at2.run()
assert not at2.exception, at2.exception
subs = [s.value for s in at2.subheader]
print("小节:", subs)
assert any("方案A" in s for s in subs) and any("方案D" in s for s in subs)
assert any("横向对比" in s for s in subs)
assert any("压力测试" in s for s in subs)
assert any("择时信号" in s for s in subs) and any("动态配置回测" in s for s in subs)
m2_metrics = [m.label for m in at2.metric]
assert any("蒙特卡洛" in l for l in m2_metrics), m2_metrics
caps = " ".join(c.value or "" for c in at2.caption)
assert "适合人群" in caps and "主要风险" in caps
assert "不构成投资建议" in caps

print("\n== 模式三：多目标规划 ==")
at3 = AppTest.from_file("app.py", default_timeout=180)
at3.run()
nav3 = next(r for r in at3.sidebar.radio if r.label == "功能导航")
nav3.set_value("多目标规划")
at3.run()
assert not at3.exception, at3.exception
assert any("多目标规划" in (h.value or "") for h in at3.header)
# 添加一个目标
_click_submit(at3, "添加目标")
assert not at3.exception, at3.exception
subs3 = [s.value for s in at3.subheader]
print("小节:", subs3)
assert any("子组合" in s for s in subs3)
assert any("目标达成概率" in (m.label or "") for m in at3.metric)
caps3 = " ".join(c.value or "" for c in at3.caption)
assert "下滑轨道" in caps3

print("\n== KYC 适当性过滤：C2 客户输入含 R3+ 产品的组合（回归：长度不匹配 bug） ==")
at4 = AppTest.from_file("app.py", default_timeout=180)
at4.run()
# 答 KYC：q0-q4 默认第1项（各1分），q5-q7 分别选 3/3/4 分 → 总分 15 → C2
q_by_label = {r.label: r for r in at4.sidebar.radio}
q_by_label["""您对股票、债券、基金等投资知识的了解程度？"""].set_value("了解基本概念")
q_by_label["您的家庭负担情况？"].set_value("有负担，应急存款够 6-12 个月")
q_by_label["如果您的持仓一个月跌了 15%，您会怎么做？"].set_value("按计划再平衡")
_click_submit(at4, "提交测评")
assert not at4.exception, at4.exception
ti4 = next(t for t in at4.text_input if "资产代码" in (t.label or ""))
ti4.set_value("SPY, GLD, BND, SHV")
_click_submit(at4, "开始计算")
assert not at4.exception, at4.exception
warns4 = [w.value for w in at4.warning]
print("警告:", warns4)
assert any("适当性" in w for w in warns4)
caps4 = " ".join(c.value or "" for c in at4.caption)
assert "C2 稳健型" in caps4 or any("C2 稳健型" in (m.value or "") for m in list(at4.info) + list(at4.warning))

print("\n界面级验证全部通过 ✔")
