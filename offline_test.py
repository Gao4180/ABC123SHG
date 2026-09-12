# -*- coding: utf-8 -*-
"""离线验证：用合成的 5 年日线数据走通 优化→指标→图表→文本 全链路（不依赖网络）。"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import data_loader
import optimizer as opt
from data_loader import load_prices, annualized_stats
from report import (
    RF, amount_table, asset_metrics_table, auto_analysis,
    nav_chart, pie_chart, portfolio_summary_row,
)
import mode2

# ---- 构造合成数据：9 个“资产”，不同收益/波动/相关性，其中 1 个历史仅 2 年 ----
rng = np.random.default_rng(7)
dates = pd.bdate_range("2021-09-01", "2026-09-10")
n = len(dates)
tickers = ["GLD", "BND", "TLT", "SPY", "QQQ", "DBC", "QAI", "HDG", "SHV"]
params = {  # (年化收益, 年化波动)
    "GLD": (.08, .15), "BND": (.02, .05), "TLT": (-.01, .13), "SPY": (.12, .17),
    "QQQ": (.16, .24), "DBC": (.05, .18), "QAI": (.04, .06), "HDG": (.03, .07),
    "SHV": (.02, .01),
}
cols = {}
for t, (m, v) in params.items():
    drift, shock = m / 252, v / np.sqrt(252)
    rets = drift + shock * rng.standard_normal(n)
    # 制造相关性：QQQ/HDG 跟随 SPY
    if t in ("QQQ", "HDG") and "SPY" in cols:
        rets = 0.6 * cols["SPY"].pct_change().fillna(0).values + 0.8 * rets
    price = 100 * np.cumprod(1 + rets)
    s = pd.Series(price, index=dates, name=t)
    if t == "HDG":
        s = s.loc[s.index >= "2024-09-01"]   # 仅约 2 年历史 → 应触发"数据较短"
    cols[t] = s

def fake_fetch_batch(tickers):
    return {t: cols[t] for t in tickers if t in cols}

data_loader.fetch_close_batch = fake_fetch_batch   # 替换网络拉取（不影响被测的对齐/优化逻辑）

print("== 1. load_prices：对齐 + 错误代码 + 短历史 ==")
prices, ok, bad, short, markets, limited, quality = load_prices(tickers + ["BADCODE123"])
print("有效:", ok)
print("跳过:", bad, "| 数据较短:", short, "| 市场:", set(markets.values()))
assert "BADCODE123" in bad and "HDG" in short and len(ok) == 9 and not prices.empty

mu, cov, _ = opt.estimate_params(prices)
idx = {t: i for i, t in enumerate(prices.columns)}
eq = [idx[t] for t in ("SPY", "QQQ")]
bd = [idx[t] for t in ("BND", "TLT")]
defensive = [idx[t] for t in ("BND", "TLT", "SHV")]

print("\n== 2. 模式一三档优化 ==")
for name, w in [("保守", opt.min_variance(mu, cov)),
                ("平衡", opt.max_sharpe(mu, cov, rf=RF)),
                ("激进", opt.max_return_capped_vol(mu, cov, vol_limit=0.25))]:
    r, v, s, m = annualized_stats(prices, w, rf=RF)
    print(f"{name}: 权重和={w.sum():.3f} 上限={w.max():.2f} 收益={r:.1%} 波动={v:.1%} 夏普={s:.2f} 回撤={m:.1%}")
    assert abs(w.sum() - 1) < 1e-3 and w.max() <= 0.401 and w.min() >= -1e-9
    if name == "激进":
        assert v <= 0.252

print("\n== 3. 模式二四套方案 ==")
plans = {
    "方案A · 稳健保本型": opt.optimize_plan_A(mu, cov, defensive, vol_limit=0.05),
    "方案B · 经典60/40": opt.optimize_plan_B(mu, cov, eq, bd, rf=RF),
    "方案C · 全天候组合": opt.risk_parity(cov, cap=0.4),
    "方案D · 积极增长型": opt.optimize_plan_D(mu, cov, eq, rf=RF, eq_floor=0.5),
}
rows = []
for name, w in plans.items():
    r, v, s, m = annualized_stats(prices, w, rf=RF)
    rows.append(portfolio_summary_row(name, prices, w))
    eqw = sum(w[i] for i in eq)
    print(f"{name}: 收益={r:.1%} 波动={v:.1%} 夏普={s:.2f} 回撤={m:.1%} 股票占比={eqw:.0%}")
va = annualized_stats(prices, plans["方案A · 稳健保本型"], rf=RF)[1]
assert va <= 0.055, f"方案A波动 {va:.1%} 应≤5%"
assert abs(sum(plans["方案B · 经典60/40"][i] for i in eq) - 0.6) < 0.02
assert sum(plans["方案D · 积极增长型"][i] for i in eq) >= 0.49

print("\n== 4. 表格 / 图表 / 文本渲染 ==")
w = plans["方案C · 全天候组合"]
t1 = asset_metrics_table(prices, w)
t2 = amount_table(prices, w, 100000, "人民币")
fig1 = pie_chart(prices.columns, w)
fig2 = nav_chart({"C": data_loader.portfolio_nav(prices, w),
                  "对比": data_loader.portfolio_nav(prices, plans["方案B · 经典60/40"])})
txt = auto_analysis(list(prices.columns), w, prices, mu, cov, label="全天候组合")
print(t1.to_string(index=False))
print(t2.head(3).to_string(index=False))
print("对比表:", pd.DataFrame(rows).to_dict("records"))
print("分析文本:", txt)
assert len(fig1.data) == 1 and len(fig2.data) == 2 and len(txt) > 80

print("\n== 5. 新增分析模块：蒙特卡洛 + 压力测试 + 扇形图 ==")
import analytics
from data_loader import load_config
from report import mc_fan_chart

cfg = load_config()
assert cfg["risk_free_rate"] == 0.02 and cfg["asset_pool"], "config.yaml 应含资产池"
assert all("name" in p and "desc" in p for p in cfg["asset_pool"]), "资产池应有具体名称与说明"

prob, fan = analytics.monte_carlo_forecast(0.10, 0.18, paths=cfg["monte_carlo_paths"])
print(f"蒙特卡洛: 正收益概率={prob:.1%}, 扇形表 {fan.shape}")
assert 0 <= prob <= 1 and list(fan.columns) == ["交易日", "p5", "p25", "p50", "p75", "p95"]
fig3 = mc_fan_chart(fan)
assert len(fig3.data) == 5

stress = analytics.stress_test(prices, plans["方案C · 全天候组合"],
                               cfg["stress_scenarios"])
print("压力测试:", stress.to_dict("records"))
# 合成数据中 HDG 仅 2 年历史（pct_change.dropna 后起点为 2024-09），
# 只能覆盖 2025 关税冲击；2020/2022 情景因数据未覆盖被正确跳过
assert len(stress) >= 1 and "2025年4月关税冲击" in list(stress["情景"])

print("\n== 6. 择时模块：信号表 + 战术权重 + 月末调仓回测 ==")
import timing
sig = timing.trend_signals(prices)
print(sig[["代码", "当前信号"]].to_string(index=False))
assert len(sig) >= 8
w_c = plans["方案C · 全天候组合"]
w_adj = timing.tactical_weights(w_c, prices, cash_ticker="SHV")
assert abs(w_adj.sum() - 1) < 1e-6, "战术调整后权重和应为 1"
nav_t, nav_s, logs = timing.tactical_backtest(prices, w_c, cash_ticker="SHV")
st_t, st_s = timing.nav_stats(nav_t), timing.nav_stats(nav_s)
print("趋势过滤:", st_t, "| 静态:", st_s)
print("最近调仓:", logs.to_dict("records") if not logs.empty else "无")
assert len(nav_t) == len(nav_s) > 250 and nav_t.iloc[-1] > 0 and nav_s.iloc[-1] > 0

print("\n== 7. 第三梯队新增：KYC / BL / 肥尾MC / 目标模拟 / 含费回测 ==")
import kyc
assert kyc.score_to_level(8) == 1 and kyc.score_to_level(40) == 5
assert kyc.score_to_level(24) == 3
lv, note = kyc.horizon_rule(1.5, 5)
assert lv == 2 and note is not None
lv, note = kyc.horizon_rule(12, 5)
assert lv == 5 and note is None
assert kyc.allowed_modes(2) == ["min_var"]
assert set(kyc.allowed_modes(5)) == {"min_var", "max_sharpe", "max_ret"}

mu_bl = opt.black_litterman(mu, cov, list(prices.columns), {"SPY", "QQQ"},
                            rf=RF, tau=0.05, view_shrink=0.5)
idx_spy = list(prices.columns).index("SPY")
assert mu_bl[idx_spy] < mu[idx_spy], "看空观点应拉低后验收益"
mu_ov = opt.apply_return_overrides(mu, list(prices.columns), {"SPY": 0.05})
assert abs(mu_ov[idx_spy] - 0.05) < 1e-9

port_rets = (prices.pct_change().dropna() * plans["方案C · 全天候组合"]).sum(axis=1)
prob_bs, fan_bs = analytics.monte_carlo_forecast(0.05, 0.05, hist_rets=port_rets,
                                                 method="bootstrap")
assert 0 <= prob_bs <= 1 and len(fan_bs) == 253
prob_g, _ = analytics.monte_carlo_forecast(0.05, 0.05, method="gbm")
print(f"蒙特卡洛 自助法={prob_bs:.0%} vs 正态={prob_g:.0%}")

prob_g2, fan_g2, med = analytics.goal_simulation(port_rets, 50000, 3000, 10, 500000)
print(f"目标模拟: 达成概率={prob_g2:.0%} 终值中位数={med:,.0f}")
assert 0 <= prob_g2 <= 1 and med > 0

nav_t2, nav_s2, _ = timing.tactical_backtest(prices, plans["方案C · 全天候组合"],
                                             cash_ticker="SHV", cost_bps=5,
                                             fees={"GLD": 0.004, "QQQ": 0.002})
assert nav_t2.iloc[-1] < nav_t.iloc[-1] * 1.001, "费率/成本应拖累净值"
print("含费回测净值:", round(float(nav_t2.iloc[-1]), 4),
      "vs 无费:", round(float(nav_t.iloc[-1]), 4))

print("\n== 8. 裸 A 股代码自动补后缀 ==")
from data_loader import normalize_ticker
assert normalize_ticker("600519") == "600519.SS"
assert normalize_ticker("002050") == "002050.SZ"
assert normalize_ticker("510300") == "510300.SS"
assert normalize_ticker("159915") == "159915.SZ"
assert normalize_ticker("spy") == "SPY"
assert normalize_ticker(" 601318 ") == "601318.SS"
print("normalize_ticker ✔")

print("\n离线全链路验证通过 ✔")
