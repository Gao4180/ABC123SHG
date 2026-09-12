# -*- coding: utf-8 -*-
"""
数据加载模块：负责资产代码识别、行情拉取（带缓存）、交易日对齐。
所有网络请求统一走这里，并用 st.cache_data 缓存，避免重复请求。
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import re
import time

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

try:
    from curl_cffi import requests as _creq   # 浏览器伪装，降低被 Yahoo 限流的概率
    _SESSION = _creq.Session(impersonate="chrome")
except Exception:                            # 未安装 curl_cffi 也能正常运行
    _SESSION = None

# yfinance 1.7 不再把失败原因写入 shared._ERRORS，限流信息只在日志里出现。
# 挂一个日志哨兵，把触发限流的代码收集到 _RATE_LIMITED 集合中。
_RATE_LIMITED: set[str] = set()

# 每个资产实际命中的数据源（供界面标注"数据来自哪里"）
_SOURCE_USED: dict[str, str] = {}


class _YfLogSentry(logging.Handler):
    def emit(self, record):
        try:
            msg = record.getMessage()
            if ("Too Many Requests" in msg or "Rate limit" in msg or "429" in msg):
                for t in re.findall(r"'([A-Za-z0-9.^=\-]+)'", msg):
                    _RATE_LIMITED.add(t.upper())
        except Exception:
            pass


_yf_logger = logging.getLogger("yfinance")
if not any(isinstance(h, _YfLogSentry) for h in _yf_logger.handlers):
    _yf_logger.addHandler(_YfLogSentry())

# 数据窗口：近 5 年日线
YEARS = 5
TRADING_DAYS = 252          # 年化因子
MIN_HISTORY_YEARS = 3       # 少于此历史长度要标注"数据较短"

# 配置默认值（config.yaml 缺失或缺字段时使用）
_DEFAULT_CONFIG = {
    "risk_free_rate": 0.02,
    "weight_cap": 0.4,
    "shrinkage": True,
    "monte_carlo_paths": 2000,
    "mc_method": "bootstrap",
    "jump_sigma": 5.0,
    "missing_warn_ratio": 0.05,
    "trade_cost_bps": 5,
    "bl_tau": 0.05,
    "bl_view_shrink": 0.5,
    "alphavantage_api_key": "",
    "stress_scenarios": [],
    "asset_pool": [],
    "asset_pool_cn": [],
}


@st.cache_data(show_spinner=False)
def load_config(path: str = "config.yaml") -> dict:
    """读取 config.yaml；文件不存在或解析失败时用内置默认值，保证应用总能启动。"""
    import yaml
    cfg = dict(_DEFAULT_CONFIG)
    try:
        with open(path, encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg.update({k: v for k, v in user_cfg.items() if v is not None})
    except Exception:
        pass
    return cfg


def pool_meta(cfg: dict) -> dict:
    """合并美股/国内两个资产池，返回 {ticker: 资产条目字典}（含 risk/fee/role/name 等）。"""
    meta = {}
    for p in (cfg.get("asset_pool") or []) + (cfg.get("asset_pool_cn") or []):
        meta[p["ticker"]] = p
    return meta


def detect_market(ticker: str) -> str:
    """根据代码后缀粗判市场：.SS/.SZ 为 A 股，其余视为美股/ETF。"""
    t = ticker.strip().upper()
    if t.endswith(".SS") or t.endswith(".SZ"):
        return "A股"
    return "美股"


def normalize_ticker(ticker: str) -> str:
    """
    用户输入规范化：裸 6 位数字代码自动补 A 股交易所后缀——
    60/68（主板/科创板）、5（场内ETF）、11（债券）→ .SS；
    00/30（主板/创业板）、15（场内基金）→ .SZ。
    """
    t = ticker.strip().upper()
    if re.fullmatch(r"\d{6}", t):
        if t[0] in "65" or t.startswith("11"):
            return t + ".SS"
        return t + ".SZ"
    return t


# ---------- 名称 → 代码 解析（腾讯联想搜索，覆盖 A股/美股/ETF） ----------

@st.cache_data(ttl=dt.timedelta(days=7), persist="disk", show_spinner=False)
def resolve_name(query: str) -> tuple[str, str] | None:
    """
    把股票/ETF 名字解析成行情代码。返回 (代码, 官方名称) 或 None。
    数据源：腾讯 smartbox 联想接口；港股暂不支持行情，跳过 hk 结果。
    """
    q = query.strip()
    if not q:
        return None
    try:
        r = _http_get(f"https://smartbox.gtimg.cn/s3/?v=2&q={q}&t=all")
        r.encoding = "gbk"
        m = re.search(r'v_hint="(.*)"', r.text, re.S)
        if not m:
            return None
        for entry in m.group(1).split("^"):
            parts = entry.split("~")
            if len(parts) < 3:
                continue
            market, code, name = parts[0], parts[1], parts[2]
            # 接口把中文转成了 \uXXXX 字面量，转回可读文字
            name = re.sub(r"\\u([0-9a-fA-F]{4})",
                          lambda mm: chr(int(mm.group(1), 16)), name)
            if market in ("sh", "sz"):
                return code.upper() + (".SS" if market == "sh" else ".SZ"), name
            if market == "us":
                return code.split(".")[0].upper(), name
    except Exception:
        pass
    return None


def resolve_tokens(tokens: list[str]):
    """
    逐个解析用户输入（代码或名字混输）。
    返回 (代码列表, 识别映射列表[(输入, 代码, 名称)], 无法识别的输入列表)。
    """
    codes, mapping, unresolved = [], [], []
    for tok in tokens:
        t = tok.strip()
        if not t:
            continue
        # 已带后缀的代码 / 裸 6 位数字 / 纯字母美股代码：直接规范化
        if re.fullmatch(r"[A-Za-z0-9.\-]+", t):
            codes.append(normalize_ticker(t))
            continue
        hit = resolve_name(t)
        if hit:
            codes.append(hit[0])
            mapping.append((t, hit[0], hit[1]))
        else:
            unresolved.append(t)
    # 去重保序
    seen, uniq = set(), []
    for c in codes:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq, mapping, unresolved


def _download_batch(tickers: list[str]) -> pd.DataFrame:
    """主数据源：Yahoo 一次请求批量下载（N 个资产只发 1 次请求，降低限流概率）。"""
    end = dt.date.today()
    start = end - dt.timedelta(days=int(YEARS * 365.25) + 10)
    kwargs = dict(start=start.isoformat(), end=end.isoformat(), interval="1d",
                  auto_adjust=True, progress=False)   # 后复权价格
    if _SESSION is not None:
        kwargs["session"] = _SESSION
    return yf.download(tickers, **kwargs)


def _http_get(url: str, headers: dict | None = None, timeout: int = 15):
    """统一 HTTP 入口：优先浏览器伪装会话，退回普通 requests。"""
    if _SESSION is not None:
        return _SESSION.get(url, headers=headers, timeout=timeout)
    h = {"User-Agent": "Mozilla/5.0"}
    if headers:
        h.update(headers)
    return requests.get(url, headers=h, timeout=timeout)


# ---------- 备用数据源（Yahoo 拉不到的资产按此顺序尝试） ----------

def _fetch_sina_us(ticker: str) -> pd.Series:
    """备用源①：新浪美股日线（不复权），覆盖美股与 ETF，历史完整。"""
    sym = ticker.lower()
    url = ("https://stock.finance.sina.com.cn/usstock/api/jsonp.php/x/"
           f"US_MinKService.getDailyK?symbol={sym}&___qn=3&dpc=1")
    r = _http_get(url, headers={"Referer": "https://finance.sina.com.cn"})
    m = re.search(r"x\((.*)\)", r.text, re.S)
    if not m:
        raise ValueError("sina parse")
    rows = json.loads(m.group(1))
    if not rows:
        raise ValueError("sina empty")
    s = pd.Series({pd.Timestamp(x["d"]): float(x["c"]) for x in rows}).sort_index()
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=int(YEARS * 365.25) + 10)
    s = s.loc[s.index >= cutoff]
    if s.empty:
        raise ValueError("sina empty")
    s.name = ticker
    return s


def _fetch_tencent_cn(ticker: str) -> pd.Series:
    """备用源②：腾讯 A 股前复权日线（每页约 640 条，5 年分两页拉取）。"""
    t = ticker.upper()
    code = ("sh" if t.endswith(".SS") else "sz") + t.split(".")[0]
    end = dt.date.today()
    start = end - dt.timedelta(days=int(YEARS * 365.25) + 10)
    mid = start + (end - start) / 2
    frames = {}
    for a, b in ((start, mid), (mid, end)):
        url = ("https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?"
               f"param={code},day,{a},{b},800,qfq")
        r = _http_get(url)
        d = r.json()["data"].get(code, {})
        rows = d.get("qfqday") or d.get("day") or []
        for x in rows:                      # 格式：[日期, 开, 收, 高, 低, ...]
            frames[pd.Timestamp(x[0])] = float(x[2])
    if not frames:
        raise ValueError("tencent empty")
    s = pd.Series(frames).sort_index()
    s.name = ticker
    return s


def _fetch_sina_cn(ticker: str) -> pd.Series:
    """备用源③：新浪 A 股日线（不复权），一次最多 1258 条，覆盖近 5 年。"""
    t = ticker.upper()
    sym = ("sh" if t.endswith(".SS") else "sz") + t.split(".")[0]
    url = ("https://quotes.sina.cn/cn/api/json_v2.php/"
           f"CN_MarketDataService.getKLineData?symbol={sym}&scale=240&ma=no&datalen=1258")
    r = _http_get(url, headers={"Referer": "https://finance.sina.com.cn"})
    rows = r.json()
    if not rows:
        raise ValueError("sina_cn empty")
    s = pd.Series({pd.Timestamp(x["day"]): float(x["close"]) for x in rows}).sort_index()
    if s.empty:
        raise ValueError("sina_cn empty")
    s.name = ticker
    return s


def _fetch_alphavantage(ticker: str) -> pd.Series:
    """
    备用源④：Alpha Vantage（免费注册 key，25 次/天，美股/ETF 全历史）。
    key 读取顺序：config.yaml 的 alphavantage_api_key → 云端 st.secrets
    （Streamlit Cloud 后台 Secrets 里配 ALPHAVANTAGE_API_KEY）；都没有则跳过。
    """
    key = load_config().get("alphavantage_api_key")
    if not key:
        try:
            key = st.secrets.get("ALPHAVANTAGE_API_KEY", "")
        except Exception:
            key = ""
    if not key:
        raise ValueError("no av key")
    url = ("https://www.alphavantage.co/query?function=TIME_SERIES_DAILY"
           f"&symbol={ticker}&outputsize=full&apikey={key}")
    r = _http_get(url, timeout=30)
    rows = r.json().get("Time Series (Daily)") or {}
    if not rows:
        raise ValueError("av empty")
    s = pd.Series({pd.Timestamp(k): float(v["4. close"]) for k, v in rows.items()}
                  ).sort_index()
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=int(YEARS * 365.25) + 10)
    s = s.loc[s.index >= cutoff]
    if s.empty:
        raise ValueError("av empty")
    s.name = ticker
    return s


_SOURCE_NAMES = {"yahoo": "Yahoo Finance", "tencent": "腾讯财经",
                 "sina_cn": "新浪财经", "sina_us": "新浪财经",
                 "alphavantage": "Alpha Vantage"}


def _fetch_fallback(ticker: str) -> pd.Series:
    """单个资产的备用源级联：A 股 → 腾讯 → 新浪；美股 → 新浪 → AlphaVantage。"""
    if detect_market(ticker) == "A股":
        for src, fn in (("tencent", _fetch_tencent_cn),
                        ("sina_cn", _fetch_sina_cn)):
            try:
                s = fn(ticker)
                _SOURCE_USED[ticker.upper()] = _SOURCE_NAMES[src]
                return s
            except Exception:
                continue
    else:
        try:
            s = _fetch_sina_us(ticker)
            _SOURCE_USED[ticker.upper()] = _SOURCE_NAMES["sina_us"]
            return s
        except Exception:
            pass
    s = _fetch_alphavantage(ticker)          # 无 key 时抛错，由上层跳过
    _SOURCE_USED[ticker.upper()] = _SOURCE_NAMES["alphavantage"]
    return s


@st.cache_data(ttl=dt.timedelta(hours=12), persist="disk", show_spinner=False)
def fetch_close_batch(tickers: tuple[str, ...]) -> dict[str, pd.Series]:
    """
    批量拉取多个资产近 5 年日线收盘价，返回 {代码: Series}。
    级联顺序：Yahoo 批量 → 新浪美股 / 腾讯A股 逐个兜底。
    - persist="disk"：缓存落到硬盘，应用重启后 12 小时内不再重复发请求；
    - 一个资产都没拉到时抛异常——失败结果绝不进缓存（否则会把"空结果"缓存 12 小时）；
    - 检测到 Yahoo 限流时立即停手（不再重试敲门），直接走备用源；
    - 非限流的偶发失败只补试一次。
    """
    tickers = list(tickers)
    df = pd.DataFrame()
    for attempt in range(2):
        if attempt:
            time.sleep(10)
        try:
            df = _download_batch(tickers)
        except Exception:
            df = pd.DataFrame()
        if df is not None and not df.empty:
            break
        if _RATE_LIMITED:            # Yahoo 已限流：立即放弃，转备用源
            break
    out: dict[str, pd.Series] = {}
    if df is not None and not df.empty:
        close = df["Close"]
        if isinstance(close, pd.Series):     # 只有 1 个资产时 yfinance 返回 Series
            close = close.to_frame(tickers[0])
        for t in tickers:
            if t in close.columns:
                s = close[t].dropna()
                if not s.empty:
                    s.name = t
                    out[t] = s
                    _SOURCE_USED[t] = "Yahoo Finance"
    # 备用源兜底：逐个补齐 Yahoo 没拉到的资产
    for t in tickers:
        if t not in out:
            try:
                out[t] = _fetch_fallback(t)
            except Exception:
                pass
            time.sleep(0.3)                  # 备用源礼貌限速
    if not out:
        raise ValueError("empty")            # 失败不缓存，下次访问会重新请求
    return out


def is_rate_limited(ticker: str) -> bool:
    """判断某资产拉取失败是否源于 Yahoo 限流（由日志哨兵收集）。"""
    return ticker.strip().upper() in _RATE_LIMITED


def last_sources() -> dict[str, str]:
    """返回上一轮 load_prices 中每个资产实际命中的数据源名。"""
    return dict(_SOURCE_USED)


def load_prices(tickers: list[str], progress=None):
    """
    批量加载并对齐交易日。
    progress：可选回调 progress(代码, 第几个, 总数)，用于界面显示逐资产进度。
    返回 (prices_df, ok, bad, short, markets, rate_limited, quality_warns)：
    - prices_df：各资产收盘价对齐后的 DataFrame（前向填充最多5天，再丢弃空缺行）
    - bad：拉不到数据、已跳过的代码
    - short：历史不足 3 年的代码
    - markets：{代码: 市场}，用于混输提示
    - rate_limited：失败原因明确是数据源限流的代码（提示用户稍后再试）
    - quality_warns：数据质量提示（缺失占比高 / 超 5σ 价格跳跃）
    """
    series, bad, short, markets, limited = {}, [], [], {}, []
    _RATE_LIMITED.clear()                    # 每轮加载前清空上次的限流记录
    _SOURCE_USED.clear()
    todo = [normalize_ticker(t) for t in tickers if t.strip()]
    if progress:
        progress("、".join(todo), 1, 1)      # 批量请求：一次取全部
    try:
        data = fetch_close_batch(tuple(todo))
    except Exception:
        data = {}                            # 全部拉取失败（失败结果不进缓存）
    batch_blocked = not data and bool(_RATE_LIMITED)   # 整批请求被限流
    for t in todo:
        s = data.get(t)
        if s is None:
            bad.append(t)
            if is_rate_limited(t) or batch_blocked:
                limited.append(t)
            continue
        series[t] = s
        markets[t] = detect_market(t)
        first_date = s.index[0]
        if (pd.Timestamp.now().normalize() - first_date).days < MIN_HISTORY_YEARS * 365:
            short.append(t)

    if not series:
        return pd.DataFrame(), [], bad, short, markets, limited, []

    raw = pd.concat(series.values(), axis=1).sort_index()
    # 对齐交易日：不同市场休市日不同，先最多前向填充 5 个交易日，再丢弃仍有空缺的行
    prices = raw.ffill(limit=5).dropna()

    # 数据质量检查（缺失占比 + 价格跳跃），参数来自 config.yaml
    cfg = load_config()
    quality = []
    for t in prices.columns:
        col = raw[t]
        fv = col.first_valid_index()
        if fv is not None:
            miss = col.loc[fv:].isna().mean()
            if miss > cfg["missing_warn_ratio"]:
                quality.append(f"{t} 有 {miss:.0%} 的交易日缺失数据（已用前值补齐），结果可能略有偏差")
    rets = prices.pct_change().dropna()
    for t in rets.columns:
        r = rets[t]
        vol = r.std()
        # 超低波动资产（如短债 ETF SHV，日波动 <0.05%）的 5σ 阈值太小，
        # 一分钱的跳动就会误报，直接跳过跳跃检测
        if vol >= 0.0005:
            jumps = int((r.abs() > cfg["jump_sigma"] * vol).sum())
            # 5 年里 1~3 次极端波动多为真实崩盘日（如 2020.3 / 2025.4），不算脏数据；
            # 出现 4 次以上才提示，避免预警噪音
            if jumps >= 4:
                quality.append(f"{t} 检出 {jumps} 次超过 {cfg['jump_sigma']:.0f}σ 的单日剧烈波动，"
                               "次数偏多，可能包含数据源脏数据，建议结合净值图核对")
    return prices, list(series.keys()), bad, short, markets, limited, quality


def max_drawdown(nav: pd.Series) -> float:
    """最大回撤（基于净值序列）。"""
    roll_max = nav.cummax()
    dd = nav / roll_max - 1.0
    return float(dd.min())


def drawdown_periods(nav: pd.Series, top_n: int = 3):
    """
    找出最深的若干个回撤区间，返回 [(峰值日, 谷底日, 回撤幅度), ...]。
    用于自动文字分析里描述"近5年经历过哪些大回撤"。
    """
    roll_max = nav.cummax()
    dd = nav / roll_max - 1.0
    is_peak = nav >= roll_max
    # 按峰-谷分段
    periods = []
    peak_idx = None
    for date, peak in is_peak.items():
        if peak:
            if peak_idx is not None and peak_idx != date:
                seg = dd.loc[peak_idx:date]
                if len(seg) > 1:
                    trough = seg.idxmin()
                    periods.append((peak_idx, trough, float(seg.min())))
            peak_idx = date
    # 末尾未修复的回撤
    if peak_idx is not None and peak_idx != nav.index[-1]:
        seg = dd.loc[peak_idx:]
        trough = seg.idxmin()
        periods.append((peak_idx, trough, float(seg.min())))
    periods.sort(key=lambda x: x[2])
    return periods[:top_n]


def portfolio_nav(prices: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """按固定权重（买入持有、每日再平衡近似）计算组合累计净值。"""
    rets = prices.pct_change().dropna()
    port = (rets * weights).sum(axis=1)
    nav = (1 + port).cumprod()
    nav = pd.concat([pd.Series([1.0], index=[rets.index[0] - pd.Timedelta(days=1)]), nav])
    return nav


def annualized_stats(prices: pd.DataFrame, weights: np.ndarray, rf: float = 0.02):
    """组合层面的年化收益/波动/夏普/最大回撤。"""
    rets = prices.pct_change().dropna()
    port = (rets * weights).sum(axis=1)
    ann_ret = float(port.mean() * TRADING_DAYS)
    ann_vol = float(port.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan
    nav = (1 + port).cumprod()
    mdd = max_drawdown(nav)
    return ann_ret, ann_vol, sharpe, mdd
