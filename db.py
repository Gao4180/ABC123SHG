# -*- coding: utf-8 -*-
"""
策略档案云端存储（Supabase Postgres）。

- 未配置 SUPABASE_URL / SUPABASE_KEY（st.secrets）时优雅降级：available() 返回 False，
  页面其余功能不受影响。
- 6 位数字代码由系统随机分配（SystemRandom），即档案的访问凭证；
  可选口令以 SHA-256 加盐哈希存储，永不保存明文。
- 档案 = 纯身份；策略与买入记录按模块归属（mode=1/2/3）分别存储，
  三个模块各自独立恢复、记账、体检（方案 A）。
- supabase 包为惰性导入：本地测试环境没装也能 import 本模块。

表结构：
  profiles(code pk, plan_name, pool, total_amount, pass_hash, created_at)
    —— 新版档案 plan_name 等字段留空字符串，仅为兼容旧版档案保留
  buys(id pk, code, mode, ticker, name, amount, buy_date, created_at)
  strategies(code, mode, payload jsonb, updated_at, pk(code, mode))
"""
from __future__ import annotations

import hashlib
import random

_CLIENT = None
_CLIENT_TRIED = False


def _secrets():
    try:
        import streamlit as st
        return (st.secrets.get("SUPABASE_URL", ""),
                st.secrets.get("SUPABASE_KEY", ""))
    except Exception:
        return "", ""


def _client():
    global _CLIENT, _CLIENT_TRIED
    if _CLIENT_TRIED:
        return _CLIENT
    _CLIENT_TRIED = True
    url, key = _secrets()
    if not url or not key:
        return None
    try:
        from supabase import create_client
        _CLIENT = create_client(url, key)
    except Exception:
        _CLIENT = None
    return _CLIENT


def available() -> bool:
    """云端数据库是否已配置且可用。"""
    return _client() is not None


def _hash_pass(passphrase: str) -> str:
    return hashlib.sha256(("wc-pin$" + passphrase).encode("utf-8")).hexdigest()


# ---------------- 档案（身份） ----------------

def create_profile(passphrase: str = "") -> str | None:
    """创建纯身份档案，返回系统分配的 6 位数字代码；失败返回 None。"""
    cli = _client()
    if cli is None:
        return None
    rng = random.SystemRandom()
    for _ in range(20):
        code = f"{rng.randint(0, 999999):06d}"
        try:
            dup = cli.table("profiles").select("code").eq("code", code).execute()
            if dup.data:
                continue
            cli.table("profiles").insert({
                "code": code,
                "plan_name": "",           # 旧字段，新版档案不再使用
                "pool": "",
                "total_amount": 0,
                "pass_hash": _hash_pass(passphrase) if passphrase else "",
            }).execute()
            return code
        except Exception:
            continue
    return None


def get_profile(code: str) -> dict | None:
    cli = _client()
    if cli is None or not code:
        return None
    try:
        r = cli.table("profiles").select("*").eq("code", code).execute()
        return r.data[0] if r.data else None
    except Exception:
        return None


def check_pass(profile: dict, passphrase: str) -> bool:
    stored = profile.get("pass_hash") or ""
    if not stored:
        return True                       # 创建时没设口令，直接放行
    return stored == _hash_pass(passphrase or "")


# ---------------- 分模块策略 ----------------

def save_strategy(code: str, mode: int, payload: dict) -> bool:
    """保存/覆盖某模块的策略配置（upsert）。"""
    cli = _client()
    if cli is None:
        return False
    try:
        cli.table("strategies").upsert({
            "code": code, "mode": int(mode), "payload": payload,
        }).execute()
        return True
    except Exception:
        return False


def get_strategy(code: str, mode: int) -> dict | None:
    """读取某模块已保存的策略；旧版档案回退到 profiles 上的方案二字段。"""
    cli = _client()
    if cli is None:
        return None
    try:
        r = cli.table("strategies").select("payload") \
               .eq("code", code).eq("mode", int(mode)).execute()
        if r.data:
            return r.data[0].get("payload") or None
    except Exception:
        return None
    if int(mode) == 2:                    # 旧版档案兼容
        prof = get_profile(code)
        if prof and prof.get("plan_name"):
            return {"plan_name": prof["plan_name"], "pool": prof.get("pool", ""),
                    "total_amount": float(prof.get("total_amount") or 0)}
    return None


# ---------------- 买入记录（按模块） ----------------

def add_buy(code: str, mode: int, ticker: str, name: str, amount: float,
            buy_date: str) -> bool:
    cli = _client()
    if cli is None:
        return False
    try:
        cli.table("buys").insert({
            "code": code, "mode": int(mode),
            "ticker": ticker, "name": name,
            "amount": float(amount), "buy_date": str(buy_date),
        }).execute()
        return True
    except Exception:
        return False


def list_buys(code: str, mode: int) -> list[dict]:
    cli = _client()
    if cli is None:
        return []
    try:
        r = cli.table("buys").select("*").eq("code", code) \
               .eq("mode", int(mode)).order("buy_date").execute()
        return r.data or []
    except Exception:
        return []


def delete_buy(buy_id) -> bool:
    cli = _client()
    if cli is None:
        return False
    try:
        cli.table("buys").delete().eq("id", buy_id).execute()
        return True
    except Exception:
        return False
