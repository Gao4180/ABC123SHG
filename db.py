# -*- coding: utf-8 -*-
"""
策略档案云端存储（Supabase Postgres）。

- 未配置 SUPABASE_URL / SUPABASE_KEY（st.secrets）时优雅降级：available() 返回 False，
  页面其余功能不受影响。
- 6 位数字代码由系统随机分配（SystemRandom），即档案的访问凭证；
  可选口令以 SHA-256 加盐哈希存储，永不保存明文。
- supabase 包为惰性导入：本地测试环境没装也能 import 本模块。

建表 SQL（在 Supabase SQL Editor 执行一次）见仓库 README 或部署指引。
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


# ---------------- 档案 ----------------

def create_profile(plan_name: str, pool_label: str, total_amount: float,
                   passphrase: str = "") -> str | None:
    """创建档案，返回系统分配的 6 位数字代码；失败返回 None。"""
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
                "plan_name": plan_name,
                "pool": pool_label,
                "total_amount": float(total_amount),
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


def update_plan(code: str, plan_name: str) -> bool:
    cli = _client()
    if cli is None:
        return False
    try:
        cli.table("profiles").update({"plan_name": plan_name}) \
           .eq("code", code).execute()
        return True
    except Exception:
        return False


# ---------------- 买入记录 ----------------

def add_buy(code: str, ticker: str, name: str, amount: float,
            buy_date: str) -> bool:
    cli = _client()
    if cli is None:
        return False
    try:
        cli.table("buys").insert({
            "code": code, "ticker": ticker, "name": name,
            "amount": float(amount), "buy_date": str(buy_date),
        }).execute()
        return True
    except Exception:
        return False


def list_buys(code: str) -> list[dict]:
    cli = _client()
    if cli is None:
        return []
    try:
        r = cli.table("buys").select("*").eq("code", code) \
               .order("buy_date").execute()
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
