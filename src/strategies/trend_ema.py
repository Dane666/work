# -*- coding: utf-8 -*-
"""策略1：趋势跟踪（EMA12/EMA30 比值排序）。

信号（每月末）：
  - 买入条件：EMA12 > EMA30（上升趋势）；
  - 持仓：选出 EMA12/EMA30 比值最高的 30 只（趋势强度排序）。
卖出：无日内规则，月末轮换（不再入选即自动卖出）——与 V8 卖出规则一致。
共享：MA240+波动率市场门控、月频调仓、分档滑点。
"""

from __future__ import annotations

import pandas as pd


def _ns_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex) and out.index.dtype != "datetime64[ns]":
        out.index = out.index.as_unit("ns").normalize()
    return out


def gen_signal(close_panel: pd.DataFrame,
               month_ends,
               top_n: int = 30,
               fast: int = 12,
               slow: int = 30) -> dict:
    """返回 {month_end: [codes]}：EMA12/EMA30 比值最高的 top_n 只（须 EMA12>EMA30）。"""
    close = _ns_index(close_panel)
    me = pd.DatetimeIndex(month_ends).normalize()
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    ratio = ema_f / ema_s          # >1 表示上升趋势，越大趋势越强
    up = ema_f > ema_s

    sel = {}
    for t in me:
        if t not in close.index:
            continue
        px = close.loc[t]
        mask = up.loc[t] & px.notna() & ratio.loc[t].notna()
        cand = px.index[mask]
        if len(cand) == 0:
            sel[t] = []
            continue
        ranked = ratio.loc[t].reindex(cand).sort_values(ascending=False)
        sel[t] = [c for c in ranked.head(top_n).index]
    return sel
