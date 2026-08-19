# -*- coding: utf-8 -*-
"""策略3：波动率突破（接近 20 日高点 + 放量）。

信号（每月末）：
  - 价格位置 pos = (close - 20日高点)/20日高点；买入条件 pos > -2%（接近/突破 20 日高点）；
  - 量能 vol_ratio = 当日成交额 / 20 日均成交额；买入条件 vol_ratio > 1.5（放量）；
  - 持仓：同时满足二者，按突破强度（pos 从高到低，即最接近/超越高点的）取 30 只。
量代理：成交额面板（amount），与 V8 抓取口径一致。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ns_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex) and out.index.dtype != "datetime64[ns]":
        out.index = out.index.as_unit("ns").normalize()
    return out


def gen_signal(close_panel: pd.DataFrame,
               amount_panel: pd.DataFrame,
               month_ends,
               top_n: int = 30,
               hi_win: int = 20,
               pos_thr: float = -0.02,
               vol_ratio_thr: float = 1.5) -> dict:
    """返回 {month_end: [codes]}：pos>-2% 且 量比>1.5，按 pos 降序取 top_n。"""
    close = _ns_index(close_panel)
    amount = _ns_index(amount_panel).reindex(close.index)
    me = pd.DatetimeIndex(month_ends).normalize()

    hi20 = close.rolling(hi_win).max().shift(1)     # 前 20 日高点（不含当日）
    pos = close / hi20 - 1.0                        # 相对 20 日高点的位置
    vol20 = amount.rolling(20).mean()
    vol_ratio = amount / vol20                      # 当日量 / 20 日均量

    sel = {}
    for t in me:
        if t not in close.index:
            continue
        p = pos.loc[t].replace([np.inf, -np.inf], np.nan)
        vr = vol_ratio.loc[t].replace([np.inf, -np.inf], np.nan)
        px = close.loc[t]
        mask = (p > pos_thr) & (vr > vol_ratio_thr) & px.notna() & p.notna() & vr.notna()
        cand = px.index[mask]
        if len(cand) == 0:
            sel[t] = []
            continue
        ranked = p.reindex(cand).sort_values(ascending=False)   # 突破强度：最接近/超越高点
        sel[t] = [c for c in ranked.head(top_n).index]
    return sel
