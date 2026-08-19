# -*- coding: utf-8 -*-
"""策略2：均值回归（MA20 偏离度）。

信号（每月末）：
  - 偏离度 dev = close/MA20 - 1（相对 20 日均线的百分比偏离）；
  - 买入条件：dev 的截面 z 分 < -2（严重超卖）；
  - 持仓：选出 dev 最低（最超卖）的 30 只。
卖出：无日内规则，月末轮换。
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
               month_ends,
               top_n: int = 30,
               ma_win: int = 20,
               z_thr: float = -2.0) -> dict:
    """返回 {month_end: [codes]}：dev 截面 z<-2 候选，dev 最低 top_n。"""
    close = _ns_index(close_panel)
    me = pd.DatetimeIndex(month_ends).normalize()
    ma20 = close.rolling(ma_win).mean()
    dev = close / ma20 - 1.0        # 百分比偏离度

    sel = {}
    for t in me:
        if t not in close.index:
            continue
        drow = dev.loc[t].replace([np.inf, -np.inf], np.nan).dropna()
        if len(drow) < 10:
            sel[t] = []
            continue
        # 截面 z 分
        mu, sd = drow.mean(), drow.std(ddof=0)
        zrow = (drow - mu) / sd if sd > 0 else pd.Series(0.0, index=drow.index)
        cand = drow[zrow < z_thr]   # 严重超卖（<-2σ）
        if len(cand) == 0:
            sel[t] = []
            continue
        ranked = cand.sort_values()  # dev 最低（最超卖）优先
        sel[t] = [c for c in ranked.head(top_n).index]
    return sel
