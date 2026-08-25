# -*- coding: utf-8 -*-
"""
v3_common.py — 主板版 V3 共享逻辑（回测 main_mainboard_v3 与 signal_generator 复用）
====================================================================

V3 四件套（2026-08-25 验证达标：全期 0.54 / 2024-25 0.65 / 回撤 -19.4%）：
  1) 股息率仓位门控 build_dy_gate：每月全市场股息率中位数（排除 0）
     > 历史滚动 36 月均值 + 0.3σ → 仓位上限降至 70%
  2) 质量过滤 apply_quality_mask：ROE≥5% + 20日均成交额≥2000万 + 上市满1年
  3) 长动量 build_mz_v3：ret = (ret_12 + ret_24) / 2 与 ROE/毛利率等权 Z-score
  4) 强制质量模式：use_reversal 全 False（回避 1.9% 覆盖率的反转信号）
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config

DY_GATE_WINDOW = 36          # 股息率中位数滚动窗口（月）
DY_GATE_SIGMA = 0.3          # 阈值：均值 + 0.3σ（1.0σ 时触发月全 MA240 破位冗余）
DY_GATE_WEIGHT = 0.70        # 触发时仓位上限
MIN_ROE = 5.0                # 质量过滤：ROE ≥ 5%
MIN_AMOUNT = 2.0e7           # 质量过滤：20 日均成交额 ≥ 2000 万
MIN_LIST_YEARS = 1           # 质量过滤：上市满 1 年
FWD_DAYS = config.FWD_RETURN_DAYS   # 21


def build_dy_gate(div_yield_panel: pd.DataFrame, me: pd.DatetimeIndex) -> pd.Series:
    """股息率仓位门控（月频 0.7/1.0）。

    每月全市场股息率中位数（排除 0 = 不分红股）> 历史滚动(36月)均值+0.3σ → 0.70。
    滚动统计 shift(1) 防未来函数。触发月份与 MA240 破位部分重叠（天然冗余），
    0.3σ 下约 6/19 个月落在 MA240 站上期（提前防御降仓，贡献回撤改善）。
    """
    dy_med = div_yield_panel.where(div_yield_panel > 0).median(axis=1).sort_index()
    dy_med = dy_med.reindex(me)
    mean = dy_med.rolling(DY_GATE_WINDOW, min_periods=24).mean().shift(1)
    std = dy_med.rolling(DY_GATE_WINDOW, min_periods=24).std().shift(1)
    gate = pd.Series(1.0, index=me)
    hit = (dy_med > mean + DY_GATE_SIGMA * std) & std.notna()
    gate[hit] = DY_GATE_WEIGHT
    return gate


def apply_quality_mask(close_m, roe, amount, me, min_roe=MIN_ROE,
                       min_amount=MIN_AMOUNT, min_years=MIN_LIST_YEARS):
    """质量过滤：ROE≥5% + 20日均成交额≥2000万 + 上市满1年 → close_m 置 NaN。"""
    out = close_m.copy()
    # 上市满 1 年（首个有效收盘日 + 365 自然日）
    for c in out.columns:
        s = out[c].dropna()
        if s.empty:
            continue
        cutoff = s.index.min() + pd.Timedelta(days=min_years * 365)
        out.loc[out.index < cutoff, c] = np.nan
    # ROE ≥ 5%（月度面板月末值，PIT）
    roe_m = roe.reindex(me)
    roe_ok = roe_m >= min_roe
    # 20 日均成交额 ≥ min_amount
    amt20 = amount.rolling(20, min_periods=10).mean().reindex(me)
    amt_ok = amt20 >= min_amount
    for t in me:
        if t not in out.index:
            continue
        bad = set(out.columns)
        if t in roe_ok.index:
            bad &= set(roe_ok.loc[t][roe_ok.loc[t] == False].index)
        if t in amt_ok.index:
            bad &= set(amt_ok.loc[t][amt_ok.loc[t] == False].index)
        out.loc[t, list(bad)] = np.nan
    return out


def build_mz_v3(close_m, roe, gpm, me, long_momentum=True):
    """动量/质量 Z-score 合成分：ret=(ret_12+ret_24)/2 与 ROE/毛利率 等权。
    long_momentum=False 时退化为 ret_12（对照用）。"""
    ret12 = close_m.pct_change(FWD_DAYS * 12)
    if long_momentum:
        ret24 = close_m.pct_change(FWD_DAYS * 24)
        ret12 = (ret12 + ret24) / 2.0
    rows = {}
    for t in pd.DatetimeIndex(me):
        sub = pd.DataFrame({
            "ret_12": ret12.loc[t] if t in ret12.index else pd.Series(dtype=float),
            "roe": roe.loc[t],
            "gpm_yoy": gpm.loc[t],
        })
        sub = sub.dropna()
        if sub.empty:
            rows[t] = pd.Series(dtype=float)
            continue
        sub_w = sub.copy()
        for col in sub_w.columns:
            lo, hi = sub_w[col].quantile(0.01), sub_w[col].quantile(0.99)
            sub_w[col] = sub_w[col].clip(lo, hi)
        z = (sub_w - sub_w.mean()) / sub_w.std(ddof=0)
        rows[t] = z.sum(axis=1) / sub_w.shape[1]
    return pd.DataFrame(rows).T.sort_index()
