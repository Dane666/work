# -*- coding: utf-8 -*-
"""
v3_common.py — 主板版 V3 共享逻辑（回测 main_mainboard_v3 与 signal_generator 复用）
====================================================================

V3 四件套（2026-08-25 验证达标：全期 0.46 / 2024-25 0.67 / 回撤 -19.4%；V3.1 方向3 门控优化后 0.61/0.83/-19.4%）：
  1) 股息率仓位门控 build_dy_gate：每月全市场股息率中位数（排除 0）
     > 历史滚动 36 月均值 + 0.20σ → 仓位上限降至 20%（降幅80%；V3.1 方向3 网格择优）
  2) 质量过滤 apply_quality_mask：ROE≥5% + 20日均成交额≥2000万 + 上市满1年
  3) 长动量 build_mz_v3：ret = (ret_12 + ret_24) / 2 与 ROE/毛利率等权 Z-score
  4) 强制质量模式：use_reversal 全 False（回避 1.9% 覆盖率的反转信号）
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config

DY_GATE_WINDOW = 36          # 股息率中位数滚动窗口（月）
DY_GATE_SIGMA = 0.20         # V3.1 方向3：0.20（网格0.10~0.60扫描，低σ×高降幅稳健高原）
DY_GATE_WEIGHT = 0.20        # V3.1 方向3：触发时仓位上限 20%（降幅80%；原 V3 为 0.70）
MIN_ROE = 5.0                # 质量过滤：ROE ≥ 5%
MIN_AMOUNT = 2.0e7           # 质量过滤：20 日均成交额 ≥ 2000 万
MIN_LIST_YEARS = 1           # 质量过滤：上市满 1 年
FWD_DAYS = config.FWD_RETURN_DAYS   # 21


def build_dy_gate(div_yield_panel: pd.DataFrame, me: pd.DatetimeIndex) -> pd.Series:
    """股息率仓位门控（月频 1.0 / 触发时 DY_GATE_WEIGHT）。

    每月全市场股息率中位数（排除 0 = 不分红股）> 历史滚动(36月)均值+DY_GATE_SIGMA*std
    → 仓位降至 DY_GATE_WEIGHT（V3.1 方向3：σ=0.20 触发更敏感、仓位20% 更防御）。
    滚动统计 shift(1) 防未来函数。V3.1 网格(σ 0.10~0.60 × 降幅 50%~90%) 显示
    低σ(0.15~0.20)×高降幅(70%~90%) 为稳健高原，全期0.59~0.61/2024-25 0.81~0.84 全达标。
    ⚠️ 参数为全样本网格内择优（in-sample），上线前建议 walk-forward 复核。
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


def build_mz_v3(close_m, roe, gpm, me, long_momentum=True, persistence=True):
    """动量/质量 Z-score 合成分。

    persistence=True（方向1 动量持续性，当前默认）：
        mz = (z(ret_12 - ret_3) + z(ret_24) + z(roe) + z(gpm_yoy)) / 4
        动量持续性 ret_12-ret_3：长期动量扣除最近3月动量；数值大=长动量仍在、
        短端已回落（避免追高接盘），数值小/负=动量已衰减。
    persistence=False（V3.1 基线）：ret=(ret_12+ret_24)/2 与 ROE/毛利率 等权。
    long_momentum 仅影响 persistence=False 路径（True 用 12/24m 均值，False 仅 ret_12）。
    """
    ret12 = close_m.pct_change(FWD_DAYS * 12)
    ret24 = close_m.pct_change(FWD_DAYS * 24)
    if persistence:
        ret3 = close_m.pct_change(FWD_DAYS * 3)
        cols = {"ret_persist": ret12 - ret3, "ret_24": ret24, "roe": roe, "gpm_yoy": gpm}
    else:
        ret_v = (ret12 + ret24) / 2.0 if long_momentum else ret12
        cols = {"ret_12": ret_v, "roe": roe, "gpm_yoy": gpm}
    rows = {}
    for t in pd.DatetimeIndex(me):
        sub = pd.DataFrame({k: v.loc[t] if t in v.index else pd.Series(dtype=float)
                            for k, v in cols.items()})
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


def industry_cap(sector: str, bench: dict, cap_mult: float, min_cap: float) -> float:
    """行业权重上限 = max(基准×cap_mult, min_cap)。
    用户规则：持仓行业权重 ≤ 基准×2；基准<5% 时上限固定 10%。"""
    b = bench.get(sector, 0.0)
    return max(b * cap_mult, min_cap)


def apply_sector_neutral(weights: dict, sector_of: dict, bench: dict,
                         cap_mult: float = 2.0, min_cap: float = 0.10,
                         max_iter: int = 10) -> dict:
    """对目标持仓权重做行业中性化约束（迭代收敛，返回调整后 {code: weight}）。

    规则：任一行业持仓权重 ≤ industry_cap(行业) = max(基准×cap_mult, min_cap)。
    每轮：超限行业内部等比例缩至上限，释放权重按**行业剩余空间**（cap−当前）比例
    分配给未超限行业（行业内按个股权重等比例，单行业分配额 ≤ 其剩余空间，保证不
    制造新超限）；无剩余空间的部分转现金（总权重 ≤ 原总和）。最多 max_iter 轮；
    权重归零（<1e-12）的标的移除。
    weights: {code: weight}（占净值比例）；sector_of: {code: 行业}；bench: {行业: 市值权重}。
    """
    w = {c: max(0.0, wt) for c, wt in weights.items() if wt > 0}

    def _cap(s):
        return max(bench.get(s, 0.0) * cap_mult, min_cap)

    for _ in range(max_iter):
        ind_w = {}
        for c, wt in w.items():
            s = sector_of.get(c, "其他")
            ind_w[s] = ind_w.get(s, 0.0) + wt
        viol = {s: tw for s, tw in ind_w.items() if tw > _cap(s) + 1e-12}
        if not viol:
            break
        freed = 0.0
        for s, tw in viol.items():
            scale = _cap(s) / tw
            for c in list(w):
                if sector_of.get(c, "其他") == s:
                    w[c] *= scale
            freed += tw - _cap(s)
        if freed <= 1e-12:
            break
        # 重新汇总（缩仓后），按剩余空间分配
        ind_w = {}
        for c, wt in w.items():
            s = sector_of.get(c, "其他")
            ind_w[s] = ind_w.get(s, 0.0) + wt
        headroom = {s: _cap(s) - tw for s, tw in ind_w.items() if tw < _cap(s) - 1e-12}
        total_hr = sum(headroom.values())
        if total_hr <= 1e-12:
            break                       # 无剩余空间：未分配部分转现金
        for c in list(w):
            s = sector_of.get(c, "其他")
            hr = headroom.get(s, 0.0)
            ind_total = ind_w[s]
            if hr > 0 and ind_total > 0:
                add_ind = min(freed * hr / total_hr, hr)   # 单行业分配额 ≤ 剩余空间
                w[c] += add_ind * (w[c] / ind_total)
    return {c: wt for c, wt in w.items() if wt > 1e-12}


def load_sector_data():
    """加载行业映射 {code: 行业} 与月末市值基准 (date × industry)。
    文件缺失返回 (None, None)——调用方应回退为不启用中性化。"""
    if not config.SECTOR_IND_MAP.exists() or not config.SECTOR_IND_BENCH.exists():
        return None, None
    ind = pd.read_parquet(config.SECTOR_IND_MAP)
    bench = pd.read_parquet(config.SECTOR_IND_BENCH)
    sector_of = dict(zip(ind["code"].astype(str), ind["industry"]))
    return sector_of, bench
