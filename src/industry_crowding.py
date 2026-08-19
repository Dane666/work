# -*- coding: utf-8 -*-
"""
V9 模块2：行业拥挤度风控（point-in-time，零未来函数）。

逻辑（每月末 / 每周末执行）：
  1. 对每个行业，计算其「过去 20 日总成交额」占「全市场过去 20 日总成交额」的比例 → 行业成交占比。
  2. 回看过去 3 年（756 交易日），取该行业成交占比的 75% 分位数（仅用 ≤ 决策日 t 的历史）。
  3. 若当前占比 > 历史 75% 分位 → 该行业标记为「拥挤」。
  4. 对拥挤行业内的入选个股，目标仓位打 7 折（100% → 70%），不改变选股集合本身。

行业分类：申万一级（由 index_component_sw 抓取，落盘 data/sw_industry_map.parquet）。
未分类股票（不在任何申万一级成分中）不参与拥挤判定，权重不打折。

输出：compute_crowding_weight_mult(...) → dict[(rebalance_date, code)] = 折扣系数(1.0 或 0.7)。
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

import config


def fetch_industry_map() -> pd.DataFrame:
    """抓取申万一级行业成分，落盘 data/sw_industry_map.parquet（code, industry, name）。

    仅 31 次调用 index_component_sw，断点续跑友好（已存在则直接读取）。
    """
    for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
              "ALL_PROXY", "all_proxy"]:
        os.environ.pop(k, None)
    out = config.DATA_DIR / "sw_industry_map.parquet"
    if out.exists():
        return pd.read_parquet(out)

    import akshare as ak
    first = ak.sw_index_first_info()  # 31 申万一级
    rows = []
    for _, r in first.iterrows():
        code = str(r["行业代码"]).split(".")[0]  # 801010
        name = r["行业名称"]
        cons = ak.index_component_sw(symbol=code)
        for _, c in cons.iterrows():
            rows.append({
                "code": str(c["证券代码"]).zfill(6),
                "industry_code": code,
                "industry": name,
            })
    df = pd.DataFrame(rows).drop_duplicates("code")
    df.to_parquet(out)
    print(f"[industry] 抓取申万一级映射完成：{df['industry'].nunique()} 行业 / "
          f"{len(df)} 只股票 -> {out}")
    return df


def load_industry_map() -> dict:
    df = fetch_industry_map()
    return dict(zip(df["code"], df["industry"]))


def compute_crowding_weight_mult(close_panel: pd.DataFrame,
                                 amount_panel: pd.DataFrame,
                                 industry_map: dict,
                                 rebalance_dates,
                                 vol_window: int = 20,
                                 lookback: int = 756,
                                 discount: float = None) -> dict:
    """返回 {(rebalance_date, code): 折扣系数}。

    折扣系数 = discount（默认 config.ANALYST_CROWDED_DISCOUNT=0.70）当该股行业在 t 拥挤，
    否则 1.0。无覆盖股票恒为 1.0。
    """
    if discount is None:
        discount = getattr(config, "ANALYST_CROWDED_DISCOUNT", 0.70)
    dates = close_panel.index
    amount = amount_panel.reindex(dates).fillna(0.0)  # 缺失成交额按 0 计（拥挤度代理）

    # 全市场 20 日成交额（分母）
    mkt_20 = amount.rolling(vol_window).sum().sum(axis=1)

    # 各行业 20 日成交额占比面板（date × industry）
    industries = sorted(set(industry_map.values()))
    code_to_ind = industry_map
    share = pd.DataFrame(index=dates, columns=industries, dtype=float)
    for ind in industries:
        members = [c for c in close_panel.columns if code_to_ind.get(c) == ind]
        if not members:
            continue
        ind_20 = amount[members].rolling(vol_window).sum().sum(axis=1)
        share[ind] = ind_20 / mkt_20
    share = share.ffill().fillna(0.0)

    # 各行业 3 年滚动 75% 分位（point-in-time：shift(1) 确保阈值仅由「t 之前」的
    # 历史构成，不含当日自身，避免自我包含削弱信号；亦严格无未来函数）
    rolling_75 = share.rolling(lookback, min_periods=250).quantile(0.75).shift(1)
    crowded = share > rolling_75  # date × industry 布尔；阈值未成熟期为 NaN → False

    # 映射回 (rebalance_date, code)
    out = {}
    for t in pd.DatetimeIndex(rebalance_dates):
        if t not in crowded.index:
            continue
        crowded_now = crowded.loc[t]
        for c in close_panel.columns:
            ind = code_to_ind.get(c)
            if ind is None:
                out[(t, c)] = 1.0
                continue
            out[(t, c)] = discount if (ind in crowded_now and bool(crowded_now[ind])) else 1.0
    return out
