# -*- coding: utf-8 -*-
"""
V5 财务数据补抓：为 V3 宇宙（中证500+创业板指，526 只）抓取
  - 净资产收益率(%) -> roe
  - 销售毛利率(%)   -> 计算毛利率同比(gpm_yoy)
按真实披露时点 + 延迟映射做点对点对齐（杜绝未来函数），构建日频面板。

运行：python fetch_fundamentals_v5.py
依赖：先运行 fetch_v3.py（确保 data/v3_close_panel.parquet 存在）。
可断点续跑：已成功抓取的代码会跳过。
"""

from __future__ import annotations

import os
import time
import json

for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
          "ALL_PROXY", "all_proxy"]:
    os.environ.pop(k, None)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import akshare as ak

import config


LAG_MAP = config.DISCLOSURE_LAG_MONTHS  # 03-31:1, 06-30:2, 09-30:1, 12-31:5


def available_date(report_date: pd.Timestamp, lag_map: dict) -> pd.Timestamp:
    """由报告期与披露延迟映射，得到数据「可用起始日」（通常为次月 1 日）。"""
    lag = lag_map.get(report_date.strftime("%m-%d"), 4)
    y, m = report_date.year, report_date.month
    m0 = m + lag
    y += (m0 - 1) // 12
    m = (m0 - 1) % 12 + 1
    return pd.Timestamp(y, m, 1)


def fetch_code(code: str):
    """返回该代码的 (raw_long_df[code,date,roe,gpm])，失败返回 None。"""
    try:
        fi = ak.stock_financial_analysis_indicator(symbol=code)
    except Exception as e:
        return None, f"fetch_err:{repr(e)[:80]}"
    if "日期" not in fi.columns:
        return None, "no_date_col"
    rows = []
    for _, r in fi.iterrows():
        rd = r["日期"]
        if rd is None or (isinstance(rd, float) and np.isnan(rd)):
            continue
        if not isinstance(rd, pd.Timestamp):
            try:
                rd = pd.Timestamp(rd)
            except Exception:
                continue
        roe = r.get("净资产收益率(%)")
        gpm = r.get("销售毛利率(%)")
        rows.append((code, rd, roe, gpm))
    if not rows:
        return None, "empty"
    df = pd.DataFrame(rows, columns=["code", "date", "roe", "gpm"])
    return df, "ok"


def build_panels(raw_long: pd.DataFrame, trade_idx: pd.DatetimeIndex, codes: list):
    """由长表构建 point-in-time 日频面板：roe_panel, gpm_yoy_panel。"""
    # 仅保留有效数值
    raw = raw_long.dropna(subset=["roe", "gpm"], how="all").copy()

    def point_in_time(col):
        out = pd.DataFrame(index=trade_idx, columns=codes, dtype=float)
        for code in codes:
            sub = raw[raw["code"] == code][["date", col]].dropna()
            if sub.empty:
                continue
            sub = sub.copy()
            sub["av"] = sub["date"].map(lambda d: available_date(d, LAG_MAP))
            s = pd.Series(sub[col].values, index=pd.DatetimeIndex(sub["av"].values))
            s = s[~s.index.duplicated(keep="last")].sort_index()
            union = trade_idx.union(s.index, sort=True)
            full = s.reindex(union).ffill().reindex(trade_idx)
            out[code] = full
        return out

    roe_panel = point_in_time("roe")

    # gpm_yoy：在「年度」粒度上做同比。先取每年最后一个报告（最完整的年报口径），再 diff(1)。
    gpm_yoy_panel = pd.DataFrame(index=trade_idx, columns=codes, dtype=float)
    for code in codes:
        sub = raw[raw["code"] == code][["date", "gpm"]].dropna()
        if sub.empty or len(sub) < 2:
            continue
        sub = sub.copy()
        sub["year"] = sub["date"].dt.year
        # 每年取最后一个报告值（年报最全）
        annual = sub.groupby("year")["gpm"].last()
        annual_yoy = annual.diff(1)  # 毛利率同比
        # 映射到「可用日」：用当年 12-31 报告的可用日（lag=5 -> 次年 5/1）
        av_dates = [available_date(pd.Timestamp(y, 12, 31), LAG_MAP) for y in annual_yoy.index]
        s = pd.Series(annual_yoy.values, index=pd.DatetimeIndex(av_dates))
        s = s[~s.index.duplicated(keep="last")].sort_index()
        union = trade_idx.union(s.index, sort=True)
        full = s.reindex(union).ffill().reindex(trade_idx)
        gpm_yoy_panel[code] = full
    return roe_panel, gpm_yoy_panel


def main():
    t0 = time.time()
    close = pd.read_parquet(config.V3_CLOSE_PANEL)
    codes = list(close.columns)
    trade_idx = pd.DatetimeIndex(close.index)
    print(f"V3 宇宙 {len(codes)} 只，交易日 {len(trade_idx)} 个")

    ckpt = config.DATA_DIR / "fin_v5_raw.parquet"
    done = set()
    raw_parts = []
    if ckpt.exists():
        prev = pd.read_parquet(ckpt)
        raw_parts.append(prev)
        done = set(prev["code"].unique().tolist())
        print(f"断点续跑：已完成 {len(done)} 只")

    ok = 0
    for i, code in enumerate(codes, 1):
        if code in done:
            ok += 1
            continue
        df, status = fetch_code(code)
        if df is not None:
            raw_parts.append(df)
            ok += 1
        else:
            print(f"  skip {code}: {status}")
        if i % 25 == 0:
            # 每 25 只 checkpoint
            combined = pd.concat(raw_parts, ignore_index=True)
            combined.to_parquet(ckpt)
            print(f"  进度 {i}/{len(codes)} 成功 {ok} 已存 checkpoint")
        time.sleep(0.25)

    combined = pd.concat(raw_parts, ignore_index=True)
    combined.to_parquet(ckpt)
    print(f"抓取完成 raw 行数={len(combined)} 成功代码={combined['code'].nunique()}")

    roe_panel, gpm_yoy_panel = build_panels(combined, trade_idx, codes)
    roe_panel.to_parquet(config.DATA_DIR / "roe_panel_v5.parquet")
    gpm_yoy_panel.to_parquet(config.DATA_DIR / "gpm_yoy_panel_v5.parquet")
    print(f"roe 非空={roe_panel.notna().sum().sum()}  gpm_yoy 非空={gpm_yoy_panel.notna().sum().sum()}")
    print(f"完成，耗时 {(time.time()-t0)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
