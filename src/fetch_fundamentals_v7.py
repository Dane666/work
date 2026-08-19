# -*- coding: utf-8 -*-
"""
V7 财务数据补抓（point-in-time，对齐到 2025Q2）：为 V3 宇宙（526 只）抓取
  - 净资产收益率(%) -> roe
  - 销售毛利率(%)   -> 计算毛利率同比(gpm_yoy)
沿用 V5 已验证的「披露延迟映射 + ffill」point-in-time 对齐逻辑（杜绝未来函数），
但输出到独立文件 roe_panel_v7 / gpm_yoy_panel_v7，不覆盖 V5 面板（保证 V5/V6 可复现）。

与 fetch_fundamentals_v5.py 的唯一差别：输出文件名 / checkpoint 名改为 v7。

运行：python fetch_fundamentals_v7.py
依赖：先运行 fetch_v6_data.py（确保 data/v6_close_panel.parquet 存在，提供交易日索引与宇宙）。
可断点续跑：已成功抓取的代码会跳过。
"""

from __future__ import annotations

import os
import time

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
    """返回该代码的 (raw_long_df[code,date,roe,gpm], status)。"""
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

    gpm_yoy_panel = pd.DataFrame(index=trade_idx, columns=codes, dtype=float)
    for code in codes:
        sub = raw[raw["code"] == code][["date", "gpm"]].dropna()
        if sub.empty or len(sub) < 2:
            continue
        sub = sub.copy()
        sub["year"] = sub["date"].dt.year
        annual = sub.groupby("year")["gpm"].last()
        annual_yoy = annual.diff(1)
        av_dates = [available_date(pd.Timestamp(y, 12, 31), LAG_MAP) for y in annual_yoy.index]
        s = pd.Series(annual_yoy.values, index=pd.DatetimeIndex(av_dates))
        s = s[~s.index.duplicated(keep="last")].sort_index()
        union = trade_idx.union(s.index, sort=True)
        full = s.reindex(union).ffill().reindex(trade_idx)
        gpm_yoy_panel[code] = full
    return roe_panel, gpm_yoy_panel


def main():
    t0 = time.time()
    close = pd.read_parquet(config.DATA_DIR / "v6_close_panel.parquet")
    codes = list(close.columns)
    trade_idx = pd.DatetimeIndex(close.index)
    print(f"V3 宇宙 {len(codes)} 只，交易日 {len(trade_idx)} 个（至 {trade_idx[-1].date()}）")

    ckpt = config.DATA_DIR / "fin_v7_raw.parquet"
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
            combined = pd.concat(raw_parts, ignore_index=True)
            combined.to_parquet(ckpt)
            print(f"  进度 {i}/{len(codes)} 成功 {ok} 已存 checkpoint")
        time.sleep(0.25)

    combined = pd.concat(raw_parts, ignore_index=True)
    combined.to_parquet(ckpt)
    print(f"抓取完成 raw 行数={len(combined)} 成功代码={combined['code'].nunique()}")

    roe_panel, gpm_yoy_panel = build_panels(combined, trade_idx, codes)
    roe_panel.to_parquet(config.DATA_DIR / "roe_panel_v7.parquet")
    gpm_yoy_panel.to_parquet(config.DATA_DIR / "gpm_yoy_panel_v7.parquet")
    print(f"roe 非空={roe_panel.notna().sum().sum()}  gpm_yoy 非空={gpm_yoy_panel.notna().sum().sum()}")
    # 覆盖到 2024-2025 的检查
    for yr in [2023, 2024, 2025]:
        m = roe_panel.loc[f"{yr}-06-01":f"{yr}-12-31"]
        print(f"  {yr} H2 roe 月均非空率={m.notna().mean().mean():.1%}")
    print(f"完成，耗时 {(time.time()-t0)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
