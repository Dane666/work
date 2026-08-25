# -*- coding: utf-8 -*-
"""
fetch_dividend.py — 股息率因子面板构建（主板版 V2）
================================================================

数据源：ak.stock_history_dividend_detail（新浪历史分红明细，2026-08-25 实测可用，~0.3s/只）
口径：
  - 每股派息 dps = 派息 / 10（接口按每 10 股计）
  - 仅统计进度='实施' 且除权除息日有效的分红（PIT：除息日起数据可得）
  - dividend_yield(t) = 过去 365 天内已实施每股分红合计 / 当日收盘价
  - 无分红记录股票 → 0（不分红 = 股息率 0，非缺失）
输出：
  data/div_yield_panel_mainboard.parquet
    (month_end × code) 月频股息率面板（%），与 mainboard_close_panel 对齐
  data/mainboard_dividend_raw.parquet  断点续传 checkpoint（long: code/ex_date/dps）

运行：cd src && python fetch_dividend.py [--limit N] [--start N]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
          "ALL_PROXY", "all_proxy"]:
    os.environ.pop(k, None)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import akshare as ak

import config

RAW_CKPT = config.DATA_DIR / "mainboard_dividend_raw.parquet"
OUT_PANEL = config.DATA_DIR / "div_yield_panel_mainboard.parquet"

RETRIES = 3
RETRY_SLEEP = 2.0
BATCH = 50
SLEEP = 0.15          # 新浪分红接口实测不限流，0.15s 稳妥


def get_v2_codes() -> list:
    """主板版 V2 选股池：V8 面板（中证500∪创业板∪中证1000 成分）∩ 主板(60/00)。
    数据复用 mainboard 面板（至 2026-08-24），免重新抓取。"""
    v8 = pd.read_parquet(config.DATA_DIR / "v8_close_panel.parquet").columns
    mb = pd.read_parquet(config.MB_CLOSE).columns
    codes = [c for c in v8 if c.startswith(("60", "00")) and c in mb]
    return sorted(codes)


def fetch_div_code(code: str):
    """单只历史分红 → [(ex_date, dps), ...]，仅已实施。

    ValueError('No tables found') 为确定性失败（新股/无分红记录），不重试直接跳过；
    其余网络异常重试 RETRIES 次。
    """
    last = None
    for attempt in range(RETRIES):
        try:
            df = ak.stock_history_dividend_detail(symbol=code, indicator="分红")
            if df is None or len(df) == 0 or "除权除息日" not in df.columns:
                return []
            impl = df[df["进度"] == "实施"].copy()
            impl["除权除息日"] = pd.to_datetime(impl["除权除息日"], errors="coerce")
            impl = impl.dropna(subset=["除权除息日"])
            if impl.empty:
                return []
            dps = pd.to_numeric(impl["派息"], errors="coerce") / 10.0   # 每10股→每股
            out = [(d, float(dps.iloc[i])) for i, d in enumerate(impl["除权除息日"])
                   if dps.iloc[i] == dps.iloc[i] and dps.iloc[i] > 0]
            return out
        except ValueError as e:
            if "No tables found" in str(e):
                return []                       # 确定性失败：不重试
            last = e
            time.sleep(RETRY_SLEEP * (attempt + 1))
        except Exception as e:
            last = e
            time.sleep(RETRY_SLEEP * (attempt + 1))
    print(f"  [warn] {code} 分红获取失败: {repr(last)[:70]}")
    return []


def fetch_all(codes, limit: int = 0, start_at: int = 0):
    if limit:
        codes = codes[start_at:start_at + limit]
    else:
        codes = codes[start_at:]

    done = set()
    parts = []
    if RAW_CKPT.exists():
        prev = pd.read_parquet(RAW_CKPT)
        parts.append(prev)
        done = set(prev["code"].unique().tolist())
        print(f"[dividend] 断点续传：已完成 {len(done)} 只")

    ok = 0
    t0 = time.time()
    for i, code in enumerate(codes, 1):
        if code in done:
            ok += 1
            continue
        rows = fetch_div_code(code)
        if rows:
            parts.append(pd.DataFrame(
                [{"code": code, "ex_date": d, "dps": v} for d, v in rows]))
            ok += 1
        else:
            parts.append(pd.DataFrame([{"code": code, "ex_date": pd.NaT, "dps": np.nan}]))
        if i % BATCH == 0:
            pd.concat(parts, ignore_index=True).to_parquet(RAW_CKPT)
            print(f"  [dividend] 进度 {i}/{len(codes)} 有分红 {ok} "
                  f"耗时 {(time.time()-t0)/60:.1f}min")
        time.sleep(SLEEP)

    combined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["code", "ex_date", "dps"])
    combined.to_parquet(RAW_CKPT)
    print(f"[dividend] raw 完成：{combined['code'].nunique()} 只，"
          f"有效分红行 {combined['dps'].notna().sum()}")
    return combined


def build_panel(raw: pd.DataFrame) -> pd.DataFrame:
    """按月频构建股息率面板（%）。"""
    close = pd.read_parquet(config.MB_CLOSE)
    codes = [c for c in close.columns if c in set(raw["code"])]
    me = pd.DatetimeIndex(close.resample("ME").last().index).normalize()
    me = me[(me >= pd.Timestamp("2018-01-01")) & (me <= close.index[-1])]

    rec = raw.dropna(subset=["dps"]).copy()
    panel = pd.DataFrame(0.0, index=me, columns=codes, dtype=float)
    for c in codes:
        sub = rec[rec["code"] == c][["ex_date", "dps"]]
        if sub.empty:
            continue
        sub = sub.sort_values("ex_date")
        ex = pd.DatetimeIndex(sub["ex_date"])
        dps_arr = sub["dps"].values
        px = close[c].reindex(me)
        for j, t in enumerate(me):
            win = (ex > t - pd.Timedelta(days=365)) & (ex <= t)
            if not win.any():
                continue
            d12 = float(dps_arr[win].sum())
            p = px.iloc[j]
            if p == p and p > 0:
                panel.loc[t, c] = d12 / p * 100.0     # %
    # 覆盖率
    print(f"[dividend] 面板 {panel.shape} 股息率非零率={panel.gt(0).mean().mean():.1%} "
          f"末日有股息率={int(panel.iloc[-1].gt(0).sum())}/{len(codes)}")
    return panel


def main():
    ap = argparse.ArgumentParser(description="股息率因子面板")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--build-only", action="store_true", help="只构建面板（用已有 raw）")
    args = ap.parse_args()

    codes = get_v2_codes()
    print(f"[main] V2 选股池 {len(codes)} 只（V8 主板成分 ∩ mainboard 面板）")

    if not args.build_only:
        raw = fetch_all(codes, limit=args.limit, start_at=args.start)
    else:
        raw = pd.read_parquet(RAW_CKPT)

    panel = build_panel(raw)
    panel.to_parquet(OUT_PANEL)
    print(f"[main] 股息率面板已落盘: {OUT_PANEL}")


if __name__ == "__main__":
    main()
