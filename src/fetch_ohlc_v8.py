# -*- coding: utf-8 -*-
"""
抓取 V8 宇宙（1539 只）2018-2025 完整 OHLC（前复权），用于 ATR 卖出规则。

关键工程点——复权基准对齐：
  V8 收盘面板（v8_close_panel.parquet）是 2024 年批次抓取的前复权价，其复权基准
  （以最新价/某除权日为基准）与「现在」重新抓取的前复权价可能不同（期间又有除权）。
  若直接用新抓 high/low 与面板 close 混算 ATR/止损价，会失真（尤其高送转）。
  因此：抓取 qfq 后，用「共同日期的 panel_close / new_close」的中位数比值做常数
  修正，把 OHLC 整体缩放到与 V8 面板同一复权基准。修正后 new_close 应≈panel_close。

落盘：data/v8_ohlcv.pkl → {code: DataFrame(index=date, open, high, low, close)}
断点续跑：每 100 只落盘一次 CKPT；已完成的 code 跳过。
"""

from __future__ import annotations

import os
import time
import pickle
from datetime import datetime

for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
          "ALL_PROXY", "all_proxy"]:
    os.environ.pop(k, None)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import akshare as ak

import config

START_D = "20180101"
END_D = "20250812"
OUT = config.DATA_DIR / "v8_ohlcv.pkl"
CKPT = config.DATA_DIR / "_v8_ohlcv_ckpt.pkl"


def _prefix_sina(code: str) -> str:
    if code.startswith(("6", "9")):
        return "sh" + code
    if code.startswith(("8", "4")):
        return "bj" + code
    return "sz" + code


def fetch_ohlc(code: str, panel_close: pd.Series, retries: int = 3):
    """抓单只 qfq OHLC，并按 panel_close 常数比例修正到 V8 复权基准。"""
    for _ in range(retries):
        try:
            df = ak.stock_zh_a_daily(symbol=_prefix_sina(code), start_date=START_D,
                                     end_date=END_D, adjust="qfq")
            if df is None or len(df) == 0:
                return None
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")[["open", "high", "low", "close"]].astype(float)
            df = df[~df.index.duplicated(keep="last")].sort_index()
            # 复权基准修正：共同日期上 panel_close/new_close 的中位数比值
            common = df.index.intersection(panel_close.dropna().index)
            if len(common) >= 20:
                ratio = float(np.median(panel_close.loc[common] / df.loc[common, "close"]))
                if np.isfinite(ratio) and 0.1 < ratio < 10.0:
                    df[["open", "high", "low", "close"]] *= ratio
            return df
        except Exception:
            time.sleep(0.6)
    return None


def main():
    t0 = datetime.now()
    codes = pd.read_json(config.DATA_DIR / "v8_universe.json").tolist() \
        if isinstance(pd.read_json(config.DATA_DIR / "v8_universe.json"), list) \
        else list(pd.read_json(config.DATA_DIR / "v8_universe.json"))
    # v8_universe.json 是 list of codes
    import json
    codes = json.load(open(config.DATA_DIR / "v8_universe.json"))
    close_panel = pd.read_parquet(config.DATA_DIR / "v8_close_panel.parquet")
    print(f"宇宙 {len(codes)} 只，面板 {close_panel.shape}")

    out: dict = {}
    done = set()
    if CKPT.exists():
        with open(CKPT, "rb") as f:
            out = pickle.load(f)
        done = set(out.keys())
        print(f"断点续跑：已完成 {len(done)} 只")

    ok = 0
    for i, code in enumerate(codes, 1):
        if code in done:
            ok += 1
            continue
        df = fetch_ohlc(code, close_panel[code])
        if df is not None and len(df):
            out[code] = df
            ok += 1
        else:
            print(f"  skip {code}: 抓取失败")
        if i % 100 == 0:
            with open(CKPT, "wb") as f:
                pickle.dump(out, f)
            print(f"  [进度] {i}/{len(codes)} 成功 {ok}  耗时 {datetime.now()-t0}")
        time.sleep(0.05)

    with open(CKPT, "wb") as f:
        pickle.dump(out, f)
    with open(OUT, "wb") as f:
        pickle.dump(out, f)
    print(f"完成：{ok}/{len(codes)} → {OUT}  总耗时 {datetime.now()-t0}")


if __name__ == "__main__":
    main()
