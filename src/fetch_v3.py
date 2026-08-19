# -*- coding: utf-8 -*-
"""
V3 数据抓取：构建「中证500 + 创业板指」扩充宇宙日线面板。

- 成分股：中证500(000905, csindex) ∪ 创业板指(399006, eastmoney 成分接口)
- 剔除：名称含 ST / *ST 的股票
- 日线：akshare 新浪源（前复权），东方财富源在本环境被网络拦截
- 输出：v3_close_panel.parquet（date×code 收盘价）、v3_ohlcv.pkl（逐股 close/volume）
        v3_universe.json（最终剔除ST后的代码列表，便于复现）
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
          "ALL_PROXY", "all_proxy"]:
    os.environ.pop(k, None)

import warnings
warnings.filterwarnings("ignore")

import akshare as ak
import numpy as np
import pandas as pd

import config


def _prefix(code: str) -> str:
    """交易所前缀：6 开头沪市(sh)，其余深市(sz，含创业板 300/301)。"""
    return "sh" + code if code.startswith("6") else "sz" + code


def get_universe() -> list:
    """取中证500 + 创业板指成分并集，剔除 ST。"""
    c500 = ak.index_stock_cons_csindex(symbol="000905")
    cne = ak.index_stock_cons(symbol="399006")  # 399006 csindex 接口解析失败，用 eastmoney 成分接口
    codes = set(c500["成分券代码"].astype(str).str.zfill(6).tolist())
    codes |= set(cne["品种代码"].astype(str).str.zfill(6).tolist())
    # ST 剔除
    names = ak.stock_info_a_code_name()
    st = set(names[names["name"].astype(str).str.contains("ST", na=False)]["code"]
             .astype(str).str.zfill(6).tolist())
    codes -= st
    codes = sorted(codes)
    print(f"宇宙：中证500 {len(c500)} + 创业板指 {len(cne)} → 去重并集 {len(codes)} → 剔除ST后 {len(codes)}")
    return codes


def fetch_one(code: str) -> pd.DataFrame | None:
    """抓取单股新浪日线（前复权）。失败返回 None。"""
    try:
        df = ak.stock_zh_a_daily(symbol=_prefix(code), start_date="20180101",
                                 end_date="20231231", adjust="qfq")
        if df is None or df.empty:
            return None
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        return df[["close", "volume"]]
    except Exception:
        return None


def main():
    t0 = time.time()
    codes = get_universe()
    (config.DATA_DIR / "v3_universe.json").write_text(
        json.dumps(codes), encoding="utf-8")

    frames = {}
    ohlcv = {}
    done = 0
    total = len(codes)
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(fetch_one, c): c for c in codes}
        for fut in as_completed(futs):
            c = futs[fut]
            df = fut.result()
            done += 1
            if df is not None and not df.empty:
                frames[c] = df["close"]
                ohlcv[c] = df[["close", "volume"]]
            if done % 50 == 0:
                print(f"进度 {done}/{total}  已成功 {len(frames)}")

    close_panel = pd.DataFrame(frames).sort_index().sort_index(axis=1)
    close_panel.to_parquet(config.V3_CLOSE_PANEL)
    pd.to_pickle(ohlcv, config.V3_OHLCV)
    print(f"完成：close_panel={close_panel.shape}  ohlcv={len(ohlcv)}  耗时 {time.time()-t0:.0f}s")
    print(f"交易日 {close_panel.index.min().date()} ~ {close_panel.index.max().date()}")


if __name__ == "__main__":
    main()
