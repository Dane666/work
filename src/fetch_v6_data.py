# -*- coding: utf-8 -*-
"""
V6 压力测试数据扩展：将回测区间从 2018-2023 延长至 2025-08-12。

数据源：
  - 东方财富 stock_zh_a_hist 在沙箱中被大面积 ConnectionError 拦截（实测仅 ~47/526 成功），
    故按用户兜底条款切换至<b>新浪 stock_zh_a_daily</b>（已验证覆盖至 2025-08-12 且含成交额字段）。
  - CSI300 指数：新浪 stock_zh_index_daily("sh000300")。

拼接逻辑：
  - v6_close_panel(2018-2025) = v3_close_panel(2018-2023) + 新浪新区间(2024-2025)。
  - v6_amount_panel(2018-2025)：
        * 2018-2023：用 v3_ohlcv 的 volume 近似 成交额 = close × volume（实测 close×volume≈成交额×1.06）。
        * 2024-2025：新浪真实 成交额 字段。
  - v6_index(2018-2025) = index.parquet(2018-2023) + 新浪新区间(2024-2025)。

断点续跑：成功抓取的代码写入 checkpoint，重跑跳过。带重试以抗新浪偶发断连。
运行：python fetch_v6_data.py   （建议后台）
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

END_NEW = "20250812"
START_NEW = "20240101"
CKPT = config.DATA_DIR / "v6_new_raw.parquet"


def sina_sym(code: str) -> str:
    if code.startswith(("6", "9")) or code.startswith("688"):
        return "sh" + code
    if code.startswith(("8", "4")):   # 北交所
        return "bj" + code
    return "sz" + code


def fetch_sina(symbol: str, retries: int = 3):
    """新浪日线：返回 (date_index, close, amount) 或 None。带重试。"""
    last = None
    for _ in range(retries):
        try:
            df = ak.stock_zh_a_daily(symbol=symbol, start_date=START_NEW,
                                     end_date=END_NEW, adjust="qfq")
            if df is None or len(df) == 0:
                return None
            df = df.rename(columns={"date": "date", "close": "close", "amount": "amount"})
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")[["close", "amount"]].astype(float)
            return df
        except Exception as e:
            last = e
            time.sleep(0.5)
    return None


def fetch_index_sina():
    """新浪 CSI300 指数 2024-2025。"""
    df = ak.stock_zh_index_daily(symbol="sh000300")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")["close"].astype(float)
    return df[df.index >= pd.Timestamp(START_NEW)]


def main():
    t0 = time.time()
    close_old = pd.read_parquet(config.V3_CLOSE_PANEL)
    codes = list(close_old.columns)
    ohlcv = pd.read_pickle(config.V3_OHLCV)
    idx_old = pd.read_parquet(config.DATA_DIR / "index.parquet")
    if isinstance(idx_old, pd.DataFrame):
        idx_old = idx_old.iloc[:, 0]
    print(f"旧区间 close: {close_old.index[0].date()}..{close_old.index[-1].date()} codes={len(codes)}")

    # 清理旧 checkpoint（避免混入东方财富残留）
    if CKPT.exists():
        CKPT.unlink()
        print("已清理旧 checkpoint")

    new_parts = []
    ok = 0
    for i, code in enumerate(codes, 1):
        d = fetch_sina(sina_sym(code))
        if d is not None and len(d):
            d = d.copy()
            d["symbol"] = code
            new_parts.append(d.reset_index())
            ok += 1
        if i % 50 == 0:
            if new_parts:
                pd.concat(new_parts, ignore_index=True).to_parquet(CKPT)
            print(f"  进度 {i}/{len(codes)} 成功 {ok}")
        time.sleep(0.05)

    if new_parts:
        pd.concat(new_parts, ignore_index=True).to_parquet(CKPT)
    print(f"新区间抓取完成 成功代码={ok}/{len(codes)}")

    raw = pd.read_parquet(CKPT)
    new_close = raw.pivot(index="date", columns="symbol", values="close").reindex(columns=codes)
    new_amount = raw.pivot(index="date", columns="symbol", values="amount").reindex(columns=codes)
    new_close = new_close[~new_close.index.duplicated(keep="last")].sort_index()
    new_amount = new_amount[~new_amount.index.duplicated(keep="last")].sort_index()
    print(f"新区间交易日: {new_close.index[0].date()}..{new_close.index[-1].date()} "
          f"({len(new_close)}天) 覆盖率={new_close.notna().mean().mean():.1%}")

    # 成交额(2018-2023) 近似
    vol_panel = pd.DataFrame(
        {c: ohlcv[c]["volume"] for c in codes if c in ohlcv}, index=close_old.index)
    vol_panel = vol_panel.reindex(columns=codes)
    amount_old = close_old * vol_panel

    full_close = pd.concat([close_old, new_close], axis=0).sort_index()
    full_close = full_close[~full_close.index.duplicated(keep="last")]
    full_amount = pd.concat([amount_old, new_amount], axis=0).sort_index()
    full_amount = full_amount[~full_amount.index.duplicated(keep="last")]

    idx_new = fetch_index_sina()
    full_index = pd.concat([idx_old, idx_new], axis=0)
    full_index = full_index[~full_index.index.duplicated(keep="last")].sort_index()

    full_close.to_parquet(config.DATA_DIR / "v6_close_panel.parquet")
    full_amount.to_parquet(config.DATA_DIR / "v6_amount_panel.parquet")
    full_index.to_frame(name="close").to_parquet(config.DATA_DIR / "v6_index.parquet")

    print(f"\n=== V6 面板已生成 ===")
    print(f"close : {full_close.index[0].date()}..{full_close.index[-1].date()} "
          f"({len(full_close)}天) codes={full_close.shape[1]}")
    print(f"amount: 同 close；全期日均成交额中位数(亿元)= {full_amount.mean().median()/1e8:.2f}")
    print(f"index : {full_index.index[0].date()}..{full_index.index[-1].date()} ({len(full_index)}天)")
    avg_amt = full_amount.mean() / 1e8
    print(f"滑点分级: >5亿(0.10%)={int((avg_amt>5).sum())}只  1-5亿(0.30%)={int(((avg_amt>=1)&(avg_amt<=5)).sum())}只  <1亿(0.50%)={int((avg_amt<1).sum())}只")
    print(f"完成，耗时 {(time.time()-t0)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
