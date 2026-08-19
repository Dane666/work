# -*- coding: utf-8 -*-
"""数据获取执行脚本：拉取全样本数据并落盘面板，供后续分析复用。"""
import os
for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
          "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy"]:
    os.environ.pop(k, None)

import time
from datetime import datetime

import numpy as np
import pandas as pd

import config
import data_fetcher as df


def main():
    t0 = datetime.now()
    data = df.load_or_fetch_all(config.START_DATE, config.END_DATE)
    print(f"[{datetime.now()}] 获取 {len(data)} 只股票原始数据")

    close = df.build_close_panel(data)
    npg = df.build_disclosure_aligned_npg(data, close.index)
    npg = npg.reindex(columns=close.columns)  # 对齐列
    ohlcv = df.build_ohlcv_dict(data)

    close.to_parquet(config.DATA_DIR / "close_panel.parquet")
    npg.to_parquet(config.DATA_DIR / "npg_panel.parquet")
    pd.to_pickle(ohlcv, config.DATA_DIR / "ohlcv.pkl")

    print(f"close_panel={close.shape}, npg_panel={npg.shape}")
    print(f"交易日范围: {close.index[0].date()} ~ {close.index[-1].date()}")
    print(f"[{datetime.now()}] 完成，耗时 {datetime.now() - t0}")


if __name__ == "__main__":
    main()
