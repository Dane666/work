# -*- coding: utf-8 -*-
"""仅抓取沪深300指数日线（市场过滤器 MA240 用），保存为 DataFrame。"""
import os
for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
          "ALL_PROXY", "all_proxy"]:
    os.environ.pop(k, None)
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import akshare as ak
import config

idx = ak.stock_zh_index_daily(symbol="sh000300")
idx["date"] = pd.to_datetime(idx["date"])
idx = idx.set_index("date")["close"].sort_index()
idx = idx.to_frame(name="close")
idx.to_parquet(config.DATA_DIR / "index.parquet")
print(f"index.parquet 已保存：{idx.shape[0]} 行，"
      f"{idx.index[0].date()} ~ {idx.index[-1].date()}")
