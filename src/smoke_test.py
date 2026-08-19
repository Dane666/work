# -*- coding: utf-8 -*-
"""冒烟测试：用已缓存的少量股票验证全链路代码正确性（数据不完整，仅查错）。"""
import os
for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]:
    os.environ.pop(k, None)
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import config
import data_fetcher as df
from factors import compute_rsi, get_month_end_dates
from factor_eval import evaluate_factors
from backtest import run_backtest, BacktestParams
from model import train_lightgbm, predict_signal_panel, run_ml_backtest
from report import compute_metrics, compute_trade_stats, generate_html

codes = [c[:-8] for c in os.listdir(config.DAILY_DIR) if c.endswith(".parquet")]
print("cached codes:", len(codes))
data = {}
for c in codes:
    d = df.fetch_daily(c, config.START_DATE, config.END_DATE)
    f = df.fetch_financials(c)
    if d is not None:
        data[c] = {"daily": d, "fin": f}
close = df.build_close_panel(data)
npg = df.build_disclosure_aligned_npg(data, close.index).reindex(columns=close.columns)
ohlcv = df.build_ohlcv_dict(data)
print("close", close.shape, "npg", npg.shape)

rsi = compute_rsi(close, 14)
me = get_month_end_dates(close.index)
me = me[(me >= pd.Timestamp(config.START_DATE)) & (me <= pd.Timestamp(config.END_DATE))]
stats, decay = evaluate_factors(close, npg)
print("factor stats:\n", stats.to_string())
eq, tr = run_backtest(close, rsi, npg, me, config.START_DATE, config.END_DATE, BacktestParams())
print("backtest equity points:", len(eq), "trades:", len(tr))
print("metrics:", compute_metrics(eq))
print("trade_stats:", compute_trade_stats(tr))
model, imp, mlm, log = train_lightgbm(close, npg, ohlcv)
print(log)
sig = predict_signal_panel(model, close, npg, ohlcv)
mleq = run_ml_backtest(sig, close, me, config.START_DATE, config.END_DATE, top_n=10)
print("ml metrics:", compute_metrics(mleq))
html = generate_html(eq, eq, tr, stats, decay, imp, compute_metrics(eq),
                     compute_trade_stats(tr), "SMOKE", "SMOKE")
config.OUTPUT_DIR.joinpath("smoke.html").write_text(html, encoding="utf-8")
print("SMOKE OK")
