# -*- coding: utf-8 -*-
"""v2 全流程冒烟测试：用合成 eps/roe/index 验证代码路径（不依赖真实抓取）。"""
import os
for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
          "ALL_PROXY", "all_proxy"]:
    os.environ.pop(k, None)
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import config
from factors import compute_rsi, compute_ep, get_month_end_dates
from factor_eval import evaluate_factors
from backtest import run_backtest_v2, BacktestParamsV2
from model import train_lightgbm, predict_signal_panel
from report import compute_metrics, generate_html_v2

np.random.seed(0)
close = pd.read_parquet(config.DATA_DIR / "close_panel.parquet").sort_index()
codes = list(close.columns)
dates = close.index
rng = np.random.default_rng(1)
eps = pd.DataFrame(rng.normal(1.0, 0.5, close.shape), index=dates, columns=codes).clip(0.1, None)
roe = pd.DataFrame(rng.normal(10.0, 5.0, close.shape), index=dates, columns=codes).clip(0.1, None)
npg = pd.DataFrame(rng.normal(20.0, 30.0, close.shape), index=dates, columns=codes)
index_series = pd.Series(rng.normal(4000, 200, len(dates)).cumsum() + 4000, index=dates)
ohlcv = pd.read_pickle(config.DATA_DIR / "ohlcv.pkl")

ep = compute_ep(eps, close)
rsi = compute_rsi(close, 14)
me = get_month_end_dates(dates)
me = me[(me >= pd.Timestamp(config.START_DATE)) & (me <= pd.Timestamp(config.END_DATE))]

# regime
idx = index_series.reindex(dates).ffill()
ma = idx.rolling(240, min_periods=120).mean()
regime_up = (idx > ma).fillna(True)

stats, decay = evaluate_factors(close, npg, ep, roe)
model, imp, mm, log = train_lightgbm(close, ep, roe, npg, ohlcv)
print("TRAIN:", log)
signal = predict_signal_panel(model, close, ep, roe, npg, ohlcv)

eq_with, tw = run_backtest_v2(close, rsi, signal, me, config.START_DATE, config.END_DATE,
                              BacktestParamsV2(use_market_filter=True), regime_up=regime_up)
eq_without, _ = run_backtest_v2(close, rsi, signal, me, config.START_DATE, config.END_DATE,
                               BacktestParamsV2(use_market_filter=False), regime_up=regime_up)
bench = (1 + close.pct_change().mean(axis=1).fillna(0)).cumprod() * 1e6
mw = compute_metrics(eq_with); mwo = compute_metrics(eq_without); mb = compute_metrics(bench)
mwt = compute_metrics(eq_with.loc[config.TRAIN_START:config.TRAIN_END])
mwte = compute_metrics(eq_with.loc[config.TEST_START:config.TEST_END])
print(f"WITH   ann={mw['annual_return']*100:.2f}% dd={mw['max_drawdown']*100:.2f}% sh={mw['sharpe']:.2f}")
print(f"WITHOUT ann={mwo['annual_return']*100:.2f}% dd={mwo['max_drawdown']*100:.2f}% sh={mwo['sharpe']:.2f}")
print(f"TEST   ann={mwte['annual_return']*100:.2f}% dd={mwte['max_drawdown']*100:.2f}% sh={mwte['sharpe']:.2f}")

html = generate_html_v2(eq_with, eq_without, bench, tw, stats, decay, imp,
                        mw, mwo, mb, mwt, mwte, "合成市场环境研判", "合成因果结论")
config.OUTPUT_DIR.joinpath("smoke_v2.html").write_text(html, encoding="utf-8")
print("SMOKE OK ->", config.OUTPUT_DIR / "smoke_v2.html")
