# -*- coding: utf-8 -*-
"""V5 引擎等价性自检：用 V3 等价选股集合跑 run_backtest_v5，应与 run_backtest_v2 的 V3 结果一致。"""
import os
for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]:
    os.environ.pop(k, None)
import numpy as np, pandas as pd
import config
from factors import compute_rsi, get_month_end_dates
from factor_eval import build_selection_reversal_v3
from backtest import run_backtest_v2, BacktestParamsV2
from backtest_v5 import run_backtest_v5
from model import train_lightgbm, predict_signal_panel
from market_filter import build_ma240_target_weight
from report import compute_metrics


def mask_new_listings(cp, min_days=60):
    out = cp.copy()
    for col in out.columns:
        s = out[col].dropna()
        if s.empty:
            out[col] = np.nan; continue
        out.loc[out.index < s.index.min() + pd.Timedelta(days=min_days), col] = np.nan
    return out


cp = pd.read_parquet(config.V3_CLOSE_PANEL)
oh = pd.read_pickle(config.V3_OHLCV)
idx = pd.read_parquet(config.DATA_DIR / "index.parquet")
if isinstance(idx, pd.DataFrame): idx = idx.iloc[:, 0]
close = mask_new_listings(cp, config.NEW_STOCK_MIN_DAYS)
rsi = compute_rsi(close, config.RSI_WINDOW)
me = get_month_end_dates(close.index)
me = me[(me >= pd.Timestamp(config.START_DATE)) & (me <= pd.Timestamp(config.END_DATE))]
tw = build_ma240_target_weight(idx, close.index, 240)

model, *_ = train_lightgbm(close, oh)
sig = predict_signal_panel(model, close, oh)
params = BacktestParamsV2(pool_pct=0.20, n_select=30, use_market_filter=True)

eq_v3, _ = run_backtest_v2(close, rsi, sig, me, config.START_DATE, config.END_DATE, params, target_weight=tw)
sel_v3 = build_selection_reversal_v3(close, rsi, sig, me, 0.20, 30)
eq_v5, _ = run_backtest_v5(close, sel_v3, me, config.START_DATE, config.END_DATE, target_weight=tw)

m_v3 = compute_metrics(eq_v3); m_v5 = compute_metrics(eq_v5)
print(f"V3(run_backtest_v2): 年化={m_v3['annual_return']*100:.2f}% 回撤={m_v3['max_drawdown']*100:.2f}% 夏普={m_v3['sharpe']:.3f}")
print(f"V3(run_backtest_v5): 年化={m_v5['annual_return']*100:.2f}% 回撤={m_v5['max_drawdown']*100:.2f}% 夏普={m_v5['sharpe']:.3f}")
print(f"夏普差={abs(m_v3['sharpe']-m_v5['sharpe']):.4f} -> {'EQUIVALENT ✓' if abs(m_v3['sharpe']-m_v5['sharpe'])<0.01 else 'MISMATCH ✗'}")
