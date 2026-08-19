# -*- coding: utf-8 -*-
"""V3 全链路冒烟测试：合成数据验证新因子/模型/回测/报告代码路径。"""
import os
for k in ["HTTP_PROXY","HTTPS_PROXY","http_proxy","https_proxy","ALL_PROXY","all_proxy"]:
    os.environ.pop(k, None)
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import config
from factors import compute_rsi, compute_skew, compute_turnover_dev, get_month_end_dates
from factor_eval import evaluate_factors
from backtest import run_backtest_v2, BacktestParamsV2
from model import train_lightgbm, predict_signal_panel
from report import compute_metrics, generate_html_v3

np.random.seed(42)
dates = pd.bdate_range("2017-01-01", "2023-12-31")
codes = [f"{i:06d}" for i in range(120)]
# 合成价格：带趋势+噪声
close = pd.DataFrame(index=dates)
ohlcv = {}
for c in codes:
    r = np.random.normal(0.0002, 0.02, len(dates))
    p = 10 * np.exp(np.cumsum(r))
    close[c] = p
    ohlcv[c] = pd.DataFrame({"close": p, "volume": np.abs(np.random.normal(1e6, 3e5, len(dates)))}, index=dates)
close = close.sort_index().sort_index(axis=1)

idx = pd.Series(close.mean(axis=1).values, index=dates)  # 伪指数
rsi = compute_rsi(close, 14)
me = get_month_end_dates(close.index)
me = me[(me >= pd.Timestamp(config.START_DATE)) & (me <= pd.Timestamp(config.END_DATE))]

# 因子评估
stats, decay = evaluate_factors(close, ohlcv)
print("因子统计:\n", stats.to_string())

# 模型
model, imp, mlm, log = train_lightgbm(close, ohlcv)
print(log)
signal = predict_signal_panel(model, close, ohlcv)
print("signal shape", signal.shape)

# 回测 V3 (n=30)
p = BacktestParamsV2(pool_pct=0.20, n_select=30, use_market_filter=True)
eq, trades = run_backtest_v2(close, rsi, signal, me, config.START_DATE, config.END_DATE, p,
                             regime_up=pd.Series(True, index=close.index))
m = compute_metrics(eq)
print(f"V3 eq: 年化={m['annual_return']*100:.2f}% 回撤={m['max_drawdown']*100:.2f}% 夏普={m['sharpe']:.2f} 交易={len(trades)}")

# 报告
html = generate_html_v3(eq, eq, idx, trades, stats, decay, imp, m, m,
                        compute_metrics(idx.reindex(close.index).pct_change().fillna(0).add(1).cumprod()*1e6),
                        m, m, "市场研判(合成)", "因果结论(合成)")
open("/tmp/smoke_v3.html","w").write(html)
print("SMOKE_V3_OK html_bytes=", len(html))
