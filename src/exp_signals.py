# -*- coding: utf-8 -*-
"""对照实验：在 RSI 前20%候选池 + MA240 过滤下，比较不同选股信号。"""
import os
for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
          "ALL_PROXY", "all_proxy"]:
    os.environ.pop(k, None)
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

import config
from factors import compute_rsi, compute_ep, get_month_end_dates
from backtest import run_backtest_v2, BacktestParamsV2
from report import compute_metrics


def zscore_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """逐日横截面 z-score（跨股票）。"""
    return panel.subtract(panel.mean(axis=1), axis=0).divide(panel.std(axis=1), axis=0)


def build_regime(index_series, trade_index, window=240):
    idx = index_series.reindex(trade_index).ffill()
    ma = idx.rolling(window, min_periods=window // 2).mean()
    return (idx > ma).fillna(True).astype(bool)


def main():
    close = pd.read_parquet(config.DATA_DIR / "close_panel.parquet")
    eps = pd.read_parquet(config.DATA_DIR / "eps_panel.parquet").reindex(columns=close.columns)
    roe = pd.read_parquet(config.DATA_DIR / "roe_panel.parquet").reindex(columns=close.columns)
    index_series = pd.read_parquet(config.DATA_DIR / "index.parquet")
    if isinstance(index_series, pd.DataFrame):
        index_series = index_series.iloc[:, 0]
    close = close.sort_index()

    ep = compute_ep(eps, close)
    rsi = compute_rsi(close, 14)
    ret = close.pct_change()
    ret_20 = close.pct_change(20)
    ret_60 = close.pct_change(60)
    ma20 = close / close.rolling(20).mean() - 1.0
    ma60 = close / close.rolling(60).mean() - 1.0

    me = get_month_end_dates(close.index)
    me = me[(me >= pd.Timestamp(config.START_DATE)) & (me <= pd.Timestamp(config.END_DATE))]
    regime = build_regime(index_series, close.index, 240)

    # 各复合信号面板（越高越好）
    z_ep = zscore_panel(ep)
    z_roe = zscore_panel(roe)
    z_rev = (zscore_panel(-ret_20) + zscore_panel(-ret_60)
             + zscore_panel(-ma60) + zscore_panel(-ma20))
    sig_value = z_ep + z_roe                       # B
    sig_rev = z_rev                                 # C
    sig_comb = z_ep + z_roe + z_rev                # D
    sig_oversold = -rsi                            # E（池内最低RSI，纯极端超跌）

    signals = {
        "B 价值质量(EP+ROE)": sig_value,
        "C 反转复合(-ret20-60,-ma)": sig_rev,
        "D 综合(价值+质量+反转)": sig_comb,
        "E 极端超跌(最低RSI)": sig_oversold,
    }

    print(f"{'信号':<28}{'年化':>9}{'回撤':>10}{'夏普':>8}{'卡玛':>8}")
    for name, sig in signals.items():
        eq, _ = run_backtest_v2(
            close, rsi, sig, me, config.START_DATE, config.END_DATE,
            BacktestParamsV2(use_market_filter=True), regime_up=regime)
        m = compute_metrics(eq)
        print(f"{name:<28}{m['annual_return']*100:>8.2f}%{m['max_drawdown']*100:>9.2f}%"
              f"{m['sharpe']:>8.2f}{m['calmar']:>8.2f}")


if __name__ == "__main__":
    main()
