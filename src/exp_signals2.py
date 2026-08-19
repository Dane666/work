# -*- coding: utf-8 -*-
"""扩展实验：真实CSI300基准 + 提升夏普的可行变体。"""
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


def zscore_panel(panel):
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

    # 真实 CSI300 指数买入持有基准（价格指数，无幸存者偏差）
    idx_ret = index_series.reindex(close.index).pct_change().fillna(0)
    idx_eq = (1 + idx_ret).cumprod() * 1_000_000
    mi = compute_metrics(idx_eq)
    print(f"【真实CSI300指数买入持有】年化={mi['annual_return']*100:.2f}% "
          f"回撤={mi['max_drawdown']*100:.2f}% 夏普={mi['sharpe']:.2f}")

    z_ep = zscore_panel(ep); z_roe = zscore_panel(roe)
    z_rev = (zscore_panel(-ret_20) + zscore_panel(-ret_60)
             + zscore_panel(-ma60) + zscore_panel(-ma20))
    sig_comb = z_ep + z_roe + z_rev

    print(f"\n{'配置':<40}{'年化':>9}{'回撤':>10}{'夏普':>8}{'卡玛':>8}")
    variants = []
    # (标签, pool_pct, n_select, signal, filter, is_ml)
    variants.append(("原策略:池20%选10 ML信号+过滤", 0.20, 10, "ML", True, True))
    # 扩大持仓分散度
    variants.append(("池20%选20 反转+过滤", 0.20, 20, sig_comb, True, False))
    variants.append(("池20%选30 反转+过滤", 0.20, 30, sig_comb, True, False))
    # 放宽候选池（含较超跌但未极端）
    variants.append(("池40%选20 反转+过滤", 0.40, 20, sig_comb, True, False))
    variants.append(("池40%选30 反转+过滤", 0.40, 30, sig_comb, True, False))
    # 全市场反转排名（放弃RSI池约束，仅作探索对照）
    variants.append(("全市场反转选20+过滤(探索)", 1.01, 20, z_rev, True, False))
    variants.append(("全市场综合选30+过滤(探索)", 1.01, 30, sig_comb, True, False))

    for name, pool, nsel, sig, filt, is_ml in variants:
        p = BacktestParamsV2(pool_pct=pool, n_select=nsel,
                             use_market_filter=filt)
        if is_ml:
            from model import train_lightgbm, predict_signal_panel
            ohlcv = pd.read_pickle(config.DATA_DIR / "ohlcv.pkl")
            npg = pd.read_parquet(config.DATA_DIR / "npg_panel.parquet").reindex(columns=close.columns)
            model, _, _, _ = train_lightgbm(close, ep, roe, npg, ohlcv)
            sig_panel = predict_signal_panel(model, close, ep, roe, npg, ohlcv)
        else:
            sig_panel = sig
        eq, _ = run_backtest_v2(close, rsi, sig_panel, me,
                                config.START_DATE, config.END_DATE, p,
                                regime_up=regime if filt else None)
        m = compute_metrics(eq)
        print(f"{name:<40}{m['annual_return']*100:>8.2f}%{m['max_drawdown']*100:>9.2f}%"
              f"{m['sharpe']:>8.2f}{m['calmar']:>8.2f}")


if __name__ == "__main__":
    main()
