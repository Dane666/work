# -*- coding: utf-8 -*-
"""
主流程 V5：多因子 Regime 切换（仅改 factor_eval.py 的因子合成逻辑）。

设计（严格隔离变量）：V5 与 V3 使用<b>同一回测引擎机制</b>、<b>同一宇宙</b>、<b>同一持仓约束</b>
（fixed_weight=0.10, Top30 候选）、<b>同一 MA240 市场过滤</b>。唯一差异是「满仓时选什么股」：
  - V3        : 始终反转因子集（RSI最低前20% 候选池 + LightGBM Top30）
  - V5        : 每月末依「反转因子 IC 滚动均值」动态切换
                （>0.05 用反转集；≤0.05 用动量/质量集 ret_12+roe+gpm_yoy 的 Z-score Top30）
  - 纯动量/质量: 全程动量/质量集（不切换），用于验证切换是否优于单一因子。

运行：在 src 目录下  python main_v5.py
依赖：先运行 fetch_v3.py 与 fetch_fundamentals_v5.py（ROE + 毛利率同比面板）。
"""

from __future__ import annotations

import os
for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
          "ALL_PROXY", "all_proxy"]:
    os.environ.pop(k, None)

from datetime import datetime

import numpy as np
import pandas as pd

import config
from factors import compute_rsi, get_month_end_dates
from factor_eval import (monthly_reversal_ic, compute_rolling_regime,
                        compute_momentum_zscore, build_selection_v5,
                        build_selection_momentum, build_selection_reversal_v3)
from backtest import run_backtest_v2, BacktestParamsV2
from backtest_v5 import run_backtest_v5
from model import train_lightgbm, predict_signal_panel
from market_filter import build_ma240_target_weight
from report import (compute_metrics, compute_trade_stats, generate_html_v5,
                    yearly_metrics)


# ---------------------------------------------------------------------------
def mask_new_listings(close_panel: pd.DataFrame, min_days: int = 60) -> pd.DataFrame:
    out = close_panel.copy()
    for c in out.columns:
        s = out[c].dropna()
        if s.empty:
            out[c] = np.nan
            continue
        cutoff = s.index.min() + pd.Timedelta(days=min_days)
        out.loc[out.index < cutoff, c] = np.nan
    return out


def month_ends_in(close_panel: pd.DataFrame) -> pd.DatetimeIndex:
    me = get_month_end_dates(close_panel.index)
    return me[(me >= pd.Timestamp(config.START_DATE)) &
              (me <= pd.Timestamp(config.END_DATE))]


def avg_hold_per_month(trades: pd.DataFrame) -> float:
    if trades is None or trades.empty:
        return 0.0
    buys = trades[trades["action"] == "buy"].copy()
    if buys.empty:
        return 0.0
    buys["ym"] = pd.to_datetime(buys["date"]).dt.to_period("M")
    return float(buys.groupby("ym")["code"].nunique().mean())


# ---------------------------------------------------------------------------
def main():
    t0 = datetime.now()
    # ---------- 加载数据 ----------
    close_v3 = pd.read_parquet(config.V3_CLOSE_PANEL)
    ohlcv_v3 = pd.read_pickle(config.V3_OHLCV)
    index_series = pd.read_parquet(config.DATA_DIR / "index.parquet")
    if isinstance(index_series, pd.DataFrame):
        index_series = index_series.iloc[:, 0]
    roe_panel = pd.read_parquet(config.DATA_DIR / "roe_panel_v5.parquet")
    gpm_yoy_panel = pd.read_parquet(config.DATA_DIR / "gpm_yoy_panel_v5.parquet")

    # ---------- 掩码新股 + 计算 RSI + 月末 ----------
    close = mask_new_listings(close_v3, config.NEW_STOCK_MIN_DAYS)
    rsi = compute_rsi(close, config.RSI_WINDOW)
    me = month_ends_in(close)

    # ---------- LightGBM 反转信号（与 V3 同模型）----------
    print(f"[{datetime.now()}] LightGBM 训练（反转因子集）...")
    model, importance, metrics, train_log = train_lightgbm(close, ohlcv_v3)
    reversal_signal = predict_signal_panel(model, close, ohlcv_v3)
    print(train_log)
    importance.to_csv(config.OUTPUT_DIR / "v5_feature_importance.csv", index=False)

    # ---------- 反转因子 IC 滚动 + Regime 判定 ----------
    print(f"[{datetime.now()}] 计算反转因子 IC 滚动与 Regime 切换判定...")
    monthly_ic = monthly_reversal_ic(close, rsi, me, config.FWD_RETURN_DAYS)
    rolling_ic, use_reversal = compute_rolling_regime(
        monthly_ic, ic_window=12, threshold=0.05, min_periods=6)
    # 用于报告的 regime 诊断表
    regime_v5 = pd.DataFrame({
        "reversal_ic": monthly_ic,
        "rolling_ic": rolling_ic,
        "use_reversal": use_reversal,
    })

    # ---------- 动量/质量 Z-score 合成 ----------
    print(f"[{datetime.now()}] 计算动量/质量 Z-score 合成...")
    momentum_zscore = compute_momentum_zscore(close, roe_panel, gpm_yoy_panel, me)

    # ---------- MA240 市场过滤（与 V3 完全相同）----------
    tw_ma240 = build_ma240_target_weight(index_series, close.index, 240)

    # ---------- 构建三条臂的选股集合 ----------
    print(f"[{datetime.now()}] 构建选股集合（V3 / V5 / 纯动量）...")
    params = BacktestParamsV2(pool_pct=0.20, n_select=30, use_market_filter=True)
    # V3 等价选股集合（用于引擎等价性校验）
    sel_v3 = build_selection_reversal_v3(close, rsi, reversal_signal, me, 0.20, 30)
    # V5 动态切换
    sel_v5, switch_log = build_selection_v5(
        close, rsi, reversal_signal, momentum_zscore, me, use_reversal, 0.20, 30)
    # 纯动量/质量（全程不切换）
    sel_mom = build_selection_momentum(momentum_zscore, me, 30)

    switch_log.to_csv(config.OUTPUT_DIR / "v5_switch_log.csv")
    regime_v5.to_csv(config.OUTPUT_DIR / "v5_regime.csv")

    # ---------- 回测 ----------
    print(f"[{datetime.now()}] 回测 V3（run_backtest_v2，权威对照臂）...")
    eq_v3, trades_v3 = run_backtest_v2(
        close, rsi, reversal_signal, me, config.START_DATE, config.END_DATE,
        params, target_weight=tw_ma240)
    trades_v3.to_csv(config.OUTPUT_DIR / "v5_trades_v3.csv", index=False)

    print(f"[{datetime.now()}] 回测 V3（run_backtest_v5 引擎校验）...")
    eq_v3_val, _ = run_backtest_v5(
        close, sel_v3, me, config.START_DATE, config.END_DATE,
        target_weight=tw_ma240)

    print(f"[{datetime.now()}] 回测 V5（动态切换）...")
    eq_v5, trades_v5 = run_backtest_v5(
        close, sel_v5, me, config.START_DATE, config.END_DATE,
        target_weight=tw_ma240)
    trades_v5.to_csv(config.OUTPUT_DIR / "v5_trades.csv", index=False)

    print(f"[{datetime.now()}] 回测 纯动量/质量（不切换）...")
    eq_mom, trades_mom = run_backtest_v5(
        close, sel_mom, me, config.START_DATE, config.END_DATE,
        target_weight=tw_ma240)
    trades_mom.to_csv(config.OUTPUT_DIR / "v5_trades_momentum.csv", index=False)

    # ---------- 指标 ----------
    m_v3 = compute_metrics(eq_v3)
    m_v5 = compute_metrics(eq_v5)
    m_mom = compute_metrics(eq_mom)
    m_v3_val = compute_metrics(eq_v3_val)

    idx_ret = index_series.reindex(close.index).pct_change().fillna(0)
    idx_eq = (1.0 + idx_ret).cumprod() * config.INIT_CAPITAL
    m_idx = compute_metrics(idx_eq)

    m_v3_train = compute_metrics(eq_v3.loc[config.TRAIN_START:config.TRAIN_END])
    m_v3_test = compute_metrics(eq_v3.loc[config.TEST_START:config.TEST_END])
    m_v5_train = compute_metrics(eq_v5.loc[config.TRAIN_START:config.TRAIN_END])
    m_v5_test = compute_metrics(eq_v5.loc[config.TEST_START:config.TEST_END])
    m_mom_train = compute_metrics(eq_mom.loc[config.TRAIN_START:config.TRAIN_END])
    m_mom_test = compute_metrics(eq_mom.loc[config.TEST_START:config.TEST_END])

    # 逐年夏普分解
    ysh_v3 = yearly_metrics(eq_v3)
    ysh_v5 = yearly_metrics(eq_v5)
    ysh_mom = yearly_metrics(eq_mom)
    yearly = pd.DataFrame({"V3": ysh_v3, "V5": ysh_v5, "PureMomentum": ysh_mom})
    yearly.index.name = "year"
    yearly.to_csv(config.OUTPUT_DIR / "v5_yearly_sharpe.csv")

    # 因子 IC 统计（与 V3 同宇宙，复用 evaluate_factors 仅作附录）
    try:
        from factor_eval import evaluate_factors as _ef
        stats_full, decay = _ef(close, ohlcv_v3)
        stats_full.to_csv(config.OUTPUT_DIR / "v5_factor_ic_ir.csv")
        decay.to_csv(config.OUTPUT_DIR / "v5_factor_ic_decay.csv")
    except Exception as e:
        print("factor eval skip:", e)
        stats_full, decay = pd.DataFrame(), pd.DataFrame()

    # 持仓集中度（诚实披露）
    avg_hold_v3 = avg_hold_per_month(trades_v3)
    avg_hold_v5 = avg_hold_per_month(trades_v5)
    avg_hold_mom = avg_hold_per_month(trades_mom)

    # 切换统计
    n_rev = int(use_reversal.sum())
    n_mom = int((~use_reversal).sum())

    # ---------- 报告 ----------
    html = generate_html_v5(
        eq_v3=eq_v3, eq_v5=eq_v5, eq_mom=eq_mom, idx_eq=idx_eq,
        trades_v5=trades_v5, regime_v5=regime_v5, switch_log=switch_log,
        importance=importance, m_v3=m_v3, m_v5=m_v5, m_mom=m_mom, m_idx=m_idx,
        m_v3_train=m_v3_train, m_v3_test=m_v3_test,
        m_v5_train=m_v5_train, m_v5_test=m_v5_test,
        m_mom_train=m_mom_train, m_mom_test=m_mom_test,
        m_v3_val=m_v3_val, yearly=yearly,
        stats_full=stats_full, decay=decay,
        avg_hold_v3=avg_hold_v3, avg_hold_v5=avg_hold_v5, avg_hold_mom=avg_hold_mom,
        n_rev=n_rev, n_mom=n_mom,
    )
    v5_report = config.OUTPUT_DIR / "report_v5.html"
    v5_report.write_text(html, encoding="utf-8")

    print("\n================ V3 vs V5 vs 纯动量 绩效汇总 ================")
    print(f"V5(动态切换)   : 年化={m_v5['annual_return']*100:.2f}% "
          f"回撤={m_v5['max_drawdown']*100:.2f}% 夏普={m_v5['sharpe']:.2f} 卡玛={m_v5['calmar']:.2f}")
    print(f"V3(纯反转满仓) : 年化={m_v3['annual_return']*100:.2f}% "
          f"回撤={m_v3['max_drawdown']*100:.2f}% 夏普={m_v3['sharpe']:.2f} 卡玛={m_v3['calmar']:.2f}")
    print(f"纯动量/质量    : 年化={m_mom['annual_return']*100:.2f}% "
          f"回撤={m_mom['max_drawdown']*100:.2f}% 夏普={m_mom['sharpe']:.2f} 卡玛={m_mom['calmar']:.2f}")
    print(f"真实CSI300指数 : 年化={m_idx['annual_return']*100:.2f}% "
          f"回撤={m_idx['max_drawdown']*100:.2f}% 夏普={m_idx['sharpe']:.2f}")
    print(f"[引擎校验] V3(run_backtest_v2)={m_v3['sharpe']:.2f} vs "
          f"V3(run_backtest_v5)={m_v3_val['sharpe']:.2f}")
    print(f"切换次数: 反转月={n_rev} 动量/质量月={n_mom}  平均持股/月: "
          f"V3≈{avg_hold_v3:.1f} V5≈{avg_hold_v5:.1f} 动量≈{avg_hold_mom:.1f}")
    print(f"[{datetime.now()}] 报告已生成: {v5_report}  耗时 {datetime.now()-t0}")


if __name__ == "__main__":
    main()
