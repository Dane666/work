# -*- coding: utf-8 -*-
"""
回测引擎模块：事件驱动式日频回测，月度调仓。

买入：月末截面满足 RSI(14) < 30 且 净利润同比增长 > 20% 的股票，按固定权重买入。
卖出（事件驱动，非月末也触发）：
  - RSI(14) > 70 止盈；
  - 自买入价回撤 <= -10% 止损。
成本：单边千二。仓位：单只固定占净值比例，总仓 <= 100%。

所有信号均基于「截至当日已可得」的数据（RSI 当日收盘、财报按披露时点对齐），
杜绝未来函数。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestParams:
    init_capital: float = 1_000_000.0
    rsi_buy: float = 30.0
    rsi_sell: float = 70.0
    npg_min: float = 20.0
    stop_loss: float = -0.10
    cost: float = 0.002
    fixed_weight: float = 0.10
    cond_rsi: bool = True   # 是否要求 RSI 超跌条件
    cond_npg: bool = True   # 是否要求净利润同比增长条件


def run_backtest(close_panel: pd.DataFrame,
                 rsi_panel: pd.DataFrame,
                 npg_panel: pd.DataFrame,
                 month_ends: pd.DatetimeIndex,
                 start: str,
                 end: str,
                 params: BacktestParams,
                 regime_up: pd.Series = None) -> tuple[pd.Series, pd.DataFrame]:
    """运行回测，返回 (权益曲线 Series, 交易明细 DataFrame)。"""
    all_dates = close_panel.index
    mask = (all_dates >= pd.Timestamp(start)) & (all_dates <= pd.Timestamp(end))
    dates = pd.DatetimeIndex(all_dates[mask])

    close = close_panel.reindex(dates)
    rsi = rsi_panel.reindex(dates)
    npg = npg_panel.reindex(dates)
    codes = list(close.columns)
    month_end_set = set(pd.DatetimeIndex(month_ends))

    cash = params.init_capital
    positions: dict[str, list] = {}  # code -> [shares, entry_price]
    equity: dict = {}
    trades: list = []

    for t in dates:
        ct = close.loc[t]
        rt = rsi.loc[t]
        nt = npg.loc[t]

        # 1) 持仓退出检查（止盈/止损，任何交易日都可能触发）
        for code in list(positions.keys()):
            px = ct.get(code)
            if px is None or (isinstance(px, float) and np.isnan(px)):
                continue
            entry = positions[code][1]
            rsi_val = rt.get(code)
            ret = px / entry - 1.0
            reason = None
            if (not _isnan(rsi_val)) and rsi_val > params.rsi_sell:
                reason = "take_profit_rsi"
            elif ret <= params.stop_loss:
                reason = "stop_loss"
            if reason:
                shares = positions[code][0]
                proceeds = px * shares * (1.0 - params.cost)
                cash += proceeds
                pnl = (px * (1.0 - params.cost) - entry * (1.0 + params.cost)) * shares
                trades.append({
                    "date": t, "code": code, "action": "sell", "price": px,
                    "shares": shares, "notional": px * shares, "reason": reason,
                    "pnl": pnl,
                })
                del positions[code]

        # 2) 月末调仓：筛选超跌绩优并买入
        if t in month_end_set:
            # 市场状态过滤（regime gating）：趋势向下时空仓，不启用策略
            if regime_up is not None and not _truthy(regime_up.get(t)):
                for code, (shares, _) in list(positions.items()):
                    px = ct.get(code)
                    if not _isnan(px):
                        cash += px * shares * (1.0 - params.cost)
                        trades.append({
                            "date": t, "code": code, "action": "sell",
                            "price": px, "shares": shares,
                            "notional": px * shares, "reason": "regime_exit",
                            "pnl": (px * (1.0 - params.cost)
                                    - positions[code][1] * (1.0 + params.cost)) * shares,
                        })
                positions = {}
                # 跳过本轮选股
                eq = cash
                equity[t] = eq
                continue

            eq = cash + sum(
                positions[c][0] * ct.get(c)
                for c in positions
                if not _isnan(ct.get(c))
            )
            selected = []
            for code in codes:
                rsi_val = rt.get(code)
                npg_val = nt.get(code)
                px = ct.get(code)
                if _isnan(px):
                    continue
                ok_rsi = (not params.cond_rsi) or (
                    (not _isnan(rsi_val)) and rsi_val < params.rsi_buy
                )
                ok_npg = (not params.cond_npg) or (
                    (not _isnan(npg_val)) and npg_val > params.npg_min
                )
                if not (ok_rsi and ok_npg):
                    continue
                if code not in positions:
                    selected.append(code)
            for code in selected:
                if cash <= 0:
                    break
                px = ct.get(code)
                target = params.fixed_weight * eq
                shares = int(target / (px * (1.0 + params.cost)))
                max_shares = int(cash / (px * (1.0 + params.cost)))
                shares = min(shares, max_shares)
                if shares <= 0:
                    continue
                cost = px * shares * (1.0 + params.cost)
                cash -= cost
                positions[code] = [shares, px]
                trades.append({
                    "date": t, "code": code, "action": "buy", "price": px,
                    "shares": shares, "notional": px * shares, "reason": "entry",
                    "pnl": 0.0,
                })

        # 3) 每日盯市
        eq = cash + sum(
            positions[c][0] * ct.get(c)
            for c in positions
            if not _isnan(ct.get(c))
        )
        equity[t] = eq

    equity_series = pd.Series(equity).sort_index()
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df["date"] = pd.to_datetime(trades_df["date"])
        trades_df = trades_df.sort_values("date").reset_index(drop=True)
    return equity_series, trades_df


def _isnan(x) -> bool:
    """安全的 NaN 判断（兼容 numpy/pandas/原生类型）。"""
    try:
        return x is None or (isinstance(x, float) and np.isnan(x))
    except (TypeError, ValueError):
        return False


def _truthy(x) -> bool:
    """将 Series 取值安全转为布尔（兼容 numpy.bool_ / NaN）。"""
    try:
        return bool(x) and not (isinstance(x, float) and np.isnan(x))
    except (TypeError, ValueError):
        return False


# ----------------------------------------------------------------------------
# v2 回测引擎：RSI 排名分位候选池 + LightGBM 选股 + MA240 市场过滤
# ----------------------------------------------------------------------------
@dataclass
class BacktestParamsV2:
    init_capital: float = 1_000_000.0
    rsi_period: int = 14
    pool_pct: float = 0.20       # 候选池：月末 RSI 最低的前 pool_pct（超跌分位）
    n_select: int = 10           # 候选池内按 ML 预测收益选 Top-N
    cost: float = 0.002          # 单边交易成本
    fixed_weight: float = 0.10   # 单只目标仓位占净值比例
    use_market_filter: bool = True   # 是否启用 MA240 市场过滤器
    ma_window: int = 240         # 市场均线窗口（≈20 个月）


def run_backtest_v2(close_panel: pd.DataFrame,
                    rsi_panel: pd.DataFrame,
                    signal_panel: pd.DataFrame,
                    month_ends: pd.DatetimeIndex,
                    start: str,
                    end: str,
                    params: BacktestParamsV2,
                    regime_up: pd.Series = None,
                    target_weight: pd.Series = None) -> tuple[pd.Series, pd.DataFrame]:
    """v2 回测：RSI 最低前 pool_pct 为候选池，池内按 ML 信号选 Top-N 等权持有。

    市场过滤（二选一）：
      - regime_up（旧，V3 兼容）：日频布尔序列，True=可满仓，False=空仓；
      - target_weight（新，V4）：日频 0/0.5/1.0 序列，精确控制月度部署资金比例，
        盘中跌破 MA240（值归 0）立即清仓。若同时传入 target_weight，优先使用它。
    二者均为 None 时不做过滤（满仓）。
    """
    all_dates = close_panel.index
    mask = (all_dates >= pd.Timestamp(start)) & (all_dates <= pd.Timestamp(end))
    dates = pd.DatetimeIndex(all_dates[mask])
    close = close_panel.reindex(dates)
    rsi = rsi_panel.reindex(dates)
    signal = signal_panel.reindex(dates) if signal_panel is not None else None
    codes = list(close.columns)
    month_end_set = set(pd.DatetimeIndex(month_ends))

    # 统一为日频权重序列 tw（None 表示不限仓，恒为 1.0）
    tw = None
    if target_weight is not None:
        tw = target_weight.reindex(dates).astype(float)
    elif (params.use_market_filter and regime_up is not None):
        tw = regime_up.reindex(dates).astype(float)  # 1.0/0.0，兼容 V3

    cash = params.init_capital
    positions: dict = {}
    equity: dict = {}
    trades: list = []

    for t in dates:
        ct = close.loc[t]
        up = (tw is None) or (float(tw.loc[t]) > 0)

        # 1) 日内：市场过滤器翻转（权重归 0，指数跌破均线）立即清仓
        if tw is not None and (float(tw.loc[t]) <= 0) and positions:
            for code, (shares, entry) in list(positions.items()):
                px = ct.get(code)
                if _isnan(px):
                    continue
                cash += px * shares * (1.0 - params.cost)
                trades.append({
                    "date": t, "code": code, "action": "sell", "price": px,
                    "shares": shares, "notional": px * shares, "reason": "regime_exit",
                    "pnl": (px * (1.0 - params.cost)
                            - entry * (1.0 + params.cost)) * shares,
                })
            positions = {}

        # 2) 月末调仓（先 rotation 清仓，再按市场状态决定是否买入）
        if t in month_end_set:
            for code, (shares, entry) in list(positions.items()):
                px = ct.get(code)
                if _isnan(px):
                    continue
                cash += px * shares * (1.0 - params.cost)
                trades.append({
                    "date": t, "code": code, "action": "sell", "price": px,
                    "shares": shares, "notional": px * shares, "reason": "rotate",
                    "pnl": (px * (1.0 - params.cost)
                            - entry * (1.0 + params.cost)) * shares,
                })
            positions = {}

            # 市场状态仓位门控：w=当日目标仓位比例（tw 为 None 时恒满仓）
            w = 1.0 if tw is None else float(tw.loc[t])
            if w <= 0:
                equity[t] = cash
                continue  # 市场向下（或过滤判空仓）：本月末不买入

            # 候选池：月末截面 RSI 最低的前 pool_pct
            rsi_row = rsi.loc[t].dropna()
            if len(rsi_row) >= 5:
                k = max(1, int(round(len(rsi_row) * params.pool_pct)))
                pool = set(rsi_row.nsmallest(k).index)
            else:
                pool = set(rsi_row.index)

            # 池内按 ML 预测下月收益选 Top-N
            if signal is not None and pool:
                sig_row = signal.loc[t].dropna()
                ranked = sig_row[sig_row.index.isin(pool)].sort_values(
                    ascending=False).head(params.n_select)
                sel = [c for c in ranked.index if (not _isnan(ct.get(c)))]
            else:
                sel = [c for c in pool if not _isnan(ct.get(c))]
            sel = sel[:params.n_select]

            if sel and cash > 0:
                eq = cash
                # 按目标仓位比例控制月度部署资金：w=1 满仓，w=0.5 半仓
                buy_budget = w * cash
                cash_floor = cash - buy_budget
                for code in sel:
                    px = ct.get(code)
                    if _isnan(px) or cash <= cash_floor:
                        continue
                    target = params.fixed_weight * eq
                    shares = int(target / (px * (1.0 + params.cost)))
                    if shares <= 0:
                        continue
                    cost_amt = px * shares * (1.0 + params.cost)
                    if cost_amt > cash:
                        continue
                    cash -= cost_amt
                    positions[code] = [shares, px]
                    trades.append({
                        "date": t, "code": code, "action": "buy", "price": px,
                        "shares": shares, "notional": px * shares,
                        "reason": "entry", "pnl": 0.0,
                    })

        # 3) 每日盯市
        eq = cash + sum(
            v[0] * ct.get(c) for c, v in positions.items() if not _isnan(ct.get(c))
        )
        equity[t] = eq

    equity_series = pd.Series(equity).sort_index()
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df["date"] = pd.to_datetime(trades_df["date"])
        trades_df = trades_df.sort_values("date").reset_index(drop=True)
    return equity_series, trades_df
