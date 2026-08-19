# -*- coding: utf-8 -*-
"""
组合回测引擎（模块B）：按「月末个股目标权重表」建仓。

与 run_backtest_v5 的机制完全一致（月末轮换、regime 清仓、分档滑点、现金上限），
唯一区别：买入目标 = 个股目标权重 × 净值（而非 fixed_weight × mult）。
权重表 weight_schedule = {month_end: [(code, weight), ...]}，各 code 权重占净值比例，
总和不强制等于 1（由市场门控 target_weight 决定实际部署，现金自然保留/超买自然截断）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_v5 import _isnan


def run_backtest_combo(close_panel: pd.DataFrame,
                       weight_schedule: dict,
                       month_ends,
                       start: str,
                       end: str,
                       target_weight: pd.Series = None,
                       cost: float = 0.002,
                       init_capital: float = 1_000_000.0,
                       slippage_map: dict = None) -> tuple[pd.Series, pd.DataFrame]:
    all_dates = close_panel.index
    mask = (all_dates >= pd.Timestamp(start)) & (all_dates <= pd.Timestamp(end))
    dates = pd.DatetimeIndex(all_dates[mask])
    close = close_panel.reindex(dates)
    month_end_set = set(pd.DatetimeIndex(month_ends))

    tw = None
    if target_weight is not None:
        tw = target_weight.reindex(dates).astype(float)

    def _c(code):
        return slippage_map.get(code, cost) if slippage_map else cost

    cash = init_capital
    positions: dict = {}   # code -> [shares, entry_px, buy_slip]
    equity: dict = {}
    trades: list = []

    for t in dates:
        ct = close.loc[t]

        # 1) 日内：市场过滤器翻转立即清仓
        if tw is not None and (float(tw.loc[t]) <= 0) and positions:
            for code in list(positions.keys()):
                px = ct.get(code)
                if _isnan(px):
                    continue
                shares, entry, buy_slip = positions[code]
                sl = _c(code)
                cash += px * shares * (1.0 - sl)
                trades.append({"date": t, "code": code, "action": "sell", "price": px,
                               "shares": shares, "notional": px * shares,
                               "reason": "regime_exit",
                               "pnl": (px * (1.0 - sl) - entry * (1.0 + buy_slip)) * shares})
            positions = {}

        # 2) 月末调仓：轮换清仓 → 按权重表重建
        if t in month_end_set:
            for code in list(positions.keys()):
                px = ct.get(code)
                if _isnan(px):
                    continue
                shares, entry, buy_slip = positions[code]
                sl = _c(code)
                cash += px * shares * (1.0 - sl)
                trades.append({"date": t, "code": code, "action": "sell", "price": px,
                               "shares": shares, "notional": px * shares,
                               "reason": "rotate",
                               "pnl": (px * (1.0 - sl) - entry * (1.0 + buy_slip)) * shares})
            positions = {}

            w = 1.0 if tw is None else float(tw.loc[t])
            if w <= 0:
                equity[t] = cash
                continue

            wts = weight_schedule.get(t, [])
            if not wts:
                equity[t] = cash
                continue
            eq = cash
            buy_budget = w * cash
            cash_floor = cash - buy_budget
            for code, wgt in wts:
                if wgt <= 0:
                    continue
                px = ct.get(code)
                if _isnan(px) or cash <= cash_floor:
                    continue
                sl = _c(code)
                target = wgt * eq
                shares = int(target / (px * (1.0 + sl)))
                if shares <= 0:
                    continue
                cost_amt = px * shares * (1.0 + sl)
                if cost_amt > cash:
                    continue
                cash -= cost_amt
                positions[code] = [shares, px, sl]
                trades.append({"date": t, "code": code, "action": "buy", "price": px,
                               "shares": shares, "notional": px * shares,
                               "reason": "entry", "pnl": 0.0})

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
