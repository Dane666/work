# -*- coding: utf-8 -*-
"""
路线2 熔断回测引擎：基于「策略自身净值回撤」的动态仓位（单遍自洽，零未来函数）。

机制（用户规格，状态由昨日收盘净值决定、今日执行，避免同日 look-ahead）：
  - 每日收盘后更新策略净值 eq 与历史峰值 cummax；
  - 回撤 dd = eq/cummax - 1：
      dd ≤ -dd_red (20%)  → 月末建仓权重 × red_mult (0.5)；
      dd ≤ -dd_clear (25%) → 次日强制清仓并暂停交易（fuse_clear），
                             直至净值创历史新高（eq > cummax）恢复 100%；
      eq 创新高 → 恢复满仓。
  - 其余机制与 run_backtest_v5 一致：月末轮换、regime 清仓、分档滑点、fixed_weight 建仓。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_v5 import _isnan


def run_backtest_fuse(close_panel: pd.DataFrame,
                      selection: dict,
                      month_ends,
                      start: str,
                      end: str,
                      target_weight: pd.Series = None,
                      fixed_weight: float = 0.10,
                      cost: float = 0.002,
                      init_capital: float = 1_000_000.0,
                      slippage_map: dict = None,
                      weight_mult: dict = None,
                      dd_red: float = 0.20,
                      dd_clear: float = 0.25,
                      red_mult: float = 0.5) -> tuple[pd.Series, pd.DataFrame, dict]:
    all_dates = close_panel.index
    mask = (all_dates >= pd.Timestamp(start)) & (all_dates <= pd.Timestamp(end))
    dates = pd.DatetimeIndex(all_dates[mask])
    close = close_panel.reindex(dates)
    codes = list(close.columns)
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
    stats = {"n_clear": 0, "n_paused_days": 0, "n_red_days": 0,
             "n_recover": 0, "max_dd_reached": 0.0}

    cummax = init_capital
    prev_eq = init_capital      # 昨日收盘净值（决定今日熔断状态）
    paused = False

    for t in dates:
        ct = close.loc[t]

        # 昨日收盘状态 → 今日仓位乘数（月末建仓用）
        dd_prev = prev_eq / cummax - 1.0 if cummax > 0 else 0.0
        red_active = (not paused) and dd_prev <= -dd_red

        # 1) 熔断清仓：昨日判定 dd<=-dd_clear → 今日开盘清仓并暂停
        if paused and positions:
            for code in list(positions.keys()):
                px = ct.get(code)
                if _isnan(px):
                    continue
                shares, entry, buy_slip = positions[code]
                sl = _c(code)
                cash += px * shares * (1.0 - sl)
                trades.append({"date": t, "code": code, "action": "sell", "price": px,
                               "shares": shares, "notional": px * shares,
                               "reason": "fuse_clear",
                               "pnl": (px * (1.0 - sl) - entry * (1.0 + buy_slip)) * shares})
            positions = {}
            stats["n_clear"] += 1

        # 2) regime 清仓（市场过滤器翻转）
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

        # 3) 月末调仓
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
            # 暂停恢复：深回撤暂停后，市场过滤器翻正（tw>0）即恢复满仓。
            # 注意：不能用「净值创历史新高」作恢复条件——暂停清仓后净值=现金常数，
            # 永远无法创新高，会导致永久暂停（实测 2024-25 净值平坦、夏普 NaN）。
            if paused:
                if w > 0:
                    paused = False
                    stats["n_recover"] += 1
                else:
                    equity[t] = cash
                    continue

            if w <= 0:
                equity[t] = cash
                continue
            if red_active:
                w *= red_mult
                stats["n_red_days"] += 1

            sel = selection.get(t, [])
            sel = [c for c in sel if not _isnan(ct.get(c))]
            if sel and cash > 0:
                eq = cash
                buy_budget = w * cash
                cash_floor = cash - buy_budget
                for code in sel:
                    px = ct.get(code)
                    if _isnan(px) or cash <= cash_floor:
                        continue
                    sl = _c(code)
                    mult = weight_mult.get((t, code), 1.0) if weight_mult else 1.0
                    target = fixed_weight * eq * mult
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

        # 4) 每日盯市 → 更新净值/峰值/熔断状态（今日状态明日生效）
        eq = cash + sum(
            v[0] * ct.get(c) for c, v in positions.items() if not _isnan(ct.get(c))
        )
        equity[t] = eq
        if eq > cummax:
            if paused:
                stats["n_recover"] += 1
            cummax = eq
            paused = False
        dd_now = eq / cummax - 1.0
        stats["max_dd_reached"] = min(stats["max_dd_reached"], dd_now)
        if dd_now <= -dd_clear:
            paused = True
            stats["n_paused_days"] += 1
        prev_eq = eq

    equity_series = pd.Series(equity).sort_index()
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df["date"] = pd.to_datetime(trades_df["date"])
        trades_df = trades_df.sort_values("date").reset_index(drop=True)
    return equity_series, trades_df, stats
