# -*- coding: utf-8 -*-
"""
V5 回测引擎（新增模块，不修改 backtest.py）。

设计目标：与 backtest.run_backtest_v2 的<b>交易机制完全等价</b>（月度轮换、fixed_weight
预算、单边千二成本、MA240 市场门控），唯一区别在于「选股来源」——本引擎直接消费由
factor_eval.py 预先算好的「每月末目标持仓列表」(selection dict)，而非在引擎内部用
RSI 候选池 + 信号排名。这样 V5 与 V3 的差异被严格隔离在「因子合成逻辑」(factor_eval.py)
这一个变量上，满足 V5 的公平对照要求。

机制要点（与 run_backtest_v2 一一对应）：
  - 每个交易日：若 target_weight<=0（指数跌破 MA240）立即清仓。
  - 月末：先轮换清仓，再依市场权重 w（MA240 站上=1.0 / 跌破=0.0）决定买盘预算
    buy_budget = w*cash；在 selection[t] 候选列表中按 fixed_weight 逐只建仓，
    受现金上限约束（fixed_weight=0.10 时实际建仓约 10 只，与 V3 一致）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _isnan(x) -> bool:
    try:
        return x is None or (isinstance(x, float) and np.isnan(x))
    except (TypeError, ValueError):
        return False


def run_backtest_v5(close_panel: pd.DataFrame,
                    selection: dict,
                    month_ends,
                    start: str,
                    end: str,
                    target_weight: pd.Series = None,
                    fixed_weight: float = 0.10,
                    cost: float = 0.002,
                    init_capital: float = 1_000_000.0,
                    slippage_map: dict = None,
                    weight_mult: dict = None) -> tuple[pd.Series, pd.DataFrame]:
    """执行 V5 回测。

    selection      : {month_end(Timestamp): [codes]}，由 factor_eval 提供。
    target_weight  : 日频 0/1.0 序列（MA240 门控）；None 表示不限仓（恒满仓）。
    cost           : 固定单边成本（slippage_map 为 None 时生效，默认 0.002）。
    slippage_map   : {code: 单边滑点} 分档流动性滑点（V6 压力测试用）。提供时按个股
                     分级滑点买卖各扣一次，覆盖 cost；不提供则维持原固定 cost（main_v5
                     行为不变）。
    """
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
        # 分档滑点：提供 slippage_map 时按个股，否则固定 cost
        return slippage_map.get(code, cost) if slippage_map else cost

    cash = init_capital
    positions: dict = {}   # code -> [shares, entry_px, buy_slip]
    equity: dict = {}
    trades: list = []

    for t in dates:
        ct = close.loc[t]

        # 1) 日内：市场过滤器翻转（权重归 0，指数跌破均线）立即清仓
        if tw is not None and (float(tw.loc[t]) <= 0) and positions:
            for code, (shares, entry, buy_slip) in list(positions.items()):
                px = ct.get(code)
                if _isnan(px):
                    continue
                sl = _c(code)
                cash += px * shares * (1.0 - sl)
                trades.append({
                    "date": t, "code": code, "action": "sell", "price": px,
                    "shares": shares, "notional": px * shares, "reason": "regime_exit",
                    "pnl": (px * (1.0 - sl) - entry * (1.0 + buy_slip)) * shares,
                })
            positions = {}

        # 2) 月末调仓（先轮换清仓，再按市场状态决定是否买入）
        if t in month_end_set:
            for code, (shares, entry, buy_slip) in list(positions.items()):
                px = ct.get(code)
                if _isnan(px):
                    continue
                sl = _c(code)
                cash += px * shares * (1.0 - sl)
                trades.append({
                    "date": t, "code": code, "action": "sell", "price": px,
                    "shares": shares, "notional": px * shares, "reason": "rotate",
                    "pnl": (px * (1.0 - sl) - entry * (1.0 + buy_slip)) * shares,
                })
            positions = {}

            w = 1.0 if tw is None else float(tw.loc[t])
            if w <= 0:
                equity[t] = cash
                continue  # 市场向下（或过滤判空仓）：本月末不买入

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
                    # V9 模块2：行业拥挤度权重折扣。weight_mult=None（V8/V9.1）时
                    # mult 恒为 1.0，本行为与 V8 引擎逐笔完全一致（零行为改变）。
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


# ---------------------------------------------------------------------------
# V9.4 / V10 模块A：ATR 卖出规则引擎（新增，不改动 run_backtest_v5）
# ---------------------------------------------------------------------------
# 事实澄清：V8（run_backtest_v5）本身【没有】日内止损止盈，只有月末轮换 + regime 清仓。
# 用户常说的「固定10%止损 + RSI>70止盈」来自 V1 引擎 backtest.run_backtest。
# 本引擎提供三种卖出规则，供模块A对比实验：
#   sell_rule=None      -> 与 run_backtest_v5 逐笔等价（回归校验锚点）
#   sell_rule="classic" -> 固定 -10% 止损 + RSI(14)>70 止盈（移植 V1 规则，即用户认知的基线）
#   sell_rule="atr"     -> ATR 动态止损（入场价-2×ATR）+ 移动止损（只上移）+ 分批止盈
#                          （TP1：盈利 2×ATR 平 50%，剩余止损移至盈亏平衡；TP2：移动止损追踪）
# 月末轮换日不触发日内卖出规则（反正全仓 rotate）；regime 清仓优先级最高（与 V8 一致）。
# 所有 ATR 计算只用截至当日的数据（TR 含当日 high/low/close，盘中实时可得），零未来函数。
# ---------------------------------------------------------------------------

def _atr_series(ohlcv_df: pd.DataFrame, window: int = 14) -> pd.Series:
    """计算 ATR(window)。ohlcv_df 需含 high/low/close 列。"""
    h = ohlcv_df["high"]
    l = ohlcv_df["low"]
    c = ohlcv_df["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()


def _precompute_atr(ohlcv: dict, window: int) -> dict:
    if ohlcv is None:
        return {}
    return {c: _atr_series(df, window) for c, df in ohlcv.items() if df is not None}


def run_backtest_v5_atr(close_panel: pd.DataFrame,
                        selection: dict,
                        month_ends,
                        start: str,
                        end: str,
                        ohlcv: dict = None,
                        target_weight: pd.Series = None,
                        fixed_weight: float = 0.10,
                        cost: float = 0.002,
                        init_capital: float = 1_000_000.0,
                        slippage_map: dict = None,
                        weight_mult: dict = None,
                        sell_rule: str = "atr",
                        atr_window: int = 14,
                        atr_mult: float = 2.0,
                        tp1_mult: float = 2.0,
                        rsi_panel: pd.DataFrame = None,
                        rsi_sell: float = 70.0,
                        stop_loss: float = -0.10) -> tuple[pd.Series, pd.DataFrame]:
    """V8 引擎 + 可选日内卖出规则（classic / atr / None）。

    sell_rule="atr"：初始止损 entry - atr_mult*ATR；移动止损 max(stop, high - atr_mult*ATR)；
        TP1 盈利 tp1_mult*ATR 平 50% 并把剩余止损抬到盈亏平衡；TP2 由移动止损追踪。
    sell_rule="classic"：RSI>rsi_sell 止盈 或 收盘回撤 <= stop_loss 止损（与 V1 引擎一致）。
    sell_rule=None：与 run_backtest_v5 完全一致（回归校验用）。
    """
    all_dates = close_panel.index
    mask = (all_dates >= pd.Timestamp(start)) & (all_dates <= pd.Timestamp(end))
    dates = pd.DatetimeIndex(all_dates[mask])
    close = close_panel.reindex(dates)
    codes = list(close.columns)
    month_end_set = set(pd.DatetimeIndex(month_ends))

    tw = None
    if target_weight is not None:
        tw = target_weight.reindex(dates).astype(float)

    atr_map = _precompute_atr(ohlcv, atr_window) if sell_rule == "atr" else {}
    rsi = rsi_panel.reindex(dates) if (sell_rule == "classic" and rsi_panel is not None) else None

    def _c(code):
        return slippage_map.get(code, cost) if slippage_map else cost

    cash = init_capital
    # positions: code -> [shares, entry_px, buy_slip, stop_price, tp1_done, atr_entry]
    positions: dict = {}
    equity: dict = {}
    trades: list = []

    def _log_sell(t, code, shares, px, buy_slip, reason):
        sl = _c(code)
        cash_add = px * shares * (1.0 - sl)
        pnl = (px * (1.0 - sl) - positions[code][1] * (1.0 + buy_slip)) * shares
        trades.append({"date": t, "code": code, "action": "sell", "price": px,
                       "shares": shares, "notional": px * shares, "reason": reason,
                       "pnl": pnl})
        return cash_add

    for t in dates:
        ct = close.loc[t]

        # 1) 日内：市场过滤器翻转（权重归 0）立即清仓（优先级最高，与 V8 一致）
        if tw is not None and (float(tw.loc[t]) <= 0) and positions:
            for code in list(positions.keys()):
                px = ct.get(code)
                if _isnan(px):
                    continue
                shares = positions[code][0]
                cash += _log_sell(t, code, shares, px, positions[code][2], "regime_exit")
            positions = {}

        # 2) 非月末交易日：日内卖出规则检查
        if t not in month_end_set and positions and sell_rule is not None:
            for code in list(positions.keys()):
                info = positions[code]
                shares, entry, buy_slip, stop, tp1_done, atr_entry = info
                px = ct.get(code)
                if _isnan(px):
                    continue

                if sell_rule == "classic":
                    rv = rsi.loc[t].get(code) if rsi is not None else None
                    ret = px / entry - 1.0
                    reason = None
                    if (not _isnan(rv)) and rv > rsi_sell:
                        reason = "take_profit_rsi"
                    elif ret <= stop_loss:
                        reason = "stop_loss"
                    if reason:
                        cash += _log_sell(t, code, shares, px, buy_slip, reason)
                        del positions[code]

                elif sell_rule == "atr":
                    odf = ohlcv.get(code) if ohlcv else None
                    if odf is None or t not in odf.index:
                        continue  # 无 OHLC：不触发（保守，等月末 rotate）
                    row = odf.loc[t]
                    hi, lo, op = row["high"], row["low"], row["open"]
                    if _isnan(hi) or _isnan(lo) or _isnan(op):
                        continue
                    atr_s = atr_map.get(code)

                    # 移动止损：用【前一交易日】的 high 与 ATR 更新（只上移）。
                    # 关键：绝不能用「当日 high 更新 stop 再用当日 low 判断触发」——同根 K 线
                    # 内 high/low 顺序未知，那样先假设"见高点再回撤"会制造海量 whipsaw。
                    # 标准做法：收盘后用当日数据更新，次日生效（天然零未来函数）。
                    if atr_s is not None:
                        prev = atr_s.loc[atr_s.index < t]  # 严格早于 t 的全部历史
                        if len(prev):
                            pt = prev.index[-1]
                            prev_hi = odf.loc[pt, "high"] if pt in odf.index else np.nan
                            prev_atr = prev.iloc[-1]
                            if (not _isnan(prev_hi)) and (not _isnan(prev_atr)) and prev_atr > 0:
                                new_stop = prev_hi - atr_mult * prev_atr
                                if not _isnan(new_stop):
                                    stop = max(stop, new_stop)

                    # 止损/移动止损触发（先止损后止盈，保守；gap down 按 min(open, stop) 成交）
                    if not _isnan(stop) and lo <= stop:
                        sell_px = min(op, stop)
                        cash += _log_sell(t, code, positions[code][0], sell_px,
                                          positions[code][2], "trailing_stop")
                        del positions[code]
                        continue

                    # TP1：盈利 tp1_mult*ATR（入场时 ATR），平仓 50%，剩余止损移至盈亏平衡
                    if not tp1_done and not _isnan(atr_entry) and atr_entry > 0:
                        target1 = entry + tp1_mult * atr_entry
                        if not _isnan(hi) and hi >= target1:
                            sell_sh = shares // 2
                            if sell_sh < 1:
                                sell_sh = shares
                            sell_px = max(op, target1)
                            cash += _log_sell(t, code, sell_sh, sell_px, buy_slip, "tp1_half")
                            info[0] -= sell_sh
                            if info[0] <= 0:
                                del positions[code]
                                continue
                            stop = max(stop, entry)  # 剩余仓位止损抬到盈亏平衡
                            info[3] = stop
                            info[4] = True

        # 3) 月末调仓（先轮换清仓，再按市场状态决定是否买入）
        if t in month_end_set:
            for code in list(positions.keys()):
                px = ct.get(code)
                if _isnan(px):
                    continue
                shares = positions[code][0]
                cash += _log_sell(t, code, shares, px, positions[code][2], "rotate")
            positions = {}

            w = 1.0 if tw is None else float(tw.loc[t])
            if w <= 0:
                equity[t] = cash
                continue  # 市场向下（或过滤判空仓）：本月末不买入

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
                    # 初始止损：entry - atr_mult*ATR(entry day)，ATR 不可用则 fallback 固定比例
                    stop0 = np.nan
                    atr_entry0 = np.nan
                    if sell_rule == "atr":
                        atr_s = atr_map.get(code)
                        if atr_s is not None:
                            prev = atr_s.loc[atr_s.index <= t]
                            if len(prev):
                                a0 = prev.iloc[-1]
                                if not _isnan(a0) and a0 > 0:
                                    stop0 = px - atr_mult * a0
                                    atr_entry0 = float(a0)
                    if _isnan(stop0):
                        stop0 = px * (1.0 + stop_loss)  # fallback -10%
                    positions[code] = [shares, px, sl, float(stop0), False, float(atr_entry0)]
                    trades.append({
                        "date": t, "code": code, "action": "buy", "price": px,
                        "shares": shares, "notional": px * shares,
                        "reason": "entry", "pnl": 0.0,
                    })

        # 4) 每日盯市
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


# ---------------------------------------------------------------------------
# V8.1 执行口径对齐引擎：next_open 成交 + 冲击成本（新增，不改动 run_backtest_v5）
# ---------------------------------------------------------------------------
# 与模拟盘/实盘口径对齐：
#   - 信号日 T 收盘生成买卖信号 → T+1 开盘价成交；open 缺失（停牌）顺延至下一交易日；
#   - 卖出信号（regime 翻转 / 月末轮换）同样 T+1 开盘卖出；
#   - 冲击成本：按个股 T 日 20 日均成交额分级（>5亿 0.05% / 1-5亿 0.15% / <1亿 0.30%），
#     单边、与分档滑点叠加（总摩擦 = 滑点 + 冲击）。
# 实现：指令队列（T 收盘生成指令、T+1 开盘执行），现金/持仓时点与真实 T+1 成交一致。
# ---------------------------------------------------------------------------

def build_open_panel(ohlcv: dict, close_panel: pd.DataFrame) -> pd.DataFrame:
    """从 OHLCV dict 构建 open 面板（index/columns 对齐 close_panel，缺失为 NaN）。"""
    op = pd.DataFrame(index=close_panel.index, columns=close_panel.columns, dtype=float)
    for c in close_panel.columns:
        d = ohlcv.get(c)
        if d is None:
            continue
        op[c] = d["open"].reindex(close_panel.index).astype(float)
    return op


def _impact_rates(amount_panel, t, tiers, lookback: int = 20) -> dict:
    """t 日为止过去 lookback 个交易日日均成交额 → {code: 冲击成本率}（point-in-time）。"""
    if amount_panel is None or t not in amount_panel.index:
        return {}
    win = amount_panel.loc[:t].tail(lookback)
    avg = win.mean()
    out = {}
    for code, a in avg.items():
        if _isnan(a) or a <= 0:
            continue
        rate = tiers[-1][1]
        for lo, r in tiers:
            if a > lo:
                rate = r
                break
        out[code] = rate
    return out


def run_backtest_v5_ne(close_panel: pd.DataFrame,
                       selection: dict,
                       month_ends,
                       start: str,
                       end: str,
                       open_panel: pd.DataFrame = None,
                       target_weight: pd.Series = None,
                       fixed_weight: float = 0.10,
                       cost: float = 0.002,
                       init_capital: float = 1_000_000.0,
                       slippage_map: dict = None,
                       weight_mult: dict = None,
                       enable_impact: bool = False,
                       impact_tiers: list = None,
                       amount_panel: pd.DataFrame = None,
                       impact_lookback: int = 20) -> tuple[pd.Series, pd.DataFrame]:
    """V8.1 正式引擎：T+1 开盘价成交 + 可选冲击成本（与 sim_tracker/实盘口径一致）。"""
    if impact_tiers is None:
        impact_tiers = [(5.0e8, 0.0005), (1.0e8, 0.0015), (0.0, 0.0030)]
    all_dates = close_panel.index
    mask = (all_dates >= pd.Timestamp(start)) & (all_dates <= pd.Timestamp(end))
    dates = pd.DatetimeIndex(all_dates[mask])
    close = close_panel.reindex(dates)
    codes = list(close.columns)
    month_end_set = set(pd.DatetimeIndex(month_ends))

    tw = None
    if target_weight is not None:
        tw = target_weight.reindex(dates).astype(float)
    op = open_panel.reindex(dates) if open_panel is not None else None
    if op is None:
        raise ValueError("run_backtest_v5_ne 需要 open_panel（T+1 开盘价成交）")

    def _slip(code):
        return slippage_map.get(code, cost) if slippage_map else cost

    cash = init_capital
    positions: dict = {}            # code -> [shares, entry_px, buy_cost_rate]
    pending_sells: dict = {}        # code -> [shares, entry_px, buy_cost_rate, impact_rate, reason]
    pending_buy = None              # dict(codes=[(code,mult)], w=w, impact_map={})
    equity: dict = {}
    trades: list = []

    for t in dates:
        ct = close.loc[t]
        opt = op.loc[t]

        # ---- 0) 执行 T-1 收盘生成的指令（T 日开盘价）----
        # 0a) 卖出（先卖回笼现金，供买入）
        for code in list(pending_sells.keys()):
            po = opt.get(code)
            if _isnan(po):
                continue                     # 停牌：顺延下一交易日
            shares, entry, bc, imp, reason = pending_sells.pop(code)
            tot = _slip(code) + imp
            proceeds = po * shares * (1.0 - tot)
            cash += proceeds
            pnl = (po * (1.0 - tot) - entry * (1.0 + bc)) * shares
            trades.append({"date": t, "code": code, "action": "sell", "price": po,
                           "shares": shares, "notional": po * shares, "reason": reason,
                           "pnl": pnl, "cost_total": tot})
            positions.pop(code, None)
        # 0b) 买入（按信号顺序 + 现金上限）
        if pending_buy is not None:
            pb = pending_buy
            pending_buy = None
            eq0 = cash
            buy_budget = pb["w"] * eq0
            cash_floor = eq0 - buy_budget
            for (code, mult) in pb["codes"]:
                po = opt.get(code)
                if _isnan(po) or cash <= cash_floor:
                    continue
                if code in positions:        # 停牌未卖出/已持仓：跳过防覆盖
                    continue
                tot = _slip(code) + pb["impact_map"].get(code, 0.0)
                target = fixed_weight * eq0 * mult
                shares = int(target / (po * (1.0 + tot)))
                if shares <= 0:
                    continue
                cost_amt = po * shares * (1.0 + tot)
                if cost_amt > cash:
                    continue
                cash -= cost_amt
                positions[code] = [shares, po, tot]
                trades.append({"date": t, "code": code, "action": "buy", "price": po,
                               "shares": shares, "notional": po * shares,
                               "reason": "entry", "pnl": 0.0, "cost_total": tot})

        # ---- 1) regime 检查（T 收盘判定 tw<=0 → 次日开盘清仓）----
        regime_clear = False
        if tw is not None and float(tw.loc[t]) <= 0 and positions:
            imp_map = _impact_rates(amount_panel, t, impact_tiers,
                                    impact_lookback) if enable_impact else {}
            for code, (shares, entry, bc) in list(positions.items()):
                pending_sells[code] = [shares, entry, bc, imp_map.get(code, 0.0), "regime_exit"]
            regime_clear = True

        # ---- 2) 月末调仓（T 收盘生成指令）----
        if t in month_end_set:
            w = 1.0 if tw is None else float(tw.loc[t])
            if w <= 0:
                pending_buy = None
            else:
                if not regime_clear:
                    imp_map = _impact_rates(amount_panel, t, impact_tiers,
                                            impact_lookback) if enable_impact else {}
                    for code, (shares, entry, bc) in list(positions.items()):
                        pending_sells[code] = [shares, entry, bc, imp_map.get(code, 0.0), "rotate"]
                sel = selection.get(t, [])
                sel = [c for c in sel if not _isnan(ct.get(c))]
                pb_codes = []
                for c in sel:
                    mult = weight_mult.get((t, c), 1.0) if weight_mult else 1.0
                    pb_codes.append((c, mult))
                imp_map = _impact_rates(amount_panel, t, impact_tiers,
                                        impact_lookback) if enable_impact else {}
                pending_buy = {"codes": pb_codes, "w": w, "impact_map": imp_map}

        # ---- 3) 盯市（close 计价；待卖持仓仍计入）----
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
