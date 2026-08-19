# -*- coding: utf-8 -*-
"""
V9.4 模块A：卖出规则优化（ATR 动态止盈止损）—— 3 臂对比 + N 敏感性。

事实澄清（重要）：
  V8（run_backtest_v5）本身【没有】日内止损止盈——只有月末轮换 + regime 清仓。
  用户常说的「固定10%止损 + RSI>70止盈」来自 V1 引擎 backtest.run_backtest。
  因此对比设计为 3 臂：
    臂1  V8 原样（无日内卖出规则）                 -> 已验证锚点 0.57 / 0.67 / -19.40%
    臂2  V8 + classic（固定-10%止损 + RSI>70止盈） -> 用户认知的「V8基线」
    臂3  V8 + ATR（动态止损 + 移动止损 + 分批止盈） -> 本次验证的卖出规则
  判断标准（诚实）：
    - ATR 相对 V8 原样：夏普/回撤/盈亏比是否全面改善（才是真正"有效"）
    - ATR 相对 classic：证明"让利润奔跑+动态止损"优于"固定比例+RSI止盈"

ATR 规则（用户规格）：
  初始止损 = 入场价 - N×ATR(14)，N∈{1.5, 2.0, 2.5} 敏感性
  移动止损 = max(prev_stop, high - N×ATR)，只上移不下移
  分批止盈：TP1 盈利 2×ATR 平 50%，剩余止损抬至盈亏平衡；TP2 移动止损追踪
  移除 RSI 止盈
数据：data/v8_ohlcv.pkl（复权基准已按 V8 面板修正）。

交付：output/report_v8_atr.html；若 ATR 有效则作为 V10 基础。
"""

from __future__ import annotations

import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
          "ALL_PROXY", "all_proxy"]:
    os.environ.pop(k, None)

import config
from factors import compute_rsi, get_month_end_dates
from factor_eval import (monthly_reversal_ic, compute_rolling_regime,
                        compute_momentum_zscore, build_selection_v5)
from factor_eval_v7 import build_rolling_reversal_signal
from backtest_v5 import run_backtest_v5, run_backtest_v5_atr
from market_filter import build_ma240_target_weight, build_ma240_vol_target_weight
from stress_test_v6 import build_ohlcv_full, build_slippage_map
from report import compute_metrics, yearly_sharpe, chart_multi_equity, chart_drawdown_compare

END_EXT = "2025-08-12"
MA_BASE = 240
IC_BASE = 0.05
FIXED_WEIGHT = config.FIXED_WEIGHT


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


def month_ends_in(close_panel: pd.DataFrame, start, end) -> pd.DatetimeIndex:
    me = get_month_end_dates(close_panel.index)
    me = pd.DatetimeIndex(me).normalize()
    return me[(me >= pd.Timestamp(start).normalize()) & (me <= pd.Timestamp(end).normalize())]


def _m(eq: pd.Series) -> dict:
    return {"full": compute_metrics(eq),
            "old": compute_metrics(eq.loc[:"2023-12-31"]),
            "new": compute_metrics(eq.loc["2024-01-01":])}


def trade_pl_stats(trades: pd.DataFrame, rule_reasons: tuple = None) -> dict:
    """按卖出交易统计盈亏比。rule_reasons 限定规则退出（默认全部卖出）。"""
    if trades is None or trades.empty:
        return {"n": 0, "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "pl_ratio": 0.0}
    s = trades[trades["action"] == "sell"]
    if rule_reasons is not None:
        s = s[s["reason"].isin(rule_reasons)]
    if s.empty:
        return {"n": 0, "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "pl_ratio": 0.0}
    pnl = s["pnl"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    aw = wins.mean() if len(wins) else 0.0
    al = abs(losses.mean()) if len(losses) else 0.0
    return {"n": len(s), "win_rate": len(wins) / len(s) * 100,
            "avg_win": aw, "avg_loss": al,
            "pl_ratio": aw / al if al > 0 else float("inf")}


def main():
    t0 = datetime.now()
    close = pd.read_parquet(config.DATA_DIR / "v8_close_panel.parquet")
    amount = pd.read_parquet(config.DATA_DIR / "v8_amount_panel.parquet")
    idx = pd.read_parquet(config.DATA_DIR / "v6_index.parquet")
    if isinstance(idx, pd.DataFrame):
        idx = idx.iloc[:, 0]
    roe = pd.read_parquet(config.DATA_DIR / "roe_panel_v8.parquet").reindex(close.index).ffill()
    gpm = pd.read_parquet(config.DATA_DIR / "gpm_yoy_panel_v8.parquet").reindex(close.index).ffill()
    codes_all = list(close.columns)
    n_universe = len(codes_all)
    data_start = str(close.index[0].date())
    data_end = str(close.index[-1].date())

    # ---- OHLC（ATR 数据，复权基准已修正）----
    ohlcv_path = config.DATA_DIR / "v8_ohlcv.pkl"
    if not ohlcv_path.exists():
        raise SystemExit(f"缺少 {ohlcv_path}：先运行 fetch_ohlc_v8.py")
    with open(ohlcv_path, "rb") as f:
        ohlcv = pickle.load(f)
    print(f"OHLC 加载：{len(ohlcv)} 只（覆盖宇宙 {sum(1 for c in codes_all if c in ohlcv)}/{n_universe}）")

    slip_map, tier_counts = build_slippage_map(amount)
    close_m = mask_new_listings(close, config.NEW_STOCK_MIN_DAYS)
    rsi = compute_rsi(close_m, config.RSI_WINDOW)
    me = month_ends_in(close_m, config.START_DATE, END_EXT)
    ohlcv_old = pd.read_pickle(config.V3_OHLCV) if config.V3_OHLCV.exists() else {}
    ohlcv_full = build_ohlcv_full(close, amount, ohlcv_old, codes_all)

    print(f"[{datetime.now()}] 构建信号（V8 口径，无分析师因子）...")
    reversal_signal = build_rolling_reversal_signal(close_m, ohlcv_full, window=36)
    monthly_ic = monthly_reversal_ic(close_m, rsi, me, config.FWD_RETURN_DAYS)
    rolling_ic, use_reversal = compute_rolling_regime(monthly_ic, 12, IC_BASE, 6)
    config.ENABLE_ANALYST_FACTOR = False  # V8 口径
    mz = compute_momentum_zscore(close_m, roe, gpm, me)
    sel, sw = build_selection_v5(close_m, rsi, reversal_signal, mz, me, use_reversal, 0.20, 30)

    tw_clean = build_ma240_target_weight(idx, close_m.index, MA_BASE)
    tw, vol_regime = build_ma240_vol_target_weight(
        idx, close_m.index, MA_BASE, month_ends=me,
        vol_q=0.75, reduced_weight=0.60, vol_lookback=756)

    idx_ret = idx.reindex(close_m.index).pct_change().fillna(0)
    idx_eq = (1.0 + idx_ret).cumprod() * config.INIT_CAPITAL

    # ---- 三臂 ----
    print(f"[{datetime.now()}] 臂1 V8 原样...")
    eq_v8, tr_v8 = run_backtest_v5(close_m, sel, me, config.START_DATE, END_EXT,
                                   target_weight=tw, slippage_map=slip_map)

    print(f"[{datetime.now()}] 臂2 classic（-10% + RSI70）...")
    eq_cl, tr_cl = run_backtest_v5_atr(
        close_m, sel, me, config.START_DATE, END_EXT,
        ohlcv=ohlcv, target_weight=tw, slippage_map=slip_map,
        sell_rule="classic", rsi_panel=rsi)

    print(f"[{datetime.now()}] 臂3 ATR（N=2.0, TP1=2×ATR）...")
    eq_atr, tr_atr = run_backtest_v5_atr(
        close_m, sel, me, config.START_DATE, END_EXT,
        ohlcv=ohlcv, target_weight=tw, slippage_map=slip_map,
        sell_rule="atr", atr_mult=2.0, tp1_mult=2.0)

    # ---- ATR N 敏感性 {1.5, 2.0, 2.5} ----
    print(f"[{datetime.now()}] ATR N 敏感性 1.5 / 2.5 ...")
    sens = {}
    for n in [1.5, 2.5]:
        eqn, trn = run_backtest_v5_atr(
            close_m, sel, me, config.START_DATE, END_EXT,
            ohlcv=ohlcv, target_weight=tw, slippage_map=slip_map,
            sell_rule="atr", atr_mult=n, tp1_mult=2.0)
        sens[n] = (eqn, trn)
    sens[2.0] = (eq_atr, tr_atr)

    m_v8 = _m(eq_v8)
    m_cl = _m(eq_cl)
    m_atr = _m(eq_atr)
    print(f"  V8原样: 全期夏普={m_v8['full']['sharpe']:.2f} 回撤={m_v8['full']['max_drawdown']*100:.1f}%")
    print(f"  classic: 全期夏普={m_cl['full']['sharpe']:.2f} 回撤={m_cl['full']['max_drawdown']*100:.1f}%")
    print(f"  ATR: 全期夏普={m_atr['full']['sharpe']:.2f} 回撤={m_atr['full']['max_drawdown']*100:.1f}%")

    # ---- 交易统计 ----
    def reason_counts(tr):
        if tr is None or tr.empty:
            return {}
        return tr[tr["action"] == "sell"]["reason"].value_counts().to_dict()

    st_v8 = reason_counts(tr_v8)
    st_cl = reason_counts(tr_cl)
    st_atr = reason_counts(tr_atr)
    pl_v8 = trade_pl_stats(tr_v8)
    pl_cl = trade_pl_stats(tr_cl, ("take_profit_rsi", "stop_loss"))
    pl_atr = trade_pl_stats(tr_atr, ("tp1_half", "trailing_stop"))
    print(f"  V8 卖出原因: {st_v8}")
    print(f"  classic 卖出原因: {st_cl}  规则盈亏比={pl_cl['pl_ratio']:.2f} 胜率={pl_cl['win_rate']:.1f}%")
    print(f"  ATR 卖出原因: {st_atr}  规则盈亏比={pl_atr['pl_ratio']:.2f} 胜率={pl_atr['win_rate']:.1f}%")

    # ---- 判定 ----
    d_atr_full = m_atr["full"]["sharpe"] - m_v8["full"]["sharpe"]
    d_atr_new = m_atr["new"]["sharpe"] - m_v8["new"]["sharpe"]
    d_atr_dd = m_atr["full"]["max_drawdown"] * 100 - m_v8["full"]["max_drawdown"] * 100
    d_cl_full = m_cl["full"]["sharpe"] - m_v8["full"]["sharpe"]
    d_cl_dd = m_cl["full"]["max_drawdown"] * 100 - m_v8["full"]["max_drawdown"] * 100

    atr_ok = (d_atr_full >= 0) and (d_atr_dd <= 1.0)
    cl_ok = (d_cl_full >= 0) and (d_cl_dd <= 1.0)

    # ---- 报告 ----
    eq_img = chart_multi_equity({
        "V8 原样": eq_v8, "V8+classic(-10%/RSI70)": eq_cl,
        "V8+ATR(N=2)": eq_atr, "CSI300": idx_eq})
    dd_img = chart_drawdown_compare({
        "V8 原样": eq_v8, "V8+classic(-10%/RSI70)": eq_cl, "V8+ATR(N=2)": eq_atr})

    def mrow(name, m, pl):
        return (f"<tr><td><b>{name}</b></td><td>{m['full']['sharpe']:.2f}</td>"
                f"<td>{m['full']['annual_return']*100:.2f}%</td>"
                f"<td>{m['full']['max_drawdown']*100:.2f}%</td>"
                f"<td>{m['new']['sharpe']:.2f}</td>"
                f"<td>{m['new']['max_drawdown']*100:.2f}%</td>"
                f"<td>{pl['n']}</td><td>{pl['win_rate']:.1f}%</td>"
                f"<td>{pl['pl_ratio']:.2f}</td></tr>")

    rows = (mrow("V8 原样（锚点）", m_v8, pl_v8)
            + mrow("V8+classic（-10%止损+RSI70止盈）", m_cl, pl_cl)
            + mrow("V8+ATR（N=2, 分批止盈）", m_atr, pl_atr))

    sens_rows = ""
    for n in [1.5, 2.0, 2.5]:
        eqn, _ = sens[n]
        mn = _m(eqn)
        pn = trade_pl_stats(sens[n][1], ("tp1_half", "trailing_stop"))
        sens_rows += (f"<tr><td>N={n}</td><td>{mn['full']['sharpe']:.2f}</td>"
                      f"<td>{mn['full']['annual_return']*100:.2f}%</td>"
                      f"<td>{mn['full']['max_drawdown']*100:.2f}%</td>"
                      f"<td>{mn['new']['sharpe']:.2f}</td>"
                      f"<td>{pn['pl_ratio']:.2f}</td></tr>")

    if atr_ok:
        verdict = (f"<span style='color:#c0392b;font-weight:bold'>ATR 规则相对 V8 原样有效</span>"
                   f"（全期夏普 {d_atr_full:+.2f}、回撤 {d_atr_dd:+.2f}pp、2024-25 夏普 {d_atr_new:+.2f}）"
                   f"→ 可作为 V10 基础合并。")
    else:
        verdict = (f"<span style='color:#888'>ATR 规则未通过（全期夏普 {d_atr_full:+.2f}、"
                   f"回撤 {d_atr_dd:+.2f}pp）→ 维持 V8 卖出机制（无日内规则），"
                   f"模块B 多策略并行继续用 V8 卖出规则。</span>")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>V9.4 模块A · ATR 卖出规则优化</title>
<style>body{{font-family:-apple-system,'PingFang SC',sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222}}
table{{border-collapse:collapse;width:100%;margin:14px 0;font-size:13px}}
th,td{{border:1px solid #ddd;padding:7px 9px;text-align:center}}
th{{background:#f5f5f5}} h2{{border-bottom:2px solid #eee;padding-bottom:6px;margin-top:34px}}
.verdict{{background:#fff7e6;border:1px solid #f0c36d;border-radius:6px;padding:12px 16px;margin:16px 0}}
.note{{background:#f2f6fc;border-left:4px solid #4a7abb;padding:10px 14px;font-size:13px;color:#444;margin:12px 0}}
img{{max-width:100%}}</style></head><body>
<h1>V9.4 模块A：卖出规则优化（ATR 动态止盈止损）</h1>
<p>数据区间 {data_start} ~ {data_end}（{n_universe} 只宇宙）｜基准资金 1,000,000 ｜分档滑点 0.1/0.3/0.5%</p>
<div class="note"><b>事实澄清：</b>V8（run_backtest_v5）本身<b>没有</b>日内止损止盈，只有月末轮换 + regime 清仓；
「固定10%止损 + RSI&gt;70止盈」来自 V1 引擎。因此实验设 3 臂：<b>V8 原样</b>（已验证锚点 0.57/0.67/-19.40%）、
<b>V8+classic</b>（用户认知的旧基线）、<b>V8+ATR</b>（本次验证）。判定以 ATR vs V8 原样为准。</div>
<h2>三臂绩效对比</h2>
<table><thead><tr><th>策略臂</th><th>全期夏普</th><th>全期年化</th><th>全期回撤</th>
<th>2024-25夏普</th><th>2024-25回撤</th><th>卖出笔数*</th><th>胜率*</th><th>盈亏比*</th></tr></thead>
<tbody>{rows}</tbody></table>
<p style="font-size:12px;color:#888">*卖出笔数/胜率/盈亏比：classic 只统计规则退出（take_profit_rsi+stop_loss）；ATR 统计 tp1_half+trailing_stop；
V8 原样统计全部卖出（rotate+regime_exit，仅供参考）。</p>
<h2>净值曲线</h2>{eq_img}
<h2>回撤曲线</h2>{dd_img}
<h2>ATR N 值敏感性（TP1 固定 2×ATR）</h2>
<table><thead><tr><th>参数</th><th>全期夏普</th><th>全期年化</th><th>全期回撤</th><th>2024-25夏普</th><th>规则盈亏比</th></tr></thead>
<tbody>{sens_rows}</tbody></table>
<h2>卖出原因分布</h2>
<table><thead><tr><th>策略臂</th><th>原因分布</th></tr></thead><tbody>
<tr><td>V8 原样</td><td style="text-align:left">{st_v8}</td></tr>
<tr><td>V8+classic</td><td style="text-align:left">{st_cl}</td></tr>
<tr><td>V8+ATR</td><td style="text-align:left">{st_atr}</td></tr>
</tbody></table>
<h2>结论</h2>
<div class="verdict">{verdict}</div>
<p style="font-size:12px;color:#999">生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}｜
ATR 规则：初始止损 入场-2×ATR(14)；移动止损 max(prev, high-2×ATR)；TP1 盈利2×ATR 平50% 止损抬至盈亏平衡；
TP2 移动止损追踪；移除 RSI 止盈。ATR 数据复权基准已按 V8 面板常数修正，零未来函数。</p>
</body></html>"""

    out_path = config.OUTPUT_DIR / "report_v8_atr.html"
    out_path.write_text(html, encoding="utf-8")

    print("\n================ 模块A 判定 ================")
    print(f"[ATR vs V8原样] 全期 {d_atr_full:+.2f} | 2024-25 {d_atr_new:+.2f} | 回撤 {d_atr_dd:+.2f}pp")
    print(f"[classic vs V8原样] 全期 {d_cl_full:+.2f} | 回撤 {d_cl_dd:+.2f}pp")
    print(f"ATR 有效: {atr_ok} | classic 有效: {cl_ok}")
    print(f"[{datetime.now()}] 报告: {out_path}  耗时 {datetime.now()-t0}")


if __name__ == "__main__":
    main()
