# -*- coding: utf-8 -*-
"""
路线1：V8.1 分区间部署（低风险，优先）。

逻辑：E 等权组合（V8+Trend+Breakout）在 2024-25 显著优于 V8（0.79 vs 0.67），全期持平，
说明 Breakout 在新区间贡献了额外 alpha → 分区间部署：
  - 2023 及以前：V8 原样（run_backtest_v5，fixed_weight=0.10 机制）
  - 2024 起    ：E 等权组合（run_backtest_combo，V8+Trend+Breakout 各 1/3，单只 cap 10%）
实现：两段各自用已验证引擎回测（段1末值作为段2初始资金），净值拼接（日期不重叠，连续）。

验证标准（用户规定）：
  全期夏普 ≥ 0.57 | 2024-25夏普 ≥ 0.79 | 全期回撤 ≤ -21.55%
交付：output/report_v8_1_split.html
"""

from __future__ import annotations

import os
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
from backtest_v5 import run_backtest_v5
from backtest_combo import run_backtest_combo
from market_filter import build_ma240_target_weight, build_ma240_vol_target_weight
from stress_test_v6 import build_ohlcv_full, build_slippage_map
from report import compute_metrics, yearly_sharpe, chart_multi_equity, chart_drawdown_compare
from strategies.trend_ema import gen_signal as sig_trend
from strategies.vol_breakout import gen_signal as sig_breakout

END_EXT = "2025-08-12"
MA_BASE = 240
IC_BASE = 0.05
TOP_N = 30
CAP = 0.10
SPLIT_DATE = "2024-01-01"     # 分区间门控


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


def build_eq_schedule(sel_map: dict, names: list, me, cap: float = CAP) -> dict:
    """等权组合权重表：各策略 1/N，策略内等权，单只 cap。"""
    schedule = {}
    n = len(names)
    for t in me:
        acc = {}
        for name in names:
            codes = sel_map[name].get(t, [])
            if not codes:
                continue
            w_each = 1.0 / n / len(codes)
            for c in codes:
                acc[c] = min(acc.get(c, 0.0) + w_each, cap)
        schedule[t] = list(acc.items())
    return schedule


def main():
    t0 = datetime.now()
    close = pd.read_parquet(config.DATA_DIR / "v8_close_panel.parquet")
    amount = pd.read_parquet(config.DATA_DIR / "v8_amount_panel.parquet")
    idx = pd.read_parquet(config.DATA_DIR / "v6_index.parquet")
    if isinstance(idx, pd.DataFrame):
        idx = idx.iloc[:, 0]
    roe = pd.read_parquet(config.DATA_DIR / "roe_panel_v8.parquet").reindex(close.index).ffill()
    gpm = pd.read_parquet(config.DATA_DIR / "gpm_yoy_panel_v8.parquet").reindex(close.index).ffill()
    n_universe = len(close.columns)
    data_start = str(close.index[0].date())
    data_end = str(close.index[-1].date())

    slip_map, tier_counts = build_slippage_map(amount)
    close_m = mask_new_listings(close, config.NEW_STOCK_MIN_DAYS)
    rsi = compute_rsi(close_m, config.RSI_WINDOW)
    me = month_ends_in(close_m, config.START_DATE, END_EXT)
    ohlcv_old = pd.read_pickle(config.V3_OHLCV) if config.V3_OHLCV.exists() else {}
    ohlcv_full = build_ohlcv_full(close, amount, ohlcv_old, list(close.columns))

    print(f"[{datetime.now()}] 构建信号（{n_universe} 只宇宙）...")
    reversal_signal = build_rolling_reversal_signal(close_m, ohlcv_full, window=36)
    monthly_ic = monthly_reversal_ic(close_m, rsi, me, config.FWD_RETURN_DAYS)
    rolling_ic, use_reversal = compute_rolling_regime(monthly_ic, 12, IC_BASE, 6)
    config.ENABLE_ANALYST_FACTOR = False
    mz = compute_momentum_zscore(close_m, roe, gpm, me)
    sel_v8, sw_v8 = build_selection_v5(close_m, rsi, reversal_signal, mz, me,
                                       use_reversal, 0.20, TOP_N)
    sel_trend = sig_trend(close_m, me, top_n=TOP_N)
    sel_brk = sig_breakout(close_m, amount, me, top_n=TOP_N)

    tw, vol_regime = build_ma240_vol_target_weight(
        idx, close_m.index, MA_BASE, month_ends=me,
        vol_q=0.75, reduced_weight=0.60, vol_lookback=756)
    idx_ret = idx.reindex(close_m.index).pct_change().fillna(0)
    idx_eq = (1.0 + idx_ret).cumprod() * config.INIT_CAPITAL

    # ---- 基准：V8 全期（锚点）----
    print(f"[{datetime.now()}] 基准 V8 全期回测...")
    eq_v8, tr_v8 = run_backtest_v5(close_m, sel_v8, me, config.START_DATE, END_EXT,
                                   target_weight=tw, slippage_map=slip_map)

    # ---- V8.1 分区间：段1 V8(≤2023) → 段2 E等权(≥2024) ----
    print(f"[{datetime.now()}] V8.1 分区间回测（SPLIT={SPLIT_DATE}）...")
    me1 = me[me < pd.Timestamp(SPLIT_DATE)]
    me2 = me[me >= pd.Timestamp(SPLIT_DATE)]
    eq1, tr1 = run_backtest_v5(close_m, sel_v8, me1, config.START_DATE, "2023-12-31",
                               target_weight=tw, slippage_map=slip_map)
    init2 = float(eq1.iloc[-1])
    sel_map = {"V8": sel_v8, "Trend": sel_trend, "Breakout": sel_brk}
    wt_sched = build_eq_schedule(sel_map, ["V8", "Trend", "Breakout"], me2)
    eq2, tr2 = run_backtest_combo(close_m, wt_sched, me2, SPLIT_DATE, END_EXT,
                                  target_weight=tw, slippage_map=slip_map,
                                  init_capital=init2)
    assert eq1.index[-1] < eq2.index[0], "两段日期必须不重叠"
    eq_split = pd.concat([eq1, eq2])
    eq_split = eq_split[~eq_split.index.duplicated(keep="first")].sort_index()

    m_v8 = _m(eq_v8)
    m_sp = _m(eq_split)
    print(f"  V8    : 全期夏普={m_v8['full']['sharpe']:.2f} 回撤={m_v8['full']['max_drawdown']*100:.1f}% "
          f"2024-25={m_v8['new']['sharpe']:.2f}")
    print(f"  V8.1  : 全期夏普={m_sp['full']['sharpe']:.2f} 回撤={m_sp['full']['max_drawdown']*100:.1f}% "
          f"2024-25={m_sp['new']['sharpe']:.2f}")
    print(f"  段1(≤2023)末值={init2:,.0f} 段2(≥2024)起点={eq2.iloc[0]:,.0f} 全期末值={eq_split.iloc[-1]:,.0f}")

    # ---- 验证标准（回撤达标 = 不深于阈值，max_drawdown >= 阈值）----
    ok1 = m_sp["full"]["sharpe"] >= 0.57
    ok2 = m_sp["new"]["sharpe"] >= 0.79
    ok3 = m_sp["full"]["max_drawdown"] >= -0.2155   # 回撤不深于 -21.55%
    print(f"\n验证: 全期≥0.57: {ok1} ({m_sp['full']['sharpe']:.2f}) | "
          f"2024-25≥0.79: {ok2} ({m_sp['new']['sharpe']:.2f}) | "
          f"回撤≤-21.55%: {ok3} ({m_sp['full']['max_drawdown']*100:.2f}%)")

    # ---- 报告 ----
    eq_img = chart_multi_equity({"V8 全期": eq_v8, "V8.1 分区间(2024起E组合)": eq_split,
                                 "CSI300": idx_eq}, "V8.1 分区间部署 vs V8")
    dd_img = chart_drawdown_compare({"V8 全期": eq_v8, "V8.1 分区间": eq_split, "CSI300": idx_eq})

    def mrow(name, m):
        return (f"<tr><td><b>{name}</b></td><td>{m['full']['sharpe']:.2f}</td>"
                f"<td>{m['full']['annual_return']*100:.2f}%</td>"
                f"<td>{m['full']['max_drawdown']*100:.2f}%</td>"
                f"<td>{m['old']['sharpe']:.2f}</td>"
                f"<td>{m['new']['sharpe']:.2f}</td>"
                f"<td>{m['new']['max_drawdown']*100:.2f}%</td></tr>")

    rows = (mrow("V8 原样（锚点）", m_v8) + mrow("V8.1 分区间部署", m_sp))

    verdict = "<br>".join([
        f"✅ 全期夏普 ≥0.57：{m_sp['full']['sharpe']:.2f}" if ok1 else
        f"❌ 全期夏普 ≥0.57：{m_sp['full']['sharpe']:.2f}",
        f"✅ 2024-25夏普 ≥0.79：{m_sp['new']['sharpe']:.2f}" if ok2 else
        f"❌ 2024-25夏普 ≥0.79：{m_sp['new']['sharpe']:.2f}",
        f"✅ 全期回撤 ≤-21.55%：{m_sp['full']['max_drawdown']*100:.2f}%" if ok3 else
        f"❌ 全期回撤 ≤-21.55%：{m_sp['full']['max_drawdown']*100:.2f}%",
    ])

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>V8.1 分区间部署 · 路线1</title>
<style>body{{font-family:-apple-system,'PingFang SC',sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222}}
table{{border-collapse:collapse;width:100%;margin:14px 0;font-size:13px}}
th,td{{border:1px solid #ddd;padding:7px 9px;text-align:center}}
th{{background:#f5f5f5}} h2{{border-bottom:2px solid #eee;padding-bottom:6px;margin-top:34px}}
.verdict{{background:#fff7e6;border:1px solid #f0c36d;border-radius:6px;padding:12px 16px;margin:16px 0}}
.note{{background:#f2f6fc;border-left:4px solid #4a7abb;padding:10px 14px;font-size:13px;color:#444;margin:12px 0}}
img{{max-width:100%}}</style></head><body>
<h1>V8.1 分区间部署（路线1）</h1>
<p>数据区间 {data_start} ~ {data_end}（{n_universe} 只）｜月频｜分档滑点｜共享 MA240+波动率门控</p>
<div class="note"><b>机制：</b>2023 及以前 = V8 原样（fixed_weight=0.10 逐只建仓）；2024 起 =
E 等权组合（V8+Trend+Breakout 各 1/3、策略内 30 只等权、单只 cap 10%）。
两段分别用已验证引擎回测，段1 末值作为段2 初始资金，净值连续拼接（SPLIT={SPLIT_DATE}）。</div>
<h2>绩效对比</h2>
<table><thead><tr><th>版本</th><th>全期夏普</th><th>全期年化</th><th>全期回撤</th>
<th>2018-23夏普</th><th>2024-25夏普</th><th>2024-25回撤</th></tr></thead><tbody>{rows}</tbody></table>
<h2>资金曲线</h2>{eq_img}
<h2>回撤曲线</h2>{dd_img}
<h2>验证标准</h2>
<div class="verdict">{verdict}</div>
<p style="font-size:12px;color:#999">生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}｜
V8 核心参数零改动；分区间门控仅改变 2024 起的选股来源。</p>
</body></html>"""

    out_path = config.OUTPUT_DIR / "report_v8_1_split.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"\n[{datetime.now()}] 报告: {out_path}  耗时 {datetime.now()-t0}")


if __name__ == "__main__":
    main()
