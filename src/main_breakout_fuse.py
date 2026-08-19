# -*- coding: utf-8 -*-
"""
路线2：Breakout + 回撤熔断（备选，进攻性方案）。

逻辑：Breakout 单策略全期夏普 0.77（高于 V8 0.57），唯一问题是回撤 -35.75%。
叠加回撤熔断（backtest_fuse.py）：
  - 策略净值回撤 > 20% → 月末建仓权重 × 0.5；
  - 回撤 > 25% → 强制清仓暂停，净值创历史新高后恢复 100%。
不改 Breakout 选股逻辑（接近20日高点+放量）、不改 V8 参数。
验证标准（用户规定）：
  全期回撤 ≤ -25% | 全期夏普 ≥ 0.60 | 2024-25夏普 ≥ 0.80
交付：output/report_breakout_fuse.html
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
from backtest_fuse import run_backtest_fuse
from market_filter import build_ma240_target_weight, build_ma240_vol_target_weight
from stress_test_v6 import build_ohlcv_full, build_slippage_map
from report import compute_metrics, yearly_sharpe, chart_multi_equity, chart_drawdown_compare
from strategies.vol_breakout import gen_signal as sig_breakout

END_EXT = "2025-08-12"
MA_BASE = 240
IC_BASE = 0.05
TOP_N = 30
DD_RED = 0.20
DD_CLEAR = 0.25
RED_MULT = 0.5


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


def main():
    t0 = datetime.now()
    close = pd.read_parquet(config.DATA_DIR / "v8_close_panel.parquet")
    amount = pd.read_parquet(config.DATA_DIR / "v8_amount_panel.parquet")
    idx = pd.read_parquet(config.DATA_DIR / "v6_index.parquet")
    if isinstance(idx, pd.DataFrame):
        idx = idx.iloc[:, 0]
    n_universe = len(close.columns)
    data_start = str(close.index[0].date())
    data_end = str(close.index[-1].date())

    slip_map, tier_counts = build_slippage_map(amount)
    close_m = mask_new_listings(close, config.NEW_STOCK_MIN_DAYS)
    me = month_ends_in(close_m, config.START_DATE, END_EXT)
    ohlcv_old = pd.read_pickle(config.V3_OHLCV) if config.V3_OHLCV.exists() else {}
    ohlcv_full = build_ohlcv_full(close, amount, ohlcv_old, list(close.columns))

    print(f"[{datetime.now()}] 构建 Breakout 信号（{n_universe} 只宇宙）...")
    rsi = compute_rsi(close_m, config.RSI_WINDOW)
    sel_brk = sig_breakout(close_m, amount, me, top_n=TOP_N)

    tw, vol_regime = build_ma240_vol_target_weight(
        idx, close_m.index, MA_BASE, month_ends=me,
        vol_q=0.75, reduced_weight=0.60, vol_lookback=756)
    idx_ret = idx.reindex(close_m.index).pct_change().fillna(0)
    idx_eq = (1.0 + idx_ret).cumprod() * config.INIT_CAPITAL

    # ---- Breakout 原样（对照）----
    print(f"[{datetime.now()}] Breakout 原样回测...")
    eq_raw, tr_raw = run_backtest_v5(close_m, sel_brk, me, config.START_DATE, END_EXT,
                                     target_weight=tw, slippage_map=slip_map)

    # ---- Breakout + 熔断 ----
    print(f"[{datetime.now()}] Breakout+熔断回测（降仓20%/清仓25%）...")
    eq_fuse, tr_fuse, st = run_backtest_fuse(
        close_m, sel_brk, me, config.START_DATE, END_EXT,
        target_weight=tw, slippage_map=slip_map,
        dd_red=DD_RED, dd_clear=DD_CLEAR, red_mult=RED_MULT)
    print(f"  熔断统计: 清仓触发 {st['n_clear']} 次 | 降仓期 {st['n_red_days']} 期 | "
          f"暂停天数 {st['n_paused_days']} | 恢复 {st['n_recover']} 次 | "
          f"触及最深回撤 {st['max_dd_reached']*100:.1f}%")

    m_raw = _m(eq_raw)
    m_fuse = _m(eq_fuse)
    print(f"  Breakout原样: 全期夏普={m_raw['full']['sharpe']:.2f} 回撤={m_raw['full']['max_drawdown']*100:.1f}% "
          f"2024-25={m_raw['new']['sharpe']:.2f}")
    print(f"  Breakout熔断: 全期夏普={m_fuse['full']['sharpe']:.2f} 回撤={m_fuse['full']['max_drawdown']*100:.1f}% "
          f"2024-25={m_fuse['new']['sharpe']:.2f}")

    # ---- 验证标准（回撤达标 = 不深于阈值）----
    ok1 = m_fuse["full"]["max_drawdown"] >= -0.25   # 回撤不深于 -25%
    ok2 = m_fuse["full"]["sharpe"] >= 0.60
    ok3 = m_fuse["new"]["sharpe"] >= 0.80
    print(f"\n验证: 回撤≤-25%: {ok1} ({m_fuse['full']['max_drawdown']*100:.2f}%) | "
          f"全期≥0.60: {ok2} ({m_fuse['full']['sharpe']:.2f}) | "
          f"2024-25≥0.80: {ok3} ({m_fuse['new']['sharpe']:.2f})")

    # ---- 报告 ----
    eq_img = chart_multi_equity({"Breakout 原样": eq_raw, "Breakout+熔断": eq_fuse,
                                 "CSI300": idx_eq}, "Breakout + 回撤熔断")
    dd_img = chart_drawdown_compare({"Breakout 原样": eq_raw, "Breakout+熔断": eq_fuse})

    def mrow(name, m):
        return (f"<tr><td><b>{name}</b></td><td>{m['full']['sharpe']:.2f}</td>"
                f"<td>{m['full']['annual_return']*100:.2f}%</td>"
                f"<td>{m['full']['max_drawdown']*100:.2f}%</td>"
                f"<td>{m['new']['sharpe']:.2f}</td>"
                f"<td>{m['new']['max_drawdown']*100:.2f}%</td></tr>")

    rows = (mrow("Breakout 原样（对照）", m_raw) + mrow("Breakout + 回撤熔断", m_fuse))

    verdict = "<br>".join([
        f"✅ 全期回撤 ≤-25%：{m_fuse['full']['max_drawdown']*100:.2f}%" if ok1 else
        f"❌ 全期回撤 ≤-25%：{m_fuse['full']['max_drawdown']*100:.2f}%",
        f"✅ 全期夏普 ≥0.60：{m_fuse['full']['sharpe']:.2f}" if ok2 else
        f"❌ 全期夏普 ≥0.60：{m_fuse['full']['sharpe']:.2f}",
        f"✅ 2024-25夏普 ≥0.80：{m_fuse['new']['sharpe']:.2f}" if ok3 else
        f"❌ 2024-25夏普 ≥0.80：{m_fuse['new']['sharpe']:.2f}",
    ])

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>Breakout + 回撤熔断 · 路线2</title>
<style>body{{font-family:-apple-system,'PingFang SC',sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222}}
table{{border-collapse:collapse;width:100%;margin:14px 0;font-size:13px}}
th,td{{border:1px solid #ddd;padding:7px 9px;text-align:center}}
th{{background:#f5f5f5}} h2{{border-bottom:2px solid #eee;padding-bottom:6px;margin-top:34px}}
.verdict{{background:#fff7e6;border:1px solid #f0c36d;border-radius:6px;padding:12px 16px;margin:16px 0}}
.note{{background:#f2f6fc;border-left:4px solid #4a7abb;padding:10px 14px;font-size:13px;color:#444;margin:12px 0}}
img{{max-width:100%}}</style></head><body>
<h1>Breakout + 回撤熔断（路线2）</h1>
<p>数据区间 {data_start} ~ {data_end}（{n_universe} 只）｜月频｜分档滑点｜共享 MA240+波动率门控</p>
<div class="note"><b>熔断机制：</b>基于策略自身净值回撤（单遍自洽，昨日收盘判定今日执行）——
回撤 &gt;20% 月末建仓权重 ×{RED_MULT}；回撤 &gt;{DD_CLEAR*100:.0f}% 强制清仓暂停，净值创历史新高后恢复 100%。
选股逻辑与 V8 参数零改动。<br>
<b>熔断统计：</b>清仓触发 {st['n_clear']} 次｜降仓期 {st['n_red_days']} 期｜暂停 {st['n_paused_days']} 天｜
恢复 {st['n_recover']} 次｜引擎内最深回撤 {st['max_dd_reached']*100:.1f}%。</div>
<h2>绩效对比</h2>
<table><thead><tr><th>版本</th><th>全期夏普</th><th>全期年化</th><th>全期回撤</th>
<th>2024-25夏普</th><th>2024-25回撤</th></tr></thead><tbody>{rows}</tbody></table>
<h2>资金曲线</h2>{eq_img}
<h2>回撤曲线</h2>{dd_img}
<h2>验证标准</h2>
<div class="verdict">{verdict}</div>
<p style="font-size:12px;color:#999">生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}｜
熔断状态逐日由净值驱动，无未来函数（T 日行动只用 ≤T-1 收盘信息）。</p>
</body></html>"""

    out_path = config.OUTPUT_DIR / "report_breakout_fuse.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"\n[{datetime.now()}] 报告: {out_path}  耗时 {datetime.now()-t0}")


if __name__ == "__main__":
    main()
