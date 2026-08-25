# -*- coding: utf-8 -*-
"""
main_mainboard.py — 主板版回测（选股池：全市场主板 60/00 开头）
================================================================

背景：V8.1 选股池含创业板(30)/科创板(688)，用户无对应交易权限。
本脚本将选股池切换为「沪深主板」（60/00 开头，剔除 ST/退市，约 3000 只），
其余机制与 V8.1 完全一致：
  - 冷启动：2018-2019 数据训练初始 LightGBM（factor_eval_v7.build_rolling_reversal_signal）
  - 滚动重训：2020Q1 起每季度末用过去 36 个月重训
  - 市场过滤器：MA240 + 波动率降仓（沪深300，v6_index）
  - Regime 切换：反转 IC 滚动均值 > 0.05 用反转，否则用动量/质量
  - 分区间部署：≤2023 V8 原样；≥2024 E 等权组合（V8+Trend+Breakout 各 1/3）
  - 月频 30 只、单只 cap 10%、分档滑点（0.1%/0.3%/0.5%）+ 冲击成本同 V8.1

验证标准（用户规定，2026-08-25）：
  全期夏普 ≥ 0.50 | 2024-25夏普 ≥ 0.50 | 全期回撤 ≤ -25%（不深于）
  对比 V8.1（全期 0.60 / 2024-25 0.79 / 回撤 -19.40%）

交付：output/report_mainboard.html
运行：cd src && python main_mainboard.py（约 20-30 分钟）
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

# ---- 数据路径：主板版 ----
CLOSE_P = config.DATA_DIR / "mainboard_close_panel.parquet"
AMOUNT_P = config.DATA_DIR / "mainboard_amount_panel.parquet"
ROE_P = config.DATA_DIR / "roe_panel_mainboard.parquet"
GPM_P = config.DATA_DIR / "gpm_yoy_panel_mainboard.parquet"

MA_BASE = 240
IC_BASE = 0.05
TOP_N = 30
CAP = 0.10
SPLIT_DATE = "2024-01-01"     # 分区间门控（与 V8.1 一致）

# 验证标准（用户规定）：全期≥0.50 / 2024-25≥0.50 / 回撤≤-25%
TARGET_FULL_SHARPE = 0.50
TARGET_NEW_SHARPE = 0.50
TARGET_DD = -0.25

# V8.1 原版对照（已回测锁定，2026-08-19）
V81_REF = {"full_sharpe": 0.60, "new_sharpe": 0.79, "max_dd": -19.40}


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
    close = pd.read_parquet(CLOSE_P)
    amount = pd.read_parquet(AMOUNT_P)
    idx = pd.read_parquet(config.DATA_DIR / "v6_index.parquet")
    if isinstance(idx, pd.DataFrame):
        idx = idx.iloc[:, 0]
    roe = pd.read_parquet(ROE_P).reindex(close.index).ffill()
    gpm = pd.read_parquet(GPM_P).reindex(close.index).ffill()
    n_universe = len(close.columns)
    data_start = str(close.index[0].date())
    data_end = str(close.index[-1].date())

    slip_map, tier_counts = build_slippage_map(amount)
    close_m = mask_new_listings(close, config.NEW_STOCK_MIN_DAYS)
    rsi = compute_rsi(close_m, config.RSI_WINDOW)
    me = month_ends_in(close_m, config.START_DATE, data_end)
    # 主板全周期 2018 起有完整成交额，直接用 amount/close 重建 volume（与 V8 2024-25 口径一致）
    ohlcv_full = build_ohlcv_full(close, amount, {}, list(close.columns))

    print(f"[{datetime.now()}] 构建信号（主板 {n_universe} 只宇宙）...")
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

    # ---- 基准：V8 原样全期（同一引擎，供对照）----
    print(f"[{datetime.now()}] 基准 V8 全期回测...")
    eq_v8, tr_v8 = run_backtest_v5(close_m, sel_v8, me, config.START_DATE, data_end,
                                   target_weight=tw, slippage_map=slip_map)

    # ---- 分区间：段1 V8(≤2023) → 段2 E等权(≥2024) ----
    print(f"[{datetime.now()}] 主板版分区间回测（SPLIT={SPLIT_DATE}）...")
    me1 = me[me < pd.Timestamp(SPLIT_DATE)]
    me2 = me[me >= pd.Timestamp(SPLIT_DATE)]
    eq1, tr1 = run_backtest_v5(close_m, sel_v8, me1, config.START_DATE, "2023-12-31",
                               target_weight=tw, slippage_map=slip_map)
    init2 = float(eq1.iloc[-1])
    sel_map = {"V8": sel_v8, "Trend": sel_trend, "Breakout": sel_brk}
    wt_sched = build_eq_schedule(sel_map, ["V8", "Trend", "Breakout"], me2)
    eq2, tr2 = run_backtest_combo(close_m, wt_sched, me2, SPLIT_DATE, data_end,
                                  target_weight=tw, slippage_map=slip_map,
                                  init_capital=init2)
    assert eq1.index[-1] < eq2.index[0], "两段日期必须不重叠"
    eq_split = pd.concat([eq1, eq2])
    eq_split = eq_split[~eq_split.index.duplicated(keep="first")].sort_index()

    m_v8 = _m(eq_v8)
    m_mb = _m(eq_split)
    print(f"  主板V8 : 全期夏普={m_v8['full']['sharpe']:.2f} 回撤={m_v8['full']['max_drawdown']*100:.1f}% "
          f"2024-25={m_v8['new']['sharpe']:.2f}")
    print(f"  主板V8.1: 全期夏普={m_mb['full']['sharpe']:.2f} 回撤={m_mb['full']['max_drawdown']*100:.1f}% "
          f"2024-25={m_mb['new']['sharpe']:.2f}")

    # ---- 验证标准（回撤达标 = 不深于阈值）----
    ok1 = m_mb["full"]["sharpe"] >= TARGET_FULL_SHARPE
    ok2 = m_mb["new"]["sharpe"] >= TARGET_NEW_SHARPE
    ok3 = m_mb["full"]["max_drawdown"] >= TARGET_DD
    passed = ok1 and ok2 and ok3
    print(f"\n验证: 全期≥{TARGET_FULL_SHARPE}: {ok1} ({m_mb['full']['sharpe']:.2f}) | "
          f"2024-25≥{TARGET_NEW_SHARPE}: {ok2} ({m_mb['new']['sharpe']:.2f}) | "
          f"回撤≤{TARGET_DD*100:.0f}%: {ok3} ({m_mb['full']['max_drawdown']*100:.2f}%)")
    print(f">>> 主板版{'✅ 通过（可切换模拟盘）' if passed else '❌ 未通过（不建议切换）'}")

    # ---- 报告 ----
    eq_img = chart_multi_equity({"主板V8(对照)": eq_v8, "主板V8.1(2024起E组合)": eq_split,
                                 "CSI300": idx_eq}, "主板版分区间部署 vs 主板V8")
    dd_img = chart_drawdown_compare({"主板V8(对照)": eq_v8, "主板V8.1": eq_split, "CSI300": idx_eq})

    def mrow(name, m):
        return (f"<tr><td><b>{name}</b></td><td>{m['full']['sharpe']:.2f}</td>"
                f"<td>{m['full']['annual_return']*100:.2f}%</td>"
                f"<td>{m['full']['max_drawdown']*100:.2f}%</td>"
                f"<td>{m['old']['sharpe']:.2f}</td>"
                f"<td>{m['new']['sharpe']:.2f}</td>"
                f"<td>{m['new']['max_drawdown']*100:.2f}%</td></tr>")

    rows = (mrow("主板V8 原样（对照）", m_v8) + mrow("主板V8.1 分区间部署", m_mb))

    ref_row = (f"<tr><td><b>V8.1 原版（锚点）</b></td>"
               f"<td>{V81_REF['full_sharpe']:.2f}</td><td>—</td>"
               f"<td>{V81_REF['max_dd']:.2f}%</td><td>—</td>"
               f"<td>{V81_REF['new_sharpe']:.2f}</td><td>—</td></tr>")

    verdict = "<br>".join([
        f"{'✅' if ok1 else '❌'} 全期夏普 ≥{TARGET_FULL_SHARPE}：{m_mb['full']['sharpe']:.2f}（V8.1原版 {V81_REF['full_sharpe']}）",
        f"{'✅' if ok2 else '❌'} 2024-25夏普 ≥{TARGET_NEW_SHARPE}：{m_mb['new']['sharpe']:.2f}（V8.1原版 {V81_REF['new_sharpe']}）",
        f"{'✅' if ok3 else '❌'} 全期回撤 ≤{TARGET_DD*100:.0f}%：{m_mb['full']['max_drawdown']*100:.2f}%（V8.1原版 {V81_REF['max_dd']}%）",
        f"<b style='font-size:14px'>结论：{'✅ 通过验证，可切换模拟盘至主板版' if passed else '❌ 未通过验证，维持 V8.1 原版'}</b>",
    ])

    # 每年夏普（主板版 vs 主板V8）
    yt = yearly_sharpe(eq_split)
    yrows = "".join(f"<tr><td>{yr}</td><td>{v:.2f}</td></tr>"
                    for yr, v in yt.items() if isinstance(yr, int))

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>主板版回测报告（V8.1 选股池切换）</title>
<style>body{{font-family:-apple-system,'PingFang SC',sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222}}
table{{border-collapse:collapse;width:100%;margin:14px 0;font-size:13px}}
th,td{{border:1px solid #ddd;padding:7px 9px;text-align:center}}
th{{background:#f5f5f5}} h2{{border-bottom:2px solid #eee;padding-bottom:6px;margin-top:34px}}
.verdict{{background:#fff7e6;border:1px solid #f0c36d;border-radius:6px;padding:12px 16px;margin:16px 0}}
.note{{background:#f2f6fc;border-left:4px solid #4a7abb;padding:10px 14px;font-size:13px;color:#444;margin:12px 0}}
.pass{{background:#e8f8e8;border:1px solid #7ab87a;border-radius:6px;padding:12px 16px;margin:16px 0}}
.fail{{background:#fdeaea;border:1px solid #d88;border-radius:6px;padding:12px 16px;margin:16px 0}}
img{{max-width:100%}}</style></head><body>
<h1>主板版回测报告（V8.1 选股池切换）</h1>
<p>数据区间 {data_start} ~ {data_end}（主板 {n_universe} 只，60/00 开头，剔除 ST/退市）｜月频｜分档滑点｜MA240+波动率门控</p>
<div class="note"><b>机制：</b>与 V8.1 完全一致——冷启动 2018-2019 训练 LightGBM、2020Q1 起每季度末
36 个月滚动重训；反转 IC 滚动均值 &gt;0.05 用反转、否则动量/质量；≤2023 V8 原样、≥2024 E 等权组合
（V8+Trend+Breakout 各 1/3，单只 cap 10%）。<b>仅选股池改为沪深主板（60/00 开头）</b>。</div>
<h2>绩效对比（主板版 vs V8.1 原版）</h2>
<table><thead><tr><th>版本</th><th>全期夏普</th><th>全期年化</th><th>全期回撤</th>
<th>2018-23夏普</th><th>2024-25夏普</th><th>2024-25回撤</th></tr></thead>
<tbody>{ref_row}{rows}</tbody></table>
<h2>验证标准</h2>
<div class="{'pass' if passed else 'fail'}">{verdict}</div>
<h2>资金曲线</h2>{eq_img}
<h2>回撤曲线</h2>{dd_img}
<h2>主板版年度夏普</h2>
<table><thead><tr><th>年份</th><th>夏普</th></tr></thead><tbody>{yrows}</tbody></table>
<p style="font-size:12px;color:#999">生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}｜
引擎零改动；选股池 = 全市场主板（60/00 开头），排除创业板/科创板/北交所。</p>
</body></html>"""

    out_path = config.OUTPUT_DIR / "report_mainboard.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"\n[{datetime.now()}] 报告: {out_path}  耗时 {datetime.now()-t0}")


if __name__ == "__main__":
    main()
