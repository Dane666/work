# -*- coding: utf-8 -*-
"""
V9.4 模块B（严格版）：多策略并行 —— 6 臂实验。

前置裁定：模块A（ATR 卖出规则）已判失败 → 全部策略使用 V8 卖出规则（无日内规则）。

策略（全部：共享 MA240+波动率市场门控 / 月频 / 30 只 / V8 卖出规则 / V8 面板数据）：
  A V8 主策略（锚点，零参数改动）
  B 策略1 趋势跟踪：EMA12/EMA30 比值最高 30 只（须 EMA12>EMA30）
  C 策略2 均值回归：dev=close/MA20-1 截面 z<-2，dev 最低 30 只
  D 策略3 波动率突破：pos=(close-20日高)/20日高>-2% 且 量比>1.5，pos 最高 30 只
  E 组合·等权：有效策略（夏普≥0.3）各 1/N，策略内 30 只等权，单只 cap 10%
  F 组合·绩效加权：每月末近 6 个月夏普归一化（负夏普置 0），单只 cap 10%

执行顺序：先 B/C/D 独立回测 → 剔除夏普<0.3 的策略 → 再 E/F 组合。
交付：report_module_b.html + output/v8_combo_weights.csv（每月权重审计）。
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
from strategies.mean_reversion import gen_signal as sig_mr
from strategies.vol_breakout import gen_signal as sig_breakout

END_EXT = "2025-08-12"
MA_BASE = 240
IC_BASE = 0.05
TOP_N = 30
ROLL_M = 6          # F 臂滚动夏普窗口（月）
MIN_SHARPE = 0.30   # 独立策略保留阈值（用户规定：低于则从组合剔除）
CAP = 0.10          # 单只股票权重上限（与 V8 一致）


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


def monthly_ret_series(eq: pd.Series, me) -> pd.Series:
    eq = eq.sort_index()
    me_used = [t for t in me if t in eq.index]
    mvals = pd.Series({t: float(eq.loc[t]) for t in me_used})
    return mvals.pct_change().dropna()


def rolling_sharpe_weights(ret_series_map: dict, me, window: int = ROLL_M) -> dict:
    """每月末各策略近 window 个月夏普归一化（point-in-time：只用 <t 的月收益）。"""
    names = list(ret_series_map.keys())
    out = {}
    for t in me:
        wts = {}
        for s in names:
            rs = ret_series_map[s]
            win = rs.loc[rs.index < t].tail(window)
            if len(win) >= 3:
                mu, sd = win.mean(), win.std(ddof=0)
                sh = mu / sd * np.sqrt(12) if sd > 0 else 0.0
            else:
                sh = 0.0
            wts[s] = max(sh, 0.0)
        tot = sum(wts.values())
        out[t] = {s: (wts[s] / tot if tot > 0 else 1.0 / len(names)) for s in names}
    return out


def build_schedule(sel_map: dict, w_map: dict, me, cap: float = CAP) -> dict:
    """合并各策略持仓为目标权重表 {t: [(code, weight)]}，单只 cap。"""
    schedule = {}
    for t in me:
        acc = {}
        for name, w_s in w_map[t].items():
            codes = sel_map[name].get(t, [])
            if not codes:
                continue
            w_each = w_s / len(codes)
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
    sel_mr = sig_mr(close_m, me, top_n=TOP_N)
    sel_brk = sig_breakout(close_m, amount, me, top_n=TOP_N)
    for nm, s in [("trend", sel_trend), ("mean_rev", sel_mr), ("breakout", sel_brk)]:
        n_empty = sum(1 for v in s.values() if not v)
        print(f"  {nm}: {len(s)} 期，空仓 {n_empty} 期，平均 {np.mean([len(v) for v in s.values()]):.1f} 只")

    tw, vol_regime = build_ma240_vol_target_weight(
        idx, close_m.index, MA_BASE, month_ends=me,
        vol_q=0.75, reduced_weight=0.60, vol_lookback=756)
    idx_ret = idx.reindex(close_m.index).pct_change().fillna(0)
    idx_eq = (1.0 + idx_ret).cumprod() * config.INIT_CAPITAL

    # ---- 独立回测 A/B/C/D ----
    print(f"[{datetime.now()}] 独立回测 A/B/C/D ...")
    def bt(sel, tag):
        eq, tr = run_backtest_v5(close_m, sel, me, config.START_DATE, END_EXT,
                                 target_weight=tw, slippage_map=slip_map)
        print(f"  [{tag}] 全期夏普={compute_metrics(eq)['sharpe']:.2f}")
        return eq, tr

    eq_v8, _ = bt(sel_v8, "A V8")
    eq_tr, _ = bt(sel_trend, "B Trend")
    eq_mr, _ = bt(sel_mr, "C MeanRev")
    eq_br, _ = bt(sel_brk, "D Breakout")
    m_v8, m_tr, m_mr, m_br = _m(eq_v8), _m(eq_tr), _m(eq_mr), _m(eq_br)
    for nm, m in [("Trend", m_tr), ("MeanRev", m_mr), ("Breakout", m_br)]:
        print(f"  {nm}: 全期夏普={m['full']['sharpe']:.2f} 回撤={m['full']['max_drawdown']*100:.1f}% "
              f"2024-25夏普={m['new']['sharpe']:.2f}")

    # ---- 剔除夏普<0.3 的策略 ----
    standalone = {"Trend": (sel_trend, m_tr), "MeanRev": (sel_mr, m_mr),
                  "Breakout": (sel_brk, m_br)}
    kept = {nm: (sel, m) for nm, (sel, m) in standalone.items()
            if m["full"]["sharpe"] >= MIN_SHARPE}
    dropped = [nm for nm in standalone if nm not in kept]
    print(f"\n剔除（夏普<{MIN_SHARPE}）: {dropped if dropped else '无'}；"
          f"保留进组合: {['V8'] + list(kept.keys())}")

    # ---- 月收益与相关性 ----
    ret_map = {"V8": monthly_ret_series(eq_v8, me),
               "Trend": monthly_ret_series(eq_tr, me),
               "MeanRev": monthly_ret_series(eq_mr, me),
               "Breakout": monthly_ret_series(eq_br, me)}
    corr = pd.DataFrame(ret_map).corr()
    print("\n月收益相关矩阵：")
    print(corr.round(3).to_string())

    # ---- E 臂：等权组合（有效策略各 1/N）----
    combo_names = ["V8"] + list(kept.keys())
    sel_map_all = {"V8": sel_v8, "Trend": sel_trend, "MeanRev": sel_mr, "Breakout": sel_brk}
    w_eq = {t: {nm: 1.0 / len(combo_names) for nm in combo_names} for t in me}
    wt_sched_eq = build_schedule(sel_map_all, w_eq, me)
    eq_cmb_eq, tr_cmb_eq = run_backtest_combo(close_m, wt_sched_eq, me,
                                              config.START_DATE, END_EXT,
                                              target_weight=tw, slippage_map=slip_map)
    m_eq = _m(eq_cmb_eq)
    print(f"[E 等权组合] 全期夏普={m_eq['full']['sharpe']:.2f} 回撤={m_eq['full']['max_drawdown']*100:.1f}% "
          f"2024-25={m_eq['new']['sharpe']:.2f}")

    # ---- F 臂：近 6 月滚动夏普加权组合 ----
    ret_c = {nm: ret_map[nm] for nm in combo_names}
    w_roll = rolling_sharpe_weights(ret_c, me, window=ROLL_M)
    wt_sched_roll = build_schedule(sel_map_all, w_roll, me)
    eq_cmb_roll, tr_cmb_roll = run_backtest_combo(close_m, wt_sched_roll, me,
                                                  config.START_DATE, END_EXT,
                                                  target_weight=tw, slippage_map=slip_map)
    m_roll = _m(eq_cmb_roll)
    print(f"[F 绩效加权组合] 全期夏普={m_roll['full']['sharpe']:.2f} 回撤={m_roll['full']['max_drawdown']*100:.1f}% "
          f"2024-25={m_roll['new']['sharpe']:.2f}")

    # ---- 权重明细（审计）----
    w_rows = []
    for t in me:
        wts = w_roll[t]
        row = {"month_end": t}
        for nm in combo_names:
            row[f"w_{nm}"] = round(wts.get(nm, 0.0), 4)
        row["n_holdings"] = len(wt_sched_roll[t])
        w_rows.append(row)
    w_df = pd.DataFrame(w_rows)
    w_df.to_csv(config.OUTPUT_DIR / "v8_combo_weights.csv", index=False)

    # ---- 判定 ----
    d_eq = m_eq["full"]["sharpe"] - m_v8["full"]["sharpe"]
    d_roll = m_roll["full"]["sharpe"] - m_v8["full"]["sharpe"]
    d_eq_new = m_eq["new"]["sharpe"] - m_v8["new"]["sharpe"]
    d_roll_new = m_roll["new"]["sharpe"] - m_v8["new"]["sharpe"]
    d_eq_dd = m_eq["full"]["max_drawdown"] * 100 - m_v8["full"]["max_drawdown"] * 100
    d_roll_dd = m_roll["full"]["max_drawdown"] * 100 - m_v8["full"]["max_drawdown"] * 100
    goal_full = m_eq["full"]["sharpe"] > 0.60 or m_roll["full"]["sharpe"] > 0.60
    goal_new = m_eq["new"]["sharpe"] > 0.70 or m_roll["new"]["sharpe"] > 0.70

    # ---- 报告 ----
    eq_img = chart_multi_equity({
        "V8 主策略": eq_v8, "Trend": eq_tr, "MeanRev": eq_mr, "Breakout": eq_br,
        "E 等权组合": eq_cmb_eq, "F 绩效加权组合": eq_cmb_roll, "CSI300": idx_eq},
        "V9.4 模块B · 6 臂资金曲线")
    dd_img = chart_drawdown_compare({
        "V8 主策略": eq_v8, "E 等权组合": eq_cmb_eq, "F 绩效加权组合": eq_cmb_roll})

    def mrow(name, m, mark=""):
        return (f"<tr><td><b>{name}</b></td><td>{m['full']['sharpe']:.2f}</td>"
                f"<td>{m['full']['annual_return']*100:.2f}%</td>"
                f"<td>{m['full']['max_drawdown']*100:.2f}%</td>"
                f"<td>{m['new']['sharpe']:.2f}</td>"
                f"<td>{m['new']['max_drawdown']*100:.2f}%</td><td>{mark}</td></tr>")

    rows = (mrow("A V8 主策略（锚点）", m_v8, "基线")
            + mrow("B 趋势跟踪（EMA12/30 比值）", m_tr,
                   "剔除" if "Trend" in dropped else "保留")
            + mrow("C 均值回归（MA20 偏离 z<-2σ）", m_mr,
                   "剔除" if "MeanRev" in dropped else "保留")
            + mrow("D 波动率突破（接近高点+放量）", m_br,
                   "剔除" if "Breakout" in dropped else "保留")
            + mrow("E 四策略等权组合", m_eq, f"Δ夏普 {d_eq:+.2f}")
            + mrow("F 绩效加权组合（近6月夏普）", m_roll, f"Δ夏普 {d_roll:+.2f}"))

    corr_html = corr.round(3).to_html(border=0, classes="corr")

    verdict_parts = []
    if goal_full:
        verdict_parts.append(f"<span style='color:#c0392b'>达成全期目标(&gt;0.60)：E={m_eq['full']['sharpe']:.2f} / F={m_roll['full']['sharpe']:.2f}</span>")
    else:
        verdict_parts.append(f"<span style='color:#888'>未达成全期目标(&gt;0.60)：E={m_eq['full']['sharpe']:.2f} / F={m_roll['full']['sharpe']:.2f}（V8={m_v8['full']['sharpe']:.2f}）</span>")
    if goal_new:
        verdict_parts.append(f"<span style='color:#c0392b'>达成2024-25目标(&gt;0.70)：E={m_eq['new']['sharpe']:.2f} / F={m_roll['new']['sharpe']:.2f}</span>")
    else:
        verdict_parts.append(f"<span style='color:#888'>未达成2024-25目标(&gt;0.70)：E={m_eq['new']['sharpe']:.2f} / F={m_roll['new']['sharpe']:.2f}（V8={m_v8['new']['sharpe']:.2f}）</span>")

    verdict = "<br>".join(verdict_parts)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>V9.4 模块B（严格版）· 多策略并行 6 臂实验</title>
<style>body{{font-family:-apple-system,'PingFang SC',sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222}}
table{{border-collapse:collapse;width:100%;margin:14px 0;font-size:13px}}
th,td{{border:1px solid #ddd;padding:7px 9px;text-align:center}}
th{{background:#f5f5f5}} h2{{border-bottom:2px solid #eee;padding-bottom:6px;margin-top:34px}}
.verdict{{background:#fff7e6;border:1px solid #f0c36d;border-radius:6px;padding:12px 16px;margin:16px 0}}
.note{{background:#f2f6fc;border-left:4px solid #4a7abb;padding:10px 14px;font-size:13px;color:#444;margin:12px 0}}
img{{max-width:100%}} .corr{{margin:0 auto}}</style></head><body>
<h1>V9.4 模块B：多策略并行（6 臂实验）</h1>
<p>数据区间 {data_start} ~ {data_end}（{n_universe} 只）｜月频｜分档滑点｜共享 MA240+波动率门控</p>
<div class="note"><b>设定：</b>前置裁定模块A（ATR）失败 → 全部策略用 V8 卖出规则（无日内规则）。
各策略独立输出 30 只持仓；E 臂等权（有效策略各 1/N）、F 臂近 6 月滚动夏普归一化；
单只权重上限 10%。独立策略全期夏普 &lt;{MIN_SHARPE} 从组合剔除（当前剔除：{dropped or '无'}）。</div>
<h2>各策略独立绩效（A/B/C/D 臂）</h2>
<table><thead><tr><th>策略</th><th>全期夏普</th><th>全期年化</th><th>全期回撤</th>
<th>2024-25夏普</th><th>2024-25回撤</th><th>组合去留</th></tr></thead><tbody>{rows}</tbody></table>
<h2>月收益相关性矩阵（组合前提：低相关）</h2>{corr_html}
<h2>资金曲线（全部 6 臂）</h2>{eq_img}
<h2>回撤对比（V8 vs 组合）</h2>{dd_img}
<h2>F 臂组合权重明细（近 6 月滚动夏普归一化，抽样）</h2>
<table><thead><tr><th>月末</th>{''.join(f'<th>w_{nm}</th>' for nm in combo_names)}<th>持仓数</th></tr></thead><tbody>
{''.join(f"<tr><td>{r['month_end'].date()}</td>{''.join(f'<td>{r[f'w_{nm}']:.3f}</td>' for nm in combo_names)}<td>{int(r['n_holdings'])}</td></tr>"
         for _, r in w_df[w_df['month_end'] >= '2023-01-01'].iterrows())}
</tbody></table>
<p style="font-size:12px;color:#888">完整权重见 output/v8_combo_weights.csv（审计用）。</p>
<h2>归因结论</h2>
<div class="verdict">{verdict}</div>
<p style="font-size:12px;color:#999">生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}｜
F 臂权重只用 ≤t-1 月收益（point-in-time）；E/F 组合总仓位受 MA240 门控 ≤100%，单只 ≤10%。</p>
</body></html>"""

    out_path = config.OUTPUT_DIR / "report_module_b.html"
    out_path.write_text(html, encoding="utf-8")

    print("\n================ 模块B 判定 ================")
    print(f"[E 等权 vs V8] 全期 {d_eq:+.2f} ({m_v8['full']['sharpe']:.2f}→{m_eq['full']['sharpe']:.2f}) "
          f"2024-25 {d_eq_new:+.2f} 回撤 {d_eq_dd:+.2f}pp")
    print(f"[F 加权 vs V8] 全期 {d_roll:+.2f} ({m_v8['full']['sharpe']:.2f}→{m_roll['full']['sharpe']:.2f}) "
          f"2024-25 {d_roll_new:+.2f} 回撤 {d_roll_dd:+.2f}pp")
    print(f"目标达成: 全期>0.60={goal_full} | 2024-25>0.70={goal_new}")
    print(f"[{datetime.now()}] 报告: {out_path}  耗时 {datetime.now()-t0}")


if __name__ == "__main__":
    main()
