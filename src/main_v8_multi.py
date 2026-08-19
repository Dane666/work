# -*- coding: utf-8 -*-
"""
V9.4 模块B：多策略并行框架（在 V8 基础上并行 3 个低相关性策略）。

前置裁定（模块A）：ATR 卖出规则已判失败（全期夏普 0.08 vs V8 0.57，回撤恶化），
classic（-10%+RSI70）亦失败（0.30）。故模块B 全部策略使用 **V8 卖出规则**
（月末轮换 + regime 清仓，无日内止损止盈）。

框架：
  1) V8 主策略（不变，零参数改动）          -> sel_v8（build_selection_v5）
  2) 策略1 趋势跟踪（EMA12/EMA30 金叉）     -> strategies/trend_ema.py
  3) 策略2 均值回归（MA20-2σ 超跌）         -> strategies/mean_reversion.py
  4) 策略3 波动率突破（缩量后放量突破20日高）-> strategies/vol_breakout.py
  5) 组合：月末按各策略「近 3 个月滚动夏普」归一化权重分配资金，
     策略内等权，总仓位受共享 MA240 门控（target_weight）约束，≤100%。
流程：先各自独立回测确认有效性 → 再组合。
交付：output/report_v8_multistrategy.html（各策略独立绩效 + 组合绩效 + 相关性矩阵）。
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
ROLL_M = 3          # 滚动夏普窗口（月）
TOP_N = 30          # 各策略选股数


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
    """日频净值 → 月末净值 → 月收益序列。"""
    eq = eq.sort_index()
    me_used = [t for t in me if t in eq.index]
    mvals = pd.Series({t: float(eq.loc[t]) for t in me_used})
    return mvals.pct_change().dropna()


def rolling_sharpe_weights(ret_series_map: dict, me) -> dict:
    """月末计算各策略近 ROLL_M 个月滚动夏普，归一化为组合权重（point-in-time）。

    权重 = max(sharpe_s, 0) / Σ max(sharpe, 0)；全 ≤0 或样本不足 → 等权。
    夏普用截至 t-1 月末的完整月收益（不含 t 月自身，零未来函数）。
    """
    names = list(ret_series_map.keys())
    out = {}
    for t in me:
        wts = {}
        for s in names:
            rs = ret_series_map[s]
            hist = rs.loc[rs.index < t]           # 严格早于 t
            win = hist.tail(ROLL_M)
            if len(win) >= 2:
                mu = win.mean()
                sd = win.std(ddof=0)
                sh = mu / sd * np.sqrt(12) if sd > 0 else 0.0
            else:
                sh = 0.0
            wts[s] = max(sh, 0.0)
        tot = sum(wts.values())
        if tot > 0:
            out[t] = {s: wts[s] / tot for s in names}
        else:
            out[t] = {s: 1.0 / len(names) for s in names}   # 全无效 → 等权
    return out


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

    slip_map, tier_counts = build_slippage_map(amount)
    close_m = mask_new_listings(close, config.NEW_STOCK_MIN_DAYS)
    rsi = compute_rsi(close_m, config.RSI_WINDOW)
    me = month_ends_in(close_m, config.START_DATE, END_EXT)
    ohlcv_old = pd.read_pickle(config.V3_OHLCV) if config.V3_OHLCV.exists() else {}
    ohlcv_full = build_ohlcv_full(close, amount, ohlcv_old, codes_all)

    print(f"[{datetime.now()}] 构建 V8 信号（{n_universe} 只宇宙）...")
    reversal_signal = build_rolling_reversal_signal(close_m, ohlcv_full, window=36)
    monthly_ic = monthly_reversal_ic(close_m, rsi, me, config.FWD_RETURN_DAYS)
    rolling_ic, use_reversal = compute_rolling_regime(monthly_ic, 12, IC_BASE, 6)
    config.ENABLE_ANALYST_FACTOR = False
    mz = compute_momentum_zscore(close_m, roe, gpm, me)
    sel_v8, sw_v8 = build_selection_v5(close_m, rsi, reversal_signal, mz, me,
                                       use_reversal, 0.20, TOP_N)

    print(f"[{datetime.now()}] 构建三策略信号...")
    sel_trend = sig_trend(close_m, me, top_n=TOP_N)
    sel_mr = sig_mr(close_m, me, top_n=TOP_N)
    sel_brk = sig_breakout(close_m, amount, me, top_n=TOP_N)
    for nm, s in [("trend", sel_trend), ("mean_rev", sel_mr), ("breakout", sel_brk)]:
        n_empty = sum(1 for v in s.values() if not v)
        print(f"  {nm}: {len(s)} 期，空仓 {n_empty} 期，平均选股 "
              f"{np.mean([len(v) for v in s.values()]):.1f} 只")

    tw_clean = build_ma240_target_weight(idx, close_m.index, MA_BASE)
    tw, vol_regime = build_ma240_vol_target_weight(
        idx, close_m.index, MA_BASE, month_ends=me,
        vol_q=0.75, reduced_weight=0.60, vol_lookback=756)

    idx_ret = idx.reindex(close_m.index).pct_change().fillna(0)
    idx_eq = (1.0 + idx_ret).cumprod() * config.INIT_CAPITAL

    # ---- 独立回测（V8 卖出规则，共享门控）----
    print(f"[{datetime.now()}] 独立回测：V8 / trend / mean_rev / breakout ...")
    def bt(sel, tag):
        eq, tr = run_backtest_v5(close_m, sel, me, config.START_DATE, END_EXT,
                                 target_weight=tw, slippage_map=slip_map)
        print(f"  [{tag}] 全期夏普={compute_metrics(eq)['sharpe']:.2f}")
        return eq, tr

    eq_v8, tr_v8 = bt(sel_v8, "V8")
    eq_tr, tr_tr = bt(sel_trend, "trend")
    eq_mr, tr_mr = bt(sel_mr, "mean_rev")
    eq_br, tr_br = bt(sel_brk, "breakout")

    m_v8 = _m(eq_v8)
    m_tr = _m(eq_tr)
    m_mr = _m(eq_mr)
    m_br = _m(eq_br)
    print(f"  trend: 全期夏普={m_tr['full']['sharpe']:.2f} 回撤={m_tr['full']['max_drawdown']*100:.1f}% | "
          f"2024-25 {m_tr['new']['sharpe']:.2f}")
    print(f"  mean_rev: 全期夏普={m_mr['full']['sharpe']:.2f} 回撤={m_mr['full']['max_drawdown']*100:.1f}% | "
          f"2024-25 {m_mr['new']['sharpe']:.2f}")
    print(f"  breakout: 全期夏普={m_br['full']['sharpe']:.2f} 回撤={m_br['full']['max_drawdown']*100:.1f}% | "
          f"2024-25 {m_br['new']['sharpe']:.2f}")

    # ---- 月收益与相关性 ----
    ret_map = {"V8": monthly_ret_series(eq_v8, me),
               "Trend": monthly_ret_series(eq_tr, me),
               "MeanRev": monthly_ret_series(eq_mr, me),
               "Breakout": monthly_ret_series(eq_br, me)}
    corr = pd.DataFrame(ret_map).corr()
    print("\n策略月收益相关矩阵（含 V8）：")
    print(corr.round(3).to_string())

    # ---- 组合权重（近3月滚动夏普归一化）----
    print(f"[{datetime.now()}] 构建组合（近{ROLL_M}月滚动夏普归一化权重）...")
    w_map = rolling_sharpe_weights(
        {"Trend": ret_map["Trend"], "MeanRev": ret_map["MeanRev"],
         "Breakout": ret_map["Breakout"]}, me)
    wt_schedule = {}
    for t in me:
        wts = w_map[t]
        picks = []
        for s, w_s in wts.items():
            codes_s = {"Trend": sel_trend, "MeanRev": sel_mr, "Breakout": sel_brk}[s].get(t, [])
            if not codes_s:
                continue
            w_each = w_s / len(codes_s)
            for c in codes_s:
                picks.append((c, w_each))
        wt_schedule[t] = picks

    eq_cmb, tr_cmb = run_backtest_combo(close_m, wt_schedule, me, config.START_DATE, END_EXT,
                                        target_weight=tw, slippage_map=slip_map)
    m_cmb = _m(eq_cmb)
    print(f"  组合: 全期夏普={m_cmb['full']['sharpe']:.2f} 回撤={m_cmb['full']['max_drawdown']*100:.1f}% | "
          f"2024-25 {m_cmb['new']['sharpe']:.2f}")

    # 组合权重演变（抽样打印）
    w_evol = []
    for t in me:
        wts = w_map[t]
        w_evol.append({"month_end": t, **{f"w_{k}": round(v, 3) for k, v in wts.items()}})
    w_df = pd.DataFrame(w_evol)
    w_df.to_csv(config.OUTPUT_DIR / "v8_multi_weights.csv", index=False)
    print("\n组合权重抽样（2024-2025）：")
    print(w_df[w_df["month_end"] >= "2024-01-01"].tail(6).to_string(index=False))

    # ---- 判定 ----
    d_cmb = m_cmb["full"]["sharpe"] - m_v8["full"]["sharpe"]
    d_cmb_new = m_cmb["new"]["sharpe"] - m_v8["new"]["sharpe"]
    d_cmb_dd = m_cmb["full"]["max_drawdown"] * 100 - m_v8["full"]["max_drawdown"] * 100
    combo_ok = (d_cmb >= 0.02) and (d_cmb_dd <= 1.0)

    # ---- 报告 ----
    eq_img = chart_multi_equity({
        "V8 主策略": eq_v8, "Trend(EMA12/30)": eq_tr, "MeanRev(MA20-2σ)": eq_mr,
        "Breakout(20日高)": eq_br, "组合(滚动夏普加权)": eq_cmb, "CSI300": idx_eq})
    dd_img = chart_drawdown_compare({
        "V8 主策略": eq_v8, "组合(滚动夏普加权)": eq_cmb, "CSI300": idx_eq})

    def mrow(name, m, note=""):
        return (f"<tr><td><b>{name}</b></td><td>{m['full']['sharpe']:.2f}</td>"
                f"<td>{m['full']['annual_return']*100:.2f}%</td>"
                f"<td>{m['full']['max_drawdown']*100:.2f}%</td>"
                f"<td>{m['new']['sharpe']:.2f}</td>"
                f"<td>{m['new']['max_drawdown']*100:.2f}%</td><td>{note}</td></tr>")

    rows = (mrow("V8 主策略（锚点）", m_v8)
            + mrow("策略1 趋势跟踪（EMA12/30 金叉）", m_tr)
            + mrow("策略2 均值回归（MA20-2σ）", m_mr)
            + mrow("策略3 波动率突破（缩量→放量）", m_br)
            + mrow("组合（近3月滚动夏普加权）", m_cmb))

    corr_html = corr.round(3).to_html(border=0, classes="corr")

    if combo_ok:
        verdict = (f"<span style='color:#c0392b;font-weight:bold'>组合有效</span>："
                   f"相对 V8 全期夏普 {d_cmb:+.2f}、回撤 {d_cmb_dd:+.2f}pp、"
                   f"2024-25 夏普 {d_cmb_new:+.2f} → 建议合并为 V10。")
    else:
        verdict = (f"<span style='color:#888'>组合未通过（全期夏普 {d_cmb:+.2f}、"
                   f"回撤 {d_cmb_dd:+.2f}pp）→ 维持 V8 单独运行；"
                   f"或仅保留组合中显著有效的策略子集。</span>")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>V9.4 模块B · 多策略并行</title>
<style>body{{font-family:-apple-system,'PingFang SC',sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222}}
table{{border-collapse:collapse;width:100%;margin:14px 0;font-size:13px}}
th,td{{border:1px solid #ddd;padding:7px 9px;text-align:center}}
th{{background:#f5f5f5}} h2{{border-bottom:2px solid #eee;padding-bottom:6px;margin-top:34px}}
.verdict{{background:#fff7e6;border:1px solid #f0c36d;border-radius:6px;padding:12px 16px;margin:16px 0}}
.note{{background:#f2f6fc;border-left:4px solid #4a7abb;padding:10px 14px;font-size:13px;color:#444;margin:12px 0}}
img{{max-width:100%}} .corr{{margin:0 auto}}</style></head><body>
<h1>V9.4 模块B：多策略并行框架</h1>
<p>数据区间 {data_start} ~ {data_end}（{n_universe} 只宇宙）｜月频调仓｜分档滑点 0.1/0.3/0.5%｜
共享 MA240+波动率市场门控</p>
<div class="note"><b>前置裁定：</b>模块A（ATR 卖出规则）已判失败（全期 0.08 vs V8 0.57、回撤恶化），
classic（-10%+RSI70）亦失败（0.30）。故本模块全部策略使用 <b>V8 卖出规则</b>（月末轮换 + regime 清仓）。
V8 主策略参数零改动。</div>
<h2>各策略独立绩效（先各自回测，确认有效后再组合）</h2>
<table><thead><tr><th>策略</th><th>全期夏普</th><th>全期年化</th><th>全期回撤</th>
<th>2024-25夏普</th><th>2024-25回撤</th><th>说明</th></tr></thead><tbody>{rows}</tbody></table>
<h2>净值曲线（全部策略 + 组合）</h2>{eq_img}
<h2>回撤对比（V8 vs 组合）</h2>{dd_img}
<h2>策略月收益相关性（低相关性是组合前提）</h2>{corr_html}
<p style="font-size:12px;color:#888">Pearson 相关，基于月末调仓日净值月收益；|ρ| 越低组合分散收益越强。</p>
<h2>组合权重演变（近 3 月滚动夏普归一化）</h2>
<table><thead><tr><th>月末</th><th>w_Trend</th><th>w_MeanRev</th><th>w_Breakout</th></tr></thead><tbody>
{''.join(f"<tr><td>{r['month_end'].date()}</td><td>{r['w_Trend']:.3f}</td>"
         f"<td>{r['w_MeanRev']:.3f}</td><td>{r['w_Breakout']:.3f}</td></tr>"
         for _, r in w_df[w_df['month_end'] >= '2023-01-01'].iterrows())}
</tbody></table>
<h2>结论</h2>
<div class="verdict">{verdict}</div>
<p style="font-size:12px;color:#999">生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}｜
组合权重 = 各策略近 3 个月滚动夏普的归一化权重（负夏普归零，全无效时等权），
月末调仓日 T 仅用 ≤T-1 的月收益，零未来函数；总仓位受 MA240 门控 ≤100%。</p>
</body></html>"""

    out_path = config.OUTPUT_DIR / "report_v8_multistrategy.html"
    out_path.write_text(html, encoding="utf-8")

    print("\n================ 模块B 判定 ================")
    print(f"[组合 vs V8] 全期 {d_cmb:+.2f} | 2024-25 {d_cmb_new:+.2f} | 回撤 {d_cmb_dd:+.2f}pp")
    print(f"组合有效: {combo_ok}")
    print(f"[{datetime.now()}] 报告: {out_path}  耗时 {datetime.now()-t0}")


if __name__ == "__main__":
    main()
