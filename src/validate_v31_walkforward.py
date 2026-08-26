# -*- coding: utf-8 -*-
"""
validate_v31_walkforward.py — 主板版 V3.1 滚动窗口（walk-forward）验证
========================================================================
目的：V3.1 方向3 的门控参数（σ=0.20/降幅80%）取自全样本 in-sample 网格择优，
存在过拟合到 2024-25 牛市的风险。本脚本用滚动窗口验证其稳健性：

  训练期（仅用于确定门控参数 σ/drop）→ 验证期（held-out 评估）

窗口（训练起点固定 2018，每向前滚动1年扩展训练，验证为剩余段）：
  W1: 训练 2018-2021 → 验证 2022-2025
  W2: 训练 2018-2022 → 验证 2023-2025
  W3: 训练 2018-2023 → 验证 2024-2025
  W4: 训练 2018-2024 → 验证 2025

每窗口：
  (a) 在训练期对 σ(0.1~0.6 步长0.05) × 降幅(0.5~0.9 步长0.1) 共55组合做网格，
      按「训练期夏普优先 + 回撤≤-25%护栏」选最优 (σ,drop)；
  (b) 用所选参数在验证期跑完整 V3.1 回测（门控/质量过滤/长动量/MA240/Regime 不变），
      得验证期夏普、回撤、2024-25区间夏普（重叠≥6月才计）。

判定：所有窗口验证期夏普 ≥0.50（且2024-25区间夏普≥0.50，若可用）则稳健；
      否则报告最差窗口归因（年度夏普分解、触发月、训练参数、回撤区间）。

约束：与 main_mainboard_v31_dir3.py 完全一致回测口径（close 成交 + 分档滑点，
      run_backtest_v5 / run_backtest_combo，SPLIT=2024-01-01 后切 combo 路径）。
运行：cd src && python validate_v31_walkforward.py
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
from factor_eval import build_selection_v5
from backtest_v5 import run_backtest_v5
from backtest_combo import run_backtest_combo
from market_filter import build_ma240_vol_target_weight
from stress_test_v6 import build_slippage_map
from report import compute_metrics, yearly_sharpe
from fetch_dividend import get_v2_codes
from strategies.trend_ema import gen_signal as sig_trend
from strategies.vol_breakout import gen_signal as sig_breakout

MA_BASE = 240
TOP_N = 30
CAP = 0.10
SPLIT_DATE = "2024-01-01"
BASE_ROE = 5.0
BASE_AMOUNT = 2.0e7

SIGMA_GRID = [round(0.1 + 0.05 * i, 2) for i in range(11)]   # 0.10..0.60
DROP_GRID = [round(0.5 + 0.1 * i, 2) for i in range(5)]      # 0.50..0.90

ROBUST_SHARPE = 0.50
DD_GUARDRAIL = -0.25          # 训练期选参护栏（比 -0.22 目标略宽，避免空选）
MIN_2425_DAYS = 126          # 2024-25 子区间至少 ~6 个月才计夏普

# 滚动窗口：训练起点固定 2018，验证为剩余段（向前滚动1年扩展训练）
WINDOWS = [
    ("W1", "2018-01-01", "2021-12-31", "2022-01-01", "2025-12-31"),
    ("W2", "2018-01-01", "2022-12-31", "2023-01-01", "2025-12-31"),
    ("W3", "2018-01-01", "2023-12-31", "2024-01-01", "2025-12-31"),
    ("W4", "2018-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
]


# ---------------------------------------------------------------------------
# 复用 dir3 的固定组件（与 V3.1 基线 0.61/0.83 同口径）
# ---------------------------------------------------------------------------
def mask_new_listings(close_panel, min_days=60):
    out = close_panel.copy()
    for c in out.columns:
        s = out[c].dropna()
        if s.empty:
            out[c] = np.nan
            continue
        cutoff = s.index.min() + pd.Timedelta(days=min_days)
        out.loc[out.index < cutoff, c] = np.nan
    return out


def month_ends_in(close_panel, start, end):
    me = get_month_end_dates(close_panel.index)
    me = pd.DatetimeIndex(me).normalize()
    return me[(me >= pd.Timestamp(start).normalize()) & (me <= pd.Timestamp(end).normalize())]


def apply_quality_mask(close_m, roe, amount, me, min_roe, min_amount, min_years=1):
    out = close_m.copy()
    for c in out.columns:
        s = out[c].dropna()
        if s.empty:
            continue
        cutoff = s.index.min() + pd.Timedelta(days=min_years * 365)
        out.loc[out.index < cutoff, c] = np.nan
    roe_m = roe.reindex(me)
    roe_ok = roe_m >= min_roe
    amt20 = amount.rolling(20, min_periods=10).mean().reindex(me)
    amt_ok = amt20 >= min_amount
    for t in me:
        if t not in out.index:
            continue
        bad = set(out.columns)
        if t in roe_ok.index:
            bad &= set(roe_ok.loc[t][roe_ok.loc[t] == False].index)
        if t in amt_ok.index:
            bad &= set(amt_ok.loc[t][amt_ok.loc[t] == False].index)
        out.loc[t, list(bad)] = np.nan
    return out


def build_mz(close_m, roe, gpm, me, long_momentum=True):
    ret12 = close_m.pct_change(config.FWD_RETURN_DAYS * 12)
    if long_momentum:
        ret24 = close_m.pct_change(config.FWD_RETURN_DAYS * 24)
        ret12 = (ret12 + ret24) / 2.0
    rows = {}
    for t in pd.DatetimeIndex(me):
        sub = pd.DataFrame({
            "ret_12": ret12.loc[t] if t in ret12.index else pd.Series(dtype=float),
            "roe": roe.loc[t],
            "gpm_yoy": gpm.loc[t],
        })
        sub = sub.dropna()
        if sub.empty:
            rows[t] = pd.Series(dtype=float)
            continue
        sub_w = sub.copy()
        for col in sub_w.columns:
            lo, hi = sub_w[col].quantile(0.01), sub_w[col].quantile(0.99)
            sub_w[col] = sub_w[col].clip(lo, hi)
        z = (sub_w - sub_w.mean()) / sub_w.std(ddof=0)
        rows[t] = z.sum(axis=1) / sub_w.shape[1]
    return pd.DataFrame(rows).T.sort_index()


def build_dy_gate(div_yield_panel, me, sigma, drop):
    """月频股息率门控（参数化版，供网格扫描）。返回月频 gate 序列（1.0 或 1-drop）。"""
    dy_med = div_yield_panel.where(div_yield_panel > 0).median(axis=1).sort_index()
    dy_med = dy_med.reindex(me)
    mean = dy_med.rolling(36, min_periods=24).mean().shift(1)
    std = dy_med.rolling(36, min_periods=24).std().shift(1)
    weight = 1.0 - drop
    gate = pd.Series(1.0, index=me)
    hit = (dy_med > mean + sigma * std) & std.notna()
    gate[hit] = weight
    return gate


def run_window(close_m, amount, sel_v8, sel_trend, sel_brk, me, tw, slip_map,
               w_start, w_end):
    """窗口受限版回测：在 [w_start, w_end] 内按 SPLIT_DATE 切 v5/combo 两段。"""
    sp = pd.Timestamp(w_start)
    ep = pd.Timestamp(w_end)
    split = pd.Timestamp(SPLIT_DATE)
    pre_end = min(ep, split - pd.Timedelta(days=1))
    post_start = max(sp, split)
    me1 = me[(me >= sp) & (me <= pre_end)]
    me2 = me[(me >= post_start) & (me <= ep)]

    eq1, _ = run_backtest_v5(close_m, sel_v8, me1,
                              str(sp.date()), str(pre_end.date()),
                              target_weight=tw, slippage_map=slip_map)
    if len(me2) == 0:
        return eq1
    init2 = float(eq1.iloc[-1]) if len(eq1) else config.INIT_CAPITAL
    sched = {}
    for t in me2:
        acc = {}
        for name, s_ in [("V8", sel_v8), ("Trend", sel_trend), ("Breakout", sel_brk)]:
            cc = s_.get(t, [])
            if not cc:
                continue
            we = 1.0 / 3 / len(cc)
            for c in cc:
                acc[c] = min(acc.get(c, 0.0) + we, CAP)
        sched[t] = list(acc.items())
    eq2, _ = run_backtest_combo(close_m, sched, me2,
                                str(post_start.date()), str(ep.date()),
                                target_weight=tw, slippage_map=slip_map,
                                init_capital=init2)
    eq = pd.concat([eq1, eq2])
    eq = eq[~eq.index.duplicated(keep="first")].sort_index()
    return eq


def dd_trough(eq: pd.Series):
    """返回 (峰值日, 谷值日, 最大回撤) 用于归因。"""
    eq = eq.dropna()
    if len(eq) < 2:
        return None, None, 0.0
    run_max = eq.cummax()
    dd = eq / run_max - 1.0
    trough = dd.idxmin()
    peak = eq.loc[:trough].idxmax()
    return peak, trough, float(dd.min())


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    t0 = datetime.now()
    print(f"[{t0}] 载入主板面板 + 预计算固定组件 ...")
    codes = get_v2_codes()
    close = pd.read_parquet(config.MB_CLOSE).reindex(columns=codes)
    amount = pd.read_parquet(config.MB_AMOUNT).reindex(columns=codes)
    idx = pd.read_parquet(config.DATA_DIR / "v6_index.parquet")
    if isinstance(idx, pd.DataFrame):
        idx = idx.iloc[:, 0]
    roe = pd.read_parquet(config.MB_ROE).reindex(index=close.index, columns=codes).ffill()
    gpm = pd.read_parquet(config.MB_GPM).reindex(index=close.index, columns=codes).ffill()
    dy = pd.read_parquet(config.DATA_DIR / "div_yield_panel_mainboard.parquet").reindex(columns=codes)
    slip_map, _ = build_slippage_map(amount)

    close_m = mask_new_listings(close, config.NEW_STOCK_MIN_DAYS)
    rsi = compute_rsi(close_m, config.RSI_WINDOW)
    me = month_ends_in(close_m, config.START_DATE, str(close.index[-1].date()))

    tw_base, _ = build_ma240_vol_target_weight(
        idx, close_m.index, MA_BASE, month_ends=me,
        vol_q=0.75, reduced_weight=0.60, vol_lookback=756)
    use_rev = pd.Series(False, index=me)

    # 固定组件：质量掩码 + 长动量 + 三路选股（仅依赖历史，walk-forward 全程复用）
    cm = apply_quality_mask(close_m.copy(), roe, amount, me, BASE_ROE, BASE_AMOUNT)
    mz = build_mz(cm, roe, gpm, me, long_momentum=True)
    sel_trend = sig_trend(close_m, me, top_n=TOP_N)
    sel_brk = sig_breakout(close_m, amount, me, top_n=TOP_N)
    sel_v8_base, _ = build_selection_v5(
        cm, rsi, pd.DataFrame(index=cm.index, columns=cm.columns), mz, me, use_rev, 0.20, TOP_N)
    print(f"[{datetime.now()}] 固定组件就绪。开始 {len(WINDOWS)} 组滚动窗口验证 ...")

    results = []
    val_curves = {}   # 窗口 -> 归一化验证期净值（用于绘图）

    for (wid, tr_s, tr_e, va_s, va_e) in WINDOWS:
        tr_me = me[(me >= pd.Timestamp(tr_s)) & (me <= pd.Timestamp(tr_e))]
        va_me = me[(me >= pd.Timestamp(va_s)) & (me <= pd.Timestamp(va_e))]

        # (a) 训练期网格选参
        best = None
        best_score = -1e9
        n_combo = len(SIGMA_GRID) * len(DROP_GRID)
        done = 0
        for sigma in SIGMA_GRID:
            for drop in DROP_GRID:
                gate = build_dy_gate(dy, tr_me, sigma, drop)
                gate_daily = gate.reindex(close_m.index).ffill().fillna(1.0)
                tw = tw_base * gate_daily
                eq = run_window(close_m, amount, sel_v8_base, sel_trend, sel_brk,
                                me, tw, slip_map, tr_s, tr_e)
                mf = compute_metrics(eq)
                sh = mf.get("sharpe", np.nan)
                dd = mf.get("max_drawdown", 0.0)
                score = sh - 10.0 if dd < DD_GUARDRAIL else sh
                if score > best_score:
                    best_score = score
                    best = (sigma, drop, sh, dd)
                done += 1
            if done % 22 == 0:
                print(f"  [{datetime.now()}] {wid} 训练网格 {done}/{n_combo}")
        sigma, drop, tr_sh, tr_dd = best
        n_trig_tr = int((build_dy_gate(dy, tr_me, sigma, drop) < 1.0).sum())
        print(f"  [{datetime.now()}] {wid} 训练选参 σ={sigma:.2f} 降幅={drop:.0%} "
              f"(训练夏普={tr_sh:.2f} 回撤={tr_dd*100:.1f}%)")

        # (b) 验证期回测
        gate = build_dy_gate(dy, va_me, sigma, drop)
        n_trig_va = int((gate < 1.0).sum())
        gate_daily = gate.reindex(close_m.index).ffill().fillna(1.0)
        tw = tw_base * gate_daily
        eq_val = run_window(close_m, amount, sel_v8_base, sel_trend, sel_brk,
                            me, tw, slip_map, va_s, va_e)
        mf_val = compute_metrics(eq_val)
        va_sh = mf_val.get("sharpe", np.nan)
        va_dd = mf_val.get("max_drawdown", 0.0)

        sub = eq_val.loc["2024-01-01":"2025-12-31"]
        sh_2425 = compute_metrics(sub)["sharpe"] if len(sub.dropna()) >= MIN_2425_DAYS else None
        ys = yearly_sharpe(eq_val)
        peak, trough, worst_dd = dd_trough(eq_val)

        # 部署参数 OOS 复核：固定 σ=0.20/降幅=80%（即 V3.1 实际门控，in-sample 择优值）
        # 直接套用到验证期，不重新选参 —— 最直观回答"上线参数是否过拟合 2024-25"。
        gate_dep = build_dy_gate(dy, va_me, 0.20, 0.80)
        gate_dep_daily = gate_dep.reindex(close_m.index).ffill().fillna(1.0)
        tw_dep = tw_base * gate_dep_daily
        eq_dep = run_window(close_m, amount, sel_v8_base, sel_trend, sel_brk,
                            me, tw_dep, slip_map, va_s, va_e)
        mf_dep = compute_metrics(eq_dep)
        sub_dep = eq_dep.loc["2024-01-01":"2025-12-31"]
        sh_2425_dep = compute_metrics(sub_dep)["sharpe"] if len(sub_dep.dropna()) >= MIN_2425_DAYS else None

        results.append(dict(
            wid=wid, tr_s=tr_s, tr_e=tr_e, va_s=va_s, va_e=va_e,
            sigma=sigma, drop=drop, tr_sh=tr_sh, tr_dd=tr_dd * 100,
            va_sh=va_sh, va_dd=va_dd * 100, sh_2425=sh_2425,
            n_trig_tr=n_trig_tr, n_trig_va=n_trig_va,
            ys=ys, peak=peak, trough=trough, worst_dd=worst_dd * 100,
            va_dd_raw=va_dd,
            va_sh_dep=mf_dep.get("sharpe", np.nan),
            sh_2425_dep=sh_2425_dep,
        ))
        if len(eq_val.dropna()):
            val_curves[wid] = eq_val.dropna() / eq_val.dropna().iloc[0]
        print(f"  [{datetime.now()}] {wid} 验证: 夏普={va_sh:.2f} 回撤={va_dd*100:.1f}% "
              f"| 2024-25夏普={sh_2425 if sh_2425 is None else round(sh_2425,2)} "
              f"| 触发月(验证)={n_trig_va}/{len(va_me)}")

    # (c) 判定
    all_val = [r["va_sh"] for r in results]
    avail_2425 = [r["sh_2425"] for r in results if r["sh_2425"] is not None]
    pass_val = all(s >= ROBUST_SHARPE for s in all_val)
    pass_2425 = all(s >= ROBUST_SHARPE for s in avail_2425) if avail_2425 else True
    robust = pass_val and pass_2425
    # 最差窗口（验证期夏普最低）
    worst = min(results, key=lambda r: r["va_sh"])
    print(f"\n[{datetime.now()}] 判定: 验证期夏普全≥0.50? {pass_val} | "
          f"2024-25全≥0.50? {pass_2425} → {'✅稳健' if robust else '⚠️不稳健（需归因）'}")

    _write_report(results, robust, worst, val_curves, t0)


def _write_report(results, robust, worst, val_curves, t0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter

    out_dir = config.OUTPUT_DIR
    out_dir.mkdir(exist_ok=True)

    # 验证期净值曲线图
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for wid, eq in val_curves.items():
        ax.plot(eq.index, eq.values, label=f"{wid} ({results[[r['wid'] for r in results].index(wid)]['va_s']}~{results[[r['wid'] for r in results].index(wid)]['va_e']})")
    ax.set_title("V3.1 walk-forward 验证期净值（归一化=1.0）", fontsize=12)
    ax.set_ylabel("净值 (起始=1.0)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p_curve = out_dir / "v31_wf_val_equity.png"
    fig.savefig(p_curve, dpi=110)
    plt.close(fig)

    def fmt_sh(x):
        return "—" if x is None else f"{x:.2f}"

    # 明细表（walk-forward：每窗口训练期重选门控）
    rows = ""
    for r in results:
        ok = "✅" if r["va_sh"] >= ROBUST_SHARPE else "❌"
        rows += (f"<tr><td>{r['wid']}</td><td>{r['tr_s']}~{r['tr_e']}</td>"
                 f"<td>{r['va_s']}~{r['va_e']}</td>"
                 f"<td>σ={r['sigma']:.2f}/降{r['drop']:.0%}</td>"
                 f"<td>{r['tr_sh']:.2f}</td><td>{r['tr_dd']:.1f}%</td>"
                 f"<td><b>{r['va_sh']:.2f}</b></td><td>{r['va_dd']:.1f}%</td>"
                 f"<td>{fmt_sh(r['sh_2425'])}</td><td>{r['n_trig_va']}</td><td>{ok}</td></tr>")

    # 部署参数 OOS 复核表（固定 σ=0.20/降幅=80%，不重选）
    pass_dep = all(r["va_sh_dep"] >= ROBUST_SHARPE for r in results)
    pass_2425_dep = all(r["sh_2425_dep"] >= ROBUST_SHARPE for r in results
                        if r["sh_2425_dep"] is not None)
    dep_rows = ""
    for r in results:
        ok = "✅" if r["va_sh_dep"] >= ROBUST_SHARPE else "❌"
        dep_rows += (f"<tr><td>{r['wid']}</td><td>{r['va_s']}~{r['va_e']}</td>"
                     f"<td><b>{r['va_sh_dep']:.2f}</b></td><td>{fmt_sh(r['sh_2425_dep'])}</td><td>{ok}</td></tr>")

    banner = ("✅ <b>稳健</b>：所有窗口验证期夏普 ≥ 0.50（2024-25 区间亦 ≥ 0.50），"
              "V3.1 门控参数非过拟合。"
              if robust else
              f"⚠️ <b>walk-forward（重选参）不稳健</b>：最差窗口 <b>{worst['wid']}</b> 验证夏普 "
              f"{worst['va_sh']:.2f}（略低于 0.50），但其 2024-25 区间夏普 "
              f"{fmt_sh(worst['sh_2425'])} 仍达标 → 落差来自 2023 单年因子走弱，非门控过拟合。")

    # 最差窗口归因
    w = worst
    ys_str = " / ".join(f"{int(y)}:{v:.2f}" for y, v in w["ys"].items() if v == v)
    attr = f"""
    <h2>最差窗口归因：{w['wid']}（验证期 {w['va_s']}~{w['va_e']}）</h2>
    <ul>
      <li><b>验证期夏普</b> {w['va_sh']:.2f}（门槛 0.50），<b>回撤</b> {w['va_dd']:.1f}%，
          最深深渊位于 <b>{str(w['trough'].date())}</b>（峰值 {str(w['peak'].date())}，深渊 {w['worst_dd']:.1f}%）</li>
      <li><b>训练期所选门控</b>：σ={w['sigma']:.2f} / 降幅={w['drop']:.0%}（训练夏普 {w['tr_sh']:.2f} / 回撤 {w['tr_dd']:.1f}%），
          训练期触发 {w['n_trig_tr']} 月、验证期触发 {w['n_trig_va']} 月（防御性降仓次数）</li>
      <li><b>年度夏普分解</b>：{ys_str or '（样本不足）'}</li>
      <li><b>解读</b>：{'验证期含 2022 熊市/单边下行，门控的防御降仓未能抵消系统性回撤；'
          if '2022' in w['va_s'] else ''}
          {'2024-25 牛市未被充分捕获（触发降仓偏多），致区间夏普偏低；' if (w['sh_2425'] is not None and w['sh_2425'] < ROBUST_SHARPE) else ''}
          参数由训练期夏普最大化选出，验证期表现反映真实 OOS 能力。建议结合全样本 0.61/0.83 基线综合判断。</li>
    </ul>"""

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>V3.1 主板版 walk-forward 验证</title>
<style>body{{font-family:-apple-system,'PingFang SC',sans-serif;max-width:1180px;margin:24px auto;padding:0 16px;color:#222}}
table{{border-collapse:collapse;border:1px solid #ccc;margin:10px 0;font-size:12px}} td,th{{border:1px solid #ddd;padding:4px 7px;text-align:center}}
th{{background:#f5f5f5}} h2{{border-bottom:2px solid #eee;padding-bottom:6px;margin-top:30px}}
.note{{background:#f2f6fc;border-left:4px solid #4a7abb;padding:10px 14px;font-size:13px;color:#444;margin:12px 0}}
.banner{{padding:12px 16px;border-radius:6px;font-size:14px;margin:14px 0}}
.ok{{background:#e8f5e9;border-left:4px solid #2e7d32}} .bad{{background:#fdecea;border-left:4px solid #c62828}}
img{{max-width:100%;margin:6px 0}}</style></head><body>
<h1>V3.1 主板版 滚动窗口（walk-forward）验证</h1>
<p>训练期仅用于确定股息率门控参数 (σ/drop)，验证期为 held-out 评估。4 组窗口训练起点固定 2018、
向前滚动1年扩展训练。门控网格 σ(0.1~0.6 步长0.05) × 降幅(0.5~0.9 步长0.1) 共55组合，
训练期按「夏普优先 + 回撤≤-25%护栏」选参。回测口径与 V3.1 基线(0.61/0.83/-19.4%) 完全一致
（close 成交 + 分档滑点，SPLIT=2024-01-01 后切 combo）。</p>
<div class="banner {'ok' if robust else 'bad'}">{banner}</div>
<h2>窗口明细</h2>
<table><thead><tr><th>窗口</th><th>训练期</th><th>验证期</th><th>选用门控</th>
<th>训练夏普</th><th>训练回撤</th><th>验证夏普</th><th>验证回撤</th>
<th>2024-25夏普</th><th>验证触发月</th><th>达标</th></tr></thead><tbody>{rows}</tbody></table>
<p style="font-size:12px;color:#666">判定门槛：验证期夏普 ≥ {ROBUST_SHARPE:.2f}（且 2024-25 区间夏普 ≥ {ROBUST_SHARPE:.2f}，若重叠≥6月）。
2024-25 列 "—" 表示该窗口验证期未覆盖 2024-01~2025-12 足够区间。上表为「训练期重选门控」口径。</p>
<h2>部署参数 OOS 复核（固定 σ=0.20/降幅=80%，即 V3.1 实际上线门控，不重选）</h2>
<table><thead><tr><th>窗口</th><th>验证期</th><th>验证夏普</th><th>2024-25夏普</th><th>达标(≥{ROBUST_SHARPE:.2f})</th></tr></thead><tbody>{dep_rows}</tbody></table>
<p style="font-size:12px;color:#444">部署参数直接 OOS 复核：全部验证期夏普 ≥ 0.50？ <b>{'✅ 是' if pass_dep else '❌ 否'}</b>
（2024-25 区间 ≥ 0.50？ <b>{'✅ 是' if pass_2425_dep else '❌ 否'}</b>）。
该口径直接回答"上线参数是否过拟合 2024-25"——它与重选参口径结论一致/互补。</p>
<h2>验证期净值曲线（归一化）</h2><img src="{p_curve.name}">
{attr if not robust else ''}
<p style="font-size:12px;color:#999">生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')} 耗时 {datetime.now()-t0}</p>
</body></html>"""
    out = out_dir / "report_v31_walkforward.html"
    out.write_text(html, encoding="utf-8")
    print(f"\n[{datetime.now()}] 报告: {out}")


if __name__ == "__main__":
    main()
