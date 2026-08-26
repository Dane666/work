# -*- coding: utf-8 -*-
"""
main_mainboard_v31_dir1.py — 主板版 V3.1 方向1：质量过滤器网格扫描
====================================================================
扫描 ROE 阈值(5%~15% 步长1%) × 成交额阈值(1000万~5000万 步长500万) 共 99 组合，
每组合独立回测，输出：
  - 全期夏普 / 2024-25 夏普 / 全期回撤 三类热力图
  - 最优组合（按 2024-25 夏普优先、回撤≤-20% 约束、全期夏普次之）
  - 相对 V3-3 基线(0.46/0.67/-19.3%) 的边际贡献

实验设计（隔离质量过滤效应）：
  - sel_trend / sel_breakout（E组合另外两个腿）在【基线质量掩码】上算一次复用，
    不随质量阈值变化 → 单独衡量"质量过滤阈值"对 V8 腿 + 组合净值的边际影响。
  - 默认组合 (ROE=5%, 额=2000万) 须精确复现 V3-3 基线，作为校验。

约束：月频不变、MA240门控不变(用 tw_base)、强制质量模式(use_rev 全 False)、
      股息率门控不变(用基线 dy_gate)。仅质量过滤器阈值变化。
运行：cd src && python main_mainboard_v31_dir1.py
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
from report import compute_metrics
from fetch_dividend import get_v2_codes
from strategies.trend_ema import gen_signal as sig_trend
from strategies.vol_breakout import gen_signal as sig_breakout

MA_BASE = 240
TOP_N = 30
CAP = 0.10
SPLIT_DATE = "2024-01-01"

# 基线阈值（V3-3 默认）→ 用于算 sel_trend/sel_brk 与校验
BASE_ROE = 5.0
BASE_AMOUNT = 2.0e7

# ---- 网格定义 ----
ROE_GRID = [float(r) for r in range(5, 16, 1)]            # 5%..15% step 1%
AMT_GRID = [float(a) * 1e6 for a in range(10, 55, 5)]     # 1000万..5000万 step 500万

# 目标（用户绝对阈值）
TARGET_FULL = 0.58
TARGET_NEW = 0.75
TARGET_DD = -0.20


# ---------------------------------------------------------------------------
# 复用 V3 函数（与 main_mainboard_v3.py 一致，参数化质量阈值）
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


def build_dy_gate(div_yield_panel, me):
    dy_med = div_yield_panel.where(div_yield_panel > 0).median(axis=1).sort_index()
    dy_med = dy_med.reindex(me)
    mean = dy_med.rolling(36, min_periods=24).mean().shift(1)
    std = dy_med.rolling(36, min_periods=24).std().shift(1)
    gate = pd.Series(1.0, index=me)
    hit = (dy_med > mean + 0.3 * std) & std.notna()
    gate[hit] = 0.70
    return gate


def run_full_precomp(close_m, amount, sel_v8, sel_trend, sel_brk, me, tw, slip_map):
    """分区间回测（与 V3 run_full 等价），sel_trend/sel_brk 由外部传入（算一次复用）。"""
    me1 = me[me < pd.Timestamp(SPLIT_DATE)]
    me2 = me[me >= pd.Timestamp(SPLIT_DATE)]
    eq1, _ = run_backtest_v5(close_m, sel_v8, me1, config.START_DATE, "2023-12-31",
                             target_weight=tw, slippage_map=slip_map)
    init2 = float(eq1.iloc[-1])
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
    eq2, _ = run_backtest_combo(close_m, sched, me2, SPLIT_DATE,
                                str(close_m.index[-1].date()),
                                target_weight=tw, slippage_map=slip_map, init_capital=init2)
    eq = pd.concat([eq1, eq2])
    eq = eq[~eq.index.duplicated(keep="first")].sort_index()
    return eq


# ---------------------------------------------------------------------------
def main():
    t0 = datetime.now()
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
    gate = build_dy_gate(dy, me)
    gate_daily = gate.reindex(close_m.index).ffill().fillna(1.0)
    tw = tw_base * gate_daily

    use_rev = pd.Series(False, index=me)

    # 基线质量掩码 → 算一次 sel_trend/sel_brk 复用
    cm_base = apply_quality_mask(close_m.copy(), roe, amount, me, BASE_ROE, BASE_AMOUNT)
    sel_trend = sig_trend(close_m, me, top_n=TOP_N)
    sel_brk = sig_breakout(close_m, amount, me, top_n=TOP_N)
    print(f"[{datetime.now()}] 数据就绪：{len(codes)} 只，月频数 {len(me)}，"
          f"sel_trend/sel_brk 已算一次复用")

    # 校验：默认组合须复现 V3-3 基线
    cm0 = cm_base
    mz0 = build_mz(cm0, roe, gpm, me, long_momentum=True)
    sel0, _ = build_selection_v5(cm0, rsi, pd.DataFrame(index=cm0.index, columns=cm0.columns),
                                 mz0, me, use_rev, 0.20, TOP_N)
    eq0 = run_full_precomp(cm0, amount, sel0, sel_trend, sel_brk, me, tw, slip_map)
    mf0 = compute_metrics(eq0)
    mn0 = compute_metrics(eq0.loc["2024-01-01":])
    print(f"  [校验] 默认(5%,2000万) 全期={mf0['sharpe']:.2f} 回撤={mf0['max_drawdown']*100:.1f}% "
          f"| 2024-25={mn0['sharpe']:.2f}  (期望 0.46/0.67/-19.3%)")

    # 网格扫描
    results = {}
    n_total = len(ROE_GRID) * len(AMT_GRID)
    done = 0
    for roe_thr in ROE_GRID:
        for amt_thr in AMT_GRID:
            cm = apply_quality_mask(close_m.copy(), roe, amount, me, roe_thr, amt_thr)
            mz = build_mz(cm, roe, gpm, me, long_momentum=True)
            sel_v8, _ = build_selection_v5(
                cm, rsi, pd.DataFrame(index=cm.index, columns=cm.columns),
                mz, me, use_rev, 0.20, TOP_N)
            eq = run_full_precomp(cm, amount, sel_v8, sel_trend, sel_brk, me, tw, slip_map)
            mf = compute_metrics(eq)
            mn = compute_metrics(eq.loc["2024-01-01":])
            results[(roe_thr, amt_thr)] = (
                mf["sharpe"], mf["max_drawdown"] * 100,
                mn["sharpe"], mn["max_drawdown"] * 100)
            done += 1
            if done % 11 == 0:
                print(f"  [{datetime.now()}] 进度 {done}/{n_total}")

    # 汇总为 DataFrame（行=ROE，列=amt）
    roe_l = ROE_GRID
    amt_l = AMT_GRID
    full_sh = pd.DataFrame(index=[f"{r:.0f}%" for r in roe_l],
                           columns=[f"{a/1e6:.0f}M" for a in amt_l])
    new_sh = full_sh.copy()
    dd = full_sh.copy()
    for (r, a), (fs, fdd, ns, ndd) in results.items():
        full_sh.loc[f"{r:.0f}%", f"{a/1e6:.0f}M"] = round(fs, 3)
        new_sh.loc[f"{r:.0f}%", f"{a/1e6:.0f}M"] = round(ns, 3)
        dd.loc[f"{r:.0f}%", f"{a/1e6:.0f}M"] = round(fdd, 1)

    # 最优组合：优先达标(全期≥0.58 & 2024-25≥0.75 & 回撤≤-20%)，否则按 2024-25 夏普排序、回撤≤-20% 约束
    best = None
    best_score = -1e9
    for (r, a), (fs, fdd, ns, ndd) in results.items():
        passed = (fs >= TARGET_FULL and ns >= TARGET_NEW and fdd >= TARGET_DD * 100)
        # 评分：达标优先，其次 2024-25 夏普（约束回撤≤-20%）
        if fdd < TARGET_DD * 100:
            continue
        score = ns + (10 if passed else 0)
        if score > best_score:
            best_score = score
            best = (r, a, fs, fdd, ns, ndd, passed)

    print(f"\n[{datetime.now()}] 网格完成 {n_total} 组合，耗时 {datetime.now()-t0}")
    if best:
        r, a, fs, fdd, ns, ndd, passed = best
        print(f">>> 最优(ROE={r:.0f}%, 额={a/1e6:.0f}M): 全期={fs:.2f} 回撤={fdd:.1f}% | "
              f"2024-25={ns:.2f} 回撤={ndd:.1f}%  {'✅达标' if passed else '未达标'}")
        print(f"    基线(0.46/0.67/-19.3%) 边际: 全期 {fs-0.46:+.2f} / 2024-25 {ns-0.67:+.2f} / "
              f"回撤 {fdd+19.3:+.1f}pp")
    else:
        print(">>> 无组合满足回撤≤-20%，取 2024-25 夏普最高者")

    _write_report(full_sh, new_sh, dd, results, best, mf0, mn0, t0)


def _write_report(full_sh, new_sh, dd, results, best, mf0, mn0, t0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    out_dir = config.OUTPUT_DIR
    out_dir.mkdir(exist_ok=True)

    def heat(df, title, fname, fmt="{:.2f}", cmap="RdYlGn"):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        data = df.astype(float).values
        im = ax.imshow(data, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(df.columns)))
        ax.set_xticklabels(df.columns)
        ax.set_yticks(range(len(df.index)))
        ax.set_yticklabels(df.index)
        ax.set_xlabel("20日成交额阈值")
        ax.set_ylabel("ROE阈值")
        ax.set_title(title, fontsize=12)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center",
                        fontsize=7, color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        p = out_dir / fname
        fig.savefig(p, dpi=110)
        plt.close(fig)
        return p

    p1 = heat(full_sh, "方向1 全期夏普 (ROE x 成交额)", "v31_dir1_heat_full.png")
    p2 = heat(new_sh, "方向1 2024-25夏普 (ROE x 成交额)", "v31_dir1_heat_new.png")
    p3 = heat(dd, "方向1 全期回撤% (ROE x 成交额)", "v31_dir1_heat_dd.png", fmt="{:.1f}", cmap="RdYlGn_r")

    # 表格 HTML
    def tbl(df):
        return df.to_html(border=1, float_format=lambda x: f"{x:.2f}")

    best_html = ""
    if best:
        r, a, fs, fdd, ns, ndd, passed = best
        best_html = (f"<tr><td>ROE={r:.0f}%, 额={a/1e6:.0f}M</td>"
                     f"<td>{fs:.2f}</td><td>{fdd:.1f}%</td><td>{ns:.2f}</td><td>{ndd:.1f}%</td>"
                     f"<td>{'✅达标' if passed else '未达标'}</td>"
                     f"<td>全期{fs-0.46:+.2f}/2024-25{ns-0.67:+.2f}/回撤{fdd+19.3:+.1f}pp</td></tr>")

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>V3.1 方向1 质量过滤器网格扫描</title>
<style>body{{font-family:-apple-system,'PingFang SC',sans-serif;max-width:1200px;margin:24px auto;padding:0 16px;color:#222}}
table{{border-collapse:collapse;border:1px solid #ccc;margin:10px 0;font-size:12px}} td,th{{border:1px solid #ddd;padding:4px 7px;text-align:center}}
th{{background:#f5f5f5}} h2{{border-bottom:2px solid #eee;padding-bottom:6px;margin-top:30px}}
.note{{background:#f2f6fc;border-left:4px solid #4a7abb;padding:10px 14px;font-size:13px;color:#444;margin:12px 0}}
.verdict{{background:#fff7e6;border:1px solid #f0c36d;border-radius:6px;padding:12px 16px;margin:16px 0}}
img{{max-width:100%;margin:6px 0}}</style></head><body>
<h1>V3.1 方向1：质量过滤器网格扫描</h1>
<p>网格：ROE 5%~15%(步长1%) × 成交额 1000万~5000万(步长500万) = 99 组合。
实验设计：sel_trend/sel_breakout 在基线质量掩码上算一次复用，隔离质量过滤边际效应。
约束：月频/MA240门控/强制质量模式/股息率门控 均不变。</p>
<div class="note"><b>V3-3 基线（默认 5%/2000万）：</b>全期 {mf0['sharpe']:.2f} / 回撤 {mf0['max_drawdown']*100:.1f}% / 2024-25 {mn0['sharpe']:.2f}
（代码可复现基线；用户背景所述 0.54/0.65 为更早快照）。
目标：全期≥0.58 / 2024-25≥0.75 / 回撤≤-20%。</div>
<h2>最优组合（回撤≤-20% 约束下按 2024-25 夏普优先）</h2>
<table><thead><tr><th>组合</th><th>全期夏普</th><th>全期回撤</th><th>2024-25夏普</th><th>2024-25回撤</th><th>达标</th><th>vs基线边际</th></tr></thead>
<tbody>{best_html}</tbody></table>
<h2>热力图</h2>
<h3>全期夏普</h3><img src="{p1.name}"><h3>2024-25 夏普</h3><img src="{p2.name}"><h3>全期回撤 %</h3><img src="{p3.name}">
<h2>全期夏普矩阵</h2>{tbl(full_sh)}<h2>2024-25 夏普矩阵</h2>{tbl(new_sh)}<h2>全期回撤% 矩阵</h2>{tbl(dd)}
<p style="font-size:12px;color:#999">生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')} 耗时 {datetime.now()-t0}</p>
</body></html>"""
    out = out_dir / "report_mainboard_v31_dir1.html"
    out.write_text(html, encoding="utf-8")
    print(f"\n[{datetime.now()}] 报告: {out}")


if __name__ == "__main__":
    main()
