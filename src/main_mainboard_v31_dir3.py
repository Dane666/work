# -*- coding: utf-8 -*-
"""
main_mainboard_v31_dir3.py — 主板版 V3.1 方向3：股息率门控参数微调
====================================================================
扫描 触发灵敏度 σ(0.1~0.6 步长0.05) × 触发降幅(drop 50%~90% 步长10%) 共 55 组合。
门控逻辑：每月全市场股息率中位数(排除0) > 历史滚动36月均值 + σ×std → 仓位上限降至 (1-drop)。
每组合独立回测，输出热力图 + 最优组合 + 相对 V3-3 基线边际贡献。

约束：月频/MA240门控/强制质量模式/质量过滤器阈值(基线5%/2000万) 均不变。仅股息率门控 σ 与降幅变化。
运行：cd src && python main_mainboard_v31_dir3.py
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
BASE_ROE = 5.0
BASE_AMOUNT = 2.0e7

SIGMA_GRID = [round(0.1 + 0.05 * i, 2) for i in range(11)]   # 0.10..0.60
DROP_GRID = [round(0.5 + 0.1 * i, 2) for i in range(5)]      # 0.50..0.90 (降幅→仓位=1-drop)

TARGET_FULL = 0.58
TARGET_NEW = 0.75
TARGET_DD = -0.20


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
    dy_med = div_yield_panel.where(div_yield_panel > 0).median(axis=1).sort_index()
    dy_med = dy_med.reindex(me)
    mean = dy_med.rolling(36, min_periods=24).mean().shift(1)
    std = dy_med.rolling(36, min_periods=24).std().shift(1)
    weight = 1.0 - drop
    gate = pd.Series(1.0, index=me)
    hit = (dy_med > mean + sigma * std) & std.notna()
    gate[hit] = weight
    return gate


def run_full_precomp(close_m, amount, sel_v8, sel_trend, sel_brk, me, tw, slip_map):
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


def main():
    t0 = datetime.now()
    codes = get_v2_codes()
    close = pd.read_parquet(config.MB_CLOSE).reindex(columns=codes)
    amount = pd.read_parquet(config.MB_AMOUNT).reindex(columns=codes)
    idx = pd.read_parquet(config.DATA_DIR / "v6_index.parquet")
    if isinstance(idx, pd.DataFrame):
        idx = idx.iloc[:, 0]
    roe = pd.read_parquet(config.MB_ROE).reindex(index=close.index, columns=codes).ffill()
    gpm = pd.read_parquet(config.MB_GPM).reindex(index=close.index, columns=codes if False else codes).ffill()
    dy = pd.read_parquet(config.DATA_DIR / "div_yield_panel_mainboard.parquet").reindex(columns=codes)
    slip_map, _ = build_slippage_map(amount)

    close_m = mask_new_listings(close, config.NEW_STOCK_MIN_DAYS)
    rsi = compute_rsi(close_m, config.RSI_WINDOW)
    me = month_ends_in(close_m, config.START_DATE, str(close.index[-1].date()))

    tw_base, _ = build_ma240_vol_target_weight(
        idx, close_m.index, MA_BASE, month_ends=me,
        vol_q=0.75, reduced_weight=0.60, vol_lookback=756)
    use_rev = pd.Series(False, index=me)

    # 质量掩码固定（基线）+ sel_trend/sel_brk 算一次复用
    cm = apply_quality_mask(close_m.copy(), roe, amount, me, BASE_ROE, BASE_AMOUNT)
    mz = build_mz(cm, roe, gpm, me, long_momentum=True)
    sel_trend = sig_trend(close_m, me, top_n=TOP_N)
    sel_brk = sig_breakout(close_m, amount, me, top_n=TOP_N)
    sel_v8_base, _ = build_selection_v5(
        cm, rsi, pd.DataFrame(index=cm.index, columns=cm.columns), mz, me, use_rev, 0.20, TOP_N)

    results = {}
    n_total = len(SIGMA_GRID) * len(DROP_GRID)
    done = 0
    for sigma in SIGMA_GRID:
        for drop in DROP_GRID:
            gate = build_dy_gate(dy, me, sigma, drop)
            gate_daily = gate.reindex(close_m.index).ffill().fillna(1.0)
            tw = tw_base * gate_daily
            eq = run_full_precomp(cm, amount, sel_v8_base, sel_trend, sel_brk, me, tw, slip_map)
            mf = compute_metrics(eq)
            mn = compute_metrics(eq.loc["2024-01-01":])
            results[(sigma, drop)] = (mf["sharpe"], mf["max_drawdown"] * 100,
                                      mn["sharpe"], mn["max_drawdown"] * 100)
            done += 1
            if done % 11 == 0:
                print(f"  [{datetime.now()}] 进度 {done}/{n_total}")

    sig_l = SIGMA_GRID
    drop_l = DROP_GRID
    full_sh = pd.DataFrame(index=[f"σ={s:.2f}" for s in sig_l],
                           columns=[f"drop={d:.0%}" for d in drop_l])
    new_sh = full_sh.copy()
    dd = full_sh.copy()
    for (s, d), (fs, fdd, ns, ndd) in results.items():
        full_sh.loc[f"σ={s:.2f}", f"drop={d:.0%}"] = round(fs, 3)
        new_sh.loc[f"σ={s:.2f}", f"drop={d:.0%}"] = round(ns, 3)
        dd.loc[f"σ={s:.2f}", f"drop={d:.0%}"] = round(fdd, 1)

    best = None
    best_score = -1e9
    for (s, d), (fs, fdd, ns, ndd) in results.items():
        if fdd < TARGET_DD * 100:
            continue
        passed = (fs >= TARGET_FULL and ns >= TARGET_NEW and fdd >= TARGET_DD * 100)
        score = ns + (10 if passed else 0)
        if score > best_score:
            best_score = score
            best = (s, d, fs, fdd, ns, ndd, passed)

    print(f"\n[{datetime.now()}] 网格完成 {n_total} 组合，耗时 {datetime.now()-t0}")
    if best:
        s, d, fs, fdd, ns, ndd, passed = best
        print(f">>> 最优(σ={s:.2f}, 降幅={d:.0%}): 全期={fs:.2f} 回撤={fdd:.1f}% | "
              f"2024-25={ns:.2f} 回撤={ndd:.1f}%  {'✅达标' if passed else '未达标'}")
        print(f"    基线(0.46/0.67/-19.3%) 边际: 全期 {fs-0.46:+.2f} / 2024-25 {ns-0.67:+.2f} / "
              f"回撤 {fdd+19.3:+.1f}pp")
    else:
        print(">>> 无组合满足回撤≤-20%")

    _write_report(full_sh, new_sh, dd, best, t0)


def _write_report(full_sh, new_sh, dd, best, t0):
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
        ax.set_xlabel("触发降幅 (drop)")
        ax.set_ylabel("触发灵敏度 sigma")
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

    p1 = heat(full_sh, "方向3 全期夏普 (σ x drop)", "v31_dir3_heat_full.png")
    p2 = heat(new_sh, "方向3 2024-25夏普 (σ x drop)", "v31_dir3_heat_new.png")
    p3 = heat(dd, "方向3 全期回撤% (σ x drop)", "v31_dir3_heat_dd.png", fmt="{:.1f}", cmap="RdYlGn_r")

    def tbl(df):
        return df.to_html(border=1, float_format=lambda x: f"{x:.2f}")

    best_html = ""
    if best:
        s, d, fs, fdd, ns, ndd, passed = best
        best_html = (f"<tr><td>σ={s:.2f}, 降幅={d:.0%}</td>"
                     f"<td>{fs:.2f}</td><td>{fdd:.1f}%</td><td>{ns:.2f}</td><td>{ndd:.1f}%</td>"
                     f"<td>{'✅达标' if passed else '未达标'}</td>"
                     f"<td>全期{fs-0.46:+.2f}/2024-25{ns-0.67:+.2f}/回撤{fdd+19.3:+.1f}pp</td></tr>")

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>V3.1 方向3 股息率门控参数扫描</title>
<style>body{{font-family:-apple-system,'PingFang SC',sans-serif;max-width:1200px;margin:24px auto;padding:0 16px;color:#222}}
table{{border-collapse:collapse;border:1px solid #ccc;margin:10px 0;font-size:12px}} td,th{{border:1px solid #ddd;padding:4px 7px;text-align:center}}
th{{background:#f5f5f5}} h2{{border-bottom:2px solid #eee;padding-bottom:6px;margin-top:30px}}
.note{{background:#f2f6fc;border-left:4px solid #4a7abb;padding:10px 14px;font-size:13px;color:#444;margin:12px 0}}
img{{max-width:100%;margin:6px 0}}</style></head><body>
<h1>V3.1 方向3：股息率门控参数微调</h1>
<p>网格：σ 0.10~0.60(步长0.05) × 降幅 50%~90%(步长10%) = 55 组合。门控：
每月全市场股息率中位数(排除0) &gt; 滚动36月均值+σ×std → 仓位降至 (1-drop)。
约束：月频/MA240/强制质量模式/质量过滤器(5%/2000万) 不变。</p>
<div class="note"><b>V3-3 基线：</b>全期 0.46 / 回撤 -19.3% / 2024-25 0.67（可复现代码基线）。
目标：全期≥0.58 / 2024-25≥0.75 / 回撤≤-20%。基线门控 σ=0.3, 降幅=30%(仓位70%)。</div>
<h2>最优组合（回撤≤-20% 约束下按 2024-25 夏普优先）</h2>
<table><thead><tr><th>组合</th><th>全期夏普</th><th>全期回撤</th><th>2024-25夏普</th><th>2024-25回撤</th><th>达标</th><th>vs基线边际</th></tr></thead>
<tbody>{best_html}</tbody></table>
<h2>热力图</h2>
<h3>全期夏普</h3><img src="{p1.name}"><h3>2024-25 夏普</h3><img src="{p2.name}"><h3>全期回撤 %</h3><img src="{p3.name}">
<h2>全期夏普矩阵</h2>{tbl(full_sh)}<h2>2024-25 夏普矩阵</h2>{tbl(new_sh)}<h2>全期回撤% 矩阵</h2>{tbl(dd)}
<p style="font-size:12px;color:#999">生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')} 耗时 {datetime.now()-t0}</p>
</body></html>"""
    out = out_dir / "report_mainboard_v31_dir3.html"
    out.write_text(html, encoding="utf-8")
    print(f"\n[{datetime.now()}] 报告: {out}")


if __name__ == "__main__":
    main()
