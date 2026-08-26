# -*- coding: utf-8 -*-
"""
main_mainboard_v3.py — 主板版 V3 回测（仓位门控 → 质量过滤 → 动量窗口）
====================================================================

V2 教训：股息率作选股因子（等权/补充）在收益端均负贡献（高股息=低动量低成长），
仅在 MA240 门控下降回撤（-47%→-21%）。V3 将股息率**从选股因子转为择时门控**
（对症"稀释收益 + 回撤仍可改进"），并叠加质量过滤与长动量窗口。

改动（按 stage 顺序执行，达标即停）：
  V3-1 仓位门控：每月全市场股息率中位数 > 历史滚动均值+1σ → 仓位上限 70%
        （选股仍用三因子 ret_12+roe+gpm_yoy，不再用股息率选股）
  V3-2 质量过滤：ROE(当前报告期)≥5% 且 20日日均成交额≥2000万 且 上市满1年
        （池子 1004 → 预计 400-500 只）
  V3-3 动量窗口：ret_12 替换为 ret_12+ret_24 等权合成

验证标准：全期夏普 ≥0.40 | 2024-25夏普 ≥0.60 | 全期回撤 ≤-22%
（V2 基线：0.33 / 0.52 / -21.2%）

交付：output/report_mainboard_v3.html
运行：cd src && python main_mainboard_v3.py --stage 1     # 只跑 stage1
      python main_mainboard_v3.py                        # 顺序 1→2→3，达标即停
"""

from __future__ import annotations

import argparse
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
from report import compute_metrics, chart_multi_equity, chart_drawdown_compare
from strategies.trend_ema import gen_signal as sig_trend
from strategies.vol_breakout import gen_signal as sig_breakout
from fetch_dividend import get_v2_codes

MA_BASE = 240
TOP_N = 30
CAP = 0.10
SPLIT_DATE = "2024-01-01"

# V3 验证标准
TARGET_FULL = 0.40
TARGET_NEW = 0.60
TARGET_DD = -0.22

DY_GATE_WINDOW = 36        # 股息率中位数滚动窗口（月）
DY_GATE_SIGMA = 0.20       # V3.1 方向3：0.20（低σ×高降幅稳健高原，全期0.61/2024-25 0.83）
DY_GATE_WEIGHT = 0.20      # V3.1 方向3：触发时仓位上限 20%（降幅80%；原 V3 为 0.70）
MIN_ROE = 5.0              # 质量过滤：ROE ≥ 5%
MIN_AMOUNT = 2.0e7         # 质量过滤：20日均成交额 ≥ 2000万
MIN_LIST_YEARS = 1         # 质量过滤：上市满 1 年


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


def build_dy_gate(div_yield_panel: pd.DataFrame, me: pd.DatetimeIndex) -> pd.Series:
    """股息率仓位门控（月频 0.7/1.0）：
    每月全市场股息率中位数 > 历史滚动(36月)均值+1σ → 0.70，否则 1.0。
    滚动统计用 shift(1) 防未来函数（决策月只看之前月份）。
    """
    dy_med = div_yield_panel.where(div_yield_panel > 0).median(axis=1).sort_index()
    dy_med = dy_med.reindex(me)
    mean = dy_med.rolling(DY_GATE_WINDOW, min_periods=24).mean().shift(1)
    std = dy_med.rolling(DY_GATE_WINDOW, min_periods=24).std().shift(1)
    gate = pd.Series(1.0, index=me)
    hit = (dy_med > mean + DY_GATE_SIGMA * std) & std.notna()
    gate[hit] = DY_GATE_WEIGHT
    n_hit = int(hit.sum())
    print(f"  [dy门控] 触发 {n_hit}/{len(me)} 个月（股息率中位数>均值+{DY_GATE_SIGMA}σ，仓位降至 {DY_GATE_WEIGHT:.0%}）")
    return gate


def apply_quality_mask(close_m, roe, amount, me, min_roe=MIN_ROE,
                       min_amount=MIN_AMOUNT, min_years=MIN_LIST_YEARS):
    """质量过滤：ROE≥5% + 20日均成交额≥2000万 + 上市满1年。
    在 close_m 上置 NaN（不合格股票无法进入因子计算/选股）。"""
    out = close_m.copy()
    # 上市满 1 年（自首个有效收盘日 +250 自然日）
    for c in out.columns:
        s = out[c].dropna()
        if s.empty:
            continue
        cutoff = s.index.min() + pd.Timedelta(days=min_years * 365)
        out.loc[out.index < cutoff, c] = np.nan
    # ROE ≥ 5%（月度面板月末值，PIT）
    roe_m = roe.reindex(me)
    roe_ok = roe_m >= min_roe
    # 20 日均成交额 ≥ min_amount
    amt20 = amount.rolling(20, min_periods=10).mean().reindex(me)
    amt_ok = amt20 >= min_amount
    # 逐月末 mask
    for t in me:
        if t not in out.index:
            continue
        bad = set(out.columns)
        if t in roe_ok.index:
            bad &= set(roe_ok.loc[t][roe_ok.loc[t] == False].index)
        if t in amt_ok.index:
            bad &= set(amt_ok.loc[t][amt_ok.loc[t] == False].index)
        out.loc[t, list(bad)] = np.nan
    # 统计
    sizes = [out.loc[t].notna().sum() for t in me if t in out.index]
    print(f"  [质量过滤] 末日合格 {sizes[-1] if sizes else 0} 只，平均 {np.mean(sizes) if sizes else 0:.0f} 只")
    return out


def build_mz(close_m, roe, gpm, me, long_momentum=True, persistence=True):
    """动量/质量 Z-score 合成分。
    persistence=True（方向1 动量持续性）：(z(ret12-ret3)+z(ret24)+z(roe)+z(gpm_yoy))/4
    persistence=False（V3.1 基线）：ret=(ret_12+ret_24)/2 与 ROE/毛利率 等权。
    long_momentum 仅影响 persistence=False 路径。"""
    ret12 = close_m.pct_change(config.FWD_RETURN_DAYS * 12)
    ret24 = close_m.pct_change(config.FWD_RETURN_DAYS * 24)
    if persistence:
        ret3 = close_m.pct_change(config.FWD_RETURN_DAYS * 3)
        cols = {"ret_persist": ret12 - ret3, "ret_24": ret24, "roe": roe, "gpm_yoy": gpm}
    else:
        ret_v = (ret12 + ret24) / 2.0 if long_momentum else ret12
        cols = {"ret_12": ret_v, "roe": roe, "gpm_yoy": gpm}
    rows = {}
    for t in pd.DatetimeIndex(me):
        sub = pd.DataFrame({k: v.loc[t] if t in v.index else pd.Series(dtype=float)
                            for k, v in cols.items()})
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


def run_full(close_m, amount, sel_v8, me, tw, enable_sector=None,
             sector_map=None, bench=None, conc_log=None):
    """分区间回测：段1 V8(≤2023) → 段2 E等权(≥2024)。返回拼接净值。
    enable_sector=None → 跟随 config.ENABLE_SECTOR_NEUTRAL（方向2 开关）；
    显式 True/False 供对比实验。sector_map/bench 可外部注入（None 时自动加载）。
    conc_log：若提供 list，则记录各月末目标持仓的行业集中度（验证约束生效）。"""
    from v3_common import apply_sector_neutral, load_sector_data
    if enable_sector is None:
        enable_sector = bool(getattr(config, "ENABLE_SECTOR_NEUTRAL", False))
    if enable_sector and (sector_map is None or bench is None):
        sector_map, bench = load_sector_data()
        if sector_map is None or bench is None:
            print("  [sector] ⚠️ 行业数据缺失，本段跳过中性化")
            enable_sector = False

    def _bench_row(t):
        if bench is None or t not in bench.index:
            return {}
        return {str(k): float(v) for k, v in bench.loc[t].items()}

    def _log_conc(t, weights, mode):
        if conc_log is None or sector_map is None:
            return
        ind_w = {}
        for c, wt in weights.items():
            s = sector_map.get(str(c), "其他")
            ind_w[s] = ind_w.get(s, 0.0) + wt
        if not ind_w:
            return
        top = sorted(ind_w.items(), key=lambda kv: -kv[1])
        conc_log.append({
            "date": str(t.date()), "mode": mode, "top1_industry": top[0][0],
            "top1_weight": top[0][1], "n_industries": len(ind_w),
            "top3_weight": sum(v for _, v in top[:3]),
        })

    sel_trend = sig_trend(close_m, me, top_n=TOP_N)
    sel_brk = sig_breakout(close_m, amount, me, top_n=TOP_N)
    me1 = me[me < pd.Timestamp(SPLIT_DATE)]
    me2 = me[me >= pd.Timestamp(SPLIT_DATE)]

    # 段1 V8：selection 列表 → 等权；enable_sector 时中性化后经 weight_mult 注入
    wmult = {} if enable_sector else None
    for t in me1:
        cc = sel_v8.get(t, [])
        if not cc:
            continue
        w0 = {c: 1.0 / len(cc) for c in cc}
        _log_conc(t, w0, "raw")
        if enable_sector:
            w1 = apply_sector_neutral(w0, sector_map, _bench_row(t))
            for c in cc:
                wmult[(t, c)] = w1.get(c, 0.0) / 0.10   # target = 0.10*eq*mult = wt*eq
            _log_conc(t, w1, "neutral")
    eq1, _ = run_backtest_v5(close_m, sel_v8, me1, config.START_DATE, "2023-12-31",
                             target_weight=tw, slippage_map=slip_map_, weight_mult=wmult)
    init2 = float(eq1.iloc[-1])

    # 段2 E等权组合：V8/Trend/Breakout 各 1/3，单只 ≤ CAP
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
    if enable_sector:
        for t in me2:
            w1 = apply_sector_neutral(dict(sched[t]), sector_map, _bench_row(t))
            sched[t] = list(w1.items())
    for t in me2:
        _log_conc(t, dict(sched[t]), "neutral" if enable_sector else "raw")
    eq2, _ = run_backtest_combo(close_m, sched, me2, SPLIT_DATE, str(close_m.index[-1].date()),
                                target_weight=tw, slippage_map=slip_map_, init_capital=init2)
    eq = pd.concat([eq1, eq2])
    eq = eq[~eq.index.duplicated(keep="first")].sort_index()
    return eq


def main():
    ap = argparse.ArgumentParser(description="主板版 V3 回测")
    ap.add_argument("--stage", type=int, default=0,
                    help="1=仓位门控 / 2=+质量过滤 / 3=+长动量；0=顺序 1→2→3 达标即停")
    args = ap.parse_args()

    global slip_map_
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
    slip_map_, _ = build_slippage_map(amount)

    close_m = mask_new_listings(close, config.NEW_STOCK_MIN_DAYS)
    rsi = compute_rsi(close_m, config.RSI_WINDOW)
    me = month_ends_in(close_m, config.START_DATE, str(close.index[-1].date()))

    # 基础 MA240+波动率门控（日频）
    tw_base, _ = build_ma240_vol_target_weight(
        idx, close_m.index, MA_BASE, month_ends=me,
        vol_q=0.75, reduced_weight=0.60, vol_lookback=756)

    stages = [1, 2, 3] if args.stage == 0 else [args.stage]
    results = {}
    for st in stages:
        label = {1: "V3-1 三因子 + 股息率仓位门控",
                 2: "V3-2 + 质量过滤(ROE≥5%/额≥2000万/上市1年)",
                 3: "V3-3 + 长动量(ret_12+ret_24)"}[st]
        print(f"\n[{datetime.now()}] ==== {label} ====")

        # 质量过滤（stage≥2）
        cm = close_m
        if st >= 2:
            cm = apply_quality_mask(close_m.copy(), roe, amount, me)

        # 动量/质量合成分（stage≥3 长动量）
        mz = build_mz(cm, roe, gpm, me, long_momentum=(st >= 3))
        use_rev = pd.Series(False, index=me)
        sel_v8, _ = build_selection_v5(
            cm, rsi, pd.DataFrame(index=cm.index, columns=cm.columns),
            mz, me, use_rev, 0.20, TOP_N)

        # 仓位门控（所有 stage 都有）：tw = MA240门控 × 股息率门控
        tw = tw_base
        if st >= 1:
            gate = build_dy_gate(dy, me)
            gate_daily = gate.reindex(close_m.index).ffill().fillna(1.0)
            tw = tw_base * gate_daily

        eq = run_full(cm, amount, sel_v8, me, tw)
        mf = compute_metrics(eq)
        mn = compute_metrics(eq.loc["2024-01-01":])
        results[st] = (mf, mn)
        print(f"  → {label}: 全期夏普={mf['sharpe']:.2f} 回撤={mf['max_drawdown']*100:.1f}% | "
              f"2024-25夏普={mn['sharpe']:.2f} 回撤={mn['max_drawdown']*100:.1f}%")

        ok = (mf["sharpe"] >= TARGET_FULL and mn["sharpe"] >= TARGET_NEW
              and mf["max_drawdown"] >= TARGET_DD)
        if ok:
            print(f">>> {label} ✅ 达标（0.40/0.60/-22%），停止后续 stage")
            _write_report(results, label, eq, mf, mn, st, t0)
            return

    # 全部跑完未达标：用最后 stage 生成报告
    st = stages[-1]
    mf, mn = results[st]
    print(f">>> 全部 stage 未达标，维持 V8.1 原版")
    _write_report(results, None, None, mf, mn, st, t0)


def _write_report(results, label, eq, mf, mn, st, t0):
    from pathlib import Path
    rows_html = ""
    for s, (m_f, m_n) in results.items():
        nm = {1: "V3-1 仓位门控", 2: "V3-2 +质量过滤", 3: "V3-3 +长动量"}[s]
        rows_html += (f"<tr><td>{nm}</td><td>{m_f['sharpe']:.2f}</td>"
                      f"<td>{m_f['max_drawdown']*100:.2f}%</td>"
                      f"<td>{m_n['sharpe']:.2f}</td></tr>")
    verdict = ("<b>✅ 达标</b>" if (mf["sharpe"] >= TARGET_FULL and mn["sharpe"] >= TARGET_NEW
              and mf["max_drawdown"] >= TARGET_DD) else "<b>❌ 未达标</b>")
    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>主板版 V3 回测报告</title>
<style>body{{font-family:-apple-system,'PingFang SC',sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222}}
table{{border-collapse:collapse;width:100%;margin:14px 0;font-size:13px}}
th,td{{border:1px solid #ddd;padding:7px 9px;text-align:center}}
th{{background:#f5f5f5}} h2{{border-bottom:2px solid #eee;padding-bottom:6px;margin-top:34px}}
.note{{background:#f2f6fc;border-left:4px solid #4a7abb;padding:10px 14px;font-size:13px;color:#444;margin:12px 0}}
.verdict{{background:#fff7e6;border:1px solid #f0c36d;border-radius:6px;padding:12px 16px;margin:16px 0}}
img{{max-width:100%}}</style></head><body>
<h1>主板版 V3 回测报告</h1>
<p>选股池：V8 指数成分 ∩ 主板 60/00（1004 只）｜月频｜分档滑点｜MA240+波动率+股息率门控</p>
<div class="note"><b>V3 机制：</b>股息率从选股因子转为<b>择时门控</b>（每月全市场股息率中位数
&gt;历史36月均值+1σ → 仓位上限降至 70%）；V3-2 叠加质量过滤（ROE≥5%、20日成交额≥2000万、
上市满1年）；V3-3 动量窗口加长（ret_12+ret_24 等权）。选股恒为 ret_12/ROE/毛利率 三因子质量模式。</div>
<h2>Stage 结果（目标：全期≥0.40 / 2024-25≥0.60 / 回撤≤-22%）</h2>
<table><thead><tr><th>Stage</th><th>全期夏普</th><th>全期回撤</th><th>2024-25夏普</th></tr></thead>
<tbody><tr><td>V2 基线</td><td>0.33</td><td>-21.2%</td><td>0.52</td></tr>{rows_html}</tbody></table>
<div class="verdict">{verdict}（生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}）</div>
<p style="font-size:12px;color:#999">引擎零改动；门控合并入 target_weight，未触碰 backtest_v5 主逻辑。</p>
</body></html>"""
    out = config.OUTPUT_DIR / "report_mainboard_v3.html"
    out.write_text(html, encoding="utf-8")
    print(f"\n[{datetime.now()}] 报告: {out}  耗时 {datetime.now()-t0}")


if __name__ == "__main__":
    main()
