# -*- coding: utf-8 -*-
"""
main_mainboard_v2.py — 主板版 V2 回测（收窄池 + 强制质量模式 + 股息率因子）
====================================================================

V1 失败教训（2026-08-25）：
  - 全市场主板 3046 只太宽、稀释 alpha；反转 IC 0.054>0.05 触发反转模式但 LightGBM
    反转信号覆盖率仅 1.9% → 反转月份空仓 54 个月 → 全期 -0.11。

V2 三项改动：
  1) 选股池收窄：V8 面板（中证500∪创业板∪中证1000 成分）∩ 主板(60/00) → 1004 只
     （数据复用 mainboard 面板，至 2026-08-24，免重新抓取）
  2) 强制质量模式：不训练反转信号、use_reversal 全 False，全程 ret_12+roe+gpm_yoy
     等权 Z-score 选股（+可选第 4 因子股息率）
  3) 股息率因子：div_yield_panel（近 12 月每股分红 / 股价，PIT，无分红=0）并入质量合成

机制其余不变（与 V8.1 一致）：
  - MA240 + 波动率降仓门控（沪深300）、分档滑点、月频 30 只、单只 cap 10%
  - ≤2023 V8 原样；≥2024 E 等权组合（V8+Trend+Breakout 各 1/3）

验证标准（用户规定）：
  全期夏普 ≥ 0.45 | 2024-25夏普 ≥ 0.70 | 全期回撤 ≤ -25%

交付：output/report_mainboard_v2.html
运行：cd src && python main_mainboard_v2.py
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
from factor_eval import compute_momentum_zscore, build_selection_v5
from backtest_v5 import run_backtest_v5
from backtest_combo import run_backtest_combo
from market_filter import build_ma240_vol_target_weight
from stress_test_v6 import build_slippage_map
from report import compute_metrics, chart_multi_equity, chart_drawdown_compare
from strategies.trend_ema import gen_signal as sig_trend
from strategies.vol_breakout import gen_signal as sig_breakout

MA_BASE = 240
TOP_N = 30
CAP = 0.10
SPLIT_DATE = "2024-01-01"

# V2 验证标准
TARGET_FULL_SHARPE = 0.45
TARGET_NEW_SHARPE = 0.70
TARGET_DD = -0.25


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

    # ---- 1) 收窄选股池：V8 面板（指数成分）∩ 主板(60/00)，数据复用 mainboard 面板 ----
    from fetch_dividend import get_v2_codes
    codes = get_v2_codes()          # 1004 只
    close = pd.read_parquet(config.MB_CLOSE).reindex(columns=codes)
    amount = pd.read_parquet(config.MB_AMOUNT).reindex(columns=codes)
    idx = pd.read_parquet(config.DATA_DIR / "v6_index.parquet")
    if isinstance(idx, pd.DataFrame):
        idx = idx.iloc[:, 0]
    roe = pd.read_parquet(config.MB_ROE).reindex(index=close.index, columns=codes).ffill()
    gpm = pd.read_parquet(config.MB_GPM).reindex(index=close.index, columns=codes).ffill()
    dy_path = config.DATA_DIR / "div_yield_panel_mainboard.parquet"
    div_yield = None
    if dy_path.exists():
        div_yield = pd.read_parquet(dy_path).reindex(columns=codes)
        print(f"  股息率面板已加载（{div_yield.shape}）")
    else:
        print("  [warn] 股息率面板缺失（先运行 fetch_dividend.py），本次用三因子质量")

    n_universe = len(codes)
    data_start = str(close.index[0].date())
    data_end = str(close.index[-1].date())

    slip_map, tier_counts = build_slippage_map(amount)
    close_m = mask_new_listings(close, config.NEW_STOCK_MIN_DAYS)
    rsi = compute_rsi(close_m, config.RSI_WINDOW)
    me = month_ends_in(close_m, config.START_DATE, data_end)

    # ---- 2) 强制质量模式：mz（含股息率第 4 因子）选股，use_reversal 全 False ----
    print(f"[{datetime.now()}] 构建动量/质量合成分（{n_universe} 只，"
          f"{'含股息率' if div_yield is not None else '三因子'}）...")
    mz = compute_momentum_zscore(close_m, roe, gpm, me, div_yield_panel=div_yield)
    use_rev_f = pd.Series(False, index=me)
    sel_v8, sw = build_selection_v5(close_m, rsi, pd.DataFrame(
        index=close_m.index, columns=close_m.columns), mz, me, use_rev_f, 0.20, TOP_N)
    sel_trend = sig_trend(close_m, me, top_n=TOP_N)
    sel_brk = sig_breakout(close_m, amount, me, top_n=TOP_N)
    sizes = [len(v) for v in sel_v8.values()]
    print(f"  V8 段每月选股：空月={sum(1 for s in sizes if s == 0)} "
          f"满仓={sum(1 for s in sizes if s == TOP_N)} 平均={np.mean(sizes):.1f}")

    tw, vol_regime = build_ma240_vol_target_weight(
        idx, close_m.index, MA_BASE, month_ends=me,
        vol_q=0.75, reduced_weight=0.60, vol_lookback=756)
    idx_ret = idx.reindex(close_m.index).pct_change().fillna(0)
    idx_eq = (1.0 + idx_ret).cumprod() * config.INIT_CAPITAL

    # ---- 3) 分区间：段1 V8(≤2023) → 段2 E等权(≥2024) ----
    print(f"[{datetime.now()}] 分区间回测（SPLIT={SPLIT_DATE}）...")
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

    m_mb = _m(eq_split)
    print(f"  主板V2: 全期夏普={m_mb['full']['sharpe']:.2f} 回撤={m_mb['full']['max_drawdown']*100:.1f}% "
          f"2024-25={m_mb['new']['sharpe']:.2f}")

    ok1 = m_mb["full"]["sharpe"] >= TARGET_FULL_SHARPE
    ok2 = m_mb["new"]["sharpe"] >= TARGET_NEW_SHARPE
    ok3 = m_mb["full"]["max_drawdown"] >= TARGET_DD
    passed = ok1 and ok2 and ok3
    print(f"验证: 全期≥{TARGET_FULL_SHARPE}: {ok1} ({m_mb['full']['sharpe']:.2f}) | "
          f"2024-25≥{TARGET_NEW_SHARPE}: {ok2} ({m_mb['new']['sharpe']:.2f}) | "
          f"回撤≤{TARGET_DD*100:.0f}%: {ok3} ({m_mb['full']['max_drawdown']*100:.2f}%)")
    print(f">>> 主板版V2{'✅ 通过（可切换模拟盘）' if passed else '❌ 未通过'}")

    # ---- 报告 ----
    eq_img = chart_multi_equity({"主板V2": eq_split, "CSI300": idx_eq},
                                "主板版 V2 分区间部署")
    dd_img = chart_drawdown_compare({"主板V2": eq_split, "CSI300": idx_eq})

    mrow = (f"<tr><td><b>主板V2（收窄池+质量+股息率）</b></td>"
            f"<td>{m_mb['full']['sharpe']:.2f}</td>"
            f"<td>{m_mb['full']['annual_return']*100:.2f}%</td>"
            f"<td>{m_mb['full']['max_drawdown']*100:.2f}%</td>"
            f"<td>{m_mb['old']['sharpe']:.2f}</td>"
            f"<td>{m_mb['new']['sharpe']:.2f}</td>"
            f"<td>{m_mb['new']['max_drawdown']*100:.2f}%</td></tr>")
    ref_row = ("<tr><td><b>V8.1 原版（锚点）</b></td><td>0.60</td><td>—</td>"
               "<td>-19.40%</td><td>—</td><td>0.79</td><td>—</td></tr>")
    v1_row = ("<tr><td><b>主板V1（全市场3046只+原版Regime）</b></td><td>-0.11</td>"
              "<td>—</td><td>-44.62%</td><td>—</td><td>-0.17</td><td>—</td></tr>")

    verdict = "<br>".join([
        f"{'✅' if ok1 else '❌'} 全期夏普 ≥{TARGET_FULL_SHARPE}：{m_mb['full']['sharpe']:.2f}",
        f"{'✅' if ok2 else '❌'} 2024-25夏普 ≥{TARGET_NEW_SHARPE}：{m_mb['new']['sharpe']:.2f}",
        f"{'✅' if ok3 else '❌'} 全期回撤 ≤{TARGET_DD*100:.0f}%：{m_mb['full']['max_drawdown']*100:.2f}%",
        f"<b style='font-size:14px'>结论：{'✅ 通过验证，可切换模拟盘至主板版V2' if passed else '❌ 未通过验证'}</b>",
    ])

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>主板版 V2 回测报告</title>
<style>body{{font-family:-apple-system,'PingFang SC',sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222}}
table{{border-collapse:collapse;width:100%;margin:14px 0;font-size:13px}}
th,td{{border:1px solid #ddd;padding:7px 9px;text-align:center}}
th{{background:#f5f5f5}} h2{{border-bottom:2px solid #eee;padding-bottom:6px;margin-top:34px}}
.verdict{{background:#fff7e6;border:1px solid #f0c36d;border-radius:6px;padding:12px 16px;margin:16px 0}}
.pass{{background:#e8f8e8;border:1px solid #7ab87a;border-radius:6px;padding:12px 16px;margin:16px 0}}
.fail{{background:#fdeaea;border:1px solid #d88;border-radius:6px;padding:12px 16px;margin:16px 0}}
.note{{background:#f2f6fc;border-left:4px solid #4a7abb;padding:10px 14px;font-size:13px;color:#444;margin:12px 0}}
img{{max-width:100%}}</style></head><body>
<h1>主板版 V2 回测报告</h1>
<p>数据区间 {data_start} ~ {data_end}（主板 V2 池 {n_universe} 只 = V8 指数成分 ∩ 主板 60/00）｜月频｜分档滑点｜MA240+波动率门控</p>
<div class="note"><b>V2 三项改动：</b>① 选股池收窄至 1004 只（V8 成分中的主板，去宽稀释）；
② 强制质量模式（use_reversal=False，回避 1.9% 覆盖率的反转信号）；
③ 质量因子新增股息率（近 12 月每股分红/股价，PIT，与 ROE/毛利率/动量等权合成）。
其余机制与 V8.1 一致：≤2023 V8 / ≥2024 E 等权组合。</div>
<h2>绩效对比</h2>
<table><thead><tr><th>版本</th><th>全期夏普</th><th>全期年化</th><th>全期回撤</th>
<th>2018-23夏普</th><th>2024-25夏普</th><th>2024-25回撤</th></tr></thead>
<tbody>{ref_row}{v1_row}{mrow}</tbody></table>
<h2>验证标准</h2>
<div class="{'pass' if passed else 'fail'}">{verdict}</div>
<h2>资金曲线</h2>{eq_img}
<h2>回撤曲线</h2>{dd_img}
<p style="font-size:12px;color:#999">生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}｜引擎零改动；V2 仅收窄池 + 强制质量 + 股息率。</p>
</body></html>"""

    out_path = config.OUTPUT_DIR / "report_mainboard_v2.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"\n[{datetime.now()}] 报告: {out_path}  耗时 {datetime.now()-t0}")


if __name__ == "__main__":
    main()
