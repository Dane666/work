# -*- coding: utf-8 -*-
"""
V8.1 执行口径对齐回测（next_open 成交 + 冲击成本）—— 与模拟盘/实盘口径一致。

对比臂：
  A  V8.1 close 口径（原版，对照）         —— 应复现 0.60 / 0.79 / -19.40%
  B  V8.1 next_open 无冲击成本            —— 隔离「执行价格」影响
  C  V8.1 next_open + 冲击成本（正式版）   —— 完整实盘口径

机制（run_backtest_v5_ne）：
  - 信号日 T 收盘生成信号 → T+1 开盘价成交（open 缺失顺延）；
  - 冲击成本按 T 日 20 日均成交额分级（>5亿 0.05% / 1-5亿 0.15% / <1亿 0.30%），与滑点叠加。
  - 分区间：≤2023 V8 原样（fixed_weight=0.10）；≥2024 E 等权组合
    （selection=三策略并集、weight_mult=该股被选中策略数、fixed_weight=1/90 → 权重 1/90×k，cap 隐含 3/90=3.33%）。
  - 两段各自回测后净值拼接（段1末值→段2初始资金），与 main_v8_1_split 同法。

验证：全期绩效对比；与模拟盘 2025-08-13 首次建仓对齐（新买入组合当日收益 vs 模拟盘 NAV 1.0297）。
交付：output/report_v8_1_next_open.html
"""

from __future__ import annotations

import os
import pickle
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
from backtest_v5 import run_backtest_v5, run_backtest_v5_ne, build_open_panel
from backtest_combo import run_backtest_combo
from market_filter import build_ma240_target_weight, build_ma240_vol_target_weight
from stress_test_v6 import build_ohlcv_full, build_slippage_map
from report import compute_metrics, yearly_sharpe, chart_multi_equity, chart_drawdown_compare
from strategies.trend_ema import gen_signal as sig_trend
from strategies.vol_breakout import gen_signal as sig_breakout

END_EXT = "2025-08-13"        # 含追加的真实行情日（next_open 引擎需在 T+1=08-13 执行 08-12 指令）
MA_BASE = 240
IC_BASE = 0.05
TOP_N = 30
CAP = 0.10
SPLIT = pd.Timestamp("2024-01-01")
IMPACT_TIERS = getattr(config, "IMPACT_TIERS", [(5e8, 0.0005), (1e8, 0.0015), (0.0, 0.0030)])
FW_COMBO = 1.0 / 3                # E 组合：策略权重 1/3，个股权重 = 1/3 × Σ(1/len_s)


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


def combo_selection_and_mult(sel_v8, sel_trend, sel_brk, me) -> tuple[dict, dict]:
    """E 等权组合：selection=三策略并集；weight_mult[(t,c)] = Σ_s 1/len_s
    （c 所属各策略的实际选股数倒数之和）；fixed_weight=1/3（策略权重）。
    与 main_v8_1_split.build_eq_schedule 逐策略 1/3×1/len_s 权重完全一致。"""
    sel_combo = {}
    wm = {}
    for t in me:
        groups = [("V8", sel_v8.get(t, [])), ("Trend", sel_trend.get(t, [])),
                  ("Breakout", sel_brk.get(t, []))]
        cs = []
        per: dict = {}
        for _name, codes_s in groups:
            n = len(codes_s)
            if n == 0:
                continue
            w_each = 1.0 / n
            for c in codes_s:
                per[c] = per.get(c, 0.0) + w_each
                if c not in cs:
                    cs.append(c)
        sel_combo[t] = cs
        for c in cs:
            wm[(t, c)] = per[c]
    return sel_combo, wm


def append_next_day(close_m, open_panel, amount, tw, codes_pick, date="2025-08-13"):
    """把真实行情日追加到面板尾部（供 next_open 引擎在 T+1 执行指令并盯市）。

    仅追加 codes_pick（相关持仓/选股）的 open/close，其余为 NaN；tw 沿用前一交易日值。
    返回追加后的 (close_m, open_panel, amount, tw)。
    """
    from sim_tracker import fetch_day_prices
    dt = pd.Timestamp(date)
    px = fetch_day_prices([str(c).zfill(6) for c in codes_pick], dt)
    print(f"  追加 {date}: 行情覆盖 {len(px)}/{len(codes_pick)} 只")
    if not px:
        return close_m, open_panel, amount, tw
    for c in close_m.columns:
        p = px.get(c)
        if p is None:
            close_m.loc[dt, c] = np.nan
            open_panel.loc[dt, c] = np.nan
        else:
            close_m.loc[dt, c] = p["close"]
            open_panel.loc[dt, c] = p["open"]
    amount.loc[dt] = amount.loc[close_m.index[close_m.index < dt][-1]] if len(close_m.index[close_m.index < dt]) else 0.0
    tw.loc[dt] = tw.loc[close_m.index[close_m.index < dt][-1]] if len(close_m.index[close_m.index < dt]) else 1.0
    return close_m, open_panel, amount, tw


def run_split_ne(close_m, me, tw, slip_map, open_panel, amount,
                 sel_v8, sel_combo, wm, enable_impact, impact_tiers=None) -> pd.Series:
    """分区间 next_open 回测：≤2023 V8(fw=0.10) → ≥2024 E组合(fw=1/3×mult)。"""
    if impact_tiers is None:
        impact_tiers = IMPACT_TIERS
    me1 = me[me < SPLIT]
    me2 = me[me >= SPLIT]
    eq1, _ = run_backtest_v5_ne(close_m, sel_v8, me1, config.START_DATE, "2023-12-31",
                                open_panel=open_panel, target_weight=tw, slippage_map=slip_map,
                                fixed_weight=0.10, enable_impact=enable_impact,
                                impact_tiers=impact_tiers, amount_panel=amount)
    init2 = float(eq1.iloc[-1])
    eq2, _ = run_backtest_v5_ne(close_m, sel_combo, me2, "2024-01-01", END_EXT,
                                open_panel=open_panel, target_weight=tw, slippage_map=slip_map,
                                fixed_weight=FW_COMBO, weight_mult=wm,
                                enable_impact=enable_impact,
                                impact_tiers=impact_tiers, amount_panel=amount,
                                init_capital=init2)
    return pd.concat([eq1, eq2]).pipe(lambda s: s[~s.index.duplicated(keep="first")]).sort_index()


def run_split_close(close_m, me, tw, slip_map, sel_v8, sel_combo, wm) -> pd.Series:
    """分区间 close 口径（对照 A）：段1 V8 原引擎；段2 combo 原引擎。"""
    me1 = me[me < SPLIT]
    me2 = me[me >= SPLIT]
    eq1, _ = run_backtest_v5(close_m, sel_v8, me1, config.START_DATE, "2023-12-31",
                             target_weight=tw, slippage_map=slip_map)
    init2 = float(eq1.iloc[-1])
    # combo 权重表（close 口径原版）：weight = 1/90 × k
    wt_sched = {}
    for t in me2:
        wt_sched[t] = [(c, FW_COMBO * wm.get((t, c), 1.0)) for c in sel_combo.get(t, [])]
    eq2, _ = run_backtest_combo(close_m, wt_sched, me2, "2024-01-01", END_EXT,
                                target_weight=tw, slippage_map=slip_map, init_capital=init2)
    return pd.concat([eq1, eq2]).pipe(lambda s: s[~s.index.duplicated(keep="first")]).sort_index()


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

    # ---- open 面板（next_open 执行价）----
    with open(config.DATA_DIR / "v8_ohlcv.pkl", "rb") as f:
        ohlcv = pickle.load(f)
    open_panel = build_open_panel(ohlcv, close)
    n_cov = int(open_panel.notna().any(axis=1).sum() > 0 and open_panel.shape[1])
    print(f"open 面板覆盖: {int((open_panel.iloc[-1].notna()).sum())}/{n_universe} 只（末日）")

    slip_map, tier_counts = build_slippage_map(amount)
    close_m = mask_new_listings(close, config.NEW_STOCK_MIN_DAYS)
    rsi = compute_rsi(close_m, config.RSI_WINDOW)
    me = month_ends_in(close_m, config.START_DATE, END_EXT)
    ohlcv_old = pd.read_pickle(config.V3_OHLCV) if config.V3_OHLCV.exists() else {}
    ohlcv_full = build_ohlcv_full(close, amount, ohlcv_old, list(close.columns))

    print(f"[{datetime.now()}] 构建信号...")
    reversal_signal = build_rolling_reversal_signal(close_m, ohlcv_full, window=36)
    monthly_ic = monthly_reversal_ic(close_m, rsi, me, config.FWD_RETURN_DAYS)
    rolling_ic, use_reversal = compute_rolling_regime(monthly_ic, 12, IC_BASE, 6)
    config.ENABLE_ANALYST_FACTOR = False
    mz = compute_momentum_zscore(close_m, roe, gpm, me)
    sel_v8, sw_v8 = build_selection_v5(close_m, rsi, reversal_signal, mz, me,
                                       use_reversal, 0.20, TOP_N)
    sel_trend = sig_trend(close_m, me, top_n=TOP_N)
    sel_brk = sig_breakout(close_m, amount, me, top_n=TOP_N)
    sel_combo, wm = combo_selection_and_mult(sel_v8, sel_trend, sel_brk, me)

    tw, vol_regime = build_ma240_vol_target_weight(
        idx, close_m.index, MA_BASE, month_ends=me,
        vol_q=0.75, reduced_weight=0.60, vol_lookback=756)

    # ---- 追加 08-13 真实行情（让 next_open 引擎执行 08-12 指令并盯市，与模拟盘对齐）----
    last_me = me[-1]
    prev_me = me[-2] if len(me) >= 2 else last_me
    codes_pick = list(dict.fromkeys(
        list(sel_combo.get(last_me, [])) + list(sel_combo.get(prev_me, []))))
    close_m, open_panel, amount, tw = append_next_day(
        close_m, open_panel, amount, tw, codes_pick, date="2025-08-13")

    idx_ret = idx.reindex(close_m.index).pct_change().fillna(0)
    idx_eq = (1.0 + idx_ret).cumprod() * config.INIT_CAPITAL

    # ---- 三臂 ----
    print(f"[{datetime.now()}] 臂A：V8.1 close 口径（对照）...")
    eq_A = run_split_close(close_m, me, tw, slip_map, sel_v8, sel_combo, wm)
    print(f"[{datetime.now()}] 臂B：V8.1 next_open 无冲击...")
    eq_B = run_split_ne(close_m, me, tw, slip_map, open_panel, amount,
                        sel_v8, sel_combo, wm, enable_impact=False)
    print(f"[{datetime.now()}] 臂C：V8.1 next_open + 冲击成本（正式版）...")
    eq_C = run_split_ne(close_m, me, tw, slip_map, open_panel, amount,
                        sel_v8, sel_combo, wm, enable_impact=True)
    print(f"[{datetime.now()}] 臂D：next_open + 0.5×冲击（敏感性，评估是否高估）...")
    half_tiers = [(lo, r * 0.5) for lo, r in IMPACT_TIERS]
    eq_D = run_split_ne(close_m, me, tw, slip_map, open_panel, amount,
                        sel_v8, sel_combo, wm, enable_impact=True, impact_tiers=half_tiers)

    m_A, m_B, m_C, m_D = _m(eq_A), _m(eq_B), _m(eq_C), _m(eq_D)
    print(f"  A close: 全期={m_A['full']['sharpe']:.2f} 回撤={m_A['full']['max_drawdown']*100:.1f}% "
          f"2024-25={m_A['new']['sharpe']:.2f}")
    print(f"  B next_open: 全期={m_B['full']['sharpe']:.2f} 回撤={m_B['full']['max_drawdown']*100:.1f}% "
          f"2024-25={m_B['new']['sharpe']:.2f}")
    print(f"  C next_open+冲击: 全期={m_C['full']['sharpe']:.2f} 回撤={m_C['full']['max_drawdown']*100:.1f}% "
          f"2024-25={m_C['new']['sharpe']:.2f}")
    print(f"  D next_open+0.5×冲击: 全期={m_D['full']['sharpe']:.2f} 回撤={m_D['full']['max_drawdown']*100:.1f}% "
          f"2024-25={m_D['new']['sharpe']:.2f}")

    # 冲击成本影响：年化收益下降（B→C 为全冲击影响，C→D 敏感性）
    d_ann = m_B["full"]["annual_return"] - m_C["full"]["annual_return"]
    d_ann_half = m_C["full"]["annual_return"] - m_D["full"]["annual_return"]
    print(f"  冲击成本(全档) → 年化收益下降 {d_ann*100:.3f}pp；"
          f"档位减半 → 再降 {d_ann_half*100:.3f}pp")

    # ---- 与模拟盘 2025-08-13 对齐（回测已追加 08-13 真实行情）----
    import json
    st = json.loads((config.OUTPUT_DIR / "sim_nav" / "sim_state.json").read_text(encoding="utf-8"))
    nav_sim = float(st["nav"])
    nav_bt_B = float(eq_B.loc["2025-08-13"] / eq_B.loc["2025-08-12"])
    nav_bt_C = float(eq_C.loc["2025-08-13"] / eq_C.loc["2025-08-12"])
    nav_bt_D = float(eq_D.loc["2025-08-13"] / eq_D.loc["2025-08-12"])
    dev_sim_B = abs(nav_bt_B - nav_sim) / nav_sim * 100
    dev_sim_C = abs(nav_bt_C - nav_sim) / nav_sim * 100
    print(f"\n模拟盘 08-13: NAV={nav_sim:.6f}（归一化，08-12=1.0）")
    print(f"回测臂B(next_open) 08-13 归一化净值: {nav_bt_B:.6f}（偏差 {dev_sim_B:.3f}%）")
    print(f"回测臂C(+冲击) 08-13 归一化净值: {nav_bt_C:.6f}（偏差 {dev_sim_C:.3f}%）")
    print(f"回测臂D(0.5×冲击) 08-13 归一化净值: {nav_bt_D:.6f}")

    # ---- 报告 ----
    eq_img = chart_multi_equity({
        "V8.1 close（原口径）": eq_A, "V8.1 next_open": eq_B,
        "V8.1 next_open+冲击（正式）": eq_C, "V8.1 next_open+0.5×冲击": eq_D,
        "CSI300": idx_eq}, "V8.1 执行口径对齐（close vs next_open vs +冲击）")
    dd_img = chart_drawdown_compare({"V8.1 close": eq_A, "V8.1 next_open": eq_B,
                                     "V8.1 next_open+冲击": eq_C})

    def mrow(name, m, note=""):
        return (f"<tr><td><b>{name}</b></td><td>{m['full']['sharpe']:.2f}</td>"
                f"<td>{m['full']['annual_return']*100:.2f}%</td>"
                f"<td>{m['full']['max_drawdown']*100:.2f}%</td>"
                f"<td>{m['new']['sharpe']:.2f}</td>"
                f"<td>{m['new']['max_drawdown']*100:.2f}%</td><td>{note}</td></tr>")

    rows = (mrow("A V8.1 close（对照）", m_A, "原口径")
            + mrow("B V8.1 next_open（无冲击）", m_B,
                   f"vs A 夏普 {m_B['full']['sharpe']-m_A['full']['sharpe']:+.2f}")
            + mrow("C V8.1 next_open+冲击（正式）", m_C,
                   f"vs B 夏普 {m_C['full']['sharpe']-m_B['full']['sharpe']:+.2f}")
            + mrow("D V8.1 next_open+0.5×冲击（敏感性）", m_D,
                   f"vs C 夏普 {m_D['full']['sharpe']-m_C['full']['sharpe']:+.2f}"))

    impact_ok = d_ann < 0.005
    sim_ok = dev_sim_C < 0.5
    verdict = "<br>".join([
        f"{'✅' if impact_ok else '⚠️'} 冲击成本(全档) 导致年化收益下降 {d_ann*100:.3f}pp"
        f"{'（<0.5pp，模型未高估）' if impact_ok else '（≥0.5pp）'}；"
        f"档位减半后再降 {d_ann_half*100:.3f}pp（线性敏感性）",
        f"{'✅' if sim_ok else '⚠️'} 与模拟盘 08-13 对齐：next_open+冲击 偏差 {dev_sim_C:.3f}%"
        f"（无冲击 {dev_sim_B:.3f}%），{'< 0.5% ✅' if sim_ok else '≥ 0.5%，需排查'}",
    ])

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>V8.1 执行口径对齐（next_open + 冲击成本）</title>
<style>body{{font-family:-apple-system,'PingFang SC',sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222}}
table{{border-collapse:collapse;width:100%;margin:14px 0;font-size:13px}}
th,td{{border:1px solid #ddd;padding:7px 9px;text-align:center}}
th{{background:#f5f5f5}} h2{{border-bottom:2px solid #eee;padding-bottom:6px;margin-top:34px}}
.verdict{{background:#fff7e6;border:1px solid #f0c36d;border-radius:6px;padding:12px 16px;margin:16px 0}}
.note{{background:#f2f6fc;border-left:4px solid #4a7abb;padding:10px 14px;font-size:13px;color:#444;margin:12px 0}}
img{{max-width:100%}}</style></head><body>
<h1>V8.1 执行口径对齐：next_open 成交 + 冲击成本</h1>
<p>数据区间 {data_start} ~ {data_end}（{n_universe} 只）｜月频｜分区间（≤2023 V8 / ≥2024 E组合）</p>
<div class="note"><b>口径：</b>信号日 T 收盘生成信号 → <b>T+1 开盘价成交</b>（open 缺失顺延至复牌）；
冲击成本按 T 日 20 日均成交额分级（&gt;5亿 0.05% / 1-5亿 0.15% / &lt;1亿 0.30%，单边，与分档滑点叠加）。
open 面板来自 v8_ohlcv.pkl（复权基准已与 V8 面板对齐）。</div>
<h2>四臂绩效对比</h2>
<table><thead><tr><th>臂</th><th>全期夏普</th><th>全期年化</th><th>全期回撤</th>
<th>2024-25夏普</th><th>2024-25回撤</th><th>说明</th></tr></thead><tbody>{rows}</tbody></table>
<h2>资金曲线</h2>{eq_img}
<h2>回撤曲线</h2>{dd_img}
<h2>与模拟盘对齐验证（2025-08-13 首次建仓）</h2>
<table><thead><tr><th>对象</th><th>08-13 归一化净值（08-12=1.0）</th></tr></thead><tbody>
<tr><td>模拟盘（sim_tracker，open 建仓含滑点）</td><td>{nav_sim:.6f}</td></tr>
<tr><td>回测臂B（next_open 无冲击）</td><td>{nav_bt_B:.6f}（偏差 {dev_sim_B:.3f}%）</td></tr>
<tr><td>回测臂C（next_open+冲击）</td><td>{nav_bt_C:.6f}（偏差 {dev_sim_C:.3f}%）</td></tr>
<tr><td>回测臂D（next_open+0.5×冲击）</td><td>{nav_bt_D:.6f}</td></tr>
</tbody></table>
<p style="font-size:12px;color:#888">注：回测 08-12→08-13 区间含旧仓换仓（07-31 持仓 08-13 开盘卖出→新仓买入），
模拟盘为纯首日建仓，两者口径在「旧仓隔夜收益」上不同源，偏差仅供参考口径对齐程度。</p>
<h2>判定</h2>
<div class="verdict">{verdict}</div>
<p style="font-size:12px;color:#999">生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}｜
冲击成本档位/开关见 config.py（EXECUTION_PRICE / ENABLE_IMPACT_COST / IMPACT_TIERS）；
open 面板 v8_ohlcv.pkl（1524/1539 末日覆盖）。</p>
</body></html>"""

    out_path = config.OUTPUT_DIR / "report_v8_1_next_open.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"\n[{datetime.now()}] 报告: {out_path}  耗时 {datetime.now()-t0}")


if __name__ == "__main__":
    main()
