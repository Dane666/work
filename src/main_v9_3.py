# -*- coding: utf-8 -*-
"""
V9.3 主流程 = 模块3（周频调仓）的 **2 臂消融**。

前置裁定：模块1（分析师因子，V9.1）与模块2（行业拥挤度，V9.2 2×2）均已实测判定失败，
因此不再把模块3 叠在失败版本之上作"边际"（会污染归因）。改为两条臂：

      臂     周频  分析师因子  拥挤度风控   含义
      E      ON     OFF        OFF       V8 基座 + 仅周频 → 模块3 的干净边际（vs A=V8）
      F      ON     ON         ON        用户原定 V9.3（周频+分析师+拥挤），作完整记录

共用同一数据、同一引擎、同一 regime（月度判定 ffill 到周频）、同一周频滑点（上浮50%）。
模块3 的裁定以 E vs V8（output/v8_equiv_nav.parquet, 0.57/0.54/0.67）为准：
  若夏普提升且回撤可控 → 保留；若被交易成本侵蚀 → 失败归档。

严禁触碰 V8 已验证核心参数。落盘：output/v9_3_nav.parquet(F)、output/v8_weekly_nav.parquet(E)。
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
from market_filter import build_ma240_target_weight, build_ma240_vol_target_weight
from stress_test_v6 import build_ohlcv_full, build_slippage_map
from industry_crowding import load_industry_map, compute_crowding_weight_mult
from report import compute_metrics, yearly_sharpe
from report_v9 import generate_html_v9
import v9_common as vc

config.ENABLE_WEEKLY_REBALANCE = True

END_EXT = "2025-08-12"
MA_BASE = 240
IC_BASE = 0.05
FIXED_WEIGHT = config.FIXED_WEIGHT


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
    me = pd.DatetimeIndex(me).normalize()  # 统一 ns 精度零点，避免 ms/ns 失配
    return me[(me >= pd.Timestamp(start).normalize()) & (me <= pd.Timestamp(end).normalize())]


def week_ends_in(close_panel: pd.DataFrame, start, end) -> pd.DatetimeIndex:
    """每周最后一个交易日（W-FRI 对齐到实际交易日）。

    注意：close_panel 索引为 datetime64[ms]（pyarrow 写入），而 pd.date_range 生成 ns。
    两者混用会导致 Series 查找（use_reversal / sel / sw 等）在 ns vs ms 精度边界失配。
    统一归一化为「ns 精度的当日零点」返回，保证与后续 DatetimeIndex 精确匹配。
    """
    idx = close_panel.index
    if isinstance(idx, pd.DatetimeIndex) and idx.dtype != "datetime64[ns]":
        idx = idx.as_unit("ns")
    fridays = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="W-FRI")
    fridays = fridays.as_unit("ns")
    we = [idx[idx <= f].max() for f in fridays]
    we = pd.DatetimeIndex([w for w in we if w is not None and not pd.isna(w)]).normalize()
    # 去重：月末与周末重合的日期会重复出现（如 2018-09-28），必须 keep=last 去重，
    # 否则后续 use_reversal.get(t) 因重复索引返回 Series 而崩溃。
    we = we[~we.duplicated(keep="last")]
    return we[(we >= pd.Timestamp(start).normalize()) & (we <= pd.Timestamp(end).normalize())]


def load_name_map():
    try:
        import akshare as ak
        df = ak.stock_info_a_code_name()
        return dict(zip(df["code"].astype(str).str.zfill(6), df["name"].astype(str)))
    except Exception:
        return {}


def _ref(eq: pd.Series, label: str) -> dict:
    return {"full": compute_metrics(eq),
            "old": compute_metrics(eq.loc[:"2023-12-31"]),
            "new": compute_metrics(eq.loc["2024-01-01":]),
            "label": label}


def _save_nav(eq: pd.Series, path):
    df = eq.reset_index()
    df.columns = ["date", "nav"]
    df.to_parquet(path, index=False)


def _turnover_stats(trades: pd.DataFrame, slip_map: dict) -> dict:
    """从成交明细统计换手与摩擦成本（解释周频收益去向）。"""
    if trades.empty:
        return {"n_trades": 0, "n_reb": 0, "avg_interval_d": 0,
                "buy_notional": 0.0, "est_cost": 0.0, "cost_pct": 0.0}
    buys = trades[trades["action"] == "buy"]
    sells = trades[trades["action"] == "sell"]
    reb_dates = sorted(set(buys["date"]))
    n_reb = len(reb_dates)
    avg_interval_d = (reb_dates[-1] - reb_dates[0]).days / max(n_reb - 1, 1) if n_reb > 1 else 0.0
    buy_notional = float(buys["notional"].sum())
    # 单边摩擦 ≈ 买滑点 + 卖滑点
    est_cost = 0.0
    for _, r in trades.iterrows():
        sl = slip_map.get(r["code"], 0.003)
        est_cost += r["notional"] * sl
    return {"n_trades": len(trades), "n_reb": n_reb, "avg_interval_d": avg_interval_d,
            "buy_notional": buy_notional, "est_cost": est_cost,
            "cost_pct": est_cost / buy_notional * 100 if buy_notional else 0.0}


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

    slip_base, tier_counts = build_slippage_map(amount)
    uplift = getattr(config, "ANALYST_WEEKLY_SLIPPAGE_UPLIFT", 1.50)
    slip_map = {c: min(s * uplift, 0.01) for c, s in slip_base.items()}  # 周频滑点上浮50%

    close_m = mask_new_listings(close, config.NEW_STOCK_MIN_DAYS)
    rsi = compute_rsi(close_m, config.RSI_WINDOW)
    me_month = month_ends_in(close_m, config.START_DATE, END_EXT)
    me_week = week_ends_in(close_m, config.START_DATE, END_EXT)
    ohlcv_old = pd.read_pickle(config.V3_OHLCV) if config.V3_OHLCV.exists() else {}
    ohlcv_full = build_ohlcv_full(close, amount, ohlcv_old, codes_all)

    print(f"[{datetime.now()}] 构建滚动反转信号（V9 宇宙 {n_universe} 只）...")
    reversal_signal = build_rolling_reversal_signal(close_m, ohlcv_full, window=36)
    # 反转信号面板是月频；周频选股在 build_selection_v5 内做 reversal_signal.loc[t] 精确查找，
    # 必须用最近前月末 ffill 重索引到周频（与 use_reversal 同一套 point-in-time 逻辑）
    reversal_signal = reversal_signal.reindex(me_week, method="ffill")
    # Regime 仍按「月度」判定，再 ffill 到周频
    monthly_ic = monthly_reversal_ic(close_m, rsi, me_month, config.FWD_RETURN_DAYS)
    rolling_ic, use_reversal_m = compute_rolling_regime(monthly_ic, 12, IC_BASE, 6)
    # Regime 仍按「月度」判定，再以最近前月末的值 ffill 到周频（point-in-time：
    # 不能用 .reindex(me_week).ffill()——那会把月末值精确匹配掉，周五全部 NaN→True，
    # 等于是废掉 regime 切换；必须 method='ffill' 找 ≤ 该周五的最近月末）。
    use_reversal = use_reversal_m.reindex(me_week, method="ffill").fillna(True).astype(bool)

    tw_clean = build_ma240_target_weight(idx, close_m.index, MA_BASE)
    tw, vol_regime = build_ma240_vol_target_weight(
        idx, close_m.index, MA_BASE, month_ends=me_month,
        vol_q=0.75, reduced_weight=0.60, vol_lookback=756)

    idx_ret = idx.reindex(close_m.index).pct_change().fillna(0)
    idx_eq = (1.0 + idx_ret).cumprod() * config.INIT_CAPITAL
    m_idx = compute_metrics(idx_eq)

    # ---- 两套周频选股：分析师 OFF / ON ----
    print(f"[{datetime.now()}] 周频选股臂1：分析师 OFF...")
    config.ENABLE_ANALYST_FACTOR = False
    mz_off = compute_momentum_zscore(close_m, roe, gpm, me_week)
    sel_off, sw_off = build_selection_v5(
        close_m, rsi, reversal_signal, mz_off, me_week, use_reversal, 0.20, 30)

    print(f"[{datetime.now()}] 周频选股臂2：分析师 ON...")
    config.ENABLE_ANALYST_FACTOR = True
    mz_on = compute_momentum_zscore(close_m, roe, gpm, me_week)
    sel_on, sw_on = build_selection_v5(
        close_m, rsi, reversal_signal, mz_on, me_week, use_reversal, 0.20, 30)

    # ---- 模块2（周频拥挤度，仅 F 臂使用）----
    print(f"[{datetime.now()}] 计算行业拥挤度（周频）...")
    industry_map = load_industry_map()
    crowding_mult = compute_crowding_weight_mult(close, amount, industry_map, me_week)

    # ---- 两条臂回测 ----
    def bt(sel, mult, tag):
        eq, tr = run_backtest_v5(
            close_m, sel, me_week, config.START_DATE, END_EXT,
            target_weight=tw, slippage_map=slip_map, weight_mult=mult)
        print(f"  [{tag}] 全期夏普={compute_metrics(eq)['sharpe']:.2f}")
        return eq, tr

    print(f"[{datetime.now()}] 回测 2 臂...")
    eq_E, tr_E = bt(sel_off, None, "E V8+周频(OFF)")
    eq_F, tr_F = bt(sel_on, crowding_mult, "F V9.3(ON/ON)")
    tr_F.to_csv(config.OUTPUT_DIR / "v9_3_trades.csv", index=False)
    tr_E.to_csv(config.OUTPUT_DIR / "v8_weekly_trades.csv", index=False)

    # 回归/参照：V8 基线（月频）
    ref_A = vc.ref_from_nav(config.OUTPUT_DIR / "v8_equiv_nav.parquet", "V8")
    ref_E = _ref(eq_E, "V8+周频")
    m_full, m_old, m_new = (compute_metrics(eq_F), compute_metrics(eq_F.loc[:"2023-12-31"]),
                            compute_metrics(eq_F.loc["2024-01-01":]))
    # 主交付 F 的 MA240-only 对照
    eq_F_clean, _ = run_backtest_v5(
        close_m, sel_on, me_week, config.START_DATE, END_EXT,
        target_weight=tw_clean, slippage_map=slip_map, weight_mult=crowding_mult)
    m_full_c, m_old_c, m_new_c = (compute_metrics(eq_F_clean),
                                  compute_metrics(eq_F_clean.loc[:"2023-12-31"]),
                                  compute_metrics(eq_F_clean.loc["2024-01-01":]))
    yearly = yearly_sharpe(eq_F)
    yearly_prev = yearly_sharpe(eq_E)
    yearly_baseline = yearly_sharpe(vc.load_nav(config.OUTPUT_DIR / "v8_equiv_nav.parquet"))
    prev_ref = _ref(eq_E, "V8+周频")
    baseline_ref = ref_A

    _save_nav(eq_F, config.OUTPUT_DIR / "v9_3_nav.parquet")
    _save_nav(eq_E, config.OUTPUT_DIR / "v8_weekly_nav.parquet")

    # ---- 换手/成本统计 ----
    st_E = _turnover_stats(tr_E, slip_map)
    st_F = _turnover_stats(tr_F, slip_map)
    print(f"  [E] 换手: {st_E['n_trades']}笔/{st_E['n_reb']}期 平均{st_E['avg_interval_d']:.0f}天 "
          f"滑点成本≈{st_E['cost_pct']:.2f}%")
    print(f"  [F] 换手: {st_F['n_trades']}笔/{st_F['n_reb']}期 平均{st_F['avg_interval_d']:.0f}天 "
          f"滑点成本≈{st_F['cost_pct']:.2f}%")

    # ---- 明细 / 最新信号（F 臂）----
    holdings = []
    for t, codes_sel in sel_on.items():
        if t < pd.Timestamp("2024-01-01"):
            continue
        fset = sw_on.loc[t, "factor_set"] if t in sw_on.index else "?"
        holdings.append({"month_end": t, "factor_set": fset,
                         "n_selected": len(codes_sel),
                         "selected_codes": ",".join(codes_sel)})
    pd.DataFrame(holdings).to_csv(config.OUTPUT_DIR / "v9_3_holdings_2024_2025.csv", index=False)

    name_map = load_name_map()
    last_me = list(sel_on.keys())[-1]
    last_codes = sel_on[last_me]
    tw_last = float(tw.loc[last_me]) if last_me in tw.index else float(tw.iloc[-1])
    fset_last = sw_on.loc[last_me, "factor_set"] if last_me in sw_on.index else "?"
    rows = []
    for i, c in enumerate(last_codes, 1):
        mult = crowding_mult.get((last_me, c), 1.0)
        rows.append({"rank": i, "code": c, "name": name_map.get(c, c),
                     "factor_set": fset_last,
                     "regime_weight": round(tw_last * 100, 2),
                     "crowding_mult": mult,
                     "target_weight": round(FIXED_WEIGHT * mult * 100, 2) if tw_last > 0 else 0.0,
                     "action": "BUY" if tw_last > 0 else "HOLD"})
    latest_df = pd.DataFrame(rows)
    latest_df.to_csv(config.OUTPUT_DIR / "v9_3_latest_signal.csv", index=False)

    # ---- 消融表 ----
    def metrics_row(label, m_f, m_n):
        return {"臂": label, "全期夏普": m_f["sharpe"], "全期年化%": m_f["annual_return"] * 100,
                "全期回撤%": m_f["max_drawdown"] * 100,
                "新区间夏普": m_n["sharpe"], "新区间回撤%": m_n["max_drawdown"] * 100}

    abl = pd.DataFrame([
        metrics_row("A  V8 基线(月频)", ref_A["full"], ref_A["new"]),
        metrics_row("E  V8+周频", ref_E["full"], ref_E["new"]),
        metrics_row("F  V9.3(周频+分析师+拥挤)", m_full, m_new),
    ])
    abl.to_csv(config.OUTPUT_DIR / "v9_3_ablation.csv", index=False)
    print("\n---------------- 模块3 消融 ----------------")
    print(abl.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    me_full = ref_E["full"]["sharpe"] - ref_A["full"]["sharpe"]
    me_new = ref_E["new"]["sharpe"] - ref_A["new"]["sharpe"]
    me_dd = ref_E["full"]["max_drawdown"] * 100 - ref_A["full"]["max_drawdown"] * 100
    mf_full = m_full["sharpe"] - ref_E["full"]["sharpe"]
    mf_new = m_new["sharpe"] - ref_E["new"]["sharpe"]
    mf_dd = m_full["max_drawdown"] * 100 - ref_E["full"]["max_drawdown"] * 100

    def _rowhtml(r):
        return (f"<tr><td style='text-align:left;padding:6px'>{r['臂']}</td>"
                f"<td>{r['全期夏普']:.2f}</td><td>{r['全期年化%']:.2f}%</td>"
                f"<td>{r['全期回撤%']:.2f}%</td><td>{r['新区间夏普']:.2f}</td>"
                f"<td>{r['新区间回撤%']:.2f}%</td></tr>")

    abl_html = (
        "<table style='width:100%;border-collapse:collapse;margin:12px 0;font-size:13px'>"
        "<thead><tr style='background:rgba(127,127,127,.15)'>"
        "<th style='text-align:left;padding:6px'>臂</th><th>全期夏普</th><th>全期年化</th>"
        "<th>全期回撤</th><th>2024-25夏普</th><th>2024-25回撤</th></tr></thead><tbody>"
        + "".join(_rowhtml(r) for _, r in abl.iterrows()) + "</tbody></table>")

    conclusion = (
        f"前置裁定：模块1（分析师因子，V9.1 全期夏普 0.57→0.48）与模块2（行业拥挤度，V9.2 干净边际 "
        f"全期 -0.05 / 2024-25 +0.05、回撤仅 -0.18pp）均已判定失败，故模块3 不再叠在失败版本之上，"
        f"改为 2 臂消融：E=V8基座+仅周频（模块3 干净边际），F=周频+分析师+拥挤（用户原定 V9.3，完整记录）。"
        f"【模块3 干净边际 E vs A】全期夏普 {me_full:+.2f}"
        f"（{ref_A['full']['sharpe']:.2f}→{ref_E['full']['sharpe']:.2f}），"
        f"新区间 {me_new:+.2f}（{ref_A['new']['sharpe']:.2f}→{ref_E['new']['sharpe']:.2f}），"
        f"回撤 {me_dd:+.2f}pp。E 臂换手：{st_E['n_trades']}笔/{st_E['n_reb']}期、"
        f"平均 {st_E['avg_interval_d']:.0f} 天调仓一次，滑点成本约 {st_E['cost_pct']:.2f}%"
        f"（月频 V8 为 {st_F['n_trades']}笔级成本量级）。"
        f"【完整 V9.3 F 臂】全期夏普 {m_full['sharpe']:.2f}（年化 {m_full['annual_return']*100:.1f}%、"
        f"回撤 {m_full['max_drawdown']*100:.1f}%），相对 E 变化 {mf_full:+.2f} / {mf_new:+.2f} / "
        f"回撤 {mf_dd:+.2f}pp。裁定：若 E 相对 A 夏普提升且回撤可控 → 周频敏捷性生效，推荐保留；"
        f"若被交易成本侵蚀或夏普下降 → 模块3 失败归档，维持 V8 月频。"
    )

    modules_text = (
        "<b>本页为模块3 的 2 臂消融报告</b>（主图/主指标为 F 臂 = 用户原定 V9.3 = 周频+分析师+拥挤）。"
        + abl_html +
        "模块3（周频调仓）：调仓由月末改为<b>每周最后交易日（W-FRI 对齐实际交易日）</b>。"
        "信号逻辑不变：每周截面重算因子、打分、选 Top30；Regime 仍按<b>月度</b>判定再 ffill 到周；"
        "LightGBM 仍按<b>季度末</b>重训。持仓周期约 7 天（原约 30 天）。"
        "交易成本压力测试：分档滑点整体上浮 50%（0.10/0.30/0.50% → 0.15/0.45/0.75%，上限 1%）。<br>"
        f"实测换手：E 臂 {st_E['n_trades']} 笔成交 / {st_E['n_reb']} 个调仓日，"
        f"平均 {st_E['avg_interval_d']:.0f} 天调仓一次，估算滑点摩擦约 {st_E['cost_pct']:.2f}%"
        f"（按成交金额计）；F 臂 {st_F['n_trades']} 笔 / {st_F['n_reb']} 期，"
        f"摩擦约 {st_F['cost_pct']:.2f}%。<br>"
        "<b>前置判断（诚实披露）</b>：本策略核心 alpha 在月频因子（ret_12/ROE/毛利率同比），"
        "周频只能加快对 regime/反转信号的响应，但同时把固定成本放大约 4 倍——"
        "2021 反转失效年、2022 熊市若周频能更快离场则回撤下降，反之成本侵蚀收益。"
        "净效果由 E vs A 的实测裁定，不预设结论。")

    sw_on.reset_index().to_csv(config.OUTPUT_DIR / "v9_3_switch_log.csv", index=False)
    html = generate_html_v9(
        eq=eq_F, eq_clean=eq_F_clean, idx_eq=idx_eq,
        slip_map=slip_map, tier_counts=tier_counts,
        m_full=m_full, m_old=m_old, m_new=m_new,
        m_full_c=m_full_c, m_old_c=m_old_c, m_new_c=m_new_c,
        prev_ref=prev_ref, baseline_ref=baseline_ref, m_idx=m_idx,
        yearly=yearly, yearly_prev=yearly_prev, yearly_baseline=yearly_baseline,
        yearly_c=yearly,
        vol_regime=vol_regime, switch_log=sw_on.reset_index(),
        latest_signal=latest_df, holdings=pd.DataFrame(holdings),
        n_universe=n_universe, data_start=data_start, data_end=data_end,
        conclusion=conclusion,
        version_label="V9.3", prev_label="V8+周频",
        modules_text=modules_text)
    out_path = config.OUTPUT_DIR / "report_v9_3.html"
    out_path.write_text(html, encoding="utf-8")

    print("\n================ 模块3 裁定 ================")
    print(f"[干净边际 E-A] 夏普 {me_full:+.2f} | 新区间 {me_new:+.2f} | 回撤 {me_dd:+.2f}pp")
    print(f"[F-E 完整V9.3] 夏普 {mf_full:+.2f} | 新区间 {mf_new:+.2f} | 回撤 {mf_dd:+.2f}pp")
    print(f"[{datetime.now()}] 报告已生成: {out_path}  耗时 {datetime.now()-t0}")


if __name__ == "__main__":
    main()
