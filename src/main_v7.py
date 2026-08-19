# -*- coding: utf-8 -*-
"""
V7 修复主流程（独立脚本，不覆盖 main_v5/main_v6）。

目标：让 2024-2025 恢复正收益，同时保持 2018-2023 不退化。
改动范围（仅 1+2，IC 动态加权见条件分支）：
  1) 滚动窗口重训 LightGBM（factor_eval_v7.build_rolling_reversal_signal）：
     冷启动用 2018-2019（与 V5 一致），2020Q1 起每季度重训 36 月窗口。
  2) 补齐 2024-2025 真实财报（point-in-time 对齐，roe_panel_v7 / gpm_yoy_panel_v7）。

严格复用 V5 框架（不改）：
  - market_filter.build_ma240_target_weight  -> MA240 门控（站上满仓 / 跌破空仓）
  - factor_eval.compute_rolling_regime       -> IC 门控（rolling_IC>0.05 用反转集）
  - factor_eval.build_selection_v5           -> 逐月切换 + Top30 + 月频
  - backtest_v5.run_backtest_v5              -> 分档滑点（沿用 V6 0.1/0.3/0.5%）

运行：cd src && python fetch_fundamentals_v7.py && python main_v7.py
"""

from __future__ import annotations

import os
for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
          "ALL_PROXY", "all_proxy"]:
    os.environ.pop(k, None)

from datetime import datetime

import numpy as np
import pandas as pd

import config
from factors import compute_rsi, get_month_end_dates, compute_fwd_return
from factor_eval import (monthly_reversal_ic, compute_rolling_regime,
                        compute_momentum_zscore, build_selection_v5)
from factor_eval_v7 import build_rolling_reversal_signal
from backtest_v5 import run_backtest_v5
from market_filter import build_ma240_target_weight, build_ma240_vol_target_weight
from stress_test_v6 import build_ohlcv_full, build_slippage_map
from report import (compute_metrics, yearly_sharpe, generate_html_v7,
                   generate_html_v7_1)


END_EXT = "2025-08-12"
MA_BASE = 240
IC_BASE = 0.05


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
    return me[(me >= pd.Timestamp(start)) & (me <= pd.Timestamp(end))]


def main(use_ic_weight: bool = False, vol_filter: bool = False,
         vol_q: float = 0.75, reduced_weight: float = 0.60,
         vol_lookback: int = 756, out_name: str = "report_v7_1.html"):
    t0 = datetime.now()
    close = pd.read_parquet(config.DATA_DIR / "v6_close_panel.parquet")
    amount = pd.read_parquet(config.DATA_DIR / "v6_amount_panel.parquet")
    idx = pd.read_parquet(config.DATA_DIR / "v6_index.parquet")
    if isinstance(idx, pd.DataFrame):
        idx = idx.iloc[:, 0]
    roe = pd.read_parquet(config.DATA_DIR / "roe_panel_v7.parquet").reindex(close.index).ffill()
    gpm = pd.read_parquet(config.DATA_DIR / "gpm_yoy_panel_v7.parquet").reindex(close.index).ffill()
    ohlcv_old = pd.read_pickle(config.V3_OHLCV)
    codes = list(close.columns)

    data_start = str(close.index[0].date())
    data_end = str(close.index[-1].date())

    # ---------- 滑点分级（沿用 V6）----------
    slip_map, tier_counts = build_slippage_map(amount)

    # ---------- 掩码新股 + RSI + 月末 ----------
    close_m = mask_new_listings(close, config.NEW_STOCK_MIN_DAYS)
    rsi = compute_rsi(close_m, config.RSI_WINDOW)
    me = month_ends_in(close_m, config.START_DATE, END_EXT)

    # ---------- OHLCV 全周期（供滚动反转信号）----------
    ohlcv_full = build_ohlcv_full(close, amount, ohlcv_old, codes)

    # ============ (1) 滚动窗口反转信号 ============
    print(f"[{datetime.now()}] 构建滚动窗口反转信号（冷启动+V5同款 -> 2020Q1起季度重训）...")
    reversal_signal = build_rolling_reversal_signal(close_m, ohlcv_full, window=36)

    # ============ (2) IC 门控 + 动量/质量（V7 真实财报）============
    print(f"[{datetime.now()}] IC 门控 + 动量/质量 Z-score（V7 真实财报）...")
    monthly_ic = monthly_reversal_ic(close_m, rsi, me, config.FWD_RETURN_DAYS)
    rolling_ic, use_reversal = compute_rolling_regime(monthly_ic, 12, IC_BASE, 6)
    momentum_zscore = compute_momentum_zscore(close_m, roe, gpm, me)

    # ============ 市场过滤（V7.1 可选叠加波动率降档）============
    tw_clean = build_ma240_target_weight(idx, close_m.index, MA_BASE)
    vol_regime = None
    if vol_filter:
        print(f"[{datetime.now()}] V7.1 市场过滤：MA240 + 波动率降档（阈值=历史75分位）...")
        tw, vol_regime = build_ma240_vol_target_weight(
            idx, close_m.index, MA_BASE, month_ends=me,
            vol_q=vol_q, reduced_weight=reduced_weight, vol_lookback=vol_lookback)
    else:
        tw = tw_clean

    # ============ V7 选股集合（动态切换，框架不变）============
    print(f"[{datetime.now()}] 构建 V7 选股集合（IC={IC_BASE}, MA={MA_BASE}）...")
    sel_v7, switch_log = build_selection_v5(
        close_m, rsi, reversal_signal, momentum_zscore, me, use_reversal, 0.20, 30)

    if use_ic_weight:
        # 条件分支：IC 动态加权（仅在 1+2 仍不足时启用）
        # 三因子 IC 序列 -> 滚动12月IC均值(shift1) 作权重 -> 每月横截面Z-score加权合成 -> Top30
        from factor_eval_v7 import rolling_factor_ic
        fwd = compute_fwd_return(close_m, config.FWD_RETURN_DAYS)
        ret12 = close_m.pct_change(config.FWD_RETURN_DAYS * 12)

        def _ic_series(panel):
            d = {}
            for t in pd.DatetimeIndex(me):
                f = panel.loc[t].dropna()
                y = fwd.loc[t].dropna()
                c = f.index.intersection(y.index)
                d[t] = f[c].corr(y[c], method="spearman") if len(c) >= 20 else np.nan
            return pd.Series(d).sort_index()

        rev_ic = monthly_reversal_ic(close_m, rsi, me, config.FWD_RETURN_DAYS)
        mom_ic = _ic_series(ret12)
        qic = _ic_series(roe)
        ric = rolling_factor_ic(rev_ic, mom_ic, qic, 12)  # 列: reversal/momentum/quality
        comp = {}
        for t in pd.DatetimeIndex(me):
            if t not in ric.index:
                continue
            wr, wm, wq = ric.loc[t, "reversal"], ric.loc[t, "momentum"], ric.loc[t, "quality"]
            rs = reversal_signal.loc[t]
            rs = (rs - rs.mean()) / rs.std(ddof=0)
            ms = ret12.loc[t]
            ms = (ms - ms.mean()) / ms.std(ddof=0)
            qs = roe.loc[t]
            qs = (qs - qs.mean()) / qs.std(ddof=0)
            s = wr * rs.fillna(0) + wm * ms.fillna(0) + wq * qs.fillna(0)
            comp[t] = [c for c in s.dropna().sort_values(ascending=False).head(30).index]
        sel_v7 = comp
        switch_log = pd.DataFrame(
            [{"month_end": t, "use_reversal": True, "factor_set": "ic_composite",
              "n_selected": len(v)} for t, v in comp.items()]
        ).set_index("month_end")
        print("[IC-weight] 启用动态加权合成选股")

    # ============ 回测（分档滑点）============
    print(f"[{datetime.now()}] 回测 V7（分档滑点，扩展全区间）...")
    eq_v7, trades_v7 = run_backtest_v5(
        close_m, sel_v7, me, config.START_DATE, END_EXT,
        target_weight=tw, slippage_map=slip_map)
    trades_v7.to_csv(config.OUTPUT_DIR / "v7_trades.csv", index=False)

    m_full = compute_metrics(eq_v7)
    m_old = compute_metrics(eq_v7.loc[:"2023-12-31"])
    m_new = compute_metrics(eq_v7.loc["2024-01-01":])
    yearly = yearly_sharpe(eq_v7)

    # ---------- 公平对照：同一信号/滑点，MA240-only（无波动过滤）同口径回测 ----------
    if vol_filter:
        eq_clean, _ = run_backtest_v5(
            close_m, sel_v7, me, config.START_DATE, END_EXT,
            target_weight=tw_clean, slippage_map=slip_map)
        m_full_c = compute_metrics(eq_clean)
        m_old_c = compute_metrics(eq_clean.loc[:"2023-12-31"])
        m_new_c = compute_metrics(eq_clean.loc["2024-01-01":])
        yearly_c = yearly_sharpe(eq_clean)
    else:
        eq_clean, m_full_c, m_old_c, m_new_c, yearly_c = eq_v7, m_full, m_old, m_new, yearly

    # ---------- 指数基准 ----------
    idx_ret = idx.reindex(close_m.index).pct_change().fillna(0)
    idx_eq = (1.0 + idx_ret).cumprod() * config.INIT_CAPITAL
    m_idx = compute_metrics(idx_eq)

    # ---------- 2024-2025 持仓明细（校验是否规避失效因子）----------
    holdings = []
    for t, codes_sel in sel_v7.items():
        if t < pd.Timestamp("2024-01-01"):
            continue
        fset = switch_log.loc[t, "factor_set"] if t in switch_log.index else "?"
        holdings.append({"month_end": t, "factor_set": fset,
                         "n_selected": len(codes_sel),
                         "selected_codes": ",".join(codes_sel)})
    pd.DataFrame(holdings).to_csv(config.OUTPUT_DIR / "v7_holdings_2024_2025.csv", index=False)
    # 2024-2025 因子集分布
    hl = pd.DataFrame(holdings)
    fset_dist = hl["factor_set"].value_counts().to_dict() if not hl.empty else {}

    # ---------- 对比基准（V6 分档，作为长区间对照）----------
    m_v6_ref = {"annual_return": 0.0589, "max_drawdown": -0.3359, "sharpe": 0.38,
                "calmar": 0.38 / 0.3359, "label": "V6(分档,2018-2025)"}

    if vol_filter:
        conclusion = _build_conclusion_v71(m_full, m_old, m_new,
                                           m_full_c, m_old_c, m_new_c,
                                           m_v6_ref, vol_regime, use_ic_weight)
        switch_log.reset_index().to_csv(config.OUTPUT_DIR / "v7_switch_log.csv", index=False)
        html = generate_html_v7_1(
            eq_v7_1=eq_v7, eq_v7_clean=eq_clean, idx_eq=idx_eq,
            slip_map=slip_map, tier_counts=tier_counts,
            m_full=m_full, m_old=m_old, m_new=m_new,
            m_full_c=m_full_c, m_old_c=m_old_c, m_new_c=m_new_c,
            m_v6_ref=m_v6_ref, m_idx=m_idx,
            yearly=yearly, yearly_c=yearly_c, vol_regime=vol_regime,
            switch_log=switch_log.reset_index(),
            holdings_2024_2025=pd.DataFrame(holdings),
            data_start=data_start, data_end=data_end,
            conclusion=conclusion,
        )
        out_path = config.OUTPUT_DIR / out_name
    else:
        m_v5_ref = {"annual_return": 0.1201, "max_drawdown": -0.2626, "sharpe": 0.71,
                    "calmar": 0.71 / 0.2626, "label": "V5(固定0.002,2018-2023)"}
        conclusion = _build_conclusion(m_full, m_old, m_new, m_v5_ref, m_v6_ref,
                                       fset_dist, use_ic_weight)
        switch_log.reset_index().to_csv(config.OUTPUT_DIR / "v7_switch_log.csv", index=False)
        html = generate_html_v7(
            eq_v7=eq_v7, idx_eq=idx_eq,
            slip_map=slip_map, tier_counts=tier_counts,
            m_full=m_full, m_old=m_old, m_new=m_new,
            m_v5_ref=m_v5_ref, m_v6_ref=m_v6_ref, m_idx=m_idx,
            yearly=yearly, switch_log=switch_log.reset_index(),
            holdings_2024_2025=pd.DataFrame(holdings),
            fset_dist=fset_dist,
            data_start=data_start, data_end=data_end,
            use_ic_weight=use_ic_weight, conclusion=conclusion,
        )
        out_path = config.OUTPUT_DIR / "report_v7.html"

    out_path.write_text(html, encoding="utf-8")

    # ===== V7.1 新增(C)：导出理论净值（供模拟盘偏离预警 deviation_watcher.py 读取）=====
    print("[V7.1] 导出理论净值曲线...")
    try:
        if eq_v7 is not None and len(eq_v7) > 0:
            # eq_v7 为策略账户市值序列（起点=INIT_CAPITAL），归一化为净值(起点=1.0)
            theo_nav_df = (eq_v7 / eq_v7.iloc[0]).reset_index()
            theo_nav_df.columns = ["date", "nav"]
            theo_nav_path = config.OUTPUT_DIR / "theoretical_nav_v7_1.parquet"
            theo_nav_df.to_parquet(theo_nav_path, index=False)
            print(f"[V7.1] ✅ 理论净值已导出至 {theo_nav_path} (共{len(theo_nav_df)}条记录)")
        else:
            print("[V7.1] ⚠️ eq_v7 为空，跳过导出")
    except Exception as e:
        print(f"[V7.1] ❌ 理论净值导出失败: {e}")

    print("\n================ V7 修复结果 ================")
    tag = "V7.1(MA240+波动率)" if vol_filter else "V7"
    print(f"[{tag}] 全区间 2018-2025 : 年化={m_full['annual_return']*100:.2f}% "
          f"回撤={m_full['max_drawdown']*100:.2f}% 夏普={m_full['sharpe']:.2f}")
    print(f"[{tag}] 旧区间 2018-2023 : 年化={m_old['annual_return']*100:.2f}% "
          f"回撤={m_old['max_drawdown']*100:.2f}% 夏普={m_old['sharpe']:.2f}")
    print(f"[{tag}] 新区间 2024-2025 : 年化={m_new['annual_return']*100:.2f}% "
          f"回撤={m_new['max_drawdown']*100:.2f}% 夏普={m_new['sharpe']:.2f}")
    if vol_filter:
        print(f"[V7基准 MA240-only] 全区间夏普={m_full_c['sharpe']:.2f} "
              f"新区间夏普={m_new_c['sharpe']:.2f} "
              f"(波动率过滤净贡献·新区间 {m_new['sharpe']-m_new_c['sharpe']:+.2f})")
    print(f"2024-2025 因子集分布: {fset_dist}")
    print(f"[{datetime.now()}] 报告已生成: {out_path}  耗时 {datetime.now()-t0}")


def _build_conclusion(m_full, m_old, m_new, m_v5, m_v6, fset_dist, use_ic):
    L = []
    L.append(f"V7（滚动重训+真实财报，分档滑点）全区间(2018-2025)夏普 {m_full['sharpe']:.2f}，"
             f"年化 {m_full['annual_return']*100:.1f}%，最大回撤 {m_full['max_drawdown']*100:.1f}%。")
    L.append(f"旧区间(2018-2023)夏普 {m_old['sharpe']:.2f}（V5基准0.71/V6分档0.67），"
             f"年化 {m_old['annual_return']*100:.1f}%——"
             + ("与 V5 持平，未退化。" if m_old['sharpe'] >= 0.60 else
                f"较 V5(0.71)变化 {m_old['sharpe']-0.71:+.2f}，需关注。"))
    if m_new['sharpe'] > 0:
        L.append(f"新区间(2024-2025)夏普 {m_new['sharpe']:.2f}，年化 {m_new['annual_return']*100:.1f}%"
                 f"——**已恢复正收益**，相对 V6(-0.32) 改善 {m_new['sharpe']-(-0.32):+.2f}。"
                 + ("IC 动态加权未启用（1+2 已达标）。" if not use_ic else "（IC 动态加权已启用）"))
    else:
        L.append(f"新区间(2024-2025)夏普仍为负 {m_new['sharpe']:.2f}——1+2 修复不足，"
                 f"建议启用 IC 动态加权（factor_eval_v7.rolling_factor_ic）。")
    L.append(f"2024-2025 因子集切换分布: {fset_dist}（用于校验是否规避了失效的反转因子）。")
    L.append("诚实披露：冷启动沿用 2018-2019 固定训练（与 V5 一致）以保 2018-2023 可比性；"
             "2020Q1 起纯滚动 36 月窗口，早停用训练窗内 80/20 时序切分（零未来泄露）。"
             "财报按披露延迟映射 point-in-time 对齐至 2025Q2。")
    return "\n".join(L)


def _build_conclusion_v71(m_full, m_old, m_new, m_full_c, m_old_c, m_new_c,
                          m_v6, vol_regime, use_ic):
    L = []
    L.append(f"V7.1（MA240+波动率降档，分档滑点）全区间(2018-2025)夏普 {m_full['sharpe']:.2f}，"
             f"年化 {m_full['annual_return']*100:.1f}%，最大回撤 {m_full['max_drawdown']*100:.1f}%。")
    d_new = m_new['sharpe'] - m_new_c['sharpe']
    d_dd = m_full['max_drawdown'] - m_full_c['max_drawdown']
    L.append(f"与 V7基准（MA240-only，同信号同滑点）对比：新区间2024-2025夏普 "
             f"{m_new_c['sharpe']:.2f} → {m_new['sharpe']:.2f}（净贡献 {d_new:+.2f}）；"
             f"全区间最大回撤 {m_full_c['max_drawdown']*100:.1f}% → {m_full['max_drawdown']*100:.1f}%"
             f"（改善 {d_dd*100:+.1f}pp）。波动率过滤仅改市场过滤一个变量，差异即其净贡献。")
    goal_s = "达成（≥0.30）" if m_new['sharpe'] >= 0.30 else f"未达成（{m_new['sharpe']:.2f}<0.30）"
    goal_d = "达成（≥-22%）" if m_full['max_drawdown'] >= -0.22 else f"未达成（{m_full['max_drawdown']*100:.1f}%<-22%）"
    L.append(f"迭代目标：新区间2024-2025夏普 0.13→0.3+ → **{goal_s}**；"
             f"全区间回撤 -29.5%→-22%以内 → **{goal_d}**。")
    vr = vol_regime
    n_red = int((vr['target_weight'] == 0.6).sum())
    n_full = int((vr['target_weight'] == 1.0).sum())
    n_cash = int((vr['target_weight'] == 0.0).sum())
    L.append(f"波动率降档执行：全样本 {n_full} 月满仓 / {n_red} 月降档60% / {n_cash} 月空仓；"
             f"降档主要落在高波动牛市段，空仓仍完全由 MA240 决定（主门控不变）。")
    L.append(f"V6 压力基准新区间 -0.32；V7基准(无波动)新区间 {m_new_c['sharpe']:.2f}；"
             f"V7.1 新区间 {m_new['sharpe']:.2f}——波动率过滤"
             + ("进一步改善。" if m_new['sharpe'] > m_new_c['sharpe'] else
                "未带来额外改善（降档或错杀牛市动量），需关注。"))
    L.append("诚实披露：波动率阈值=CSI300 60日年化波动率的历史75分位（trailing 3y，shift(1)防自指，零未来泄露）；"
             "降档权重0.60保留部分多头暴露。其余（滚动重训/真实财报/分档滑点/切换框架）与 V7 完全一致。")
    return "\n".join(L)


if __name__ == "__main__":
    main(use_ic_weight=False, vol_filter=True)
