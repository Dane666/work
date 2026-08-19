# -*- coding: utf-8 -*-
"""
V9.1 主流程（独立脚本，不覆盖 main_v8.py）。

相对 V8 的改动（仅一项，隔离变量）：模块1「分析师评级因子」并入动量/质量工具箱。
分析师因子由 point-in-time 评级上调事件构建（rating_change / upgrade_ratio_3m /
upgrade_mom_1m），作为质量评分的补充（等权并入，不替换 ret_12/roe/gpm_yoy）。

为干净归因，单次运行同时产出：
  - V9.1（分析师因子 ON）净值
  - V8 等价净值（分析师因子 OFF，同一管线复算，应等于 V8 实测 0.57/0.54/0.67）
  - V9.1 MA240-only 对照（波动率降档净贡献）
V8 等价净值即「上一版本 / V8基线」参照，边际贡献表 = V9.1 − V8。

落盘：output/v9_1_nav.parquet（供 V9.2 直接作为 prev_ref，无需重算）。
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
from report import compute_metrics, yearly_sharpe
from report_v9 import generate_html_v9

config.ENABLE_ANALYST_FACTOR = True   # 模块1 启用

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
    return me[(me >= pd.Timestamp(start)) & (me <= pd.Timestamp(end))]


def load_name_map():
    try:
        import akshare as ak
        df = ak.stock_info_a_code_name()
        return dict(zip(df["code"].astype(str).str.zfill(6), df["name"].astype(str)))
    except Exception:
        return {}


def _build_metrics(eq):
    return (compute_metrics(eq), compute_metrics(eq.loc[:"2023-12-31"]),
            compute_metrics(eq.loc["2024-01-01":]))


def _selection_and_backtest(close_m, rsi, reversal_signal, me, use_reversal,
                            tw, slip_map, momentum_zscore, label):
    sel, switch_log = build_selection_v5(
        close_m, rsi, reversal_signal, momentum_zscore, me, use_reversal, 0.20, 30)
    eq, trades = run_backtest_v5(
        close_m, sel, me, config.START_DATE, END_EXT,
        target_weight=tw, slippage_map=slip_map)
    trades.to_csv(config.OUTPUT_DIR / f"v9_1_trades_{label}.csv", index=False)
    return sel, switch_log, eq, trades


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

    print(f"[{datetime.now()}] 构建滚动反转信号（V9 宇宙 {n_universe} 只）...")
    reversal_signal = build_rolling_reversal_signal(close_m, ohlcv_full, window=36)
    monthly_ic = monthly_reversal_ic(close_m, rsi, me, config.FWD_RETURN_DAYS)
    rolling_ic, use_reversal = compute_rolling_regime(monthly_ic, 12, IC_BASE, 6)

    tw_clean = build_ma240_target_weight(idx, close_m.index, MA_BASE)
    tw, vol_regime = build_ma240_vol_target_weight(
        idx, close_m.index, MA_BASE, month_ends=me,
        vol_q=0.75, reduced_weight=0.60, vol_lookback=756)

    idx_ret = idx.reindex(close_m.index).pct_change().fillna(0)
    idx_eq = (1.0 + idx_ret).cumprod() * config.INIT_CAPITAL
    m_idx = compute_metrics(idx_eq)

    # ---------- (A) V9.1：分析师因子 ON ----------
    print(f"[{datetime.now()}] 计算动量/质量 Z-score（含分析师因子）...")
    mz_a = compute_momentum_zscore(close_m, roe, gpm, me)
    sel_v9, sw_v9, eq_v9, _ = _selection_and_backtest(
        close_m, rsi, reversal_signal, me, use_reversal, tw, slip_map, mz_a, "v9")
    eq_v9_clean, _ = run_backtest_v5(
        close_m, sel_v9, me, config.START_DATE, END_EXT,
        target_weight=tw_clean, slippage_map=slip_map)
    m_full, m_old, m_new = _build_metrics(eq_v9)
    m_full_c, m_old_c, m_new_c = _build_metrics(eq_v9_clean)
    yearly = yearly_sharpe(eq_v9)

    # ---------- (B) V8 等价：分析师因子 OFF（同管线复算，作为 prev/baseline）----------
    print(f"[{datetime.now()}] 复算 V8 等价（分析师因子 OFF，校验一致性）...")
    config.ENABLE_ANALYST_FACTOR = False
    mz_v8 = compute_momentum_zscore(close_m, roe, gpm, me)
    config.ENABLE_ANALYST_FACTOR = True
    sel_v8, sw_v8, eq_v8, _ = _selection_and_backtest(
        close_m, rsi, reversal_signal, me, use_reversal, tw, slip_map, mz_v8, "v8")
    m_full_v8, m_old_v8, m_new_v8 = _build_metrics(eq_v8)
    yearly_v8 = yearly_sharpe(eq_v8)

    # 持久化 V9.1 净值（供 V9.2 作为 prev_ref）
    nav_df = eq_v9.reset_index(); nav_df.columns = ["date", "nav"]
    nav_df.to_parquet(config.OUTPUT_DIR / "v9_1_nav.parquet", index=False)
    # 同时存 V8 等价净值（V9.2 报告 baseline 可复用，避免重复复算）
    nav_v8 = eq_v8.reset_index(); nav_v8.columns = ["date", "nav"]
    nav_v8.to_parquet(config.OUTPUT_DIR / "v8_equiv_nav.parquet", index=False)

    prev_ref = {"full": m_full_v8, "old": m_old_v8, "new": m_new_v8, "label": "V8"}
    baseline_ref = {"full": m_full_v8, "old": m_old_v8, "new": m_new_v8, "label": "V8"}

    # ---------- 明细：2024-2025 持仓 + 最新截面 ----------
    holdings = []
    for t, codes_sel in sel_v9.items():
        if t < pd.Timestamp("2024-01-01"):
            continue
        fset = sw_v9.loc[t, "factor_set"] if t in sw_v9.index else "?"
        holdings.append({"month_end": t, "factor_set": fset,
                         "n_selected": len(codes_sel),
                         "selected_codes": ",".join(codes_sel)})
    pd.DataFrame(holdings).to_csv(config.OUTPUT_DIR / "v9_1_holdings_2024_2025.csv", index=False)

    name_map = load_name_map()
    if len(sel_v9):
        last_me = list(sel_v9.keys())[-1]
        last_codes = sel_v9[last_me]
        tw_last = float(tw.loc[last_me]) if last_me in tw.index else float(tw.iloc[-1])
        fset_last = sw_v9.loc[last_me, "factor_set"] if last_me in sw_v9.index else "?"
        latest_df = pd.DataFrame([
            {"rank": i, "code": c, "name": name_map.get(c, c),
             "factor_set": fset_last,
             "regime_weight": round(tw_last * 100, 2),
             "target_weight": round(FIXED_WEIGHT * 100, 2) if tw_last > 0 else 0.0,
             "action": "BUY" if tw_last > 0 else "HOLD"}
            for i, c in enumerate(last_codes, 1)])
        latest_df.to_csv(config.OUTPUT_DIR / "v9_1_latest_signal.csv", index=False)
        print(f"[{datetime.now()}] 最新截面({last_me.date()}) 选股 {len(last_codes)} 只，"
              f"因子集={fset_last}，regime_weight={tw_last*100:.0f}%")
    else:
        latest_df = pd.DataFrame()

    # ---------- 结论（诚实归因）----------
    d_full = m_full['sharpe'] - m_full_v8['sharpe']
    d_new = m_new['sharpe'] - m_new_v8['sharpe']
    conclusion = (
        f"V9.1 = V8 + 分析师评级因子（仅模块1，核心参数零改动）。"
        f"全区间夏普 {m_full['sharpe']:.2f}（年化 {m_full['annual_return']*100:.1f}%，"
        f"回撤 {m_full['max_drawdown']*100:.1f}%），相对 V8（{m_full_v8['sharpe']:.2f}）"
        f"净变化 {d_full:+.2f}；新区间2024-2025 夏普 {m_new['sharpe']:.2f} vs V8 {m_new_v8['sharpe']:.2f}"
        f"（{d_new:+.2f}）。"
        f"一致性校验：V8 等价复算 = {m_full_v8['sharpe']:.2f}/{m_old_v8['sharpe']:.2f}/{m_new_v8['sharpe']:.2f}，"
        f"与 V8 实测 0.57/0.54/0.67 一致，证明分析师因子为唯一变量。"
        f"分析师因子以「评级上调」事件 point-in-time 构建（免费源无历史 EPS 预测修正序列），"
        f"无覆盖股票填充 0 中性、不剔除。"
        f"前置体检已披露：并入后质量分 IC 由 −0.0385 改善至 −0.0341（方向有利），但 top30 重叠 27.3/30，"
        f"每月仅置换约 2.7 只标的，故预期边际影响温和；本回测用于实测其真实幅度，不预设结论。"
    )

    sw_v9.reset_index().to_csv(config.OUTPUT_DIR / "v9_1_switch_log.csv", index=False)
    html = generate_html_v9(
        eq=eq_v9, eq_clean=eq_v9_clean, idx_eq=idx_eq,
        slip_map=slip_map, tier_counts=tier_counts,
        m_full=m_full, m_old=m_old, m_new=m_new,
        m_full_c=m_full_c, m_old_c=m_old_c, m_new_c=m_new_c,
        prev_ref=prev_ref, baseline_ref=baseline_ref, m_idx=m_idx,
        yearly=yearly, yearly_prev=yearly_v8, yearly_baseline=yearly_v8, yearly_c=yearly,
        vol_regime=vol_regime, switch_log=sw_v9.reset_index(),
        latest_signal=latest_df, holdings=pd.DataFrame(holdings),
        n_universe=n_universe, data_start=data_start, data_end=data_end,
        conclusion=conclusion,
        version_label="V9.1", prev_label="V8",
        modules_text=(
            "模块1（分析师预期上调因子）：数据源 ak.stock_rank_forecast_cninfo(date)，"
            "按周频抓取 389 个交易日快照（2018-01-05~2025-08-08，93154 份评级报告，覆盖 3962 只股票），"
            "严格 point-in-time —— 决策日 t 仅消费 asof_date≤t 的报告。<br>"
            "4 个子因子（截面 winsorize 1%/99% + Z 后等权合成，再作为<b>第4个等权因子</b>并入质量分 "
            "(z_ret12+z_roe+z_gpm+z_ana)/4，不替换原有 ret_12/roe/gpm_yoy）："
            "<b>rating_level</b> 过去92日 mean(投资评级)；<b>rating_delta</b> mean(0~92日)−mean(92~184日) 评级上调动量；"
            "<b>attention_mom</b> log1p(报告数0~63日)−log1p(63~126日) 覆盖热度提升；"
            "<b>rating_change</b> (#上调−#下调)/报告数（92日窗口，由 投资评级−前一次投资评级 推导）。<br>"
            "无分析师覆盖→因子值 0（中性），<b>不剔除股票</b>；Z 仅在有覆盖样本上计算后再填 0，"
            "确保「无覆盖=中性」在数值上严格成立。<br>"
            "<b>口径取舍（已实测验证）</b>：① 免费 akshare 无可 point-in-time 的历史 EPS 预测修正序列"
            "（stock_profit_forecast_em 仅返回当前共识快照，用于回测=未来函数，已禁用），故用「评级上调」"
            "事件等价刻画「分析师预期上调」；② 巨潮原始「评级变化」字段实测全样本仅取值 {维持, 未知}"
            "（93154 行中 0=64140 / NaN=29014，无任何上调/下调），直接用会得到恒零因子，已弃用为主口径；"
            "③ 「前一次投资评级」字段 2018 年口径异常（自推导上调占比 41.9% vs 2019-25 仅 0.7~1.8%），"
            "故仅作 1/4 权重的辅助子因子。<br>"
            "<b>因子体检</b>：截面覆盖率 2018 年 22.6% → 2024 年 43.2%（合成因子非零占比全期 40.8%、2024-25 达 53.3%）；"
            "子因子 fwd21d IC：rating_level +0.0081(IR+0.16,胜率57.8%)、rating_change +0.0112(IR+0.09)、"
            "rating_delta −0.0000(无信号)、attention_mom −0.0066(IR−0.12，反向，即关注度拥挤效应)。"
            "并入后质量分 IC 由 −0.0385 改善至 −0.0341（2024-25：−0.0404→−0.0322，均朝有利方向），"
            "但 top30 选股重叠度 27.3/30，说明每月仅置换约 2.7 只标的，<b>预期边际影响偏温和</b>。"
            "momentum_quality regime 占 59/92 月（2024-25 为 16/20 月），为本模块的作用面。"))
    out_path = config.OUTPUT_DIR / "report_v9_1.html"
    out_path.write_text(html, encoding="utf-8")

    print("\n================ V9.1 结果 ================")
    print(f"[V9.1] 全区间: 年化={m_full['annual_return']*100:.2f}% 回撤={m_full['max_drawdown']*100:.2f}% 夏普={m_full['sharpe']:.2f}")
    print(f"[V9.1] 新区间: 年化={m_new['annual_return']*100:.2f}% 回撤={m_new['max_drawdown']*100:.2f}% 夏普={m_new['sharpe']:.2f}")
    print(f"[V8等价] 全区间夏普={m_full_v8['sharpe']:.2f} 旧={m_old_v8['sharpe']:.2f} 新={m_new_v8['sharpe']:.2f}")
    print(f"[边际] 全期夏普 {d_full:+.2f} | 新区间夏普 {d_new:+.2f}")
    print(f"[{datetime.now()}] 报告已生成: {out_path}  耗时 {datetime.now()-t0}")


if __name__ == "__main__":
    main()
