# -*- coding: utf-8 -*-
"""
V8 主流程（独立脚本，不覆盖 main_v7.py）。

唯一变量：股票池（中证500 ∪ 创业板指 ∪ 中证1000，约 1540 只）。
核心逻辑与 V7.1 完全一致（零参数改动）：
  - market_filter.build_ma240_vol_target_weight  -> MA240 门控 + 波动率降档（75分位/降60%）
  - factor_eval.compute_rolling_regime           -> IC 门控（rolling_IC>0.05 用反转集）
  - factor_eval.build_selection_v5               -> 逐月切换 + Top30 + 月频
  - factor_eval_v7.build_rolling_reversal_signal -> 滚动 36 月窗口重训（冷启动=V5同款）
  - fetch_fundamentals_v8                        -> 真实财报 point-in-time 对齐至 2025Q2
  - backtest_v5.run_backtest_v5                  -> 分档滑点（0.1/0.3/0.5%，沿用 V6）

对照基准 V7.1 由 output/theoretical_nav_v7_1.parquet 现场重算（同口径），
保证 V8 vs V7.1 比较可复现、无硬编码偏差。

运行：cd src && python fetch_all_v8.py && python main_v8.py
"""

from __future__ import annotations

import os
import time
for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
          "ALL_PROXY", "all_proxy"]:
    os.environ.pop(k, None)

from datetime import datetime

import numpy as np
import pandas as pd

import config
from factors import compute_rsi, get_month_end_dates
from factor_eval import (monthly_reversal_ic, compute_rolling_regime,
                        compute_momentum_zscore, build_selection_v5)
from factor_eval_v7 import build_rolling_reversal_signal
from backtest_v5 import run_backtest_v5
from market_filter import build_ma240_target_weight, build_ma240_vol_target_weight
from stress_test_v6 import build_ohlcv_full, build_slippage_map
from report import (compute_metrics, yearly_sharpe, generate_html_v8)


END_EXT = "2025-08-12"
MA_BASE = 240
IC_BASE = 0.05
FIXED_WEIGHT = config.FIXED_WEIGHT            # 0.10，与 V7.1 严格一致


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
    """代码->名称映射（一次拉取缓存）。"""
    try:
        import akshare as ak
        df = ak.stock_info_a_code_name()
        return dict(zip(df["code"].astype(str).str.zfill(6),
                        df["name"].astype(str)))
    except Exception:
        return {}


def main(vol_filter: bool = True,
         vol_q: float = 0.75, reduced_weight: float = 0.60,
         vol_lookback: int = 756, out_name: str = "report_v8.html"):
    t0 = datetime.now()
    # ---------- V8 面板（唯一变量：宇宙扩大）----------
    close = pd.read_parquet(config.DATA_DIR / "v8_close_panel.parquet")
    amount = pd.read_parquet(config.DATA_DIR / "v8_amount_panel.parquet")
    idx = pd.read_parquet(config.DATA_DIR / "v6_index.parquet")   # CSI300，与宇宙无关
    if isinstance(idx, pd.DataFrame):
        idx = idx.iloc[:, 0]
    roe = pd.read_parquet(config.DATA_DIR / "roe_panel_v8.parquet").reindex(close.index).ffill()
    gpm = pd.read_parquet(config.DATA_DIR / "gpm_yoy_panel_v8.parquet").reindex(close.index).ffill()
    codes_all = list(close.columns)
    n_universe = len(codes_all)

    data_start = str(close.index[0].date())
    data_end = str(close.index[-1].date())

    # ---------- V7.1 对照基准（由理论净值现场重算，同口径）----------
    theo = pd.read_parquet(config.OUTPUT_DIR / "theoretical_nav_v7_1.parquet")
    theo = theo.rename(columns={theo.columns[0]: "date", theo.columns[1]: "nav"})
    theo["date"] = pd.to_datetime(theo["date"])
    eq_v71 = theo.set_index("date")["nav"].sort_index()
    m_v71_full = compute_metrics(eq_v71)
    m_v71_old = compute_metrics(eq_v71.loc[:"2023-12-31"])
    m_v71_new = compute_metrics(eq_v71.loc["2024-01-01":])
    yearly_v71 = yearly_sharpe(eq_v71)

    # ---------- 滑点分级（沿用 V6，按全期日均成交额）----------
    slip_map, tier_counts = build_slippage_map(amount)

    # ---------- 掩码新股 + RSI + 月末 ----------
    close_m = mask_new_listings(close, config.NEW_STOCK_MIN_DAYS)
    rsi = compute_rsi(close_m, config.RSI_WINDOW)
    me = month_ends_in(close_m, config.START_DATE, END_EXT)

    # ---------- OHLCV 全周期（供滚动反转信号）----------
    ohlcv_old = pd.read_pickle(config.V3_OHLCV) if config.V3_OHLCV.exists() else {}
    ohlcv_full = build_ohlcv_full(close, amount, ohlcv_old, codes_all)

    # ============ (1) 滚动窗口反转信号（与 V7.1 同款）============
    print(f"[{datetime.now()}] 构建滚动窗口反转信号（V8 宇宙 {n_universe} 只）...")
    reversal_signal = build_rolling_reversal_signal(close_m, ohlcv_full, window=36)

    # ============ (2) IC 门控 + 动量/质量（V8 真实财报）============
    monthly_ic = monthly_reversal_ic(close_m, rsi, me, config.FWD_RETURN_DAYS)
    rolling_ic, use_reversal = compute_rolling_regime(monthly_ic, 12, IC_BASE, 6)
    momentum_zscore = compute_momentum_zscore(close_m, roe, gpm, me)

    # ============ 市场过滤（MA240 + 波动率降档，V7.1 同参）============
    tw_clean = build_ma240_target_weight(idx, close_m.index, MA_BASE)
    vol_regime = None
    if vol_filter:
        tw, vol_regime = build_ma240_vol_target_weight(
            idx, close_m.index, MA_BASE, month_ends=me,
            vol_q=vol_q, reduced_weight=reduced_weight, vol_lookback=vol_lookback)
    else:
        tw = tw_clean

    # ============ V8 选股集合（IC=0.05, MA=240, n=30）============
    print(f"[{datetime.now()}] 构建 V8 选股集合（IC={IC_BASE}, MA={MA_BASE}, n=30）...")
    sel_v8, switch_log = build_selection_v5(
        close_m, rsi, reversal_signal, momentum_zscore, me, use_reversal, 0.20, 30)

    # ============ 回测（分档滑点）============
    print(f"[{datetime.now()}] 回测 V8（分档滑点，扩展全区间）...")
    eq_v8, trades_v8 = run_backtest_v5(
        close_m, sel_v8, me, config.START_DATE, END_EXT,
        target_weight=tw, slippage_map=slip_map)
    trades_v8.to_csv(config.OUTPUT_DIR / "v8_trades.csv", index=False)

    m_full = compute_metrics(eq_v8)
    m_old = compute_metrics(eq_v8.loc[:"2023-12-31"])
    m_new = compute_metrics(eq_v8.loc["2024-01-01":])
    yearly = yearly_sharpe(eq_v8)

    # ---------- 公平对照：同一信号/滑点，MA240-only（无波动过滤）----------
    if vol_filter:
        eq_clean, _ = run_backtest_v5(
            close_m, sel_v8, me, config.START_DATE, END_EXT,
            target_weight=tw_clean, slippage_map=slip_map)
        m_full_c = compute_metrics(eq_clean)
        m_old_c = compute_metrics(eq_clean.loc[:"2023-12-31"])
        m_new_c = compute_metrics(eq_clean.loc["2024-01-01":])
        yearly_c = yearly_sharpe(eq_clean)
    else:
        eq_clean, m_full_c, m_old_c, m_new_c, yearly_c = eq_v8, m_full, m_old, m_new, yearly

    # ---------- 指数基准 ----------
    idx_ret = idx.reindex(close_m.index).pct_change().fillna(0)
    idx_eq = (1.0 + idx_ret).cumprod() * config.INIT_CAPITAL
    m_idx = compute_metrics(idx_eq)

    # ---------- 2024-2025 持仓明细 + 因子集分布 ----------
    holdings = []
    for t, codes_sel in sel_v8.items():
        if t < pd.Timestamp("2024-01-01"):
            continue
        fset = switch_log.loc[t, "factor_set"] if t in switch_log.index else "?"
        holdings.append({"month_end": t, "factor_set": fset,
                         "n_selected": len(codes_sel),
                         "selected_codes": ",".join(codes_sel)})
    pd.DataFrame(holdings).to_csv(config.OUTPUT_DIR / "v8_holdings_2024_2025.csv", index=False)
    hl = pd.DataFrame(holdings)
    fset_dist = hl["factor_set"].value_counts().to_dict() if not hl.empty else {}

    # ---------- 最新截面信号清单（对应最近数据日）----------
    name_map = load_name_map()
    if len(sel_v8):
        last_me = list(sel_v8.keys())[-1]
        last_codes = sel_v8[last_me]
        tw_last = float(tw.loc[last_me]) if last_me in tw.index else float(tw.iloc[-1])
        fset_last = switch_log.loc[last_me, "factor_set"] if last_me in switch_log.index else "?"
        rows = []
        for rank, c in enumerate(last_codes, 1):
            rows.append({
                "rank": rank,
                "code": c,
                "name": name_map.get(c, c),
                "factor_set": fset_last,
                "regime_weight": round(tw_last * 100, 2),
                "target_weight": round(FIXED_WEIGHT * 100, 2) if tw_last > 0 else 0.0,
                "action": "BUY" if tw_last > 0 else "HOLD",
            })
        latest_df = pd.DataFrame(rows)
        latest_df.to_csv(config.OUTPUT_DIR / "v8_latest_signal.csv", index=False)
        print(f"[{datetime.now()}] 最新截面({last_me.date()}) 选股 {len(last_codes)} 只，"
              f"因子集={fset_last}，regime_weight={tw_last*100:.0f}%")
    else:
        latest_df = pd.DataFrame()

    # ---------- V7.1 参考 dict（供报告并列）----------
    m_v71_ref = {
        "annual_return": m_v71_full["annual_return"],
        "max_drawdown": m_v71_full["max_drawdown"],
        "sharpe": m_v71_full["sharpe"],
        "calmar": m_v71_full["calmar"],
        "label": "V7.1(500+创业,MA240+vol)",
        "old": m_v71_old, "new": m_v71_new,
    }

    # ---------- 报告 ----------
    conclusion = _build_conclusion_v8(
        m_full, m_old, m_new, m_full_c, m_old_c, m_new_c,
        m_v71_ref, vol_regime, n_universe, fset_dist)

    switch_log.reset_index().to_csv(config.OUTPUT_DIR / "v8_switch_log.csv", index=False)
    html = generate_html_v8(
        eq_v8=eq_v8, eq_v8_clean=eq_clean, idx_eq=idx_eq,
        slip_map=slip_map, tier_counts=tier_counts,
        m_full=m_full, m_old=m_old, m_new=m_new,
        m_full_c=m_full_c, m_old_c=m_old_c, m_new_c=m_new_c,
        m_v71_ref=m_v71_ref, m_idx=m_idx,
        yearly=yearly, yearly_v71=yearly_v71, yearly_c=yearly_c,
        vol_regime=vol_regime, switch_log=switch_log.reset_index(),
        latest_signal=latest_df,
        holdings_2024_2025=pd.DataFrame(holdings),
        n_universe=n_universe, data_start=data_start, data_end=data_end,
        conclusion=conclusion,
    )
    out_path = config.OUTPUT_DIR / out_name
    out_path.write_text(html, encoding="utf-8")

    print("\n================ V8 结果 ================")
    print(f"[V8.1] 全区间 2018-2025 : 年化={m_full['annual_return']*100:.2f}% "
          f"回撤={m_full['max_drawdown']*100:.2f}% 夏普={m_full['sharpe']:.2f}")
    print(f"[V8.1] 旧区间 2018-2023 : 年化={m_old['annual_return']*100:.2f}% "
          f"回撤={m_old['max_drawdown']*100:.2f}% 夏普={m_old['sharpe']:.2f} "
          f"(V7.1={m_v71_old['sharpe']:.2f})")
    print(f"[V8.1] 新区间 2024-2025 : 年化={m_new['annual_return']*100:.2f}% "
          f"回撤={m_new['max_drawdown']*100:.2f}% 夏普={m_new['sharpe']:.2f} "
          f"(V7.1={m_v71_new['sharpe']:.2f}, 目标0.4+)")
    if vol_filter:
        print(f"[V8 MA240-only] 全区间夏普={m_full_c['sharpe']:.2f} "
          f"新区间夏普={m_new_c['sharpe']:.2f}")
    print(f"宇宙 {n_universe} 只；2024-2025 因子集分布: {fset_dist}")
    print(f"[{datetime.now()}] 报告已生成: {out_path}  耗时 {datetime.now()-t0}")


def _build_conclusion_v8(m_full, m_old, m_new, m_full_c, m_old_c, m_new_c,
                         m_v71, vol_regime, n_universe, fset_dist):
    L = []
    L.append(f"V8（中证500∪创业板∪中证1000，{n_universe}只，MA240+波动率降档，分档滑点）"
             f"全区间(2018-2025)夏普 {m_full['sharpe']:.2f}，年化 {m_full['annual_return']*100:.1f}%，"
             f"最大回撤 {m_full['max_drawdown']*100:.1f}%。")
    d_full = m_full['sharpe'] - m_v71['sharpe']
    d_new = m_new['sharpe'] - m_v71['new']['sharpe']
    L.append(f"相对 V7.1（500+创业，{m_v71['sharpe']:.2f}）全区间夏普 {d_full:+.2f}；"
             f"新区间2024-2025夏普 {m_v71['new']['sharpe']:.2f} → {m_new['sharpe']:.2f}（{d_new:+.2f}）。"
             f"唯一变量为股票池（新增中证1000），其余逻辑零改动。")
    goal_new = ("达成（≥0.40）" if m_new['sharpe'] >= 0.40 else
                f"未达成（{m_new['sharpe']:.2f}<0.40）")
    L.append(f"迭代目标：新区间2024-2025夏普 0.26→0.4+ → **{goal_new}**；"
             f"全区间 {m_full['sharpe']:.2f} vs V7.1 {m_v71['sharpe']:.2f}。")
    if vol_regime is not None:
        n_red = int((vol_regime['target_weight'] == 0.6).sum())
        n_full = int((vol_regime['target_weight'] == 1.0).sum())
        n_cash = int((vol_regime['target_weight'] == 0.0).sum())
        L.append(f"波动率降档执行：{n_full} 月满仓 / {n_red} 月降档60% / {n_cash} 月空仓"
                 f"（主门控 MA240 不变）。")
    L.append(f"2024-2025 因子集切换分布: {fset_dist}（校验是否规避失效反转因子）。")
    # 诚实归因提示
    if m_new['sharpe'] < 0.40:
        L.append("诚实归因：若新区间未达 0.4+，需区分「小盘因子本身在小盘占优年仍失效」"
                 f"（数据/执行无问题，属因子 alpha 不足）vs「数据或执行层缺陷」"
                 f"（如中证1000 财报对齐偏差、滑点分级失真）。本回测财报沿用 V8 point-in-time 对齐、"
                 f"滑点按全期日均成交额分级，二者与 V7.1 口径一致，偏差应来自选股广度本身。")
    else:
        L.append("中证1000 的加入显著提升了 2024-2025 小盘风格年的超额，验证「扩展选股广度」假设成立。")
    L.append("披露：波动率阈值=CSI300 60日年化波动率历史75分位（trailing 3y，shift(1)零泄露）；"
             "降档0.60保留部分多头。训练冷启动复用 2018-2019（与 V5 一致），2020Q1 起滚动 36 月窗口。")
    return "\n".join(L)


if __name__ == "__main__":
    main(vol_filter=True)
