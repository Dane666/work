# -*- coding: utf-8 -*-
"""
因子评估 / 因子合成模块（V5：多因子 Regime 切换）。

本模块是 V5 唯一改动点（不改动 backtest.py 与 market_filter.py）。

核心逻辑：每月末依据「反转因子(neg_rsi_14) IC 的 12 个月滚动均值」决定下月使用的
因子集：
  - rolling_IC > 0.05  -> 反转因子集（RSI + 偏度 + 成交量乖离 + 历史收益，即 V3 现有因子，
                          LightGBM 重训），选股方式同 V3：RSI 最低前 20% 候选池内取 Top30。
  - rolling_IC ≤ 0.05  -> 动量/质量因子集（ret_12 + roe + gross_profit_margin_yoy，
                          三因子等权 Z-score，跨全宇宙取 Top30，不跑 LightGBM，避免小样本过拟合）。

MA240 市场过滤（站上满仓、跌破空仓）由 market_filter.build_ma240_target_weight 提供，
本模块不改其逻辑，仅负责「满仓时选什么股」。

无未来函数保证：
  - 反转 IC[t] 用到 t 月之后的下月收益；滚动均值做 shift(1)，使决策日 T 只消费 IC[T-1]，
    绝不窥探 IC[T]。
  - ret_12 / roe / gpm_yoy 均为截至决策日 T 已可得数据；财报面板已按披露时点 + 延迟映射
    做点对点前向填充（见 fetch_fundamentals_v5.py）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from factors import compute_rsi, compute_fwd_return


# ---------------------------------------------------------------------------
# 1) 反转因子 IC 滚动与 Regime 判定
# ---------------------------------------------------------------------------
def monthly_reversal_ic(close_panel: pd.DataFrame,
                        rsi_panel: pd.DataFrame,
                        month_ends,
                        fwd_days: int = 21) -> pd.Series:
    """neg_rsi_14 的月度 Rank IC 序列（月频索引）。

    IC[t] = 当月截面 (-RSI) 与 下月收益 的 Spearman 相关。
    要求截面有效股票数 >= 20 才计算，否则置 NaN。
    """
    fwd = compute_fwd_return(close_panel, fwd_days)  # 下月收益面板
    ics = {}
    for t in pd.DatetimeIndex(month_ends):
        f = (-rsi_panel.loc[t]).dropna()
        y = fwd.loc[t].dropna()
        common = f.index.intersection(y.index)
        if len(common) < 20:
            ics[t] = np.nan
            continue
        ics[t] = f[common].corr(y[common], method="spearman")
    return pd.Series(ics, name="reversal_ic").sort_index()


def compute_rolling_regime(monthly_ic: pd.Series,
                           ic_window: int = 12,
                           threshold: float = 0.05,
                           min_periods: int = 6) -> tuple[pd.Series, pd.Series]:
    """由月度反转 IC 计算滚动均值与「是否使用反转因子集」判定。

    返回 (rolling_ic, use_reversal)：
      rolling_ic   : 12 个月滚动均值，shift(1) 不含当前月（杜绝未来函数）。
      use_reversal : 布尔序列，rolling_ic > threshold 为 True（使用反转集）；
                     历史不足(min_periods) 时默认 True（反转是常态，无信号时沿用）。
    """
    rolling = monthly_ic.rolling(ic_window, min_periods=min_periods).mean().shift(1)
    use_reversal = (rolling > threshold)
    use_reversal = use_reversal.fillna(True)  # 早期无信号 -> 默认反转
    return rolling, use_reversal.astype(bool)


# ---------------------------------------------------------------------------
# 2) 动量/质量因子集：三因子等权 Z-score 合成
# ---------------------------------------------------------------------------
def compute_momentum_zscore(close_panel: pd.DataFrame,
                            roe_panel: pd.DataFrame,
                            gpm_yoy_panel: pd.DataFrame,
                            month_ends) -> pd.DataFrame:
    """在每月末计算 ret_12 + roe + gpm_yoy 的等权 Z-score 合成分（date x code）。

    仅依赖截至决策日已可得数据；截面标准化后等权求和。返回月频面板（行=month_end）。
    """
    ret12 = close_panel.pct_change(config.FWD_RETURN_DAYS * 12)  # ≈12 个月动量
    panels = {"ret_12": ret12, "roe": roe_panel, "gpm_yoy": gpm_yoy_panel}

    # ---- V9 模块1：分析师评级因子（point-in-time，零未来函数）----
    # 仅作「补充」并入质量评分，绝不替换原有 ret_12/roe/gpm_yoy 逻辑。
    ana_comp = None
    if getattr(config, "ENABLE_ANALYST_FACTOR", False):
        from analyst_factors import (load_raw_ratings, build_analyst_panels,
                                     build_analyst_composite)
        raw = load_raw_ratings()
        codes = list(close_panel.columns)
        ap = build_analyst_panels(raw, month_ends, codes)
        ana_comp = build_analyst_composite(ap, month_ends)

    rows = {}
    for t in pd.DatetimeIndex(month_ends):
        sub = pd.DataFrame({
            "ret_12": ret12.loc[t],
            "roe": roe_panel.loc[t],
            "gpm_yoy": gpm_yoy_panel.loc[t],
        })
        sub = sub.dropna()  # 三因子均有效才参与（避免单因子缺失导致偏置）
        if sub.empty:
            rows[t] = pd.Series(dtype=float)
            continue
        # 截面 winsorize（1%/99%）：财务因子存在极端离群（负净资产 ROE<-5000、
        # 毛利率同比因数据单位不一致出现 ±1000 量级），直接 Z-score 会被离群点主导。
        sub_w = sub.copy()
        for col in sub_w.columns:
            lo, hi = sub_w[col].quantile(0.01), sub_w[col].quantile(0.99)
            sub_w[col] = sub_w[col].clip(lo, hi)
        z = (sub_w - sub_w.mean()) / sub_w.std(ddof=0)
        # 补充分析师因子：作为「第4个等权因子」并入，绝不替换原 ret_12/roe/gpm_yoy。
        # 用 /4 而非直接相加，保持合成分尺度与 V8 可比（若直接加满 Z，分析师因子
        # 会占 60%+ 权重，等同于替换原逻辑，违背"补充"约束）。
        # 无分析师覆盖的股票 z_ana=0（中性），不改变其入选资格，只是不获得加分。
        if ana_comp is not None and t in ana_comp.index:
            z_ana = ana_comp.loc[t].reindex(sub_w.index).fillna(0.0)
            comp = (z["ret_12"] + z["roe"] + z["gpm_yoy"] + z_ana) / 4.0
        else:
            comp = (z["ret_12"] + z["roe"] + z["gpm_yoy"]) / 3.0  # 等权
        rows[t] = comp
    return pd.DataFrame(rows).T.sort_index()  # 行=month_end, 列=code


# ---------------------------------------------------------------------------
# 3) 选股集合构建（供 backtest_v5.run_backtest_v5 消费）
# ---------------------------------------------------------------------------
def _rsi_pool(rsi_panel, t, pool_pct: float):
    """月末 RSI 最低前 pool_pct 候选池。"""
    rsi_row = rsi_panel.loc[t].dropna()
    if len(rsi_row) < 5:
        return set(rsi_row.index)
    k = max(1, int(round(len(rsi_row) * pool_pct)))
    return set(rsi_row.nsmallest(k).index)


def build_selection_reversal_v3(close_panel: pd.DataFrame,
                                rsi_panel: pd.DataFrame,
                                reversal_signal: pd.DataFrame,
                                month_ends,
                                pool_pct: float = 0.20,
                                n_select: int = 30) -> dict:
    """V3 等价选股集合（用于引擎等价性校验）：RSI 最低前20% 候选池内取 LightGBM Top-N。

    与 backtest.run_backtest_v2（pool_pct=0.20, n_select=30）的选股逻辑完全一致。
    """
    sel = {}
    for t in pd.DatetimeIndex(month_ends):
        pool = _rsi_pool(rsi_panel, t, pool_pct)
        sig = reversal_signal.loc[t].dropna()
        ranked = sig[sig.index.isin(pool)].sort_values(ascending=False).head(n_select)
        sel[t] = [c for c in ranked.index]
    return sel


def build_selection_momentum(momentum_zscore: pd.DataFrame,
                             month_ends,
                             n_select: int = 30) -> dict:
    """纯动量/质量对照臂：全程使用动量/质量 Z-score，跨全宇宙取 Top-N（不切换）。"""
    sel = {}
    for t in pd.DatetimeIndex(month_ends):
        comp = momentum_zscore.loc[t].dropna().sort_values(ascending=False).head(n_select)
        sel[t] = [c for c in comp.index]
    return sel


def build_selection_v5(close_panel: pd.DataFrame,
                       rsi_panel: pd.DataFrame,
                       reversal_signal: pd.DataFrame,
                       momentum_zscore: pd.DataFrame,
                       month_ends,
                       use_reversal: pd.Series,
                       pool_pct: float = 0.20,
                       n_select: int = 30) -> tuple[dict, pd.DataFrame]:
    """V5 动态切换选股集合：依 use_reversal 在「反转集」与「动量/质量集」间逐月切换。

    返回 (selection_dict, switch_log)：
      selection_dict : {month_end: [codes]}
      switch_log     : 月频 DataFrame（use_reversal / factor_set / n_selected）
    """
    sel = {}
    log_rows = []
    for t in pd.DatetimeIndex(month_ends):
        if bool(use_reversal.get(t, True)):
            pool = _rsi_pool(rsi_panel, t, pool_pct)
            sig = reversal_signal.loc[t].dropna()
            ranked = sig[sig.index.isin(pool)].sort_values(ascending=False).head(n_select)
            chosen = [c for c in ranked.index]
            fset = "reversal"
        else:
            comp = momentum_zscore.loc[t].dropna().sort_values(ascending=False).head(n_select)
            chosen = [c for c in comp.index]
            fset = "momentum_quality"
        sel[t] = chosen
        log_rows.append({
            "month_end": t,
            "use_reversal": bool(use_reversal.get(t, True)),
            "factor_set": fset,
            "n_selected": len(chosen),
        })
    switch_log = pd.DataFrame(log_rows).set_index("month_end")
    return sel, switch_log


# ---------------------------------------------------------------------------
# 附录：因子 IC / IR 评估（与 V3 同宇宙，仅作诊断，非 V5 切换逻辑）
# ---------------------------------------------------------------------------
def evaluate_factors(close_panel: pd.DataFrame,
                     ohlcv: dict = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (因子统计表, IC 衰减表)。

    基于 build_factor_long 派生的全部因子计算 Rank IC / IR 与因子收益衰减。
    因子统计表列：ic_mean, ic_std, ir, ic_pos_ratio, n_months。
    IC 衰减表：行=因子，列=未来第 h 个月（1..12）的 Rank IC 均值。
    """
    from factors import build_factor_long, compute_fwd_return
    flong = build_factor_long(close_panel, ohlcv=ohlcv)
    factor_cols = [c for c in flong.columns if c not in ("date", "code")]
    horizons = list(range(1, 13))
    fwd_map = {}
    for h in horizons:
        fr = compute_fwd_return(close_panel, config.FWD_RETURN_DAYS * h)
        s = fr.stack().reset_index()
        s.columns = ["date", "code", f"fwd_{h}"]
        fwd_map[h] = s

    stats_rows = []
    decay_rows = {}
    for name in factor_cols:
        fcol = flong[["date", "code", name]]
        m1 = fcol.merge(fwd_map[1], on=["date", "code"], how="left").dropna()
        if m1.empty:
            continue
        m1["month"] = pd.to_datetime(m1["date"]).dt.to_period("M")
        icm = m1.groupby("month")[name].corr(m1["fwd_1"], method="spearman").dropna()
        if len(icm) == 0:
            continue
        ic_mean, ic_std = icm.mean(), icm.std()
        ir = ic_mean / ic_std if (ic_std and not np.isnan(ic_std)) else np.nan
        stats_rows.append({
            "factor": name, "ic_mean": ic_mean, "ic_std": ic_std, "ir": ir,
            "ic_pos_ratio": (icm > 0).mean(), "n_months": len(icm),
        })
        decay = []
        for h in horizons:
            mh = fcol.merge(fwd_map[h], on=["date", "code"], how="left").dropna()
            mh["month"] = pd.to_datetime(mh["date"]).dt.to_period("M")
            icmh = mh.groupby("month")[name].corr(mh[f"fwd_{h}"], method="spearman").dropna()
            decay.append(icmh.mean() if len(icmh) > 0 else np.nan)
        decay_rows[name] = decay

    stats = pd.DataFrame(stats_rows).set_index("factor")
    decay = pd.DataFrame(decay_rows, index=[f"fwd_{h}m" for h in horizons]).T
    return stats, decay
