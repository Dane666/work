# -*- coding: utf-8 -*-
"""
V4 市场状态过滤器模块（独立承载市场状态判定逻辑）。

在 V3 的 MA240 趋势门控基础上，新增「反转因子 IC 滚动均值」门控，
实现更精细的仓位管理：

判定逻辑（每月末）：
  1) 计算全市场 RSI(14) 截面标准差（cross_sectional_std）作为恐慌/分歧度（诊断量）。
  2) 计算 neg_rsi_14（反转因子）过去 12 个月 IC 滚动均值（反转因子当前是否有效）。
  3) 规则：
       MA240 站上 且 rolling_IC > 0  -> 仓位 100%
       MA240 站上 且 rolling_IC <= 0 -> 仓位 50%（半仓，剩余持有现金）
       MA240 跌破                    -> 仓位 0%（空仓）

输出：
  - regime_df        : 月频 DataFrame（ma_above / rsi_dispersion / rolling_ic / target_weight）
  - target_weight    : 日频 Series（0 / 0.5 / 1.0），月末调仓按月末值定档，
                       盘中若指数跌破 MA240 立即归 0（硬空仓）。

无未来函数保证：
  - 月度 IC[t] 用到 t 月之后的下月收益；生成 target_weight 时对滚动 IC 做 shift(1)，
    使决策日 T 只消费到 IC[T-1]，绝不窥探 IC[T]（其需未来收益）。
  - MA240 使用日频真实收盘，与 V3 build_regime 完全一致，隔离「过滤器」这一唯一变量。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from factors import compute_rsi, compute_fwd_return


def compute_monthly_reversal_ic(close_panel: pd.DataFrame,
                                 rsi_panel: pd.DataFrame,
                                 month_ends) -> pd.Series:
    """neg_rsi_14 的月度 Rank IC 序列（月频索引）。

    IC[t] = 当月截面 RSI 反转因子(-RSI) 与 下月收益 的 Spearman 相关。
    要求截面有效股票数 >= 20 才计算，否则置 NaN。
    """
    fwd = compute_fwd_return(close_panel, config.FWD_RETURN_DAYS)  # 下月收益面板
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


def compute_rsi_dispersion(rsi_panel: pd.DataFrame, month_ends) -> pd.Series:
    """每月末全市场 RSI(14) 截面标准差（离散度）。

    离散度越高，说明个股超买/超卖分化越大（常对应系统性恐慌或极端行情）。
    """
    rows = {}
    for t in pd.DatetimeIndex(month_ends):
        r = rsi_panel.loc[t].dropna()
        rows[t] = r.std() if len(r) >= 10 else np.nan
    return pd.Series(rows, name="rsi_dispersion").sort_index()


def compute_ma240_daily(index_series: pd.Series,
                         daily_index: pd.DatetimeIndex,
                         window: int = 240):
    """日频 MA240 状态（指数 > 均线）。与 V3 build_regime 完全一致以隔离改动。"""
    idx = index_series.reindex(daily_index).ffill()
    ma = idx.rolling(window, min_periods=window // 2).mean()
    ma_above = (idx > ma).fillna(True)
    return ma_above, idx


def compute_market_regime_v4(close_panel: pd.DataFrame,
                              index_series: pd.Series,
                              rsi_panel: pd.DataFrame,
                              month_ends,
                              ma_window: int = 240,
                              ic_window: int = 12) -> tuple[pd.DataFrame, pd.Series]:
    """生成 V4 市场状态序列。

    返回 (regime_df, target_weight_daily)：
      regime_df        : 月频，列 ma_above / rsi_dispersion / rolling_ic / target_weight
      target_weight_daily: 日频 0/0.5/1.0（月末调仓定档，盘中破 MA240 归 0）
    """
    daily_index = close_panel.index
    ma_above_daily, _ = compute_ma240_daily(index_series, daily_index, ma_window)

    ic = compute_monthly_reversal_ic(close_panel, rsi_panel, month_ends)
    disp = compute_rsi_dispersion(rsi_panel, month_ends)

    # 滚动 12 个月 IC 均值；shift(1) 确保不含当前月，杜绝未来函数
    rolling_ic = ic.rolling(ic_window, min_periods=ic_window // 2).mean().shift(1)

    ma_above_me = ma_above_daily.reindex(pd.DatetimeIndex(month_ends)).ffill()

    rows = []
    for t in pd.DatetimeIndex(month_ends):
        up = bool(ma_above_me.get(t, True))
        ric = rolling_ic.get(t, np.nan)
        if not up:
            w = 0.0
        elif (ric is not None) and (not np.isnan(ric)) and ric > 0:
            w = 1.0
        else:
            w = 0.5
        rows.append({
            "ma_above": up,
            "rsi_dispersion": disp.get(t, np.nan),
            "rolling_ic": ric,
            "target_weight": w,
        })
    regime_df = pd.DataFrame(rows, index=pd.DatetimeIndex(month_ends))
    regime_df.index.name = "month_end"

    # 日频 target_weight：月末值前向填充；盘中 MA240 跌破则归 0
    tw_daily = regime_df["target_weight"].reindex(daily_index).ffill()
    tw_daily = tw_daily.where(ma_above_daily, 0.0)  # 盘中破均线 -> 空仓
    tw_daily = tw_daily.fillna(1.0)                 # 极早期无信号默认满仓
    return regime_df, tw_daily


def build_ma240_target_weight(index_series: pd.Series,
                              daily_index: pd.DatetimeIndex,
                              window: int = 240) -> pd.Series:
    """V3 等价 target_weight：仅 MA240 门控（站上=1.0，跌破=0.0）。

    用于 V4 报告中与 V4 新过滤器做「仅改过滤器」的公平对照。
    """
    ma_above, _ = compute_ma240_daily(index_series, daily_index, window)
    return ma_above.astype(float)


def csi300_annualized_vol(index_series: pd.Series,
                          daily_index: pd.DatetimeIndex,
                          vol_window: int = 60) -> pd.Series:
    """沪深300 过去 vol_window 日收益的<b>年化波动率</b>（日频）。

    无未来函数：rolling(vol_window).std() 仅用到 T 及之前的数据。
    """
    idx = index_series.reindex(daily_index).ffill()
    ret = idx.pct_change()
    vol = ret.rolling(vol_window, min_periods=vol_window // 2).std() * np.sqrt(252)
    return vol


def build_ma240_vol_target_weight(index_series: pd.Series,
                                  daily_index: pd.DatetimeIndex,
                                  window: int = 240,
                                  vol_window: int = 60,
                                  vol_lookback: int = 756,
                                  vol_q: float = 0.75,
                                  reduced_weight: float = 0.60,
                                  month_ends=None) -> tuple[pd.Series, pd.DataFrame]:
    """V7.1 目标权重：MA240 门控之上<b>叠加波动率过滤</b>。

    逻辑（每月末定档，日频前向填充；盘中 MA240 跌破仍硬归 0，保持主门控不变）：
      - MA240 跌破                 -> 仓位 0%（空仓，继承 V3 框架）
      - MA240 站上 且 波动率≤历史分位 -> 仓位 100%
      - MA240 站上 且 波动率>历史分位 -> 仓位 reduced_weight（默认60%，保留部分多头）

    vol 历史分位：以「截至上一交易日的过去 vol_lookback 日年化波动率」的 vol_q 分位数
    为阈值（trailing 窗口、shift(1) 防自指，零未来泄露）。

    返回 (target_weight_daily, vol_regime_df)：
      target_weight_daily : 日频 0 / 0.6 / 1.0
      vol_regime_df       : 月频，列 ma_above / vol60 / vol_thr / elevated / target_weight
    """
    ma_above, _ = compute_ma240_daily(index_series, daily_index, window)
    vol60 = csi300_annualized_vol(index_series, daily_index, vol_window)
    # 阈值：trailing vol_lookback 日年化波动率的 vol_q 分位（用 t-1 及之前，防自指）
    vol_thr = (vol60.rolling(vol_lookback, min_periods=vol_lookback // 3)
               .quantile(vol_q).shift(1))

    tw_daily = pd.Series(1.0, index=daily_index, dtype=float)
    vol_regime_rows = []

    if month_ends is not None:
        ma_me = ma_above.reindex(pd.DatetimeIndex(month_ends)).ffill()
        for t in pd.DatetimeIndex(month_ends):
            up = bool(ma_me.get(t, True))
            v = vol60.get(t, np.nan)
            thr = vol_thr.get(t, np.nan)
            elevated = (not _isnan_v(v)) and (not _isnan_v(thr)) and (v > thr)
            if not up:
                w = 0.0
            elif elevated:
                w = float(reduced_weight)
            else:
                w = 1.0
            vol_regime_rows.append({
                "ma_above": up,
                "vol60": v,
                "vol_thr": thr,
                "elevated": elevated,
                "target_weight": w,
            })
        tw_me = pd.DataFrame(vol_regime_rows, index=pd.DatetimeIndex(month_ends))
        tw_me.index.name = "month_end"
        tw_daily = tw_me["target_weight"].reindex(daily_index).ffill().astype(float)
        tw_daily = tw_daily.where(ma_above, 0.0)        # 盘中破均线 -> 空仓（主门控不变）
        tw_daily = tw_daily.fillna(1.0)                 # 极早期默认满仓
        return tw_daily, tw_me

    # 无月末输入：仅按日频 MA240 门控 + 波动率（末日分位）降级
    tw_daily = tw_daily.where(ma_above, 0.0).fillna(1.0)
    return tw_daily, pd.DataFrame()


def _isnan_v(x) -> bool:
    try:
        return x is None or (isinstance(x, float) and np.isnan(x))
    except (TypeError, ValueError):
        return False
