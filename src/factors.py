# -*- coding: utf-8 -*-
"""
因子计算模块：实现「超跌绩优」核心因子与扩展因子集。

核心因子：
- RSI(14)：Wilder 平滑的相对强弱指标，低于 30 视为超跌。
- 净利润同比增长率(np_growth)：财报披露的同比增长，>20% 视为绩优。

扩展因子（用于 LightGBM 特征与 IC 评估）：
- 动量：ret_5 / ret_20 / ret_60
- 波动率：vol_20（20 日收益标准差）
- 估值/趋势代理：close_ma20_ratio / close_ma60_ratio
- 流动性：volume_ratio（量比）
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rsi(close_panel: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """计算 RSI(14) 面板（行=交易日，列=股票）。Wilder 平滑。"""
    delta = close_panel.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder 平滑：alpha = 1/window 的指数加权移动平均
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def compute_fwd_return(close_panel: pd.DataFrame, days: int = 21) -> pd.DataFrame:
    """计算未来 N 日收益率面板（用于标签与 IC 评估）。"""
    return close_panel.shift(-days) / close_panel - 1.0


def compute_skew(close_panel: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """计算过去 window 日收益率的偏度面板（增强反转信号质量）。

    偏度衡量收益分布的非对称性：极端负偏（罕见大跌）常伴随后续均值回复。
    作为 LightGBM 特征，方向由模型自行学习。
    """
    ret = close_panel.pct_change()
    return ret.rolling(window).skew()


def compute_turnover_dev(ohlcv: dict, dates: pd.DatetimeIndex, codes: list,
                         short: int = 5, long: int = 60) -> pd.DataFrame:
    """换手率乖离率（代理）：短周期成交量相对长周期均值的偏离程度。

    说明：本沙箱环境下东方财富换手率接口被网络拦截、且无逐日流通股数据源，
    真实「换手率」= 成交量/流通股 不可得。鉴于单只股票流通股在短期内变化极小，
    换手率乖离率与「成交量乖离率」高度共线，故以
        (近 short 日均量 / 近 long 日均量 - 1) 作为等价代理，用于捕捉异常放量。
    """
    panel = pd.DataFrame(index=dates, columns=codes, dtype=float)
    for code in codes:
        if code in ohlcv:
            v = ohlcv[code]["volume"]
            dev = v.rolling(short).mean() / v.rolling(long).mean() - 1.0
            panel[code] = dev.reindex(dates)
    return panel


def build_factor_long(close_panel: pd.DataFrame,
                      extra_panels: dict = None,
                      ohlcv: dict = None) -> pd.DataFrame:
    """构建扩展因子长表：行=(date, code)，列=各因子。

    extra_panels: {因子名: 面板(DataFrame, 行=交易日, 列=股票)}，
                  如 {"ep": ep面板, "roe": roe面板, "np_growth": npg面板}。
    ohlcv: {code: DataFrame(索引 trade_date, 含 close/volume)}，用于量比。
    """
    codes = close_panel.columns.tolist()
    dates = close_panel.index
    ret = close_panel.pct_change()

    # 基础价格类因子（方向与收益无关，由 LightGBM 自行学习正负）
    base_panels = {
        "rsi_14": compute_rsi(close_panel, 14),
        "ret_5": close_panel.pct_change(5),
        "ret_20": close_panel.pct_change(20),
        "ret_60": close_panel.pct_change(60),
        "vol_20": ret.rolling(20).std(),
        "ma20_ratio": close_panel / close_panel.rolling(20).mean() - 1.0,
        "ma60_ratio": close_panel / close_panel.rolling(60).mean() - 1.0,
        "skew_60": compute_skew(close_panel, 60),     # V3 新增：60日偏度
    }
    if ohlcv:
        base_panels["turnover_dev"] = compute_turnover_dev(ohlcv, dates, codes)  # V3 新增：换手率乖离率(代理)
    if extra_panels:
        base_panels.update(extra_panels)

    # 量比（需逐只计算，因 volume 不在 close_panel 中）
    vol_ratio = pd.DataFrame(index=dates, columns=codes, dtype=float)
    if ohlcv:
        for code in codes:
            if code in ohlcv:
                v = ohlcv[code]["volume"]
                vr = v / v.rolling(20).mean()
                vol_ratio[code] = vr.reindex(dates)

    # 拼接为长表
    frames = []
    for code in codes:
        data = {"date": dates, "code": code}
        for name, panel in base_panels.items():
            data[name] = panel[code].reindex(dates).values if code in panel else np.nan
        data["volume_ratio"] = vol_ratio[code].reindex(dates).values if code in vol_ratio else np.nan
        frames.append(pd.DataFrame(data))
    long = pd.concat(frames, ignore_index=True)
    long = long.sort_values(["date", "code"]).reset_index(drop=True)
    return long


def compute_ep(eps_panel: pd.DataFrame,
               close_panel: pd.DataFrame) -> pd.DataFrame:
    """由每股收益(元)与日线收盘价计算盈利收益率 EP = EPS / 价格（等价于 1/PE）。"""
    return eps_panel / close_panel.replace(0.0, np.nan)


def get_month_end_dates(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """返回每个自然月的最后一个交易日。"""
    s = pd.Series(index=dates, data=dates)
    month_end = s.groupby(s.index.to_period("M")).last()
    return pd.DatetimeIndex(month_end.values)


def get_trade_dates_in_range(dates: pd.DatetimeIndex,
                             start: str, end: str) -> pd.DatetimeIndex:
    """截取时间范围内的交易日。"""
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    return pd.DatetimeIndex(dates[mask])
