# -*- coding: utf-8 -*-
"""
V7 滚动窗口反转信号（仅改信号生成，不动 V5 的 Regime 切换框架）。

设计（严格 point-in-time，零未来函数）：
  - 冷启动：用固定窗口 [START, 2019-12-31] 训练（与 V5 完全一致），覆盖 2018 ~ 2020Q1 前，
    以保证 2018-2023 区间与 V5 可比、不退化。
  - 滚动：自 2020-03-31 起每季度末重训，训练窗口 = 过去 36 个月（cap 在可用历史），
    月度调仓复用最近一次训练的模型预测下月信号。
  - 早停验证：用「训练窗口内最后 20%」做 early-stopping（纯窗口内时序切分，零未来泄露），
    绝不拿窗口外数据定轮次。
  - 预测：仅消费截至预测月已可得的价格/量特征（build_factor_long 全为 point-in-time 因子）。

输出 reversal_signal 面板（date x code），格式与 V5 的 predict_signal_panel 完全一致，
因此 factor_eval.build_selection_v5 / backtest_v5.run_backtest_v5 可零改动复用。
"""

from __future__ import annotations

import os
for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
          "ALL_PROXY", "all_proxy"]:
    os.environ.pop(k, None)

import numpy as np
import pandas as pd
import lightgbm as lgb

import config
from factors import build_factor_long, compute_fwd_return
from model import FEATURES, train_lightgbm, predict_signal_panel


SEED = getattr(config, "RANDOM_SEED", 42)


def _make_flong(close_m: pd.DataFrame, ohlcv: dict) -> pd.DataFrame:
    """构建全期因子长表（含未来收益标签）。"""
    flong = build_factor_long(close_m, ohlcv=ohlcv)
    fwd = compute_fwd_return(close_m, config.FWD_RETURN_DAYS).stack().rename("fwd_ret")
    fwd = fwd.reset_index()
    fwd.columns = ["date", "code", "fwd_ret"]
    flong = flong.merge(fwd, on=["date", "code"], how="left")
    return flong


def _train_window(flong: pd.DataFrame, win_start, win_end, seed: int = SEED):
    """在 [win_start, win_end] 窗口内训练 LightGBM（窗口内 80/20 时序切分做早停）。

    返回模型或 None（样本不足时）。
    """
    w = flong[(flong["date"] >= pd.Timestamp(win_start)) &
              (flong["date"] <= pd.Timestamp(win_end)) &
              (~flong["fwd_ret"].isna())].copy()
    w = w.sort_values("date")
    if len(w) < 300:
        return None
    n = max(50, int(len(w) * 0.8))
    tr, va = w.iloc[:n], w.iloc[n:]
    Xtr, ytr = tr[FEATURES], tr["fwd_ret"]
    Xva, yva = va[FEATURES], va["fwd_ret"]
    params = {
        "objective": "regression",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": seed,
        "n_jobs": -1,
        "verbose": -1,
    }
    ds = lgb.Dataset(Xtr, ytr)
    if len(va) >= 50:
        vds = lgb.Dataset(Xva, yva)
        model = lgb.train(
            params, ds, num_boost_round=300,
            valid_sets=[ds, vds],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
        )
    else:
        model = lgb.train(params, ds, num_boost_round=200,
                          valid_sets=[ds], callbacks=[lgb.log_evaluation(0)])
    return model


def _fill_segment(sig: pd.DataFrame, flong_feat: pd.DataFrame, model,
                  seg_start, seg_end):
    """用模型预测 [seg_start, seg_end) 段信号，写入 sig 面板。"""
    seg = flong_feat[(flong_feat["date"] >= pd.Timestamp(seg_start)) &
                     (flong_feat["date"] < pd.Timestamp(seg_end))]
    if seg.empty:
        return
    pred = model.predict(seg[FEATURES], num_iteration=model.best_iteration)
    p = pd.DataFrame({"date": seg["date"].values, "code": seg["code"].values, "pred": pred})
    panel = p.pivot(index="date", columns="code", values="pred")
    panel = panel.reindex(index=sig.index)
    sig.loc[panel.index, panel.columns] = panel


def build_rolling_reversal_signal(close_m: pd.DataFrame,
                                  ohlcv: dict,
                                  window: int = 36) -> pd.DataFrame:
    """构建滚动窗口反转信号面板（date x code）。

    window : 滚动训练窗口长度（月），默认 36。
    """
    flong = _make_flong(close_m, ohlcv)
    flong_feat = flong.dropna(subset=FEATURES).copy()
    flong_feat["date"] = pd.to_datetime(flong_feat["date"])

    start_ts = pd.Timestamp(config.START_DATE)
    end_ts = close_m.index[-1]

    sig = pd.DataFrame(index=close_m.index, columns=close_m.columns, dtype=float)

    # ---------- 冷启动（与 V5 完全一致：固定 2018-2019 训练 + 2020 验证）----------
    # 直接复用 model.train_lightgbm / predict_signal_panel，保证 2018-2023 与 V5 可比对。
    cold_model, _, _, _ = train_lightgbm(close_m, ohlcv)
    cold_panel = predict_signal_panel(cold_model, close_m, ohlcv)
    sig.loc[cold_panel.index, cold_panel.columns] = cold_panel
    cold = cold_model

    # ---------- 滚动重训（每季度末）----------
    q_ends = pd.date_range("2020-03-31", end_ts, freq="QE")
    for i, q in enumerate(q_ends):
        nxt = q_ends[i + 1] if i + 1 < len(q_ends) else end_ts + pd.Timedelta(days=1)
        win_start = q - pd.DateOffset(months=window)
        win_start = max(win_start, start_ts)
        mdl = _train_window(flong, win_start, q, SEED)
        if mdl is None:
            mdl = cold  # 退化保护：沿用冷启动
        _fill_segment(sig, flong_feat, mdl, q, nxt)

    return sig


# ---------------------------------------------------------------------------
# IC 动态加权（条件执行，仅当滚动+真实财报仍不足时启用）
# ---------------------------------------------------------------------------
def compute_factor_ic_series(close_m: pd.DataFrame, factor_panel: pd.DataFrame,
                             month_ends, fwd_days: int = 21) -> pd.Series:
    """计算某因子面板的月度 Rank IC 序列（用于 IC 动态加权）。"""
    from factors import compute_fwd_return
    fwd = compute_fwd_return(close_m, fwd_days)
    ics = {}
    for t in pd.DatetimeIndex(month_ends):
        f = factor_panel.loc[t].dropna()
        y = fwd.loc[t].dropna()
        common = f.index.intersection(y.index)
        if len(common) < 20:
            ics[t] = np.nan
            continue
        ics[t] = f[common].corr(y[common], method="spearman")
    return pd.Series(ics, name="ic").sort_index()


def rolling_factor_ic(reversal_ic: pd.Series,
                      momentum_ic: pd.Series,
                      quality_ic: pd.Series,
                      window: int = 12) -> pd.DataFrame:
    """返回三因子过去 12 月滚动 IC 均值（用于动态加权）。

    列：reversal / momentum / quality，行：month_end。
    """
    r = reversal_ic.rolling(window, min_periods=6).mean().shift(1)
    m = momentum_ic.rolling(window, min_periods=6).mean().shift(1)
    q = quality_ic.rolling(window, min_periods=6).mean().shift(1)
    out = pd.DataFrame({"reversal": r, "momentum": m, "quality": q})
    return out.sort_index()
