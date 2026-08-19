# -*- coding: utf-8 -*-
"""
模型训练模块：基于扩展因子训练 LightGBM，预测下月收益。

- 训练集 2018-2020，测试集 2021-2023（严格时间隔离）。
- 输出训练日志、特征重要性、样本内外 IC。
- 提供 ML 增强选股（月度 Top-N）回测，与主策略「超跌绩优」对照。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import lightgbm as lgb
from sklearn.metrics import mean_squared_error

import config
from factors import build_factor_long, compute_fwd_return


FEATURES = [
    # V3：剔除样本外失效的 EP/ROE/净利润增长，改用以「反转」为主轴的因子集
    "rsi_14", "skew_60", "turnover_dev",
    "ret_5", "ret_20", "ret_60",
    "vol_20", "ma20_ratio", "ma60_ratio", "volume_ratio",
]


def prepare_ml_data(close_panel: pd.DataFrame, ohlcv: dict):
    """构建 ML 数据集，返回 (X, y, meta)。

    V3 特征全部可由价格/成交量派生（含 60日偏度、换手率乖离率代理），
    不再依赖财报（EP/ROE/净利润增长），规避财报时点对齐带来的未来函数与失效因子。
    """
    flong = build_factor_long(close_panel, ohlcv=ohlcv)
    fwd = compute_fwd_return(close_panel, config.FWD_RETURN_DAYS).stack().rename("fwd_ret")
    fwd = fwd.reset_index()
    fwd.columns = ["date", "code", "fwd_ret"]
    flong = flong.merge(fwd, on=["date", "code"], how="left")
    flong = flong.dropna(subset=FEATURES + ["fwd_ret"])
    X = flong[FEATURES].copy()
    y = flong["fwd_ret"].copy()
    meta = flong[["date", "code", "fwd_ret"]].copy()
    return X, y, meta


def _rank_ic(y_true: pd.Series, y_pred: pd.Series) -> float:
    """计算秩相关系数（IC）。"""
    df = pd.DataFrame({"y": y_true, "p": y_pred}).dropna()
    if len(df) < 10:
        return np.nan
    return df["y"].corr(df["p"], method="spearman")


def train_lightgbm(close_panel: pd.DataFrame, ohlcv: dict):
    """训练 LightGBM，返回 (model, importance_df, metrics, train_log)。"""
    X, y, meta = prepare_ml_data(close_panel, ohlcv)
    meta = meta.copy()
    d = pd.to_datetime(meta["date"])
    # 训练 2018-2019，验证 2020（用于早停，杜绝用测试集定轮次），测试 2021-2023
    is_train = d <= pd.Timestamp("2019-12-31")
    is_val = (d >= pd.Timestamp("2020-01-01")) & (d <= pd.Timestamp("2020-12-31"))
    is_test = d >= pd.Timestamp(config.TEST_START)
    train_mask = is_train
    val_mask = is_val
    test_mask = is_test

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    params = {
        "objective": "regression",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": config.RANDOM_SEED,
        "n_jobs": -1,
        "verbose": -1,
    }
    train_set = lgb.Dataset(X_train, y_train)
    valid_set = lgb.Dataset(X_val, y_val)
    model = lgb.train(
        params, train_set, num_boost_round=300,
        valid_sets=[train_set, valid_set],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )

    pred_train = model.predict(X_train, num_iteration=model.best_iteration)
    pred_test = model.predict(X_test, num_iteration=model.best_iteration)

    metrics = {
        "train_samples": int(train_mask.sum()),
        "val_samples": int(val_mask.sum()),
        "test_samples": int(test_mask.sum()),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, pred_train))),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
        "train_ic": float(_rank_ic(y_train, pred_train)),
        "test_ic": float(_rank_ic(y_test, pred_test)),
        "best_iteration": int(model.best_iteration),
    }

    imp = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    train_log = (
        f"LightGBM 训练完成 | 训练(18-19)={metrics['train_samples']} "
        f"验证(20)={metrics['val_samples']} 测试(21-23)={metrics['test_samples']} "
        f"| best_iter={metrics['best_iteration']}\n"
        f"训练 RMSE={metrics['train_rmse']:.4f} 测试 RMSE={metrics['test_rmse']:.4f}\n"
        f"训练 IC={metrics['train_ic']:.4f} 测试 IC={metrics['test_ic']:.4f}"
    )
    return model, imp, metrics, train_log


def predict_signal_panel(model, close_panel: pd.DataFrame, ohlcv: dict) -> pd.DataFrame:
    """用训练好的模型生成预测信号面板（date x code）。"""
    flong = build_factor_long(close_panel, ohlcv=ohlcv)
    flong = flong.dropna(subset=FEATURES)
    if flong.empty:
        return pd.DataFrame()
    X = flong[FEATURES]
    pred = model.predict(X, num_iteration=model.best_iteration)
    sig = flong[["date", "code"]].copy()
    sig["pred"] = pred
    panel = sig.pivot(index="date", columns="code", values="pred").sort_index()
    return panel.reindex(close_panel.index)


def run_ml_backtest(signal_panel: pd.DataFrame,
                    close_panel: pd.DataFrame,
                    month_ends: pd.DatetimeIndex,
                    start: str, end: str,
                    top_n: int = 10,
                    cost: float = 0.002,
                    init_capital: float = 1_000_000.0) -> pd.Series:
    """ML 增强回测：每月末按预测信号选 Top-N，等权持有至下月末。"""
    all_dates = close_panel.index
    mask = (all_dates >= pd.Timestamp(start)) & (all_dates <= pd.Timestamp(end))
    dates = pd.DatetimeIndex(all_dates[mask])
    close = close_panel.reindex(dates)
    signal = signal_panel.reindex(dates)
    month_end_set = set(pd.DatetimeIndex(month_ends))

    cash = init_capital
    positions: dict = {}
    equity: dict = {}

    for t in dates:
        ct = close.loc[t]
        # 退出上一期持仓（月度换仓）
        if t in month_end_set and positions:
            for code, (shares, _) in list(positions.items()):
                px = ct.get(code)
                if not _bad(px):
                    cash += px * shares * (1.0 - cost)
            positions = {}
        # 选股并买入
        if t in month_end_set:
            eq = cash  # 月初现金即权益（持仓已在月初卖出）
            sel = signal.loc[t].dropna().sort_values(ascending=False).head(top_n)
            sel = [c for c in sel.index if not _bad(ct.get(c)) and c not in positions]
            n = max(1, len(sel))
            w = 1.0 / n
            for code in sel:
                px = ct.get(code)
                if _bad(px) or cash <= 0:
                    continue
                target = w * eq
                shares = int(target / (px * (1.0 + cost)))
                if shares <= 0:
                    continue
                cost_amt = px * shares * (1.0 + cost)
                if cost_amt > cash:
                    continue
                cash -= cost_amt
                positions[code] = [shares, px]
        eq = cash + sum(
            v[0] * ct.get(c) for c, v in positions.items() if not _bad(ct.get(c))
        )
        equity[t] = eq
    return pd.Series(equity).sort_index()


def _bad(x) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x))
