# -*- coding: utf-8 -*-
"""
V9 模块1：分析师预期上调因子（point-in-time，零未来函数）。

数据源：data/analyst_ratings_raw.parquet（fetch_analyst_v9.py 用
ak.stock_rank_forecast_cninfo(date) 按周频抓取；每行 = 某交易日发布的某份评级报告）。

────────────────────────────────────────────────────────────────────────
【口径说明：为何不用 EPS 预测修正 / 巨潮"评级变化"字段】

1) 免费 akshare 无可 point-in-time 的历史 EPS 预测修正序列
   （stock_profit_forecast_em 仅返回"当前共识快照"，用于回测=未来函数，禁用）。
   → 用「分析师评级上调」事件等价刻画"分析师预期上调"的经济含义。

2) 巨潮原始字段「评级变化」实测全样本仅取值 {维持, 未知}（93154 行中
   0.0=64140 / NaN=29014，无任何 上调/下调），直接用会得到恒零因子。
   → 弃用该字段作为主口径。

3) 字段「前一次投资评级」在 2018 年口径异常（自推导上调占比 41.9%，
   而 2019-2025 仅 0.7~1.8%），跨期不可比。
   → 仅作为一个辅助子因子（用户明确要求的 rating_change 口径），
     且与其他子因子等权，降低其单点影响。

因此主口径改为「自算评级动量」：只消费干净且跨期一致的 `投资评级` 字段
（买入2/增持1/中性0/减持-1/卖出-2）与报告条数，在我们自己的
point-in-time 时间序列上做窗口对比。
────────────────────────────────────────────────────────────────────────

子因子（截至决策日 t 仅消费 asof_date <= t 的报告，严格无泄露）：
  rating_level  : 过去 92 日内 mean(投资评级)         —— 分析师看好程度（水平）
  rating_delta  : mean(0~92日) − mean(92~184日)       —— 评级上调动量（主口径）
  attention_mom : log1p(cnt 0~63日) − log1p(cnt 63~126日) —— 覆盖热度提升
  rating_change : (#上调 − #下调)/报告数, 92 日窗口     —— 用户要求的评级变动口径
                  （由 投资评级 − 前一次投资评级 推导）

无分析师覆盖 → 因子值 = 0（中性），不剔除股票（符合用户约束）。
Z-score 只在「有覆盖」的股票上计算，未覆盖股票再填 0，保证"无覆盖=中性"
在数值上严格成立（若把 0 混入均值/标准差会污染分布）。
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

import config

SUBFACTORS = ["rating_level", "rating_delta", "attention_mom", "rating_change"]


def load_raw_ratings() -> pd.DataFrame:
    """读取原始评级报告（代理关闭，巨潮资讯独立 host）。"""
    for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
              "ALL_PROXY", "all_proxy"]:
        os.environ.pop(k, None)
    p = config.DATA_DIR / "analyst_ratings_raw.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"缺失分析师原始数据: {p}（请先运行 fetch_analyst_v9.py）")
    df = pd.read_parquet(p)
    df["asof_date"] = pd.to_datetime(df["asof_date"])
    # 自推导评级变动方向（仅当 当前评级 与 前一次评级 均为有效评级时）
    d = df["rating_num"] - df["prev_rating_num"]
    df["chg_dir"] = np.sign(d)          # +1 上调 / 0 维持 / -1 下调 / NaN 无法判定
    return df.sort_values("asof_date").reset_index(drop=True)


def _winsorize_z(series: pd.Series) -> pd.Series:
    """截面 winsorize(1%/99%) + Z-score（与 factor_eval 财务因子处理一致）。

    仅对非 NaN（=有覆盖）样本计算，返回的 Series 在无覆盖处保持 NaN。
    """
    s = series.dropna()
    if len(s) < 10:
        return pd.Series(np.nan, index=series.index)
    lo, hi = s.quantile(0.01), s.quantile(0.99)
    sw = s.clip(lo, hi)
    mu, sd = sw.mean(), sw.std(ddof=0)
    if sd == 0 or not np.isfinite(sd):
        return pd.Series(np.nan, index=series.index)
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    out.loc[sw.index] = (sw - mu) / sd
    return out


def build_analyst_panels(raw: pd.DataFrame,
                         rebalance_dates,
                         universe_codes) -> dict:
    """构建分析师子因子面板（index=rebalance_dates, columns=universe_codes）。

    未覆盖位置保持 NaN（交由 composite 阶段在 Z 之后填 0）。
    """
    dates = pd.DatetimeIndex(rebalance_dates)
    codes = pd.Index(universe_codes)

    # 只保留宇宙内股票，显著降低分组开销
    raw = raw[raw["code"].isin(set(codes))].sort_values("asof_date")

    W_LVL = pd.Timedelta(days=config.ANALYST_UPGRADE_WINDOW_DAYS)      # 92d
    W_ATT = pd.Timedelta(days=63)
    rows = {k: {} for k in SUBFACTORS}

    # 预排序数组，用 searchsorted 做窗口切片（比逐次布尔筛选快很多）
    ad = raw["asof_date"].values
    for t in dates:
        t_np = np.datetime64(t)
        hi = np.searchsorted(ad, t_np, side="right")          # asof_date <= t
        if hi == 0:
            for k in SUBFACTORS:
                rows[k][t] = pd.Series(np.nan, index=codes, dtype="float64")
            continue

        def _slice(lo_ts, hi_ts):
            lo_i = np.searchsorted(ad, np.datetime64(lo_ts), side="right")
            hi_i = np.searchsorted(ad, np.datetime64(hi_ts), side="right")
            return raw.iloc[lo_i:hi_i]

        w_lvl = _slice(t - W_LVL, t)                 # 近 92d
        w_lvl_prev = _slice(t - 2 * W_LVL, t - W_LVL)  # 前 92d
        w_att = _slice(t - W_ATT, t)                 # 近 63d
        w_att_prev = _slice(t - 2 * W_ATT, t - W_ATT)  # 前 63d

        lvl = w_lvl.groupby("code")["rating_num"].mean()
        lvl_prev = w_lvl_prev.groupby("code")["rating_num"].mean()
        # rating_delta：两期都有评级才有意义（否则 NaN=无覆盖）
        delta = (lvl - lvl_prev).dropna()

        cnt = w_att.groupby("code").size().astype(float)
        cnt_prev = w_att_prev.groupby("code").size().astype(float)
        att_idx = cnt.index.union(cnt_prev.index)
        att = (np.log1p(cnt.reindex(att_idx).fillna(0.0))
               - np.log1p(cnt_prev.reindex(att_idx).fillna(0.0)))

        chg = w_lvl.groupby("code")["chg_dir"].mean()   # (#上调−#下调)/可判定报告数

        rows["rating_level"][t] = lvl.reindex(codes)
        rows["rating_delta"][t] = delta.reindex(codes)
        rows["attention_mom"][t] = att.reindex(codes)
        rows["rating_change"][t] = chg.reindex(codes)

    panels = {}
    for k in SUBFACTORS:
        panels[k] = pd.DataFrame(rows[k]).T.reindex(index=dates, columns=codes)
    return panels


def build_analyst_composite(panels: dict, rebalance_dates) -> pd.DataFrame:
    """子因子逐期 winsorize+Z 后等权合成；无覆盖处填 0（中性）。

    返回面板（行=rebalance_date, 列=code），数值为可直接叠加到质量分的 Z 分。
    """
    dates = pd.DatetimeIndex(rebalance_dates)
    comp_rows = {}
    for t in dates:
        zs = []
        for name in SUBFACTORS:
            if t not in panels[name].index:
                continue
            z = _winsorize_z(panels[name].loc[t])
            if z.notna().sum() >= 10:
                zs.append(z)
        if not zs:
            comp_rows[t] = pd.Series(0.0, index=panels[SUBFACTORS[0]].columns)
            continue
        # 逐股票对「该股票有值的子因子」取均值，避免个别子因子缺失拉低强度
        z = pd.concat(zs, axis=1).mean(axis=1, skipna=True)
        comp_rows[t] = z.fillna(0.0)          # 无覆盖 → 0（中性）
    comp = pd.DataFrame(comp_rows).T.reindex(index=dates)
    return comp


def coverage_report(panels: dict) -> pd.DataFrame:
    """各子因子逐期截面覆盖率（用于报告披露数据质量）。"""
    out = {}
    for k, p in panels.items():
        out[k] = p.notna().sum(axis=1) / max(p.shape[1], 1)
    return pd.DataFrame(out)
