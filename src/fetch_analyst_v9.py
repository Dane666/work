# -*- coding: utf-8 -*-
"""
V9 模块1：分析师评级数据收集（point-in-time）。

数据源：ak.stock_rank_forecast_cninfo(date) —— 巨潮资讯-投资评级，返回「指定日期」
全部分析师评级报告快照（含 投资评级 / 评级变化 / 前一次投资评级）。

关键设计（杜绝未来函数）：
  - 按周频(W-FRI) 在 2018-01-01 ~ 2025-08-12 的每一个「该周最后交易日」调用一次，
    得到该日发布的评级报告。把这些报告「归属」到该交易日的截面，下游再以
    「截至月末 t 的过去 N 日报告」构建因子 —— 仅消费 t 之前已发布的数据。
  - 不依赖任何「当前共识快照」类接口（如 stock_profit_forecast_em 只返回最新，
    无法 point-in-time，禁用）。

输出：data/analyst_ratings_raw.parquet
  列：asof_date(归属交易日), code(6位), name, rating_num, change_num,
      prev_rating_num, is_first
  rating_num : 买入=2/增持=1/中性=0/减持=-1/卖出=-2/不评级=NaN
  change_num : 上调=+1/维持=0/下调=-1/未知=NaN

断点续跑：若已存在输出文件，跳过其中已收集的 asof_date。
代理必须关闭（巨潮资讯走独立 host）。
"""

from __future__ import annotations

import os
import time
import random

for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
          "ALL_PROXY", "all_proxy"]:
    os.environ.pop(k, None)

import numpy as np
import pandas as pd
import akshare as ak

import config

RAW_OUT = config.DATA_DIR / "analyst_ratings_raw.parquet"

RATING_MAP = {
    "买入": 2.0, "增持": 1.0, "中性": 0.0, "减持": -1.0,
    "卖出": -2.0, "不评级": np.nan, "": np.nan,
}
CHANGE_MAP = {
    "上调": 1.0, "维持": 0.0, "下调": -1.0, "未知": np.nan, "": np.nan,
}


def _target_dates():
    """每个自然周五对应的「该周最后交易日」序列。"""
    close = pd.read_parquet(config.DATA_DIR / "v8_close_panel.parquet")
    trading = pd.DatetimeIndex(close.index)
    fridays = pd.date_range(trading.min(), trading.max(), freq="W-FRI")
    targets = []
    for f in fridays:
        prior = trading[trading <= f]
        if len(prior) == 0:
            continue
        targets.append(prior.max())
    # 去重（避免跨年边界重复）
    targets = list(dict.fromkeys(targets))
    return targets


def _collect_one(d: pd.Timestamp) -> pd.DataFrame:
    date_str = d.strftime("%Y%m%d")
    df = ak.stock_rank_forecast_cninfo(date=date_str)
    if df is None or df.empty:
        return pd.DataFrame()
    out = pd.DataFrame()
    out["asof_date"] = [d] * len(df)
    code = df["证券代码"].astype(str).str.zfill(6)
    out["code"] = code
    out["name"] = df.get("证券简称", pd.Series([""] * len(df)))
    out["rating_num"] = df["投资评级"].map(RATING_MAP).astype("float")
    out["change_num"] = df["评级变化"].map(CHANGE_MAP).astype("float")
    prev = df.get("前一次投资评级")
    out["prev_rating_num"] = prev.map(RATING_MAP).astype("float") if prev is not None else np.nan
    isf = df.get("是否首次评级")
    out["is_first"] = (isf == "是首次评级") if isf is not None else False
    # 仅保留 6 位纯数字代码（剔除指数/异常行）
    out = out[out["code"].str.fullmatch(r"\d{6}")]
    return out


def main():
    targets = _target_dates()
    print(f"[collector] 计划抓取周频交易日 {len(targets)} 个 "
          f"({targets[0].date()} ~ {targets[-1].date()})")

    done = set()
    existing = None
    if RAW_OUT.exists():
        existing = pd.read_parquet(RAW_OUT)
        done = set(pd.to_datetime(existing["asof_date"]).unique())
        print(f"[collector] 断点续跑：已存在 {len(done)} 个日期，"
              f"跳过。")

    collected = [] if existing is None else [existing]
    n_fail = 0
    for i, d in enumerate(targets):
        if d in done:
            continue
        ok = False
        for attempt in range(4):
            try:
                sub = _collect_one(d)
                if not sub.empty:
                    collected.append(sub)
                ok = True
                break
            except Exception as e:
                wait = 1.5 * (attempt + 1) + random.uniform(0, 0.5)
                print(f"  ! {d.date()} 第{attempt+1}次失败: {repr(e)[:120]}; "
                      f"{wait:.1f}s 后重试")
                time.sleep(wait)
        if not ok:
            n_fail += 1
            print(f"  X {d.date()} 持续失败，跳过")
        if (i + 1) % 10 == 0:
            # 周期性落盘（防中断丢失）
            combined = pd.concat(collected, ignore_index=True)
            combined.to_parquet(RAW_OUT)
            print(f"[progress] {i+1}/{len(targets)} | 累计报告行 "
                  f"{len(combined)} | 失败 {n_fail}")
        time.sleep(random.uniform(0.25, 0.55))

    combined = pd.concat(collected, ignore_index=True)
    combined = combined.sort_values(["asof_date", "code"]).reset_index(drop=True)
    combined.to_parquet(RAW_OUT)
    print(f"[collector] 完成。总报告行 {len(combined)} | "
          f"覆盖股票 {combined['code'].nunique()} 只 | 失败日期 {n_fail}")
    print(f"[collector] 落盘: {RAW_OUT}")


if __name__ == "__main__":
    main()
