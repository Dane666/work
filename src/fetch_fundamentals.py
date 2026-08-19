# -*- coding: utf-8 -*-
"""
补充抓取脚本：从财报分析指标中提取 EPS / ROE / 净利润增长率，
按披露时点对齐（点对点，杜绝未来函数），构建日频面板；
并抓取沪深300指数日线（市场过滤器 MA240 用）。

日线行情已在 fetch_all.py 中缓存为 close_panel.parquet，本脚本
仅依赖其列名（成分股代码）与交易日历，避免重复抓取行情。

运行：python fetch_fundamentals.py
"""

from __future__ import annotations

import os
import time

for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
          "ALL_PROXY", "all_proxy"]:
    os.environ.pop(k, None)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import akshare as ak

import config


def available_date(report_date: pd.Timestamp, lag_map: dict) -> pd.Timestamp:
    """由报告期与披露延迟映射，得到数据「可用起始日」（通常为次月 1 日）。"""
    lag = lag_map.get(report_date.strftime("%m-%d"), 4)
    y, m = report_date.year, report_date.month
    m0 = m + lag
    y += (m0 - 1) // 12
    m = (m0 - 1) % 12 + 1
    return pd.Timestamp(y, m, 1)


def main():
    close = pd.read_parquet(config.DATA_DIR / "close_panel.parquet")
    codes = list(close.columns)
    trade_idx = pd.DatetimeIndex(close.index)
    print(f"成分股 {len(codes)} 只，交易日 {len(trade_idx)} 个")

    rows_eps, rows_roe, rows_npg = [], [], []
    ok = 0
    for i, code in enumerate(codes, 1):
        try:
            fi = ak.stock_financial_analysis_indicator(symbol=code)
        except Exception as e:
            print(f"  skip {code}: {repr(e)[:80]}")
            continue
        if "日期" not in fi.columns:
            continue
        for _, r in fi.iterrows():
            rd = r["日期"]
            if rd is None or (isinstance(rd, float) and np.isnan(rd)):
                continue
            if not isinstance(rd, pd.Timestamp):
                try:
                    rd = pd.Timestamp(rd)
                except Exception:
                    continue
            av = available_date(rd, config.DISCLOSURE_LAG_MONTHS)
            rows_eps.append((code, av, r.get("摊薄每股收益(元)")))
            rows_roe.append((code, av, r.get("净资产收益率(%)")))
            rows_npg.append((code, av, r.get("净利润增长率(%)")))
        ok += 1
        if i % 30 == 0:
            print(f"  进度 {i}/{len(codes)} 已成功 {ok}")

    print(f"财务抓取完成：eps={len(rows_eps)} roe={len(rows_roe)} npg={len(rows_npg)}")

    def build_panel(rows, min_obs=5):
        df = pd.DataFrame(rows, columns=["code", "date", "val"]).dropna(subset=["val"])
        out = pd.DataFrame(index=trade_idx, columns=codes, dtype=float)
        for code in codes:
            sub = df[df["code"] == code]
            if sub.empty or len(sub) < min_obs:
                continue
            s = pd.Series(sub["val"].values,
                         index=pd.DatetimeIndex(sub["date"].values))
            s = s[~s.index.duplicated(keep="last")].sort_index()
            # 在「可用日」置值，前向填充到每个交易日（可用日后首个交易日才生效）
            union = trade_idx.union(s.index, sort=True)
            full = s.reindex(union).ffill().reindex(trade_idx)
            out[code] = full
        return out

    eps_panel = build_panel(rows_eps)
    roe_panel = build_panel(rows_roe)
    npg_panel = build_panel(rows_npg)

    eps_panel.to_parquet(config.DATA_DIR / "eps_panel.parquet")
    roe_panel.to_parquet(config.DATA_DIR / "roe_panel.parquet")
    npg_panel.to_parquet(config.DATA_DIR / "npg_panel.parquet")
    print(f"面板已保存：eps={eps_panel.notna().sum().sum()} roe={roe_panel.notna().sum().sum()} "
          f"npg={npg_panel.notna().sum().sum()} 非空单元")

    # 沪深300指数日线（市场过滤器用）
    idx = ak.stock_zh_index_daily(symbol="sh000300")
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.set_index("date")["close"].sort_index()
    idx = idx[idx.index >= trade_idx[0]]
    idx.to_frame(name="close").to_parquet(config.DATA_DIR / "index.parquet")
    print(f"指数已保存：{len(idx)} 行，区间 {idx.index[0].date()} ~ {idx.index[-1].date()}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"fetch_fundamentals 完成，耗时 {(time.time()-t0)/60:.1f} 分钟")
