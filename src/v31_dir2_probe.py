# -*- coding: utf-8 -*-
"""
v31_dir2_probe.py — 方向2 北向因子抽样验证（IC 检验，不抓全量）
====================================================================
对抽样股票抓取 stock_hsgt_individual_em（逐股每日北向），构建
  factor = 过去20日北向净流入(今日增持资金滚动和) / 流通市值proxy
并检验其在截面上的 Rank IC（vs 未来21日收益），判断因子是否有信号。

流通市值proxy = 当日收盘价 × (持股数量 / (持股数量占A股百分比/100))
若 IC 为正且显著 → 值得全量抓取做正式回测；否则报告无稳健信号。
运行：cd src && python v31_dir2_probe.py --n 40
"""

from __future__ import annotations
import os, argparse, socket
from datetime import datetime
import numpy as np
import pandas as pd

for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
          "ALL_PROXY", "all_proxy"]:
    os.environ.pop(k, None)
socket.setdefaulttimeout(30)

import akshare as ak
import config
from factors import get_month_end_dates, compute_fwd_return


def fetch_northbound_code(code):
    """返回逐股每日北向：date, net_inflow(今日增持资金), close, float_mv_proxy。"""
    df = ak.stock_hsgt_individual_em(symbol=code)
    if df is None or len(df) == 0:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["持股日期"])
    df = df.set_index("date").sort_index()
    ni = pd.to_numeric(df["今日增持资金"], errors="coerce")          # 北向净流入(元)
    close = pd.to_numeric(df["当日收盘价"], errors="coerce")
    shares = pd.to_numeric(df["持股数量"], errors="coerce")
    pct_a = pd.to_numeric(df["持股数量占A股百分比"], errors="coerce")
    float_shares = shares / (pct_a / 100.0)                          # 近似流通A股
    float_mv = close * float_shares                                  # 流通市值proxy(元)
    out = pd.DataFrame({
        "net_inflow": ni,
        "close": close,
        "float_mv": float_mv,
    })
    return out


def build_factor_panel(codes, max_n):
    rows = []
    t0 = datetime.now()
    for i, code in enumerate(codes[:max_n], 1):
        try:
            d = fetch_northbound_code(code)
        except Exception as e:
            print(f"  skip {code}: {repr(e)[:60]}")
            continue
        if d is None or d.shape[0] < 60:
            continue
        d["ni20"] = d["net_inflow"].rolling(20, min_periods=10).sum()
        d["factor"] = d["ni20"] / d["float_mv"]
        d["code"] = code
        rows.append(d.reset_index()[["date", "code", "factor"]])
        if i % 10 == 0:
            print(f"  [{datetime.now()}] 完成 {i}/{min(max_n,len(codes))} 只")
    if not rows:
        return None
    long = pd.concat(rows, ignore_index=True)
    panel = long.pivot(index="date", columns="code", values="factor").sort_index()
    return panel


def ic_test(factor_panel, close_panel):
    """截面 Rank IC（月度）：factor[t] vs 未来21日收益[t]。"""
    fwd = compute_fwd_return(close_panel, 21)
    me = pd.DatetimeIndex(get_month_end_dates(close_panel.index)).normalize()
    me = me[(me >= factor_panel.index.min()) & (me <= factor_panel.index.max())]
    ics = []
    for t in me:
        if t not in factor_panel.index or t not in fwd.index:
            continue
        f = factor_panel.loc[t].dropna()
        y = fwd.loc[t].dropna()
        common = f.index.intersection(y.index)
        if len(common) < 20:
            continue
        ic = f[common].corr(y[common], method="spearman")
        if pd.notna(ic):
            ics.append(ic)
    ics = pd.Series(ics)
    return ics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()
    t0 = datetime.now()

    codes = list(pd.read_parquet(config.MB_CLOSE).columns)
    # 优先大票（北向覆盖更好）：用主板宇宙前 N 只
    uni = json_load_universe()
    codes = [c for c in uni if c in set(codes)][:args.n]

    close = pd.read_parquet(config.MB_CLOSE).reindex(columns=codes)

    print(f"[{datetime.now()}] 抽样 {len(codes)} 只，抓取北向...")
    fp = build_factor_panel(codes, args.n)
    if fp is None or fp.shape[1] < 10:
        print(">>> 抽样北向覆盖不足，无法检验")
        return
    print(f"  因子面板: {fp.shape[0]} 日 × {fp.shape[1]} 只，"
          f"覆盖 {fp.index[0].date()}~{fp.index[-1].date()}")

    ics = ic_test(fp, close)
    if len(ics) == 0:
        print(">>> 无有效 IC 月份")
        return
    mean_ic = ics.mean()
    std_ic = ics.std()
    ir = mean_ic / std_ic if std_ic else np.nan
    pos = (ics > 0).mean()
    print(f"\n=== 北向因子 Rank IC 检验（{len(ics)} 个月）===")
    print(f"  mean IC = {mean_ic:.4f}")
    print(f"  std IC  = {std_ic:.4f}")
    print(f"  IR      = {ir:.3f}")
    print(f"  正IC占比 = {pos:.1%}")
    verdict = "✅ 有正向信号，值得全量抓取做正式回测" if (mean_ic > 0.02 and ir > 0.2) else \
              ("⚠️ 弱信号" if mean_ic > 0 else "❌ 无稳健正向信号，不建议全量抓取")
    print(f"  {verdict}")
    print(f"  耗时 {datetime.now()-t0}")


def json_load_universe():
    import json
    p = config.DATA_DIR / "mainboard_universe.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return list(pd.read_parquet(config.MB_CLOSE).columns)


if __name__ == "__main__":
    main()
