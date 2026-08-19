# -*- coding: utf-8 -*-
"""
V8 数据抓取：扩展选股宇宙至 中证500(000905) ∪ 创业板指(399006) ∪ 中证1000(000852)，剔除ST。
仅改「股票池」，其余逻辑（因子/回测/滑点/Regime）一律沿用 V7.1。

数据源：
  - 日线：优先东方财富 stock_zh_a_hist（含成交额），沙箱若拦截则回退新浪 stock_zh_a_daily（含成交额）。
  - 指数：CSI300 沿用 v6_index.parquet（与宇宙无关，已抓至 2025-08-12）。
  - 财报：stock_financial_analysis_indicator 按季拉取，复用 V7「披露延迟映射 + ffill」point-in-time 对齐。

输出：
  v8_close_panel.parquet    (date × code 收盘价，2018-2025)
  v8_amount_panel.parquet   (date × code 成交额，2018-2025，真实额)
  v8_universe.json          (最终剔除ST后的代码列表，便于信号器复用)
  roe_panel_v8.parquet      (date × code 净资产收益率%)
  gpm_yoy_panel_v8.parquet  (date × code 毛利率同比)

断点续跑：日线与财报各有独立 checkpoint，重跑跳过已完成代码。
运行：python fetch_all_v8.py   （建议后台：数据量≈V7 的 3 倍，预计 1-2 小时）
"""

from __future__ import annotations

import os
import json
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


START_D = "20180101"
END_D = "20250812"
LAG_MAP = config.DISCLOSURE_LAG_MONTHS

CKPT_DAILY = config.DATA_DIR / "v8_daily_raw.parquet"
CKPT_FIN = config.DATA_DIR / "v8_fin_raw.parquet"


# ============================================================================
# 1. 选股宇宙：中证500 ∪ 创业板指 ∪ 中证1000
# ============================================================================
def _prefix_sina(code: str) -> str:
    if code.startswith(("6", "9")) or code.startswith("688"):
        return "sh" + code
    if code.startswith(("8", "4")):   # 北交所
        return "bj" + code
    return "sz" + code


def _cons_csindex(symbol: str) -> set:
    df = ak.index_stock_cons_csindex(symbol=symbol)
    col = "成分券代码" if "成分券代码" in df.columns else df.columns[1]
    return set(df[col].astype(str).str.zfill(6).tolist())


def _cons_eastmoney(symbol: str) -> set:
    df = ak.index_stock_cons(symbol=symbol)
    col = "品种代码" if "品种代码" in df.columns else df.columns[1]
    return set(df[col].astype(str).str.zfill(6).tolist())


def get_universe_v8() -> list:
    """取 中证500 ∪ 创业板指 ∪ 中证1000 并集，剔除 ST。"""
    c500 = _cons_csindex("000905")                 # 中证500 (csindex)
    try:
        cne = _cons_eastmoney("399006")            # 创业板指 (eastmoney)
    except Exception:
        cne = _cons_csindex("399006")
    # 中证1000：csindex 优先，失败回退 eastmoney
    try:
        c1000 = _cons_csindex("000852")
        src1000 = "csindex"
    except Exception as e:
        print(f"  [warn] 中证1000 csindex 失败({repr(e)[:60]})，回退 eastmoney")
        c1000 = _cons_eastmoney("000852")
        src1000 = "eastmoney"

    union = c500 | cne | c1000
    # ST 剔除
    names = ak.stock_info_a_code_name()
    st = set(names[names["name"].astype(str).str.contains("ST", na=False)]["code"]
             .astype(str).str.zfill(6).tolist())
    codes = sorted(union - st)
    print(f"宇宙：500={len(c500)} 创业板={len(cne)} 1000({src1000})={len(c1000)} "
          f"→ 并集 {len(union)} → 剔除ST后 {len(codes)}")
    return codes


# ============================================================================
# 2. 日线：双源（东方财富优先 / 新浪回退）
# ============================================================================
def fetch_east(code: str, retries: int = 2):
    for _ in range(retries):
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                    start_date=START_D, end_date=END_D, adjust="qfq")
            if df is None or len(df) == 0:
                return None
            df = df.rename(columns={"日期": "date", "收盘": "close", "成交额": "amount"})
            df["date"] = pd.to_datetime(df["date"])
            return df.set_index("date")[["close", "amount"]].astype(float)
        except Exception:
            time.sleep(0.5)
    return None


def fetch_sina(code: str, retries: int = 3):
    last = None
    for _ in range(retries):
        try:
            df = ak.stock_zh_a_daily(symbol=_prefix_sina(code), start_date=START_D,
                                     end_date=END_D, adjust="qfq")
            if df is None or len(df) == 0:
                return None
            df = df.rename(columns={"date": "date", "close": "close", "amount": "amount"})
            df["date"] = pd.to_datetime(df["date"])
            return df.set_index("date")[["close", "amount"]].astype(float)
        except Exception as e:
            last = e
            time.sleep(0.6)
    return None


def fetch_daily(codes: list):
    """抓取全部代码 2018-2025 日线收盘价+成交额，落盘面板。"""
    # 探测东方财富可用性（仅首只试探一次，避免 1500 次无谓失败）
    use_east = False
    probe = fetch_east(codes[0])
    if probe is not None and len(probe) > 100:
        use_east = True
        print(f"东方财富源可用，优先使用（首只 {codes[0]} 取到 {len(probe)} 行）")
    else:
        print(f"东方财富源不可用（沙箱拦截/限流），回退新浪源；数据截止 {END_D}")

    done = set()
    parts = []
    if CKPT_DAILY.exists():
        prev = pd.read_parquet(CKPT_DAILY)
        parts.append(prev)
        done = set(prev["code"].unique().tolist())
        print(f"断点续跑：已完成 {len(done)} 只")

    ok = 0
    for i, code in enumerate(codes, 1):
        if code in done:
            ok += 1
            continue
        d = fetch_east(code) if use_east else None
        if d is None:
            d = fetch_sina(code)
        if d is not None and len(d):
            dd = d.reset_index()
            dd["code"] = code
            parts.append(dd[["date", "code", "close", "amount"]])
            ok += 1
        else:
            print(f"  skip {code}: 双源均无数据")
        if i % 50 == 0:
            pd.concat(parts, ignore_index=True).to_parquet(CKPT_DAILY)
            print(f"  [日线] 进度 {i}/{len(codes)} 成功 {ok}")
        time.sleep(0.05)

    pd.concat(parts, ignore_index=True).to_parquet(CKPT_DAILY)
    raw = pd.read_parquet(CKPT_DAILY)
    close = raw.pivot(index="date", columns="code", values="close").reindex(columns=codes)
    amount = raw.pivot(index="date", columns="code", values="amount").reindex(columns=codes)
    close = close[~close.index.duplicated(keep="last")].sort_index()
    amount = amount[~amount.index.duplicated(keep="last")].sort_index()
    close.to_parquet(config.DATA_DIR / "v8_close_panel.parquet")
    amount.to_parquet(config.DATA_DIR / "v8_amount_panel.parquet")
    print(f"日线面板：close={close.shape} amount={amount.shape} "
          f"交易日 {close.index[0].date()}~{close.index[-1].date()} "
          f"全样本覆盖率={close.notna().mean().mean():.1%}")
    return close, amount


# ============================================================================
# 3. 财报：point-in-time（披露延迟映射 + ffill），复用 V7 逻辑
# ============================================================================
def available_date(report_date: pd.Timestamp, lag_map: dict) -> pd.Timestamp:
    lag = lag_map.get(report_date.strftime("%m-%d"), 4)
    y, m = report_date.year, report_date.month
    m0 = m + lag
    y += (m0 - 1) // 12
    m = (m0 - 1) % 12 + 1
    return pd.Timestamp(y, m, 1)


def fetch_fin_code(code: str):
    try:
        fi = ak.stock_financial_analysis_indicator(symbol=code)
    except Exception as e:
        return None, f"fetch_err:{repr(e)[:80]}"
    if "日期" not in fi.columns:
        return None, "no_date_col"
    rows = []
    for _, r in fi.iterrows():
        rd = r["日期"]
        if rd is None or (isinstance(rd, float) and np.isnan(rd)):
            continue
        if not isinstance(rd, pd.Timestamp):
            try:
                rd = pd.Timestamp(rd)
            except Exception:
                continue
        roe = r.get("净资产收益率(%)")
        gpm = r.get("销售毛利率(%)")
        rows.append((code, rd, roe, gpm))
    if not rows:
        return None, "empty"
    return pd.DataFrame(rows, columns=["code", "date", "roe", "gpm"]), "ok"


def build_fin_panels(raw_long: pd.DataFrame, trade_idx: pd.DatetimeIndex, codes: list):
    raw = raw_long.dropna(subset=["roe", "gpm"], how="all").copy()

    def point_in_time(col):
        out = pd.DataFrame(index=trade_idx, columns=codes, dtype=float)
        for code in codes:
            sub = raw[raw["code"] == code][["date", col]].dropna()
            if sub.empty:
                continue
            sub = sub.copy()
            sub["av"] = sub["date"].map(lambda d: available_date(d, LAG_MAP))
            s = pd.Series(sub[col].values, index=pd.DatetimeIndex(sub["av"].values))
            s = s[~s.index.duplicated(keep="last")].sort_index()
            union = trade_idx.union(s.index, sort=True)
            full = s.reindex(union).ffill().reindex(trade_idx)
            out[code] = full
        return out

    roe_panel = point_in_time("roe")

    gpm_yoy_panel = pd.DataFrame(index=trade_idx, columns=codes, dtype=float)
    for code in codes:
        sub = raw[raw["code"] == code][["date", "gpm"]].dropna()
        if sub.empty or len(sub) < 2:
            continue
        sub = sub.copy()
        sub["year"] = sub["date"].dt.year
        annual = sub.groupby("year")["gpm"].last()
        annual_yoy = annual.diff(1)
        av_dates = [available_date(pd.Timestamp(y, 12, 31), LAG_MAP) for y in annual_yoy.index]
        s = pd.Series(annual_yoy.values, index=pd.DatetimeIndex(av_dates))
        s = s[~s.index.duplicated(keep="last")].sort_index()
        union = trade_idx.union(s.index, sort=True)
        full = s.reindex(union).ffill().reindex(trade_idx)
        gpm_yoy_panel[code] = full
    return roe_panel, gpm_yoy_panel


def fetch_fundamentals(codes: list, trade_idx: pd.DatetimeIndex):
    done = set()
    parts = []
    if CKPT_FIN.exists():
        prev = pd.read_parquet(CKPT_FIN)
        parts.append(prev)
        done = set(prev["code"].unique().tolist())
        print(f"断点续跑(财报)：已完成 {len(done)} 只")

    ok = 0
    for i, code in enumerate(codes, 1):
        if code in done:
            ok += 1
            continue
        df, status = fetch_fin_code(code)
        if df is not None:
            parts.append(df)
            ok += 1
        else:
            print(f"  skip {code}: {status}")
        if i % 25 == 0:
            pd.concat(parts, ignore_index=True).to_parquet(CKPT_FIN)
            print(f"  [财报] 进度 {i}/{len(codes)} 成功 {ok}")
        time.sleep(0.25)

    combined = pd.concat(parts, ignore_index=True)
    combined.to_parquet(CKPT_FIN)
    print(f"财报抓取完成 raw 行数={len(combined)} 成功代码={combined['code'].nunique()}")

    roe_panel, gpm_yoy_panel = build_fin_panels(combined, trade_idx, codes)
    roe_panel.to_parquet(config.DATA_DIR / "roe_panel_v8.parquet")
    gpm_yoy_panel.to_parquet(config.DATA_DIR / "gpm_yoy_panel_v8.parquet")
    print(f"roe 非空={roe_panel.notna().sum().sum()}  gpm_yoy 非空={gpm_yoy_panel.notna().sum().sum()}")
    for yr in [2023, 2024, 2025]:
        m = roe_panel.loc[f"{yr}-06-01":f"{yr}-12-31"]
        print(f"  {yr} H2 roe 月均非空率={m.notna().mean().mean():.1%}")
    return roe_panel, gpm_yoy_panel


# ============================================================================
def main():
    t0 = time.time()
    codes = get_universe_v8()
    (config.DATA_DIR / "v8_universe.json").write_text(json.dumps(codes), encoding="utf-8")

    close, amount = fetch_daily(codes)

    # 指数 CSI300 复用 v6_index（与宇宙无关）
    if (config.DATA_DIR / "v6_index.parquet").exists():
        print("CSI300 指数复用 v6_index.parquet（与宇宙无关，数据截止 2025-08-12）")
    else:
        print("[warn] v6_index.parquet 缺失，请先运行 fetch_v6_data.py")

    fetch_fundamentals(codes, pd.DatetimeIndex(close.index))

    print(f"\n=== V8 数据全部就绪 ===  耗时 {(time.time()-t0)/60:.1f} 分钟")
    print(f"宇宙 {len(codes)} 只，交易日 {close.index[0].date()}~{close.index[-1].date()}")


if __name__ == "__main__":
    main()
