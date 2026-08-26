# -*- coding: utf-8 -*-
"""
fetch_mainboard.py — 主板版数据抓取（选股池：全市场主板 60开头 + 00开头）
====================================================================

背景：V8.1 选股池为中证500∪创业板∪中证1000（含 300/688/8 开头，用户无这些板块交易权限）。
本脚本改为抓取「沪深主板」股票：
  - 沪市主板：60 开头
  - 深市主板：00 开头（含 000/001/002/003，中小板并入深主板）
  - 排除：30（创业板）/ 688（科创板）/ 8/4/920（北交所）
  - 排除 ST / *ST / 退市股（名称含 ST、退）
  - 上市不满 60 天：由回测 mask_new_listings 统一处理（与 V8 一致）
  - 停牌股：面板构建后剔除最近 60 交易日无有效成交的列（近似过滤）

数据源（沙箱实测 2026-08-25）：
  - 东方财富全市场快照 stock_zh_a_spot_em / 日线 stock_zh_a_hist 被沙箱网络拦截（不可用）
  - 名称列表：stock_info_a_code_name（可用，缓存于 data/stock_names.json）
  - 日线：新浪 stock_zh_a_daily（可用，qfq，含 open/high/low/close/volume/amount）
  - 财报：stock_financial_analysis_indicator（可用，V8 已验证）

输出（data/）：
  mainboard_universe.json        主板股票池（3046 只，剔除 ST）
  mainboard_close_panel.parquet  (date × code 收盘价，qfq，2018-01-01 起)
  mainboard_amount_panel.parquet (date × code 成交额，元)
  mainboard_ohlcv.pkl            {code: DataFrame(index=date, open/high/low/close)}（dashboard/ATR 用）
  roe_panel_mainboard.parquet    (date × code 净资产收益率%)，PIT（披露延迟映射+ffill）
  gpm_yoy_panel_mainboard.parquet(date × code 毛利率同比%)，PIT

断点续跑：日线/财报各有独立 checkpoint（mainboard_daily_raw / mainboard_fin_raw），重跑跳过已完成。
运行：cd src && python fetch_mainboard.py            # 全量（约 1.5 小时）
      python fetch_mainboard.py --limit 20          # 试跑 20 只（验证链路）
      python fetch_mainboard.py --start N           # 从第 N 只续跑（按宇宙顺序）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
          "ALL_PROXY", "all_proxy"]:
    os.environ.pop(k, None)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import akshare as ak

import config

START_D = "20180101"                       # 回测起点（与 V8 一致）
END_D = datetime.now().strftime("%Y%m%d")  # 最新（2026-08-25）
LAG_MAP = config.DISCLOSURE_LAG_MONTHS

CKPT_DAILY = config.DATA_DIR / "mainboard_daily_raw.parquet"
CKPT_FIN = config.DATA_DIR / "mainboard_fin_raw.parquet"

OUT_CLOSE = config.DATA_DIR / "mainboard_close_panel.parquet"
OUT_AMOUNT = config.DATA_DIR / "mainboard_amount_panel.parquet"
OUT_OHLCV = config.DATA_DIR / "mainboard_ohlcv.pkl"
OUT_ROE = config.DATA_DIR / "roe_panel_mainboard.parquet"
OUT_GPM = config.DATA_DIR / "gpm_yoy_panel_mainboard.parquet"
OUT_UNIVERSE = config.DATA_DIR / "mainboard_universe.json"

# 限流控制：新浪单只失败重试次数与间隔（用户允许加长重试间隔）
SINA_RETRIES = 5
SINA_SLEEP = 1.0
BATCH_SAVE = 50                           # 每 50 只落盘 checkpoint
FIN_SLEEP = 0.3                           # 财报请求间隔（同花顺源实测不限流，0.3s 稳妥）
FIN_RETRIES = 3                           # 财报失败重试次数（指数退避）
FIN_RETRY_SLEEP = 2.0                     # 财报重试基础间隔（s）


# ============================================================================
# 1. 主板股票池
# ============================================================================
def get_universe_mainboard(refresh: bool = False) -> list:
    """全市场主板（60/00 开头）剔除 ST。名称优先读本地缓存，refresh=True 强制刷新。"""
    names_path = config.DATA_DIR / "stock_names.json"
    if refresh or not names_path.exists():
        df = ak.stock_info_a_code_name()
        names = dict(zip(df["code"].astype(str).str.zfill(6), df["name"].astype(str)))
        names_path.write_text(json.dumps(names, ensure_ascii=False), encoding="utf-8")
        print(f"[universe] 名称列表已刷新：{len(names)} 只")
    else:
        names = json.loads(names_path.read_text(encoding="utf-8"))

    codes = []
    for c, name in names.items():
        if not c.startswith(("60", "00")):
            continue                      # 排除 30/688/8/4/920（创业/科创/北交）
        if "ST" in name.upper() or "退" in name:
            continue                      # 排除 ST/*ST/退市
        codes.append(c)
    codes = sorted(set(codes))
    print(f"[universe] 主板非ST：{len(codes)} 只（60/00 开头，剔除 ST/退市）")
    return codes


# ============================================================================
# 2. 日线（新浪单源，qfq；EM 被沙箱拦截）
# ============================================================================
def _prefix_sina(code: str) -> str:
    return "sh" + code if code.startswith("6") else "sz" + code


def fetch_sina(code: str):
    """新浪日线（qfq，2018-01-01 起全历史）。失败重试 SINA_RETRIES 次，间隔 SINA_SLEEP。"""
    last = None
    for _ in range(SINA_RETRIES):
        try:
            df = ak.stock_zh_a_daily(symbol=_prefix_sina(code), start_date=START_D,
                                     end_date=END_D, adjust="qfq")
            if df is None or len(df) == 0:
                return None
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            cols = ["open", "high", "low", "close", "amount"]
            if "volume" in df.columns:
                cols.append("volume")
            return df[cols].astype(float)
        except Exception as e:
            last = e
            time.sleep(SINA_SLEEP)
    print(f"  [warn] {code} 新浪日线失败: {repr(last)[:80]}")
    return None


def fetch_daily(codes: list, limit: int = 0, start_at: int = 0):
    """抓取日线 → close/amount 面板 + ohlcv dict。断点续传。"""
    if limit:
        codes = codes[start_at:start_at + limit]
    else:
        codes = codes[start_at:]

    done = set()
    parts = []
    if CKPT_DAILY.exists():
        prev = pd.read_parquet(CKPT_DAILY)
        parts.append(prev)
        done = set(prev["code"].unique().tolist())
        print(f"[daily] 断点续传：已完成 {len(done)} 只")

    ok = 0
    t0 = time.time()
    for i, code in enumerate(codes, 1):
        if code in done:
            ok += 1
            continue
        d = fetch_sina(code)
        if d is not None and len(d):
            dd = d.reset_index()
            dd["code"] = code
            parts.append(dd[["date", "code", "open", "high", "low", "close", "amount"]])
            ok += 1
        else:
            print(f"  skip {code}: 无数据（可能长期停牌/退市）")
        if i % BATCH_SAVE == 0:
            pd.concat(parts, ignore_index=True).to_parquet(CKPT_DAILY)
            el = (time.time() - t0) / 60
            rate = ok / el if el > 0 else 0
            print(f"  [daily] 进度 {i}/{len(codes)} 成功 {ok} 耗时 {el:.1f}min "
                  f"({rate:.0f} 只/min)")
        time.sleep(0.05)                  # 控制请求频率，避免新浪限流

    if parts:
        pd.concat(parts, ignore_index=True).to_parquet(CKPT_DAILY)

    raw = pd.read_parquet(CKPT_DAILY)
    raw = raw[raw["code"].isin(codes) | raw["code"].isin(done)] if False else raw
    close = raw.pivot(index="date", columns="code", values="close").sort_index()
    amount = raw.pivot(index="date", columns="code", values="amount").sort_index()
    close = close[~close.index.duplicated(keep="last")]
    amount = amount[~amount.index.duplicated(keep="last")]

    # 停牌/退市近似过滤：最近 60 交易日无有效成交的列剔除
    last60 = close.tail(60)
    valid = last60.notna().mean() > 0.2   # 近 60 日至少 20% 有价
    close = close.loc[:, valid]
    amount = amount.loc[:, valid]
    dropped = len(valid) - int(valid.sum())
    print(f"[daily] 停牌/退市近似过滤剔除 {dropped} 只（近60交易日 <20% 有价）")

    close.to_parquet(OUT_CLOSE)
    amount.to_parquet(OUT_AMOUNT)

    # OHLCV dict（dashboard / ATR 用）
    ohlcv = {}
    for code in close.columns:
        sub = raw[raw["code"] == code].set_index("date")
        ohlcv[code] = sub[["open", "high", "low", "close"]].astype(float)
    with open(OUT_OHLCV, "wb") as f:
        import pickle
        pickle.dump(ohlcv, f)

    print(f"[daily] 面板落盘：close={close.shape} amount={amount.shape} "
          f"交易日 {close.index[0].date()}~{close.index[-1].date()} "
          f"覆盖率={close.notna().mean().mean():.1%}")
    return close, amount, ohlcv


# ============================================================================
# 3. 财报：point-in-time（披露延迟映射 + ffill），复用 V8 逻辑
# ============================================================================
def available_date(report_date: pd.Timestamp, lag_map: dict) -> pd.Timestamp:
    lag = lag_map.get(report_date.strftime("%m-%d"), 4)
    y, m = report_date.year, report_date.month
    m0 = m + lag
    y += (m0 - 1) // 12
    m = (m0 - 1) % 12 + 1
    return pd.Timestamp(y, m, 1)


def _to_num(x):
    """清洗 THS 财报数值：'2.82%' / '1,234.5' → float；空/异常 → NaN。"""
    if x is None:
        return np.nan
    if isinstance(x, (int, float)):
        return float(x) if x == x else np.nan
    s = str(x).strip().replace("%", "").replace(",", "")
    if not s or s.lower() in ("nan", "none", "-"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def fetch_fin_code(code: str):
    """财报：stock_financial_abstract_ths（同花顺源，'按报告期'）。

    源选择记录（2026-08-25 实测）：
      - V8 用的 stock_financial_analysis_indicator（新浪）当前返回空（接口失效）
      - stock_financial_abstract（东财）ROE 季度齐全但**累计请求触发 IP 级封禁**
        （~450 只后全量 JSONDecodeError，3s 间隔也无效），不可靠
      - stock_financial_abstract_ths（同花顺 '按报告期'）**ROE/毛利率 100% 覆盖、
        不限流、~0.7s/只** ✅；注意其数值为带 % 字符串，需 _to_num 清洗
    返回 long 格式 (code, 报告期, roe, gpm)。
    """
    last = None
    for attempt in range(FIN_RETRIES):
        try:
            df = ak.stock_financial_abstract_ths(symbol=code, indicator="按报告期")
        except Exception as e:
            last = f"fetch_err:{repr(e)[:80]}"
            time.sleep(FIN_RETRY_SLEEP * (attempt + 1))
            continue
        if df is None or len(df) == 0 or "报告期" not in df.columns:
            last = "empty"
            time.sleep(FIN_RETRY_SLEEP * (attempt + 1))
            continue
        roe_col = "净资产收益率" if "净资产收益率" in df.columns else None
        gpm_col = "销售毛利率" if "销售毛利率" in df.columns else None
        rows = []
        for _, r in df.iterrows():
            rd = r["报告期"]
            try:
                rd = pd.Timestamp(str(rd))
            except Exception:
                continue
            roe = _to_num(r.get(roe_col)) if roe_col else None
            gpm = _to_num(r.get(gpm_col)) if gpm_col else None
            rows.append((code, rd, roe, gpm))
        if not rows:
            last = "empty"
            continue
        return pd.DataFrame(rows, columns=["code", "date", "roe", "gpm"]), "ok"
    return None, str(last)


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


def fetch_fundamentals(codes: list, trade_idx: pd.DatetimeIndex, limit: int = 0):
    if limit:
        codes = codes[:limit]
    done = set()
    parts = []
    if CKPT_FIN.exists():
        prev = pd.read_parquet(CKPT_FIN)
        parts.append(prev)
        done = set(prev["code"].unique().tolist())
        print(f"[fin] 断点续传：已完成 {len(done)} 只")

    ok = 0
    t0 = time.time()
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
            print(f"  [fin] 进度 {i}/{len(codes)} 成功 {ok} "
                  f"耗时 {(time.time()-t0)/60:.1f}min")
        time.sleep(FIN_SLEEP)

    combined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["code", "date", "roe", "gpm"])
    combined.to_parquet(CKPT_FIN)
    print(f"[fin] 财报 raw 行数={len(combined)} 成功代码={combined['code'].nunique()}")

    roe_panel, gpm_yoy_panel = build_fin_panels(combined, trade_idx, list(codes))
    roe_panel.to_parquet(OUT_ROE)
    gpm_yoy_panel.to_parquet(OUT_GPM)
    print(f"[fin] roe 非空={roe_panel.notna().sum().sum()}  "
          f"gpm_yoy 非空={gpm_yoy_panel.notna().sum().sum()}")
    return roe_panel, gpm_yoy_panel


# ============================================================================
# 方向2：行业中性化数据（2026-08-26 追加）
#   industry_map.parquet       pool 股票 → 申万一级行业（缺失归"其他"）
#   industry_capital.parquet   pool 股票 → 注册资金（万元，≈总股本×面值，巨潮快照）
#   industry_benchmark.parquet 月末 × 行业 市值权重（市值=注册资金×1e4×收盘价）
# ============================================================================
OUT_IND_MAP = config.DATA_DIR / "industry_map.parquet"
OUT_IND_CAP = config.DATA_DIR / "industry_capital.parquet"
OUT_IND_BENCH = config.DATA_DIR / "industry_benchmark.parquet"
SW_MAP_LOCAL = config.DATA_DIR / "sw_industry_map.parquet"


def get_pool_codes() -> list:
    """方向2 基准池 = 回测选股池（V8 成分 ∩ 主板 60/00 ∩ mainboard 面板，1004 只）。"""
    from fetch_dividend import get_v2_codes
    return sorted(get_v2_codes())


def fetch_industry_map(pool_codes: list = None, force: bool = False) -> pd.DataFrame:
    """pool 股票 → 申万一级行业。优先复用本地 sw_industry_map.parquet（5207 只全市场，
    2026-08-19 已抓）；缺失时尝试 ak.stock_industry_category_cninfo 增量，仍缺失归"其他"。
    落盘 data/industry_map.parquet（code, industry）。"""
    if pool_codes is None:
        pool_codes = get_pool_codes()
    out = None
    if SW_MAP_LOCAL.exists():
        sw = pd.read_parquet(SW_MAP_LOCAL)
        sw = sw[["code", "industry"]].drop_duplicates("code")
        out = sw[sw["code"].isin(pool_codes)].copy()
    if out is None or out.empty:
        print("[ind] 本地 sw_industry_map 缺失，尝试巨潮行业分类抓取")
        try:
            df = ak.stock_industry_category_cninfo(symbol="A股", date="2026-08-25")
            out = df[["股票代码", "行业"]].rename(
                columns={"股票代码": "code", "行业": "industry"})
            out["code"] = out["code"].astype(str).str.zfill(6)
            out = out[out["code"].isin(pool_codes)]
        except Exception as e:
            print(f"[ind] 巨潮行业分类失败: {e}")
    missing = [c for c in pool_codes if c not in set(out["code"])]
    if missing:
        pad = pd.DataFrame({"code": missing, "industry": ["其他"] * len(missing)})
        out = pd.concat([out, pad], ignore_index=True)
    out.to_parquet(OUT_IND_MAP)
    print(f"[ind] industry_map: {len(out)} 只（pool {len(pool_codes)}），"
          f"行业数 {out['industry'].nunique()}，缺失归'其他' {len(missing)} 只")
    return out


def fetch_registered_capital(pool_codes: list = None, limit: int = 0,
                             start_at: int = 0) -> pd.DataFrame:
    """pool 股票注册资金（巨潮 stock_profile_cninfo 快照，万元 ≈ 总股本×面值 元口径）。
    断点续跑：data/industry_capital.parquet 已存在的 code 跳过。
    落盘 data/industry_capital.parquet（code, capital_wan）。"""
    if pool_codes is None:
        pool_codes = get_pool_codes()
    codes = pool_codes[start_at:]
    if limit:
        codes = codes[:limit]
    parts = []
    done = set()
    if OUT_IND_CAP.exists():
        prev = pd.read_parquet(OUT_IND_CAP)
        parts.append(prev)
        done = set(prev["code"].unique())
        print(f"[cap] 断点续传：已完成 {len(done)} 只")

    ok, fail = 0, 0
    t0 = time.time()
    for i, code in enumerate(codes, 1):
        if code in done:
            ok += 1
            continue
        cap = None
        for attempt in range(3):
            try:
                df = ak.stock_profile_cninfo(symbol=code)
                if df is not None and len(df):
                    raw = str(df.iloc[0]["注册资金"]).strip()
                    cap = float(raw) if raw and raw.lower() not in ("nan", "none", "-") else None
                break
            except Exception:
                time.sleep(1.5 * (attempt + 1))
        if cap is not None:
            parts.append(pd.DataFrame([{"code": code, "capital_wan": cap}]))
            ok += 1
        else:
            fail += 1
            print(f"  skip {code}（注册资金不可得）")
        if i % 50 == 0:
            pd.concat(parts, ignore_index=True).to_parquet(OUT_IND_CAP)
            print(f"  [cap] 进度 {i}/{len(codes)} 成功 {ok} 失败 {fail} "
                  f"耗时 {(time.time()-t0)/60:.1f}min")
        time.sleep(0.15)
    if parts:
        pd.concat(parts, ignore_index=True).to_parquet(OUT_IND_CAP)
    comb = pd.read_parquet(OUT_IND_CAP) if OUT_IND_CAP.exists() else pd.DataFrame(
        columns=["code", "capital_wan"])
    print(f"[cap] 注册资金快照完成：{comb['code'].nunique()} 只（本批成功 {ok} 失败 {fail}）")
    return comb


def build_industry_benchmark(close_panel: pd.DataFrame = None,
                             month_ends=None) -> pd.DataFrame:
    """月末 × 行业 市值权重基准。市值 = 注册资金(万元)×1e4 × 收盘价（qfq）。
    权重 = 行业市值 / 全池市值。落盘 data/industry_benchmark.parquet。"""
    if close_panel is None:
        close_panel = pd.read_parquet(config.MB_CLOSE)
    from factors import get_month_end_dates
    ind = pd.read_parquet(OUT_IND_MAP)
    cap = pd.read_parquet(OUT_IND_CAP)
    cap_map = dict(zip(cap["code"], cap["capital_wan"]))
    ind_map = dict(zip(ind["code"], ind["industry"]))
    pool = [c for c in close_panel.columns if c in cap_map and c in ind_map]
    if month_ends is None:
        me = get_month_end_dates(close_panel.index)
        month_ends = pd.DatetimeIndex(me).normalize()
    me = pd.DatetimeIndex(month_ends).normalize()

    shares = pd.Series({c: cap_map[c] * 1e4 for c in pool})   # 股本(股)，静态快照
    rows = []
    for t in me:
        if t not in close_panel.index:
            continue
        px = close_panel.loc[t, pool].astype(float)
        mcap = (px * shares).dropna()
        if mcap.empty or mcap.sum() <= 0:
            continue
        total = mcap.sum()
        by_ind = {}
        for c, mv in mcap.items():
            s = ind_map.get(c, "其他")
            by_ind[s] = by_ind.get(s, 0.0) + mv
        row = {s: v / total for s, v in by_ind.items()}
        row["date"] = t
        rows.append(row)
    bench = pd.DataFrame(rows).set_index("date").sort_index()
    bench.to_parquet(OUT_IND_BENCH)
    print(f"[ind] industry_benchmark: {bench.shape[0]} 个月末 × {bench.shape[1]} 行业，"
          f"覆盖 pool {len(pool)} 只")
    return bench


# ============================================================================
def main():
    ap = argparse.ArgumentParser(description="主板版数据抓取")
    ap.add_argument("--limit", type=int, default=0, help="试跑：只抓前 N 只")
    ap.add_argument("--start", type=int, default=0, help="从宇宙第 N 只开始续跑")
    ap.add_argument("--daily-only", action="store_true", help="只抓日线")
    ap.add_argument("--fin-only", action="store_true", help="只抓财报")
    ap.add_argument("--refresh-names", action="store_true", help="强制刷新名称列表")
    ap.add_argument("--industry-map", action="store_true",
                    help="方向2：行业映射 + 注册资金快照 + 市值基准（不抓行情）")
    args = ap.parse_args()

    t0 = time.time()

    if args.industry_map:
        pool = get_pool_codes()
        print(f"[main] 方向2 基准池 {len(pool)} 只")
        fetch_industry_map(pool)
        fetch_registered_capital(pool, limit=args.limit, start_at=args.start)
        build_industry_benchmark()
        print(f"\n=== 方向2 行业数据就绪 === 耗时 {(time.time()-t0)/60:.1f} 分钟")
        return

    codes = get_universe_mainboard(refresh=args.refresh_names)
    (config.DATA_DIR / "mainboard_universe.json").write_text(
        json.dumps(codes), encoding="utf-8")
    print(f"[main] 主板宇宙 {len(codes)} 只，数据区间 {START_D} ~ {END_D}")

    if args.daily_only:
        fetch_daily(codes, limit=args.limit, start_at=args.start)
        return
    if args.fin_only:
        close = pd.read_parquet(OUT_CLOSE)
        fetch_fundamentals(codes, pd.DatetimeIndex(close.index), limit=args.limit)
        return

    close, amount, ohlcv = fetch_daily(codes, limit=args.limit, start_at=args.start)
    fetch_fundamentals(codes, pd.DatetimeIndex(close.index), limit=args.limit)

    print(f"\n=== 主板数据全部就绪 ===  耗时 {(time.time()-t0)/60:.1f} 分钟")
    print(f"宇宙 {len(close.columns)} 只，交易日 {close.index[0].date()}~{close.index[-1].date()}")


if __name__ == "__main__":
    main()
