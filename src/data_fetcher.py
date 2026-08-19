# -*- coding: utf-8 -*-
"""
数据获取模块：通过 akshare 获取沪深300成分股日线与财报，并做本地缓存与
点对点（point-in-time）对齐，杜绝未来函数。

说明：
- 日线使用新浪源（ak.stock_zh_a_daily，前复权），东方财富 push2his 源在本环境被网络拦截。
- 财报使用 ak.stock_financial_analysis_indicator，提取「净利润增长率(%)」。
- 财报按真实披露截止日 + 延迟映射，得到「数据可用起始日」，再做前向填充，
  确保回测在某交易日只能看到当时已披露的财务数据。
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import akshare as ak

import config


# ----------------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------------
def _code_to_sina_symbol(code: str) -> str:
    """将 6 位代码转换为新浪日线接口所需的 sh/sz 前缀符号。"""
    code = code.strip()
    if code.startswith("6"):
        return "sh" + code
    return "sz" + code


def _add_months(dt: pd.Timestamp, months: int) -> pd.Timestamp:
    """日期加 N 个月，返回该月首日的 Timestamp。"""
    y = dt.year + (dt.month - 1 + months) // 12
    m = (dt.month - 1 + months) % 12 + 1
    return pd.Timestamp(year=y, month=m, day=1)


# ----------------------------------------------------------------------------
# 成分股宇宙
# ----------------------------------------------------------------------------
def get_universe() -> List[str]:
    """获取沪深300当前成分股代码列表（6 位字符串）。"""
    cache = config.DATA_DIR / "universe.parquet"
    if cache.exists():
        return pd.read_parquet(cache)["code"].astype(str).tolist()
    cons = ak.index_stock_cons_csindex(symbol=config.INDEX_CODE)
    codes = cons["成分券代码"].astype(str).str.zfill(6).tolist()
    pd.DataFrame({"code": codes}).to_parquet(cache)
    return codes


# ----------------------------------------------------------------------------
# 单只股票日线（带缓存 + 重试）
# ----------------------------------------------------------------------------
def fetch_daily(code: str, start: str, end: str,
                max_retry: int = 3) -> Optional[pd.DataFrame]:
    """获取单只股票日线，结果缓存到 data/daily/{code}.parquet。"""
    path = config.DAILY_DIR / f"{code}.parquet"
    if path.exists():
        return pd.read_parquet(path)

    sym = _code_to_sina_symbol(code)
    for attempt in range(max_retry):
        try:
            df = ak.stock_zh_a_daily(
                symbol=sym, start_date=start, end_date=end, adjust="qfq"
            )
            if df is None or df.empty:
                return None
            df = df.rename(columns={"date": "trade_date"})
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df[["trade_date", "open", "high", "low", "close", "volume"]].copy()
            df = df.sort_values("trade_date").reset_index(drop=True)
            df.to_parquet(path)
            return df
        except Exception as exc:  # noqa: BLE001
            wait = 1.5 * (attempt + 1)
            print(f"  [warn] {code} 日线第 {attempt + 1} 次失败: {exc!r}; {wait}s 后重试")
            time.sleep(wait)
    return None


# ----------------------------------------------------------------------------
# 单只股票财报（带缓存 + 重试）
# ----------------------------------------------------------------------------
def fetch_financials(code: str, max_retry: int = 3) -> Optional[pd.DataFrame]:
    """获取单只股票财务指标，提取报告期与净利润同比增长率，缓存到 data/fin/{code}.parquet。"""
    path = config.FIN_DIR / f"{code}.parquet"
    if path.exists():
        return pd.read_parquet(path)

    for attempt in range(max_retry):
        try:
            fi = ak.stock_financial_analysis_indicator(symbol=code)
            if fi is None or fi.empty or "净利润增长率(%)" not in fi.columns:
                return None
            out = fi[["日期", "净利润增长率(%)"]].copy()
            out = out.rename(columns={"日期": "report_date",
                                      "净利润增长率(%)": "npg"})
            out["report_date"] = pd.to_datetime(out["report_date"])
            out["npg"] = pd.to_numeric(out["npg"], errors="coerce")
            out = out.dropna(subset=["report_date"]).sort_values("report_date")
            out.to_parquet(path)
            return out
        except Exception as exc:  # noqa: BLE001
            wait = 1.5 * (attempt + 1)
            print(f"  [warn] {code} 财报第 {attempt + 1} 次失败: {exc!r}; {wait}s 后重试")
            time.sleep(wait)
    return None


# ----------------------------------------------------------------------------
# 批量获取（可重入：已缓存的跳过）
# ----------------------------------------------------------------------------
def fetch_universe_data(codes: List[str], start: str, end: str,
                        pause: float = 0.15) -> Dict[str, dict]:
    """批量获取全部成分股的日线与财报，返回 {code: {'daily': df, 'fin': df}}。

    pause 为每次请求间的节流间隔，降低被限频风险。
    """
    result: Dict[str, dict] = {}
    total = len(codes)
    for i, code in enumerate(codes, 1):
        daily = fetch_daily(code, start, end)
        fin = fetch_financials(code)
        if daily is not None:
            result[code] = {"daily": daily, "fin": fin}
        if i % 20 == 0:
            print(f"  进度 {i}/{total} 已获取 {len(result)} 只有效")
        time.sleep(pause)
    print(f"批量获取完成：有效股票 {len(result)}/{total}")
    return result


# ----------------------------------------------------------------------------
# 面板组装
# ----------------------------------------------------------------------------
def build_close_panel(data: Dict[str, dict]) -> pd.DataFrame:
    """组装收盘价面板：行=交易日，列=股票代码，值=前复权收盘价。"""
    frames = []
    for code, d in data.items():
        daily = d["daily"]
        if daily is None or daily.empty:
            continue
        s = daily.set_index("trade_date")["close"].rename(code)
        frames.append(s)
    panel = pd.concat(frames, axis=1).sort_index()
    return panel


def build_disclosure_aligned_npg(data: Dict[str, dict],
                                 trade_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """组装净利润同比增长率面板（点对点对齐）。

    对每只股票，将财报报告期按披露延迟映射为「可用起始日」，再前向填充到交易日序列，
    使回测在交易日 t 只能使用 t 之前已披露的财务数据。
    """
    trade_dates = pd.DatetimeIndex(sorted(set(trade_dates)))
    cols = {}
    for code, d in data.items():
        fin = d.get("fin")
        daily = d.get("daily")
        if fin is None or fin.empty or daily is None or daily.empty:
            continue
        fin = fin.copy()
        fin["month_day"] = fin["report_date"].dt.strftime("%m-%d")
        fin["lags"] = fin["month_day"].map(config.DISCLOSURE_LAG_MONTHS)
        fin = fin.dropna(subset=["lags"])
        fin["avail_from"] = [
            _add_months(rd, int(lg))
            for rd, lg in zip(fin["report_date"], fin["lags"])
        ]
        fin = fin.sort_values("avail_from")[["avail_from", "npg"]]
        fin_idx = fin.set_index("avail_from")["npg"].sort_index()
        series = fin_idx.reindex(trade_dates, method="ffill")
        first_avail = fin_idx.index.min()
        if first_avail is not None:
            series = series.where(trade_dates >= first_avail)
        cols[code] = series
    return pd.DataFrame(cols, index=trade_dates).sort_index()


def build_ohlcv_dict(data: Dict[str, dict]) -> Dict[str, pd.DataFrame]:
    """返回 {code: 日线 DataFrame(含 trade_date 索引)}。"""
    out = {}
    for code, d in data.items():
        daily = d.get("daily")
        if daily is not None and not daily.empty:
            out[code] = daily.set_index("trade_date").sort_index()
    return out


def load_or_fetch_all(start: str, end: str) -> Dict[str, dict]:
    """优先从缓存加载；若缓存不足则批量获取。"""
    codes = get_universe()
    cached = [c for c in codes if (config.DAILY_DIR / f"{c}.parquet").exists()]
    if len(cached) >= max(50, int(len(codes) * 0.9)):
        print(f"缓存已较完整（{len(cached)}/{len(codes)}），直接加载。")
        data = {}
        for code in codes:
            daily = fetch_daily(code, start, end)
            fin = fetch_financials(code)
            if daily is not None:
                data[code] = {"daily": daily, "fin": fin}
        return data
    print("缓存不足，开始批量获取（可重入，失败可续跑）...")
    return fetch_universe_data(codes, start, end)


if __name__ == "__main__":
    t0 = datetime.now()
    data = load_or_fetch_all(config.START_DATE, config.END_DATE)
    print(f"加载 {len(data)} 只股票，耗时 {datetime.now() - t0}")
