# -*- coding: utf-8 -*-
"""
update_daily_panel.py — 主板面板每日增量更新（轻量，替代全量 fetch_mainboard.py）
====================================================================================

背景：fetch_mainboard.py 全量抓取约 1.5 小时，不适合每日自动跑。
本脚本只做增量：以现有 mainboard_close_panel.parquet 为准，对每只股票
从 last_date 往前 30 天开始拉新浪日线（qfq），追加/替换新交易日。

关键设计：
1. 复权因子检测：qfq 除权会使整段历史缩放。增量拉取包含 last_date 当天的数据，
   与面板 last_date 收盘对比——一致 → 纯追加新行（快）；不一致 → 复权因子变化，
   重拉 365 天窗口替换该列尾部（准确，无需全历史）。
2. 盘中保护：若最新数据日期 == 今天 且 当前北京时间 < 15:30，丢弃今天未收盘行
   （拉到盘中快照不会污染面板）。
3. 幂等：面板已最新（无新交易日）→ 不产生任何写入，退出码 0。
4. 容错：单只失败重试 SINA_RETRIES 次，仍失败记 warn 跳过，不影响整体。

运行：
    cd src && python update_daily_panel.py            # 全量增量（约 25-45 分钟）
    python update_daily_panel.py --limit 20           # 小批量试跑（验证链路）
    python update_daily_panel.py --check              # 仅打印现状与是否需更新，不写盘
    python update_daily_panel.py --data-dir /path/data # 指定面板目录（默认 ../data）

注意：本脚本【不依赖 src/config.py】——data 分支只有 data/ 目录（无代码），
由 scripts/run_update_data.sh（本机）或 daily_run.yml（GitHub Actions）在
data 面板所在工作树运行，故路径自包含。已加 socket 20s 全局超时，
Actions 美国 runner 若无法访问新浪源会快速失败（不会卡死数小时）。

输出（原地更新，仅当有新数据时）：
    data/mainboard_close_panel.parquet   (date × code 收盘价，qfq)
    data/mainboard_amount_panel.parquet  (date × code 成交额，元)
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# 新浪源必须关闭代理（沙箱/本机代理会拦截）
for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
          "ALL_PROXY", "all_proxy"]:
    os.environ.pop(k, None)

# 全局 socket 超时：防止对每个请求无限等待（Actions/网络异常时快速失败，避免任务卡死数小时）
socket.setdefaulttimeout(20)

# 行缓冲：重定向到日志文件时进度实时可见
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import akshare as ak

SINA_RETRIES = 5          # 单只失败重试次数
SINA_SLEEP = 1.0          # 重试间隔（秒）
REQ_GAP = 0.08            # 正常请求间隔（秒），防新浪限流
FWD_WINDOW = 30           # 增量拉取窗口（自然日，覆盖 last_date 当天，用于复权对比）
RESET_WINDOW = 365        # 复权变化时重拉窗口（自然日）
TRADE_CLOSE_HM = (15, 30)  # 北京收盘保护时刻：晚于此且最新行==今天 才视为已收盘


def _prefix_sina(code: str) -> str:
    return "sh" + code if code.startswith("6") else "sz" + code


def fetch_sina_incremental(code: str, start_d: str) -> pd.DataFrame | None:
    """新浪日线（qfq），start_d=YYYYMMDD 起。失败重试，返回 index=date 的 df 或 None。"""
    last = None
    for _ in range(SINA_RETRIES):
        try:
            df = ak.stock_zh_a_daily(symbol=_prefix_sina(code),
                                     start_date=start_d,
                                     end_date=datetime.now().strftime("%Y%m%d"),
                                     adjust="qfq")
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
    print(f"  [warn] {code} 新浪增量失败: {repr(last)[:80]}")
    return None


def _drop_intraday_rows(df: pd.DataFrame) -> pd.DataFrame:
    """盘中保护：最新行若为今天且当前未过收盘时刻，丢弃（避免未收盘快照污染面板）。"""
    if len(df) == 0:
        return df
    last_date = df.index[-1]
    now = datetime.now()
    if last_date.date() == now.date() and now.weekday() < 5:
        now_min = now.hour * 60 + now.minute
        if now_min < TRADE_CLOSE_HM[0] * 60 + TRADE_CLOSE_HM[1]:
            print(f"  [guard] 丢弃未收盘日 {last_date.date()}（当前 {now.strftime('%H:%M')} < 15:30）")
            df = df.iloc[:-1]
    return df


def update_panel(codes: list, close_panel: pd.DataFrame, amount_panel: pd.DataFrame,
                 limit: int = 0, start_at: int = 0):
    """对每只股票做增量，返回 (new_close, new_amount, updated_codes, new_rows)。"""
    if limit:
        codes = codes[start_at:start_at + limit]
    else:
        codes = codes[start_at:]

    last_date = close_panel.index[-1]          # 面板最新交易日
    new_close = close_panel.copy()
    new_amount = amount_panel.copy()
    updated = []                               # 有新增/替换的股票
    total_new_rows = 0

    t0 = time.time()
    for i, code in enumerate(codes, 1):
        if code not in new_close.columns:
            print(f"  [warn] {code} 不在面板列中，跳过")
            continue
        # 面板该列最后有效值（用于复权对比）
        col = new_close[code].dropna()
        panel_last = float(col.iloc[-1]) if len(col) else np.nan

        start_d = (last_date - timedelta(days=FWD_WINDOW)).strftime("%Y%m%d")
        d = fetch_sina_incremental(code, start_d)
        if d is None or len(d) == 0:
            print(f"  skip {code}: 增量无数据（可能长期停牌）")
            time.sleep(REQ_GAP)
            continue
        d = _drop_intraday_rows(d)
        if len(d) == 0:
            print(f"  skip {code}: 无有效新数据")
            time.sleep(REQ_GAP)
            continue

        # 复权因子检测：对比 last_date 当天收盘
        idx_last = d.index[d.index <= last_date]
        adj_changed = False
        if len(idx_last) and not np.isnan(panel_last):
            latest_in_panel = float(d.loc[idx_last[-1], "close"])
            if abs(latest_in_panel - panel_last) / max(abs(panel_last), 1e-9) > 1e-3:
                adj_changed = True

        if adj_changed:
            # 复权因子变化 → 重拉较长窗口，替换该列尾部
            start_d2 = (last_date - timedelta(days=RESET_WINDOW)).strftime("%Y%m%d")
            d2 = fetch_sina_incremental(code, start_d2)
            if d2 is None or len(d2) == 0:
                print(f"  skip {code}: 复权重拉失败，保留原数据")
                time.sleep(REQ_GAP)
                continue
            d2 = _drop_intraday_rows(d2)
            d2 = d2[d2.index <= last_date]     # 只替换到面板最新日，避免旧窗口覆盖错误
            # 与现有历史拼接（面板中 > window_start 的行替换为新数据）
            window_start = d2.index[0]
            keep = new_close.index < window_start
            new_close.loc[window_start:, code] = d2["close"].reindex(new_close.index[new_close.index >= window_start])
            new_amount.loc[window_start:, code] = d2["amount"].reindex(new_amount.index[new_amount.index >= window_start])
            # 上面 reindex 未覆盖的行保持 NaN，属正常（停牌日）
            updated.append(code)
            print(f"  {code}: 复权因子变化 → 重拉 {RESET_WINDOW} 天替换（{d2.index[0].date()}~{d2.index[-1].date()}）")
        else:
            # 纯追加 last_date 之后的新行
            new_rows = d[d.index > last_date]
            if len(new_rows) == 0:
                time.sleep(REQ_GAP)
                continue
            for dt in new_rows.index:
                if dt not in new_close.index:
                    # 新日期行（其他股票可能已有该日期）：扩展索引
                    new_close.loc[dt, :] = np.nan
                    new_amount.loc[dt, :] = np.nan
                new_close.loc[dt, code] = new_rows.loc[dt, "close"]
                new_amount.loc[dt, code] = new_rows.loc[dt, "amount"]
            updated.append(code)
            total_new_rows += len(new_rows)
            print(f"  {code}: 追加 {len(new_rows)} 行（{new_rows.index[0].date()}~{new_rows.index[-1].date()}）")

        time.sleep(REQ_GAP)
        if i % 200 == 0:
            el = (time.time() - t0) / 60
            rate = i / el if el > 0 else 0
            print(f"  [progress] {i}/{len(codes)} 更新 {len(updated)} 只 耗时 {el:.1f}min ({rate:.0f} 只/min)")

    # 保证索引排序
    new_close = new_close.sort_index()
    new_amount = new_amount.sort_index()
    return new_close, new_amount, updated, total_new_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只更新前 N 只（试跑）")
    ap.add_argument("--start", type=int, default=0, help="从第 N 只开始")
    ap.add_argument("--check", action="store_true", help="仅打印现状与是否需更新，不写盘")
    ap.add_argument("--data-dir", default=None,
                    help="面板目录（默认脚本仓库的 ../data，即 <repo>/data）")
    args = ap.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else Path(__file__).resolve().parent.parent / "data"
    mb_close = data_dir / "mainboard_close_panel.parquet"
    mb_amount = data_dir / "mainboard_amount_panel.parquet"

    if not mb_close.exists():
        print(f"[ERROR] 面板不存在: {mb_close}（先跑全量 fetch_mainboard.py）")
        return 1

    close = pd.read_parquet(mb_close)
    amount = pd.read_parquet(mb_amount)
    last_date = close.index[-1]
    codes = list(close.columns)
    print(f"[info] 现有面板: {close.shape} 交易日 {close.index[0].date()}~{last_date.date()} 覆盖 {close.notna().mean().mean():.1%}")

    # 最新完整交易日估算：今天收盘后 = 今天（工作日）；否则回溯到上一工作日
    now = datetime.now()
    if now.weekday() < 5 and now.hour * 60 + now.minute >= TRADE_CLOSE_HM[0] * 60 + TRADE_CLOSE_HM[1]:
        latest_trade = now.date()
    else:
        d = now.date() - timedelta(days=1)
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        latest_trade = d
    need = last_date.date() < latest_trade
    print(f"[info] 最新完整交易日估算: {latest_trade} | 面板最后: {last_date.date()} | {'需更新' if need else '已最新（无需更新）'}")

    if args.check or not need:
        return 0

    new_close, new_amount, updated, nrows = update_panel(codes, close, amount,
                                                         args.limit, args.start)
    if not updated:
        print("[done] 无股票有新数据（已最新），未写盘")
        return 0

    new_close.to_parquet(mb_close)
    new_amount.to_parquet(mb_amount)
    print(f"[done] 面板更新完成: {len(updated)}/{len(codes)} 只有新数据，共新增 {nrows} 行")
    print(f"       close={new_close.shape} 交易日 {new_close.index[0].date()}~{new_close.index[-1].date()} "
          f"覆盖 {new_close.notna().mean().mean():.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
