# -*- coding: utf-8 -*-
"""
sim_tracker.py — V8.1 模拟盘净值跟踪器（每日盘后执行）
================================================================

与 signal_generator.py 配套：读取信号清单，按「买入价=次日开盘价 / 卖出价=当日收盘价」
并扣除分档滑点（中证500>5亿=0.10% / 创业板1-5亿=0.30% / 其他<1亿=0.50%，买卖各一次）
模拟交易，逐日记录账户净值，并与 CSI300 基准对比。

V8.1 升级（2026-08-19）：BUY 建仓按信号 CSV 的 target_weight 列分配现金
  （V8 段每只 10%；combo 段每只约 1.11%、重叠股 2.22%，单只上限 10%），
  总部署受 regime_weight 市场门控约束 —— 与 V8.1 回测权重口径一致。
  调用方式不变（run_daily.sh 无需改动）。

执行时序（T+1 风格，与 A 股实际成交一致）：
  - 盘后 signal_generator 生成「次日信号」（dated D）。
  - 次日（D+1）盘后 sim_tracker 读取该信号：
        * 开盘价 执行 BUY（按 target_weight 权重、regime_weight 现金上限建仓）
        * 收盘价 执行 SELL（清仓跌出目标持仓的标的）
        * 以收盘价标记持仓市值 → 记录当日 NAV

输出：
  output/sim_nav/YYYY-MM-DD_nav.csv
    列: date, nav, cash, position_value, benchmark_nav,
        daily_return, cum_return, tracking_error(滚动20日 vs CSI300)

状态（跨运行持久化于 data/state/，随 data 分支持久化；output/sim_nav/ 仅本地缓存）：
  data/state/sim_state.json  —— 跨日持仓 / 现金 / 待成交单（保证每日增量正确）。
  data/state/sim_nav_history.csv —— 净值历史（逐日 NAV 序列，供 Pages 连续展示）。

用法：
  cd src
  python sim_tracker.py --init --date 2025-08-12   # 初始化（记录开盘前状态，挂起待成交）
  python sim_tracker.py --date 2025-08-13          # 次日执行（需 D+1 行情）
  python sim_tracker.py                            # 默认用最新信号 + 今日日期
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from datetime import datetime, timedelta

for _k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
           "ALL_PROXY", "all_proxy"]:
    os.environ.pop(_k, None)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import config
from stress_test_v6 import build_slippage_map
from push_utils import push_to_bark, format_stock_list, format_percent


NAV_DIR = config.OUTPUT_DIR / "sim_nav"          # 本地缓存目录（供本地查看明细，output/ 被 gitignore）
SIGNAL_DIR = config.OUTPUT_DIR / "signals"
# 跨运行持久化目录：data/state（随 data 分支持久化），使 Actions 全新 checkout 也能
# 恢复上一日持仓 / 现金 / 净值历史，与本地持续模拟盘保持一致。
STATE_DIR = config.SIM_STATE_DIR
STATE_FILE = STATE_DIR / "sim_state.json"
NAV_HIST_FILE = STATE_DIR / "sim_nav_history.csv"   # 净值历史持久化路径（原 output/sim_nav/sim_nav_history.csv）

# 锁定滑点模型（与 V7.1 完全一致）
FIXED_WEIGHT = config.FIXED_WEIGHT          # 0.10
TRACK_WINDOW = 20                          # 滚动跟踪误差窗口（交易日）


# ---------------------------------------------------------------------------
# 价格获取（实时模式用 akshare 双源；离线/初始化无未来数据则回退 None → 挂起）
# ---------------------------------------------------------------------------
def fetch_day_prices(codes, date: pd.Timestamp) -> dict:
    """返回 {code: {'open':, 'close':}}；联网失败或当日无数据返回 {}。

    V8.1 修复①：新浪兜底必须用「带日期范围」调用（stock_zh_a_daily(start_date/end_date)），
    其返回 df 的 date 是普通列而非索引；旧版用 df.index 过滤（全历史版 index 为
    RangeIndex，与 Timestamp 比较抛异常）导致静默失败、永远拉不到行情。
    V8.1 修复②：单只请求可能瞬时失败（新浪对部分代码响应超时/限流），
    失败重试 retries 次（间隔 0.4s），确保批量 88 只时覆盖完整。
    """
    import time
    import socket
    # 防御：Actions 美国 runner 无法稳定连接中国行情源时，akshare 底层请求可能无限挂起
    # （无超时）。这里设置全局 socket 超时，使其快速失败而非阻塞整条流水线。
    _prev_to = socket.getdefaulttimeout()
    socket.setdefaulttimeout(8)
    try:
        import akshare as ak
        out = {}
        d0 = (date - pd.Timedelta(days=5)).strftime("%Y%m%d")
        d1 = date.strftime("%Y%m%d")
        for code in codes:
            df = None
            try:
                df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                        start_date=d0, end_date=d1, adjust="qfq")
            except Exception:
                df = None
            if df is None or len(df) == 0:
                prefix = ("sh" if code.startswith(("6", "9")) else
                          "bj" if code.startswith(("8", "4")) else "sz")
                for attempt in range(3):
                    try:
                        df = ak.stock_zh_a_daily(symbol=prefix + code,
                                                 start_date=d0, end_date=d1, adjust="qfq")
                    except Exception:
                        df = None
                    if df is not None and len(df) > 0:
                        break
                    time.sleep(0.4)
            if df is None or len(df) == 0:
                continue
            df = df.rename(columns={"日期": "date", "开盘": "open", "收盘": "close"})
            if "date" not in df.columns:          # 新浪源：date 已是列名；东财源已 rename
                df = df.reset_index().rename(columns={"index": "date"})
            df["date"] = pd.to_datetime(df["date"])
            row = df[df["date"] == date]
            if len(row) == 1:
                out[code] = {"open": float(row["open"].iloc[0]),
                             "close": float(row["close"].iloc[0])}
        return out
    except Exception as e:
        print(f"[{datetime.now()}] 行情获取失败（{e}），本次仅记录挂起状态。")
        return {}
    finally:
        socket.setdefaulttimeout(_prev_to)


# ---------------------------------------------------------------------------
# 离线价格（P0-2：--offline 用收盘面板价真实建仓/盯市，替代实时行情）
# ---------------------------------------------------------------------------
_close_panel_cache = None


def _load_close_panel() -> pd.DataFrame:
    """主板收盘面板（懒加载缓存）。"""
    global _close_panel_cache
    if _close_panel_cache is None:
        _close_panel_cache = pd.read_parquet(config.MB_CLOSE).sort_index()
    return _close_panel_cache


def load_offline_prices(codes, sig_date, as_of, state) -> dict:
    """P0-2：从主板收盘面板构造执行价 {code: {'open': px, 'close': px}}。

    背景：Actions 美国 runner 无法访问实时行情源，原 --offline 只记录「挂起」从不
    成交，净值恒为 1.0。现改由收盘面板（data 分支每日就绪）提供执行价：

    执行日选取（与回测「T 日收盘成交」口径一致，无前视）：
      - 优先：信号日 < d <= as_of 的【最早】面板交易日（真 T+1 收盘执行）；
      - 回退：面板尚无次日行时（Actions cron 于北京时间 0:00 运行，信号 dated=当日、
        数据截面=前一交易日），取面板中 <= 信号日的最近交易日 = 实际截面日收盘
        —— 与回测「月末截面收盘选股、同日收盘成交」完全一致。
    open=close=该日收盘价（面板无 open 列）；滑点在 execute_day 内另行叠加。
    持仓代码一并纳入（HOLD 股每日盯市），停牌/无价股不在返回中（execute_day 沿用成本）。

    返回空 dict 时调用方走原有「挂起」兜底路径。
    """
    if not config.USE_MAINBOARD:
        print(f"[sim_tracker] --offline 面板价成交当前仅支持主板模式"
              f"（config.USE_MAINBOARD=True），本次挂起")
        return {}
    if not config.MB_CLOSE.exists():
        print(f"[sim_tracker] --offline 需主板收盘面板 {config.MB_CLOSE.name}"
              f"（main 分支不跟踪 data/，可 git checkout origin/data -- data/ 恢复），本次挂起")
        return {}
    try:
        panel = _load_close_panel()
    except Exception as e:
        print(f"[sim_tracker] 收盘面板读取失败（{e}），本次挂起")
        return {}
    sig_ts = pd.Timestamp(sig_date).normalize()
    asof_ts = pd.Timestamp(as_of).normalize()
    after = panel.index[(panel.index > sig_ts) & (panel.index <= asof_ts)]
    if len(after):
        d, mode = after[0], "T+1收盘(面板)"
    else:
        prior = panel.index[panel.index <= sig_ts]
        if not len(prior):
            print(f"[sim_tracker] --offline 面板无 ≤{sig_ts.date()} 的交易日"
                  f"（面板最早 {panel.index[0].date()}），本次挂起")
            return {}
        d, mode = prior[-1], ("T收盘(面板)" if prior[-1] == sig_ts
                              else f"截面回退({prior[-1].date()})")
    # 逐股取 <= 执行上限日 d 的最近有效收盘（面板若部分交易日整列缺失——如某次增量拉取
    # 中断只覆盖部分股票——仍以该股最近可得价成交/盯市，避免大面积跳过）。
    need = sorted(set(str(c) for c in codes)
                  | set(state.get("positions", {}).keys()))
    prices = {}
    for c in need:
        if c not in panel.columns:
            continue
        s = panel.loc[:d, c].dropna()
        if len(s):
            v = float(s.iloc[-1])
            if v > 0:
                prices[c] = {"open": v, "close": v}
    print(f"[sim_tracker] --offline 面板价成交：执行日上限={d.date()}（{mode}，信号日={sig_ts.date()}），"
          f"可得价 {len(prices)}/{len(need)} 只（逐股最近有效收盘，含持仓盯市）")
    return prices


# ---------------------------------------------------------------------------
# 状态读写
# ---------------------------------------------------------------------------
def load_state() -> dict:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if not STATE_FILE.exists():
        return _empty_state()
    st = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    _apply_start_date(st)
    return st


def _empty_state() -> dict:
    return {"init_date": None, "cash": 1.0, "positions": {},
            "nav": 1.0, "pending_buys": [], "pending_sells": [],
            "last_nav_date": None,
            "last_regime_weight": None}   # P0-3：上次信号的 regime_weight（%），避免重复清仓/供追溯


def _apply_start_date(st: dict) -> None:
    """SIM_START_DATE 模式：清理起始日之前建仓的持仓，必要时完全重置。

    约定（V8.1 起始日期功能）：
      - position 缺失 entry_date（旧数据结构）→ 视为起始日之前建仓，清理
      - entry_date < SIM_START_DATE → 清理
      - 无保留持仓 → 完全重置（NAV=1.0 / 现金 100% / 空仓 / nav_history 清空）
      - 部分保留 → 被清旧仓按成本变现进 cash（起始日之前的历史盈亏忽略），nav_history 截断
    """
    start = getattr(config, "SIM_START_DATE", None)
    if start is None:
        return
    start_ts = pd.Timestamp(start).normalize()
    positions = dict(st.get("positions", {}))
    kept, dropped = {}, []
    for c, h in positions.items():
        ed = h.get("entry_date")
        ok = ed is not None and pd.Timestamp(ed).normalize() >= start_ts
        if ok:
            kept[c] = h
        else:
            dropped.append(c)
    if dropped:
        print(f"[sim_tracker] SIM_START_DATE={start}：清理起始日之前建仓 {len(dropped)} 只"
              f"（{', '.join(dropped[:8])}{' ...' if len(dropped) > 8 else ''}）")

    if not kept:
        if positions:
            print("[sim_tracker] 无起始日之后的有效持仓 → 模拟盘重置：NAV=1.0 / 现金 100% / 空仓")
            st.update(_empty_state())
            st["init_date"] = str(start_ts.date())
            _reset_nav_history()
        elif st.get("init_date") != str(start_ts.date()):
            st["init_date"] = str(start_ts.date())
            _truncate_nav_history(start_ts)
    else:
        st["positions"] = kept
        for c in dropped:
            h = positions[c]
            st["cash"] = float(st.get("cash", 0.0)) + float(h["shares"]) * float(h["cost"])
        _truncate_nav_history(start_ts)


def _truncate_nav_history(start_ts) -> None:
    """删除 sim_nav_history.csv 中早于 start_ts 的记录（保留其余）。"""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    p = NAV_HIST_FILE
    if not p.exists():
        return
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"])
    df_new = df[df["date"] >= start_ts]
    if len(df_new) != len(df):
        df_new.to_csv(p, index=False, encoding="utf-8")
        print(f"[sim_tracker] 清理 nav_history 中早于 {start_ts.date()} 的记录"
              f"（移除 {len(df) - len(df_new)} 条）")


def _reset_nav_history() -> None:
    """完全重置：清空净值历史（从起始日起重建）。"""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    p = NAV_HIST_FILE
    if p.exists():
        p.unlink()
        print("[sim_tracker] nav_history 已清空（模拟盘从起始日重建净值曲线）")


def _ensure_start_nav(as_of: pd.Timestamp, state: dict) -> None:
    """起始日模式：若净值历史尚无 ≥ SIM_START_DATE 的记录，则补记一条空仓起点 NAV。"""
    start = getattr(config, "SIM_START_DATE", None)
    if start is None:
        return
    start_ts = pd.Timestamp(start).normalize()
    if pd.Timestamp(as_of).normalize() < start_ts:
        return
    hist = _read_nav()
    if len(hist) and pd.to_datetime(hist["date"]).max() >= start_ts:
        return
    bench = load_benchmark(state.get("init_date") or as_of)
    b_nav = float(bench.get(as_of, bench.iloc[-1])) if as_of in bench.index else 1.0
    nav = float(state.get("nav", 1.0))
    row = {
        "date": as_of.strftime("%Y-%m-%d"),
        "nav": round(nav, 6),
        "cash": round(float(state.get("cash", 1.0)), 6),
        "position_value": 0.0,
        "benchmark_nav": round(float(b_nav), 6),
        "daily_return": 0.0,
        "cum_return": round(nav - 1.0, 6),
        "tracking_error": float("nan"),
        "note": "start(模拟盘起点，空仓)",
    }
    _append_nav(row)
    print(f"[sim_tracker] 记录模拟盘起点 NAV：{as_of.date()} NAV={row['nav']:.4f}（空仓，待起始日之后信号建仓）")


def save_state(st):
    # 主持久化位置：data/state/（随 data 分支持久化，跨 Actions 运行共享）
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(st, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    # 同步缓存到 output/sim_nav/，便于本地查看（不影响持久化逻辑；output/ 被 gitignore）
    NAV_DIR.mkdir(parents=True, exist_ok=True)
    (NAV_DIR / "sim_state.json").write_text(
        json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# 基准（CSI300）净值
# ---------------------------------------------------------------------------
def load_benchmark(base_date=None) -> pd.Series:
    """CSI300 净值，重定基到模拟盘起始日（base_date）为 1.0，便于跟踪误差对比。

    若 base_date 超出数据末日（如初始化日 2025-08-14 > 数据末日 2025-08-12），
    则取末日定基，起始日净值记为 1.0。
    """
    idx = pd.read_parquet(config.DATA_DIR / "v6_index.parquet")
    if isinstance(idx, pd.DataFrame):
        idx = idx.iloc[:, 0]
    idx = idx.dropna()
    if base_date is not None:
        bd = min(pd.Timestamp(base_date), idx.index[-1])
        # 休市日容错：bd 不在交易日（如 SIM_START_DATE=2026-01-01 元旦）时
        # 取 <= bd 的最近交易日定基，避免 KeyError
        if bd not in idx.index:
            prior = idx.index[idx.index <= bd]
            bd = prior[-1] if len(prior) else idx.index[0]
        base_val = idx.loc[bd]
    else:
        base_val = idx.iloc[0]
    return (idx / base_val).rename("benchmark_nav")


# ---------------------------------------------------------------------------
# 执行一笔信号日的模拟交易
# ---------------------------------------------------------------------------
def _sig_name(signal_df: pd.DataFrame, code: str) -> str:
    """从信号表反查股票名称（regime 清仓时持仓股可能不在 BUY/SELL 行）。"""
    hit = signal_df[signal_df["code"].astype(str) == code]
    if len(hit) and "name" in hit.columns:
        return str(hit["name"].iloc[0])
    return code


def execute_day(signal_df: pd.DataFrame, date: pd.Timestamp,
                state: dict, slip_map: dict, prices: dict):
    """按信号在 date 日成交，更新 state，返回当日 NAV 信息。

    V8.1 升级：BUY 建仓按信号 CSV 的 target_weight 列分配现金
    （V8 段每只 10%；combo 段每只约 1.11%，重叠股 2.22%；单只上限 10%），
    总部署受 regime_weight 现金上限约束（与 V8.1 回测一致）。
    """
    # 开盘：执行 BUY（按信号顺序 + target_weight 权重 + regime_weight 现金上限）
    regime_w = float(signal_df["regime_weight"].iloc[0]) / 100.0 if len(signal_df) else 1.0
    buys = [r for _, r in signal_df.iterrows() if r["action"] == "BUY"]
    sells = [r for _, r in signal_df.iterrows() if r["action"] == "SELL"]

    cash = state["cash"]
    positions = dict(state["positions"])
    cap = state["nav"]  # 当前账户总净值（Fraction 基准 1.0）

    budget = regime_w * cap          # 本周期可部署上限（现金口径，市场门控）
    deployed = sum(h["shares"] * h["cost"] for h in positions.values())  # 现有持仓实际占用现金

    buys_executed, sells_executed = [], []   # 实际成交明细（仅用于 Bark 推送，不影响策略）
    regime_exit = False

    # P0-3（保真度审计）：regime 门控归零（MA240 跌破）→ 与回测 regime_exit 一致，全部清仓避险。
    # 回测在 regime<=0 的交易日以收盘价清仓退出；模拟盘此前仅把 BUY 预算置 0（=0 不买），
    # 存量持仓继续 HOLD → 熊市段模拟盘满仓而回测空仓，净值不可比（回测高估）。
    # 空仓时自然无操作（buys 预算=0 亦不会建仓）；残留未成交股（停牌）会在后续 regime<=0
    # 日继续尝试补清，直至清空。
    if regime_w <= 1e-9 and positions:
        regime_exit = True
        for c in list(positions.keys()):
            p = prices.get(c)
            if p is None:
                continue            # 无当日价：本次无法成交（停牌/无价），保留待下日补清
            slip = slip_map.get(c, 0.0050)
            sell_price = p["close"] * (1 - slip)          # 卖出滑点（减价）
            proceeds = positions[c]["shares"] * sell_price
            cash += proceeds
            del positions[c]
            sells_executed.append({"code": c, "name": _sig_name(signal_df, c),
                                   "price": sell_price})

    for r in buys:
        c = str(r["code"])
        if c in positions:
            continue
        if deployed >= budget - 1e-9:               # 浮点容差，确保不超预算
            break
        p = prices.get(c)
        if p is None:
            continue
        slip = slip_map.get(c, 0.0050)
        buy_price = p["open"] * (1 + slip)          # 买入滑点（加价）
        # V8.1：按信号 target_weight 分配现金（缺失时回退固定 10%）
        try:
            w = float(r["target_weight"]) / 100.0
            if not (w > 0) or w != w:
                w = FIXED_WEIGHT
        except (KeyError, TypeError, ValueError):
            w = FIXED_WEIGHT
        w = min(w, FIXED_WEIGHT)                    # 单只上限 10%（与 V8.1 一致）
        alloc = min(w * cap, budget - deployed)
        if alloc <= 1e-9 or buy_price <= 0:
            continue
        shares = alloc / buy_price
        positions[c] = {"shares": shares, "cost": buy_price,
                        "entry_date": str(date.date())}   # 起始日期功能：记录建仓日
        cash -= alloc
        deployed += alloc
        buys_executed.append({"code": c, "name": str(r["name"]),
                              "price": buy_price, "shares": shares})

    # 收盘：执行 SELL（清仓跌出 Top30 的标的）
    for r in sells:
        c = str(r["code"])
        if c not in positions:
            continue
        p = prices.get(c)
        if p is None:
            continue
        slip = slip_map.get(c, 0.0050)
        sell_price = p["close"] * (1 - slip)        # 卖出滑点（减价）
        proceeds = positions[c]["shares"] * sell_price
        cash += proceeds
        del positions[c]
        sells_executed.append({"code": c, "name": str(r["name"]), "price": sell_price})

    # 以收盘价标记持仓市值
    pos_val = 0.0
    for c, h in positions.items():
        p = prices.get(c)
        if p is not None:
            pos_val += h["shares"] * p["close"]
        else:
            pos_val += h["shares"] * h["cost"]      # 无当日价则沿用成本
    nav = cash + pos_val
    return {"cash": cash, "positions": positions, "nav": nav, "pos_val": pos_val,
            "buys_executed": buys_executed, "sells_executed": sells_executed,
            "regime_exit": regime_exit}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="V7.1 模拟盘净值跟踪器")
    ap.add_argument("--init", action="store_true", help="初始化（记录开盘前状态）")
    ap.add_argument("--offline", action="store_true",
                    help="离线模式：跳过实时行情拉取（Actions 无中国源），直接挂起待成交")
    ap.add_argument("--live", action="store_true",
                    help="实盘模式：读取最新信号 + sim_state + 行情，调 risk_manager 生成可执行委托单 CSV"
                         "（不实际成交、不修改 sim_state；券商端执行由人工/SDK 完成）")
    ap.add_argument("--date", type=str, default=None, help="执行日（默认=今日）")
    ap.add_argument("--signal-date", type=str, default=None,
                    help="指定读取的信号文件日期（默认=执行日之前最近一份）")
    args = ap.parse_args()

    NAV_DIR.mkdir(parents=True, exist_ok=True)
    as_of = (pd.Timestamp(args.date) if args.date
             else pd.Timestamp(datetime.now().strftime("%Y-%m-%d")))

    # 状态加载提前：SIM_START_DATE 模式会在此清理旧持仓/重置（与信号文件无关）
    state = load_state()
    if args.init and state["init_date"] is None:
        state["init_date"] = str(as_of.date())

    # 定位信号文件
    sig_files = sorted(f for f in os.listdir(SIGNAL_DIR)
                       if f.endswith("_signal.csv")) if os.path.isdir(SIGNAL_DIR) else []
    if not sig_files:
        print(f"[{datetime.now()}] 无信号文件，请先运行 signal_generator.py")
        _ensure_start_nav(as_of, state)     # 起始日模式：补记空仓起点 NAV
        save_state(state)                   # 确保 data/state/sim_state.json 初始化落地
        return
    if args.signal_date:
        sig_name = f"{args.signal_date}_signal.csv"
    else:
        # 取执行日之前（含）最近一份
        cand = [f for f in sig_files
                if pd.Timestamp(f.replace("_signal.csv", "")) <= as_of]
        sig_name = cand[-1] if cand else sig_files[-1]

    # ---- 起始日期门控：早于 SIM_START_DATE 的信号不消费 ----
    start = getattr(config, "SIM_START_DATE", None)
    sig_date = pd.Timestamp(sig_name.replace("_signal.csv", ""))
    if start is not None and sig_date < pd.Timestamp(start).normalize():
        print(f"[{datetime.now()}] 信号 {sig_name} 早于 SIM_START_DATE={start}，忽略"
              f"（模拟盘起始日之前不消费任何信号）")
        _ensure_start_nav(as_of, state)
        return

    sig_df = pd.read_csv(SIGNAL_DIR / sig_name, dtype={"code": str})
    print(f"[{datetime.now()}] 读取信号: {sig_name}（执行日={as_of.date()}）")
    # P0-3：regime_weight（%制，0=MA240 跌破门控归零）——用于清仓决策与状态追溯
    raw_regime = (float(sig_df["regime_weight"].iloc[0])
                  if len(sig_df) and "regime_weight" in sig_df.columns else 100.0)

    # 滑点分级（与 V7.1 一致，仅成本模型，非信号）
    # P0-1（保真度审计）：主板模式（USE_MAINBOARD=True）必须用主板成交额面板 MB_AMOUNT。
    # 旧代码用 v6_amount_panel（仅 526 只 V8 池）→ 主板信号 75.6% 落入 0.5% fallback，
    # 滑点被系统性高估（回测同批股按真实分档 0.1%-0.3%）。V8 链路（USE_MAINBOARD=False）
    # 保持 v6_amount_panel 不变（与回测段1/段2 数据源一致）。
    amount = pd.read_parquet(config.MB_AMOUNT if config.USE_MAINBOARD
                             else config.V6_AMOUNT_PANEL)
    slip_map, _ = build_slippage_map(amount)

    codes_all = list(sig_df["code"].astype(str))
    # 执行价来源：
    #   - 实时模式（本地盘后/云服务器）：akshare 拉取执行日当日 open/close
    #   - --live（实盘委托单）：仍需实时行情（委托单面向真实成交）
    #   - P0-2 --offline（Actions 美国 runner 无实时源）：改用收盘面板价真实建仓 + 每日盯市
    #     （load_offline_prices 按"信号日之后首个面板交易日，否则截面日收盘"取价），不再挂起
    #   - --init：仅记录初始状态不成交
    if args.init:
        prices = {}
    elif args.live:
        prices = fetch_day_prices(codes_all, as_of)
    elif args.offline:
        prices = load_offline_prices(codes_all, sig_date, as_of, state)
    else:
        prices = fetch_day_prices(codes_all, as_of)

    # ---- 实盘模式（方向C）：仅生成委托单，不修改 sim_state、不实际成交 ----
    if args.live:
        import risk_manager as _risk
        if not prices:
            print(f"[sim_tracker] --live 但 fetch_day_prices 返回空（无当日行情），"
                  f"跳过生成委托单")
            return
        nav = float(state.get("nav", 1.0))
        # 同步当日收益率用于熔断（来自净值历史末行）
        try:
            hist = _read_nav()
            if len(hist) and "daily_return" in hist.columns:
                state["daily_return"] = float(hist["daily_return"].iloc[-1])
        except Exception:
            pass
        orders = _risk.build_orders(sig_df, state, prices, as_of,
                               nav=nav, nav_unit=config.INIT_CAPITAL)
        out = _risk.write_orders(orders, as_of)
        n_buy = int((orders["direction"] == "BUY").sum()) if len(orders) else 0
        n_sell = int((orders["direction"] == "SELL").sum()) if len(orders) else 0
        total_amt = float(orders["amount"].sum()) if len(orders) else 0.0
        print(f"[sim_tracker] --live 委托单：BUY={n_buy}  SELL={n_sell}  "
              f"金额合计={total_amt:,.0f} 元  NAV={nav:.4f}")
        print(f"[sim_tracker] 写入: {out}")
        return

    if not prices:
        # 无可成交行情（初始化 / 离线且面板不可用）：记录挂起状态，不实际建仓
        state["pending_buys"] = list(sig_df[sig_df["action"] == "BUY"]["code"].astype(str))
        state["pending_sells"] = list(sig_df[sig_df["action"] == "SELL"]["code"].astype(str))
        state["last_regime_weight"] = raw_regime
        save_state(state)
        bench = load_benchmark(state.get("init_date") or as_of)
        b_nav = float(bench.get(as_of, bench.iloc[-1])) if as_of in bench.index else 1.0
        row = {
            "date": as_of.strftime("%Y-%m-%d"),
            "nav": round(state["nav"], 6),
            "cash": round(state["cash"], 6),
            "position_value": 0.0,
            "benchmark_nav": round(b_nav, 6),
            "daily_return": 0.0,
            "cum_return": round(state["nav"] - 1.0, 6),
            "tracking_error": float("nan"),
            "note": "pending(无执行日行情，挂起待成交)",
        }
        _append_nav(row)
        print(f"[{datetime.now()}] 初始化：账户净值={state['nav']:.4f} 现金=100% "
              f"挂起买入 {len(state['pending_buys'])} 只（待下一交易日开盘价成交）")
        return

    # 有行情：执行成交
    res = execute_day(sig_df, as_of, state, slip_map, prices)
    state["cash"] = res["cash"]
    # 注意：必须保留 entry_date（起始日期功能依赖），缺省回填执行日
    state["positions"] = {c: {"shares": float(h["shares"]), "cost": float(h["cost"]),
                              "entry_date": h.get("entry_date", str(as_of.date()))}
                          for c, h in res["positions"].items()}
    state["nav"] = res["nav"]
    state["pending_buys"] = []
    state["pending_sells"] = []
    state["last_regime_weight"] = raw_regime          # P0-3：记录本次 regime，供追溯/防重复
    save_state(state)

    # P0-3：regime 清仓触发提示（与回测 regime_exit 对齐）
    if res.get("regime_exit"):
        n_sold = len(res.get("sells_executed", []))
        print(f"[{datetime.now()}] ⚠️ regime 清仓触发（regime_weight={raw_regime:g}，"
              f"MA240 门控归零）：{n_sold} 只持仓按执行日收盘价全部卖出，仓位归零"
              f"（与回测 regime_exit 一致）")

    # ---- Bark 推送①：今日执行（盘前成交清单；可选，未配置 key 自动跳过）----
    bark_key = getattr(config, "BARK_DEVICE_KEY", "") or os.environ.get("BARK_DEVICE_KEY", "")
    if bark_key:
        buys_ex = res.get("buys_executed", [])
        sells_ex = res.get("sells_executed", [])
        if buys_ex or sells_ex:
            try:
                body_parts = []
                if buys_ex:
                    body_parts.append("买入：")
                    body_parts.append(format_stock_list(
                        [(b["code"], b["name"], f"@{b['price']:.2f}", f"{b['shares']:.0f}股")
                         for b in buys_ex], max_display=10))
                if sells_ex:
                    body_parts.append("卖出：")
                    body_parts.append(format_stock_list(
                        [(s["code"], s["name"], f"@{s['price']:.2f}")
                         for s in sells_ex], max_display=10))
                push_to_bark(f"💰 今日执行 {as_of.strftime('%Y-%m-%d')}",
                             "\n".join(body_parts), key=bark_key)
            except Exception as e:
                print(f"[bark] 执行推送异常（静默跳过）: {e}")

    # 基准 & 跟踪误差
    bench = load_benchmark(state.get("init_date") or as_of)
    b_nav = float(bench.get(as_of, bench.iloc[-1])) if as_of in bench.index else bench.iloc[-1]
    nav_hist = _read_nav()
    prev_nav = nav_hist["nav"].iloc[-2] if len(nav_hist) >= 2 else 1.0
    daily_ret = res["nav"] / prev_nav - 1 if prev_nav else 0.0
    cum_ret = res["nav"] - 1.0
    te = _rolling_tracking_error(nav_hist, bench, as_of)
    row = {
        "date": as_of.strftime("%Y-%m-%d"),
        "nav": round(res["nav"], 6),
        "cash": round(res["cash"], 6),
        "position_value": round(res["pos_val"], 6),
        "benchmark_nav": round(float(b_nav), 6),
        "daily_return": round(float(daily_ret), 6),
        "cum_return": round(float(cum_ret), 6),
        "tracking_error": round(float(te), 6) if te == te else float("nan"),
    }
    _append_nav(row)
    print(f"[{datetime.now()}] 执行完成：NAV={res['nav']:.4f} 现金={res['cash']:.2%} "
          f"持仓市值={res['pos_val']:.4f} 累计收益={cum_ret:+.2%} "
          f"跟踪误差(20日)={te if te==te else 'NA'}")

    # ---- Bark 推送②：收盘净值（可选）----
    if bark_key:
        try:
            holdings_count = len(state["positions"])
            cash_ratio = state["cash"] / res["nav"] if res["nav"] else 0.0
            bench_ret = float(b_nav) - 1.0 if b_nav == b_nav else None
            body = (f"NAV: {res['nav']:.4f}\n"
                    f"当日: {format_percent(daily_ret, signed=True)}   "
                    f"累计: {format_percent(cum_ret, signed=True)}\n"
                    f"持仓: {holdings_count} 只  现金: {format_percent(cash_ratio)}\n"
                    f"基准(CSI300): {format_percent(bench_ret, signed=True)}")
            push_to_bark(f"📊 收盘净值 {as_of.strftime('%Y-%m-%d')}", body, key=bark_key)
        except Exception as e:
            print(f"[bark] 收盘推送异常（静默跳过）: {e}")


def _read_nav() -> pd.DataFrame:
    p = NAV_HIST_FILE
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame(columns=["date", "nav", "cash", "position_value",
                                 "benchmark_nav", "daily_return", "cum_return",
                                 "tracking_error"])


def _append_nav(row: dict):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    hist = _read_nav()
    # 同日期覆盖
    hist = hist[hist["date"] != row["date"]]
    hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    hist = hist.sort_values("date").reset_index(drop=True)
    hist.to_csv(NAV_HIST_FILE, index=False, encoding="utf-8")
    # 同步缓存到 output/sim_nav/，便于本地查看
    NAV_DIR.mkdir(parents=True, exist_ok=True)
    hist.to_csv(NAV_DIR / "sim_nav_history.csv", index=False, encoding="utf-8")
    # 同时输出按日的明细文件（用户要求的 YYYY-MM-DD_nav.csv）
    pd.DataFrame([row]).to_csv(
        NAV_DIR / f"{row['date']}_nav.csv", index=False, encoding="utf-8")
    print(f"[{datetime.now()}] NAV 已记录: {NAV_HIST_FILE}")


def _rolling_tracking_error(nav_hist: pd.DataFrame, bench: pd.Series,
                            as_of: pd.Timestamp) -> float:
    """滚动 20 日 sim 日收益与 CSI300 日收益的跟踪误差（std）。"""
    if len(nav_hist) < 2 or as_of not in bench.index:
        return float("nan")
    nav_hist = nav_hist.copy()
    nav_hist["date"] = pd.to_datetime(nav_hist["date"])
    # 对齐到交易日序列
    merged = pd.DataFrame({"nav": nav_hist.set_index("date")["nav"]})
    merged["bench"] = bench.reindex(merged.index).ffill()
    merged["r_sim"] = merged["nav"].pct_change()
    merged["r_b"] = merged["bench"].pct_change()
    merged["te"] = merged["r_sim"] - merged["r_b"]
    win = merged["te"].dropna().tail(TRACK_WINDOW)
    if len(win) < 2:
        return float("nan")
    return float(win.std(ddof=0))


if __name__ == "__main__":
    main()
