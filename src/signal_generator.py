# -*- coding: utf-8 -*-
"""
signal_generator.py — V8.1 每日信号生成器（模拟盘部署版）
================================================================

V8.1 = 分区间部署：
  - 最新截面月末 ≤ 2023-12-31：V8 原样信号（Regime 切换 + 反转/动量质量 Top30，每只 10%）
  - 最新截面月末 ≥ 2024-01-01：E 等权组合信号（V8 + Trend + Breakout 各 1/3，
    策略内 30 只等权，重叠叠加、单只 cap 10%）

复用组件（全部锁定 V8.1 参数，未做任何优化）：
  - factor_eval（regime 切换 / 动量质量 zscore）          —— V8 主策略选股
  - strategies/trend_ema.py, strategies/vol_breakout.py   —— 两个并行策略
  - market_filter.build_ma240_vol_target_weight           —— MA240 + 波动率降档门控
  - stress_test_v6.build_ohlcv_full / build_slippage_map  —— 反转信号 / 滑点（诊断用）

锁定参数（与 V8.1 完全一致）：MA_BASE=240, IC_BASE=0.05, VOL_Q=0.75, REDUCED_WEIGHT=0.60,
VOL_LOOKBACK=756, FIXED_WEIGHT=0.10, N_SELECT=30, POOL_PCT=0.20, ROLL_WINDOW=36,
SPLIT_DATE=2024-01-01, COMBO_CAP=0.10, N_COMBO=3。

用法：
  cd src
  python signal_generator.py --init --date 2025-08-12   # 首次初始化（用本地最新数据截面）
  python signal_generator.py                             # 每日盘后（自动取数据末日截面）
  python signal_generator.py --live                      # 先拉取最新行情再生成（需联网）

输出：
  output/signals/YYYY-MM-DD_signal.csv
    列: code, name, action(BUY/SELL/HOLD), target_weight(%), factor_set, regime_weight
    - target_weight : 单只目标仓位（V8 段=10%；E 组合段=策略内等权后的真实权重，重叠叠加 cap 10%）
    - regime_weight : 当日市场状态目标总仓位（MA240 跌破=0 / 波动率高位=0.6 / 常态=1.0）
    行序：E 组合段按总权重降序（重叠股优先，便于 sim_tracker 按序建仓）。
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from datetime import datetime, timedelta

# akshare 部分接口走代理会被沙箱拦，强制清掉代理环境变量（与回测脚本一致）
for _k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
           "ALL_PROXY", "all_proxy"]:
    os.environ.pop(_k, None)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import config
from factors import compute_rsi, get_month_end_dates
from factor_eval import (monthly_reversal_ic, compute_rolling_regime,
                         compute_momentum_zscore, build_selection_v5)
from factor_eval_v7 import build_rolling_reversal_signal
from market_filter import build_ma240_vol_target_weight
from stress_test_v6 import build_ohlcv_full, build_slippage_map
from strategies.trend_ema import gen_signal as sig_trend
from strategies.vol_breakout import gen_signal as sig_breakout


# ---------------------------------------------------------------------------
# 锁定参数（与 V8.1 完全一致，禁止修改）
# ---------------------------------------------------------------------------
MA_BASE = 240
IC_BASE = 0.05
VOL_Q = 0.75
REDUCED_WEIGHT = 0.60
VOL_LOOKBACK = 756
FIXED_WEIGHT = config.FIXED_WEIGHT          # 0.10
N_SELECT = 30
POOL_PCT = 0.20
ROLL_WINDOW = 36
SPLIT_DATE = pd.Timestamp("2024-01-01")     # 分区间门控
N_COMBO = 3                                  # E 组合策略数（V8+Trend+Breakout）
COMBO_CAP = 0.10                             # 单只权重上限

REVERSAL_SIGNAL_CACHE = config.DATA_DIR / "_v8_revsignal_cache.parquet"
NAMES_CACHE = config.DATA_DIR / "stock_names.json"


# ---------------------------------------------------------------------------
# 面板路径：USE_MAINBOARD=True 时切主板版（60/00 开头选股池）
# ---------------------------------------------------------------------------
def _panel_paths():
    if getattr(config, "USE_MAINBOARD", False):
        rev_cache = config.DATA_DIR / "_mainboard_revsignal_cache.parquet"
        return dict(close=config.MB_CLOSE, amount=config.MB_AMOUNT,
                    roe=config.MB_ROE, gpm=config.MB_GPM,
                    ohlcv_old={}, rev_cache=rev_cache,
                    label="主板版(60/00)")
    return dict(close=config.DATA_DIR / "v8_close_panel.parquet",
                amount=config.DATA_DIR / "v8_amount_panel.parquet",
                roe=config.DATA_DIR / "roe_panel_v8.parquet",
                gpm=config.DATA_DIR / "gpm_yoy_panel_v8.parquet",
                ohlcv_old=(pd.read_pickle(config.V3_OHLCV)
                           if config.V3_OHLCV.exists() else {}),
                rev_cache=REVERSAL_SIGNAL_CACHE,
                label="V8(中证500+创业板+中证1000)")


# ---------------------------------------------------------------------------
# 工具函数（与 main_v8_1_split.py 完全一致）
# ---------------------------------------------------------------------------
def mask_new_listings(close_panel: pd.DataFrame, min_days: int = 60) -> pd.DataFrame:
    out = close_panel.copy()
    for c in out.columns:
        s = out[c].dropna()
        if s.empty:
            out[c] = np.nan
            continue
        cutoff = s.index.min() + pd.Timedelta(days=min_days)
        out.loc[out.index < cutoff, c] = np.nan
    return out


def month_ends_in(close_panel: pd.DataFrame, start, end) -> pd.DatetimeIndex:
    me = get_month_end_dates(close_panel.index)
    me = pd.DatetimeIndex(me).normalize()
    return me[(me >= pd.Timestamp(start).normalize()) & (me <= pd.Timestamp(end).normalize())]


def load_panels(as_of: pd.Timestamp, live: bool = False) -> dict:
    """装载面板（V8 或主板版，由 config.USE_MAINBOARD 决定）。--live 时尝试追加最新行情（失败回退本地）。"""
    P = _panel_paths()
    close = pd.read_parquet(P["close"])
    amount = pd.read_parquet(P["amount"])
    idx = pd.read_parquet(config.DATA_DIR / "v6_index.parquet")
    if isinstance(idx, pd.DataFrame):
        idx = idx.iloc[:, 0]
    roe = pd.read_parquet(P["roe"]).reindex(close.index).ffill()
    gpm = pd.read_parquet(P["gpm"]).reindex(close.index).ffill()
    ohlcv_old = P["ohlcv_old"]
    codes = list(close.columns)

    if live:
        try:
            close, amount, idx = _refresh_latest(close, amount, idx, codes, as_of)
            print(f"[{datetime.now()}] --live：行情刷新成功，数据截止 {close.index[-1].date()}")
        except Exception as e:
            print(f"[{datetime.now()}] --live：行情刷新失败，回退本地数据（{e}）")

    return dict(close=close, amount=amount, idx=idx, roe=roe, gpm=gpm,
                ohlcv_old=ohlcv_old, codes=codes, panel_label=P["label"],
                rev_cache=P["rev_cache"],
                data_start=str(close.index[0].date()),
                data_end=str(close.index[-1].date()))


def _refresh_latest(close, amount, idx, codes, as_of):
    """akshare 双源刷新（东方财富优先，新浪兜底）；仅追加未纳入交易日。"""
    import akshare as ak
    end_str = as_of.strftime("%Y%m%d")
    start_str = (close.index[-1] + pd.Timedelta(days=1)).strftime("%Y%m%d")
    if start_str > end_str:
        return close, amount, idx
    new_close, new_amt = {}, {}
    for code in codes:
        df = None
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                    start_date=start_str, end_date=end_str, adjust="qfq")
        except Exception:
            try:
                df = ak.stock_zh_a_daily(symbol=("sh" + code if code.startswith("6")
                                                 else "sz" + code), adjust="qfq")
                if df is not None:
                    df = df[(df.index >= pd.Timestamp(start_str)) &
                            (df.index <= pd.Timestamp(end_str))]
            except Exception:
                df = None
        if df is None or len(df) == 0:
            continue
        df = df.rename(columns={"日期": "date", "收盘": "close", "成交额": "amount"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        new_close[code] = df["close"]
        if "amount" in df:
            new_amt[code] = df["amount"]
    if not new_close:
        raise RuntimeError("无新行情可追加")
    close = pd.concat([close, pd.DataFrame(new_close)]).pipe(
        lambda d: d[~d.index.duplicated(keep="last")]).sort_index()
    if new_amt:
        amount = pd.concat([amount, pd.DataFrame(new_amt)]).pipe(
            lambda d: d[~d.index.duplicated(keep="last")]).sort_index()
    try:
        idf = ak.stock_zh_index_daily(symbol="sh000300")
        idf.index = pd.to_datetime(idf.index)
        idf = idf[(idf.index >= pd.Timestamp(start_str)) & (idf.index <= pd.Timestamp(end_str))]
        if len(idf):
            idx = pd.concat([idx, idf["close"]]).pipe(
                lambda s: s[~s.index.duplicated(keep="last")]).sort_index()
    except Exception:
        pass
    return close, amount, idx


def _get_reversal_signal(close_m, ohlcv_full, cache_path=None):
    """构建/加载滚动反转信号面板（V8/主板版缓存隔离）。"""
    cache_path = cache_path or REVERSAL_SIGNAL_CACHE
    if cache_path.exists():
        try:
            cached = pd.read_parquet(cache_path)
            if (cached.index.equals(close_m.index) and
                    list(cached.columns) == list(close_m.columns)):
                return cached
        except Exception:
            pass
    sig = build_rolling_reversal_signal(close_m, ohlcv_full, window=ROLL_WINDOW)
    try:
        sig.to_parquet(cache_path)
    except Exception:
        pass
    return sig


# ---------------------------------------------------------------------------
# 核心：V8.1 分区间选股（返回最新截面的目标持仓与权重）
# ---------------------------------------------------------------------------
def compute_v81_targets(panels: dict, as_of: pd.Timestamp) -> dict:
    """复算 V8.1 链路，返回最新 month_end 的目标持仓 dict(code -> 目标权重)。

    返回 dict：
      last_me, targets({code: weight}), segment("v8"|"combo"), tw_last,
      factor_label, me, tw_daily, switch_log, sel_v8, sel_trend, sel_brk
    """
    close = panels["close"]; amount = panels["amount"]; idx = panels["idx"]
    roe = panels["roe"]; gpm = panels["gpm"]; ohlcv_old = panels["ohlcv_old"]
    codes = panels["codes"]

    close_m = mask_new_listings(close, config.NEW_STOCK_MIN_DAYS)
    rsi = compute_rsi(close_m, config.RSI_WINDOW)
    me = month_ends_in(close_m, config.START_DATE,
                       min(as_of, pd.Timestamp(panels["data_end"])))

    ohlcv_full = build_ohlcv_full(close, amount, ohlcv_old, codes)
    reversal_signal = _get_reversal_signal(close_m, ohlcv_full,
                                           cache_path=panels["rev_cache"])

    monthly_ic = monthly_reversal_ic(close_m, rsi, me, config.FWD_RETURN_DAYS)
    rolling_ic, use_reversal = compute_rolling_regime(monthly_ic, 12, IC_BASE, 6)
    config.ENABLE_ANALYST_FACTOR = False        # V8 口径（无分析师因子）
    mz = compute_momentum_zscore(close_m, roe, gpm, me)
    sel_v8, switch_log = build_selection_v5(close_m, rsi, reversal_signal, mz, me,
                                            use_reversal, POOL_PCT, N_SELECT)
    sel_trend = sig_trend(close_m, me, top_n=N_SELECT)
    sel_brk = sig_breakout(close_m, amount, me, top_n=N_SELECT)

    tw_daily, vol_regime = build_ma240_vol_target_weight(
        idx, close_m.index, MA_BASE, month_ends=me,
        vol_q=VOL_Q, reduced_weight=REDUCED_WEIGHT, vol_lookback=VOL_LOOKBACK)

    last_me = me[-1]
    tw_last = float(tw_daily.get(last_me, 1.0))

    # ---- 分区间门控 ----
    if last_me < SPLIT_DATE:
        segment = "v8"
        codes_s = sel_v8.get(last_me, [])
        targets = {c: FIXED_WEIGHT for c in codes_s}
        factor_label = (switch_log.loc[last_me, "factor_set"]
                        if last_me in switch_log.index else "?")
    else:
        segment = "combo"
        acc: dict = {}
        combos = [("V8", sel_v8), ("Trend", sel_trend), ("Breakout", sel_brk)]
        for name, sel_s in combos:
            codes_s = sel_s.get(last_me, [])
            if not codes_s:
                print(f"  [combo] {name} 空仓（{last_me.date()}）")
                continue
            w_each = 1.0 / N_COMBO / len(codes_s)
            for c in codes_s:
                acc[c] = min(acc.get(c, 0.0) + w_each, COMBO_CAP)
        targets = acc
        factor_label = "combo(V8+Trend+Breakout)"

    return dict(last_me=last_me, targets=targets, segment=segment, tw_last=tw_last,
                factor_label=factor_label, me=me, tw_daily=tw_daily,
                switch_log=switch_log, vol_regime=vol_regime,
                sel_v8=sel_v8, sel_trend=sel_trend, sel_brk=sel_brk,
                slip_map=None, tier_counts=None,
                rolling_ic=rolling_ic, use_reversal=use_reversal)


# ---------------------------------------------------------------------------
# 股票名称 / 上一日持仓 / 信号清单
# ---------------------------------------------------------------------------
def get_names(codes) -> dict:
    if NAMES_CACHE.exists():
        try:
            return json.loads(NAMES_CACHE.read_text(encoding="utf-8"))
        except Exception:
            pass
    names = {}
    try:
        import akshare as ak
        df = ak.stock_info_a_code_name()
        names = dict(zip(df["code"].astype(str), df["name"]))
        NAMES_CACHE.write_text(json.dumps(names, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print(f"[{datetime.now()}] 名称获取失败（{e}），回退使用代码本身。")
    return names


def load_prev_holdings(signal_dir: str, as_of: pd.Timestamp) -> list:
    p = os.path.join(signal_dir)
    if not os.path.isdir(p):
        return []
    files = sorted(f for f in os.listdir(p) if f.endswith("_signal.csv"))
    prev = None
    for f in files:
        d = pd.Timestamp(f.replace("_signal.csv", ""))
        if d < as_of:
            prev = f
    if prev is None:
        return []
    df = pd.read_csv(os.path.join(p, prev))
    out = []
    for c in df["code"]:
        try:
            out.append(f"{int(c):06d}")
        except (ValueError, TypeError):
            out.append(str(c))
    return out


def build_signal_df(targets: dict, prev_holdings, tw_last, factor_label, names):
    """按目标权重表生成 action 清单（权重降序；与上一持仓对比 BUY/SELL/HOLD）。"""
    def _code(c) -> str:
        try:
            return f"{int(c):06d}"
        except (ValueError, TypeError):
            return str(c)

    prev_set = set(_code(c) for c in prev_holdings)
    sel_set = set(_code(c) for c in targets)
    rows = []
    # 按权重降序（E 组合重叠股优先；V8 段等权 10% 无差异）
    for c, w in sorted(targets.items(), key=lambda kv: -kv[1]):
        cc = _code(c)
        action = "HOLD" if cc in prev_set else "BUY"
        rows.append({
            "code": cc,
            "name": names.get(cc, cc),
            "action": action,
            "target_weight": round(w * 100, 2),
            "factor_set": factor_label,
            "regime_weight": round(tw_last * 100, 2),
        })
    # 上一日持有但不在目标持仓 → SELL
    for c in prev_holdings:
        cc = _code(c)
        if cc not in sel_set:
            rows.append({
                "code": cc,
                "name": names.get(cc, cc),
                "action": "SELL",
                "target_weight": 0.0,
                "factor_set": factor_label,
                "regime_weight": round(tw_last * 100, 2),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="V8.1 每日信号生成器")
    ap.add_argument("--init", action="store_true", help="首次全量初始化")
    ap.add_argument("--date", type=str, default=None,
                    help="信号基准日（默认=数据末日）；截面取该日之前最近月末")
    ap.add_argument("--live", action="store_true", help="先拉取最新行情再生成")
    ap.add_argument("--no-cache", action="store_true", help="禁用反转信号缓存，强制重算")
    args = ap.parse_args()

    if args.no_cache:
        for p in [REVERSAL_SIGNAL_CACHE,
                  config.DATA_DIR / "_mainboard_revsignal_cache.parquet"]:
            if p.exists():
                p.unlink()

    as_of = (pd.Timestamp(args.date) if args.date
             else pd.Timestamp(datetime.now().strftime("%Y-%m-%d")))
    signal_dir = config.OUTPUT_DIR / "signals"
    os.makedirs(signal_dir, exist_ok=True)

    print(f"[{datetime.now()}] === V8.1 信号生成（{'INIT' if args.init else 'DAILY'}）===")
    print(f"[{datetime.now()}] 信号基准日 as_of={as_of.date()}  live={args.live}")

    panels = load_panels(as_of, live=args.live)
    print(f"[{datetime.now()}] 选股池: {panels['panel_label']}  数据区间 "
          f"{panels['data_start']} ~ {panels['data_end']}  共 {len(panels['codes'])} 只")

    res = compute_v81_targets(panels, as_of)
    last_me = res["last_me"]
    targets = res["targets"]
    tw_last = res["tw_last"]
    seg = res["segment"]
    flabel = res["factor_label"]

    # ---- 模拟盘起始日期门控：起始日之前不输出任何信号 ----
    start = getattr(config, "SIM_START_DATE", None)
    if start is not None and pd.Timestamp(last_me).normalize() < pd.Timestamp(start).normalize():
        print(f"[{datetime.now()}] ⛔ 截面月末 {last_me.date()} < SIM_START_DATE={start}，"
              f"起始日之前不输出信号（模拟盘纯净起点，等待 ≥ {start} 的截面）")
        return None

    print(f"[{datetime.now()}] 截面月末={last_me.date()}  分段={seg}  因子集={flabel}  "
          f"目标持仓={len(targets)}  市场目标仓位 regime_weight={tw_last:.2f}")
    # 审计字段：signal_date（信号日=截面月末收盘）/ execution_date（T+1 开盘执行）
    close = panels["close"]
    nxt = close.index[close.index > last_me]
    exec_date = nxt[0].date() if len(nxt) else "(数据末日，待下交易日)"
    print(f"[{datetime.now()}] AUDIT signal_date={last_me.date()} | "
          f"execution_date={exec_date} | 执行价=T+1 开盘价（open，停牌顺延）")
    if seg == "combo":
        w_sorted = sorted(targets.items(), key=lambda kv: -kv[1])
        print(f"[{datetime.now()}] 组合权重: 重叠股 {sum(1 for _, w in w_sorted if w > 1.0/N_COMBO/N_SELECT + 1e-9)} 只，"
              f"权重范围 {w_sorted[-1][1]*100:.2f}%~{w_sorted[0][1]*100:.2f}%")

    names = get_names(list(targets.keys()))
    prev = load_prev_holdings(str(signal_dir), as_of)
    df = build_signal_df(targets, prev, tw_last, flabel, names)

    out_csv = signal_dir / f"{as_of.strftime('%Y-%m-%d')}_signal.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")

    n_buy = int((df["action"] == "BUY").sum())
    n_sell = int((df["action"] == "SELL").sum())
    n_hold = int((df["action"] == "HOLD").sum())
    print(f"[{datetime.now()}] 信号已写入: {out_csv}")
    print(f"[{datetime.now()}] 动作分布: BUY={n_buy}  SELL={n_sell}  HOLD={n_hold}")

    print(f"[{datetime.now()}] 目标持仓(Top 15, 按权重序):")
    for _, r in df[df["action"].isin(["BUY", "HOLD"])].head(15).iterrows():
        print(f"    {r['code']}  {r['name']}  {r['action']}  目标{float(r['target_weight']):.2f}%")
    return df


if __name__ == "__main__":
    main()
