# -*- coding: utf-8 -*-
"""
main_mainboard_v3.py — 主板版 V3 回测（仓位门控 → 质量过滤 → 动量窗口）
====================================================================

V2 教训：股息率作选股因子（等权/补充）在收益端均负贡献（高股息=低动量低成长），
仅在 MA240 门控下降回撤（-47%→-21%）。V3 将股息率**从选股因子转为择时门控**
（对症"稀释收益 + 回撤仍可改进"），并叠加质量过滤与长动量窗口。

改动（按 stage 顺序执行，达标即停）：
  V3-1 仓位门控：每月全市场股息率中位数 > 历史滚动均值+1σ → 仓位上限 70%
        （选股仍用三因子 ret_12+roe+gpm_yoy，不再用股息率选股）
  V3-2 质量过滤：ROE(当前报告期)≥5% 且 20日日均成交额≥2000万 且 上市满1年
        （池子 1004 → 预计 400-500 只）
  V3-3 动量窗口：ret_12 替换为 ret_12+ret_24 等权合成

验证标准：全期夏普 ≥0.40 | 2024-25夏普 ≥0.60 | 全期回撤 ≤-22%
（V2 基线：0.33 / 0.52 / -21.2%）

交付：output/report_mainboard_v3.html
运行：cd src && python main_mainboard_v3.py --stage 1     # 只跑 stage1
      python main_mainboard_v3.py                        # 顺序 1→2→3，达标即停
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd

for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
          "ALL_PROXY", "all_proxy"]:
    os.environ.pop(k, None)

import config
from factors import compute_rsi, get_month_end_dates
from factor_eval import build_selection_v5
from backtest_v5 import run_backtest_v5
from backtest_combo import run_backtest_combo
from market_filter import build_ma240_vol_target_weight
from stress_test_v6 import build_slippage_map
from report import compute_metrics, chart_multi_equity, chart_drawdown_compare
from strategies.trend_ema import gen_signal as sig_trend
from strategies.vol_breakout import gen_signal as sig_breakout
from fetch_dividend import get_v2_codes

# 均线门控基准窗口（V3.2 基线 = MA240）。--gate ma120/ma60/vol_dynamic 在
# build_market_gate() 内按 config.GATE_* 切换；MA_BASE 保留供 opt_direction1/2 等
# 外部模块引用（其值恒等于 config.GATE_MA_WINDOWS[config.GATE_DEFAULT]）。
MA_BASE = config.GATE_MA_WINDOWS[config.GATE_DEFAULT]
TOP_N = 30
CAP = 0.10
SPLIT_DATE = "2024-01-01"

# V3.2 实盘门槛（compare 报告结论判定用；与工作记忆一致 0.66/0.85/-20%）
TARGET_V32_FULL = 0.66
TARGET_V32_NEW = 0.85
TARGET_V32_DD = -0.20

# V3 验证标准
TARGET_FULL = 0.40
TARGET_NEW = 0.60
TARGET_DD = -0.22

DY_GATE_WINDOW = 36        # 股息率中位数滚动窗口（月）
DY_GATE_SIGMA = 0.20       # V3.1 方向3：0.20（低σ×高降幅稳健高原，全期0.61/2024-25 0.83）
DY_GATE_WEIGHT = 0.20      # V3.1 方向3：触发时仓位上限 20%（降幅80%；原 V3 为 0.70）
MIN_ROE = 5.0              # 质量过滤：ROE ≥ 5%
MIN_AMOUNT = 2.0e7         # 质量过滤：20日均成交额 ≥ 2000万
MIN_LIST_YEARS = 1         # 质量过滤：上市满 1 年


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


def build_dy_gate(div_yield_panel: pd.DataFrame, me: pd.DatetimeIndex) -> pd.Series:
    """股息率仓位门控（月频 0.7/1.0）：
    每月全市场股息率中位数 > 历史滚动(36月)均值+1σ → 0.70，否则 1.0。
    滚动统计用 shift(1) 防未来函数（决策月只看之前月份）。
    """
    dy_med = div_yield_panel.where(div_yield_panel > 0).median(axis=1).sort_index()
    dy_med = dy_med.reindex(me)
    mean = dy_med.rolling(DY_GATE_WINDOW, min_periods=24).mean().shift(1)
    std = dy_med.rolling(DY_GATE_WINDOW, min_periods=24).std().shift(1)
    gate = pd.Series(1.0, index=me)
    hit = (dy_med > mean + DY_GATE_SIGMA * std) & std.notna()
    gate[hit] = DY_GATE_WEIGHT
    n_hit = int(hit.sum())
    print(f"  [dy门控] 触发 {n_hit}/{len(me)} 个月（股息率中位数>均值+{DY_GATE_SIGMA}σ，仓位降至 {DY_GATE_WEIGHT:.0%}）")
    return gate


def build_market_gate(gate: str, idx, daily_index, me) -> pd.Series:
    """市场门控 target_weight（日频，0 / GATE_VOL_REDUCED_WEIGHT / 1.0）。

    gate 语义（config.GATE_CHOICES）：
      - ma240 / ma120 / ma60：指数站上对应 MA{窗口} 才持仓（跌破清仓 0），
        站上但 60 日年化波动率 > 历史(756日)75 分位 → 降档 0.60；
      - vol_dynamic：去掉均线硬门控（enable_ma_gate=False，不再破位清仓），
        仅按波动率分位动态降档（>分位 → 0.60，否则 1.0）。
    仅替换「市场门控」这一变量；股息率门控（build_dy_gate）在调用方另行相乘。
    """
    w = config.GATE_MA_WINDOWS
    if gate == "vol_dynamic":
        tw, _ = build_ma240_vol_target_weight(
            idx, daily_index, window=w[config.GATE_DEFAULT], month_ends=me,
            vol_q=config.GATE_VOL_Q, reduced_weight=config.GATE_VOL_REDUCED_WEIGHT,
            vol_lookback=config.GATE_VOL_LOOKBACK, enable_ma_gate=False)
    else:
        tw, _ = build_ma240_vol_target_weight(
            idx, daily_index, window=w[gate], month_ends=me,
            vol_q=config.GATE_VOL_Q, reduced_weight=config.GATE_VOL_REDUCED_WEIGHT,
            vol_lookback=config.GATE_VOL_LOOKBACK)
    return tw


def apply_quality_mask(close_m, roe, amount, me, min_roe=MIN_ROE,
                       min_amount=MIN_AMOUNT, min_years=MIN_LIST_YEARS):
    """质量过滤：ROE≥5% + 20日均成交额≥2000万 + 上市满1年。
    在 close_m 上置 NaN（不合格股票无法进入因子计算/选股）。"""
    out = close_m.copy()
    # 上市满 1 年（自首个有效收盘日 +250 自然日）
    for c in out.columns:
        s = out[c].dropna()
        if s.empty:
            continue
        cutoff = s.index.min() + pd.Timedelta(days=min_years * 365)
        out.loc[out.index < cutoff, c] = np.nan
    # ROE ≥ 5%（月度面板月末值，PIT）
    roe_m = roe.reindex(me)
    roe_ok = roe_m >= min_roe
    # 20 日均成交额 ≥ min_amount
    amt20 = amount.rolling(20, min_periods=10).mean().reindex(me)
    amt_ok = amt20 >= min_amount
    # 逐月末 mask
    for t in me:
        if t not in out.index:
            continue
        bad = set(out.columns)
        if t in roe_ok.index:
            bad &= set(roe_ok.loc[t][roe_ok.loc[t] == False].index)
        if t in amt_ok.index:
            bad &= set(amt_ok.loc[t][amt_ok.loc[t] == False].index)
        out.loc[t, list(bad)] = np.nan
    # 统计
    sizes = [out.loc[t].notna().sum() for t in me if t in out.index]
    print(f"  [质量过滤] 末日合格 {sizes[-1] if sizes else 0} 只，平均 {np.mean(sizes) if sizes else 0:.0f} 只")
    return out


def build_mz(close_m, roe, gpm, me, long_momentum=True, persistence=True):
    """动量/质量 Z-score 合成分。
    persistence=True（方向1 动量持续性）：(z(ret12-ret3)+z(ret24)+z(roe)+z(gpm_yoy))/4
    persistence=False（V3.1 基线）：ret=(ret_12+ret_24)/2 与 ROE/毛利率 等权。
    long_momentum 仅影响 persistence=False 路径。"""
    ret12 = close_m.pct_change(config.FWD_RETURN_DAYS * 12)
    ret24 = close_m.pct_change(config.FWD_RETURN_DAYS * 24)
    if persistence:
        ret3 = close_m.pct_change(config.FWD_RETURN_DAYS * 3)
        cols = {"ret_persist": ret12 - ret3, "ret_24": ret24, "roe": roe, "gpm_yoy": gpm}
    else:
        ret_v = (ret12 + ret24) / 2.0 if long_momentum else ret12
        cols = {"ret_12": ret_v, "roe": roe, "gpm_yoy": gpm}
    rows = {}
    for t in pd.DatetimeIndex(me):
        sub = pd.DataFrame({k: v.loc[t] if t in v.index else pd.Series(dtype=float)
                            for k, v in cols.items()})
        sub = sub.dropna()
        if sub.empty:
            rows[t] = pd.Series(dtype=float)
            continue
        sub_w = sub.copy()
        for col in sub_w.columns:
            lo, hi = sub_w[col].quantile(0.01), sub_w[col].quantile(0.99)
            sub_w[col] = sub_w[col].clip(lo, hi)
        z = (sub_w - sub_w.mean()) / sub_w.std(ddof=0)
        rows[t] = z.sum(axis=1) / sub_w.shape[1]
    return pd.DataFrame(rows).T.sort_index()


def run_full(close_m, amount, sel_v8, me, tw, enable_sector=None,
             sector_map=None, bench=None, conc_log=None):
    """分区间回测：段1 V8(≤2023) → 段2 E等权(≥2024)。返回拼接净值。
    enable_sector=None → 跟随 config.ENABLE_SECTOR_NEUTRAL（方向2 开关）；
    显式 True/False 供对比实验。sector_map/bench 可外部注入（None 时自动加载）。
    conc_log：若提供 list，则记录各月末目标持仓的行业集中度（验证约束生效）。"""
    from v3_common import apply_sector_neutral, load_sector_data
    if enable_sector is None:
        enable_sector = bool(getattr(config, "ENABLE_SECTOR_NEUTRAL", False))
    if enable_sector and (sector_map is None or bench is None):
        sector_map, bench = load_sector_data()
        if sector_map is None or bench is None:
            print("  [sector] ⚠️ 行业数据缺失，本段跳过中性化")
            enable_sector = False

    def _bench_row(t):
        if bench is None or t not in bench.index:
            return {}
        return {str(k): float(v) for k, v in bench.loc[t].items()}

    def _log_conc(t, weights, mode):
        if conc_log is None or sector_map is None:
            return
        ind_w = {}
        for c, wt in weights.items():
            s = sector_map.get(str(c), "其他")
            ind_w[s] = ind_w.get(s, 0.0) + wt
        if not ind_w:
            return
        top = sorted(ind_w.items(), key=lambda kv: -kv[1])
        conc_log.append({
            "date": str(t.date()), "mode": mode, "top1_industry": top[0][0],
            "top1_weight": top[0][1], "n_industries": len(ind_w),
            "top3_weight": sum(v for _, v in top[:3]),
        })

    sel_trend = sig_trend(close_m, me, top_n=TOP_N)
    sel_brk = sig_breakout(close_m, amount, me, top_n=TOP_N)
    me1 = me[me < pd.Timestamp(SPLIT_DATE)]
    me2 = me[me >= pd.Timestamp(SPLIT_DATE)]

    # 段1 V8：selection 列表 → 等权；enable_sector 时中性化后经 weight_mult 注入
    wmult = {} if enable_sector else None
    for t in me1:
        cc = sel_v8.get(t, [])
        if not cc:
            continue
        w0 = {c: 1.0 / len(cc) for c in cc}
        _log_conc(t, w0, "raw")
        if enable_sector:
            w1 = apply_sector_neutral(w0, sector_map, _bench_row(t))
            for c in cc:
                wmult[(t, c)] = w1.get(c, 0.0) / 0.10   # target = 0.10*eq*mult = wt*eq
            _log_conc(t, w1, "neutral")
    eq1, _ = run_backtest_v5(close_m, sel_v8, me1, config.START_DATE, "2023-12-31",
                             target_weight=tw, slippage_map=slip_map_, weight_mult=wmult)
    init2 = float(eq1.iloc[-1])

    # 段2 E等权组合：V8/Trend/Breakout 各 1/3，单只 ≤ CAP
    sched = {}
    for t in me2:
        acc = {}
        for name, s_ in [("V8", sel_v8), ("Trend", sel_trend), ("Breakout", sel_brk)]:
            cc = s_.get(t, [])
            if not cc:
                continue
            we = 1.0 / 3 / len(cc)
            for c in cc:
                acc[c] = min(acc.get(c, 0.0) + we, CAP)
        sched[t] = list(acc.items())
    if enable_sector:
        for t in me2:
            w1 = apply_sector_neutral(dict(sched[t]), sector_map, _bench_row(t))
            sched[t] = list(w1.items())
    for t in me2:
        _log_conc(t, dict(sched[t]), "neutral" if enable_sector else "raw")
    eq2, _ = run_backtest_combo(close_m, sched, me2, SPLIT_DATE, str(close_m.index[-1].date()),
                                target_weight=tw, slippage_map=slip_map_, init_capital=init2)
    eq = pd.concat([eq1, eq2])
    eq = eq[~eq.index.duplicated(keep="first")].sort_index()
    return eq


def main():
    ap = argparse.ArgumentParser(description="主板版 V3 回测（--gate 门控可配置 / --gate-compare 门控消融对比）")
    ap.add_argument("--stage", type=int, default=0,
                    help="1=仓位门控 / 2=+质量过滤 / 3=+长动量；0=顺序 1→2→3 达标即停")
    ap.add_argument("--gate", choices=config.GATE_CHOICES, default=None,
                    help="市场门控（默认 config.GATE_DEFAULT=%s）：ma240/ma120/ma60=指数站上对应均线才持仓"
                         "（跌破清仓+波动率降档），vol_dynamic=去掉均线硬门仅按波动率动态降档。"
                         "指定时以完整 V3.2 链（等价 --stage 3）跑单门控实验。" % config.GATE_DEFAULT)
    ap.add_argument("--gate-compare", action="store_true",
                    help="门控消融对比：对基线 ma240 + ma120 + ma60 + vol_dynamic 各跑一次完整 V3.2 链，"
                         "输出 output/report_gate_compare.html（含净值/回撤曲线对比图）")
    ap.add_argument("--export-nav", action="store_true",
                    help="跑完把回测净值序列导出为 data/state/theoretical_nav.csv（周报对比基准）")
    args = ap.parse_args()

    global slip_map_
    t0 = datetime.now()
    codes = get_v2_codes()
    close = pd.read_parquet(config.MB_CLOSE).reindex(columns=codes)
    amount = pd.read_parquet(config.MB_AMOUNT).reindex(columns=codes)
    idx = pd.read_parquet(config.DATA_DIR / "v6_index.parquet")
    if isinstance(idx, pd.DataFrame):
        idx = idx.iloc[:, 0]
    roe = pd.read_parquet(config.MB_ROE).reindex(index=close.index, columns=codes).ffill()
    gpm = pd.read_parquet(config.MB_GPM).reindex(index=close.index, columns=codes).ffill()
    dy = pd.read_parquet(config.DATA_DIR / "div_yield_panel_mainboard.parquet").reindex(columns=codes)
    slip_map_, _ = build_slippage_map(amount)

    close_m = mask_new_listings(close, config.NEW_STOCK_MIN_DAYS)
    rsi = compute_rsi(close_m, config.RSI_WINDOW)
    me = month_ends_in(close_m, config.START_DATE, str(close.index[-1].date()))

    # ---- 门控实验分支（--gate-compare 全量消融 / --gate 单门控）----
    if args.gate_compare:
        _run_gate_compare(close_m, amount, roe, gpm, dy, idx, rsi, me, t0)
        return
    if args.gate is not None:
        _run_gate_single(args.gate, close_m, amount, roe, gpm, dy, idx, rsi, me,
                         args.export_nav, t0)
        return

    # 默认 stage 流程：基础市场门控（日频，V3.2 基线 ma240）
    tw_base = build_market_gate(config.GATE_DEFAULT, idx, close_m.index, me)

    stages = [1, 2, 3] if args.stage == 0 else [args.stage]
    results = {}
    for st in stages:
        label = {1: "V3-1 三因子 + 股息率仓位门控",
                 2: "V3-2 + 质量过滤(ROE≥5%/额≥2000万/上市1年)",
                 3: "V3-3 + 长动量(ret_12+ret_24)"}[st]
        print(f"\n[{datetime.now()}] ==== {label} ====")

        # 质量过滤（stage≥2）
        cm = close_m
        if st >= 2:
            cm = apply_quality_mask(close_m.copy(), roe, amount, me)

        # 动量/质量合成分（stage≥3 长动量）
        mz = build_mz(cm, roe, gpm, me, long_momentum=(st >= 3))
        use_rev = pd.Series(False, index=me)
        sel_v8, _ = build_selection_v5(
            cm, rsi, pd.DataFrame(index=cm.index, columns=cm.columns),
            mz, me, use_rev, 0.20, TOP_N)

        # 仓位门控（所有 stage 都有）：tw = 市场门控 × 股息率门控
        tw = tw_base
        if st >= 1:
            dy_gate = build_dy_gate(dy, me)
            gate_daily = dy_gate.reindex(close_m.index).ffill().fillna(1.0)
            tw = tw_base * gate_daily

        eq = run_full(cm, amount, sel_v8, me, tw)
        mf = compute_metrics(eq)
        mn = compute_metrics(eq.loc["2024-01-01":])
        results[st] = (mf, mn)
        print(f"  → {label}: 全期夏普={mf['sharpe']:.2f} 回撤={mf['max_drawdown']*100:.1f}% | "
              f"2024-25夏普={mn['sharpe']:.2f} 回撤={mn['max_drawdown']*100:.1f}%")

        ok = (mf["sharpe"] >= TARGET_FULL and mn["sharpe"] >= TARGET_NEW
              and mf["max_drawdown"] >= TARGET_DD)
        if ok:
            print(f">>> {label} ✅ 达标（0.40/0.60/-22%），停止后续 stage")
            if args.export_nav:
                _export_nav(eq)
            _write_report(results, label, eq, mf, mn, st, t0)
            return

    # 全部跑完未达标：用最后 stage 生成报告
    st = stages[-1]
    mf, mn = results[st]
    print(f">>> 全部 stage 未达标，维持 V8.1 原版")
    if args.export_nav:
        _export_nav(eq)
    _write_report(results, None, None, mf, mn, st, t0)


def _run_full_chain(gate, close_m, amount, roe, gpm, dy, idx, rsi, me):
    """完整 V3.2 链（等价 --stage 3 语义）+ 指定市场门控 gate：
    质量过滤 → 动量合成分(持久性,persistence=True) → V8 选股 →
    市场门控 × 股息率门控 → 分段回测。返回 (eq, tw_daily)。
    tw_daily 供报告统计仓位分布（门控行为特征：空仓/降档/满仓占比）。"""
    cm = apply_quality_mask(close_m.copy(), roe, amount, me)
    mz = build_mz(cm, roe, gpm, me, long_momentum=True)   # persistence=True（方向1 动量持续性）
    use_rev = pd.Series(False, index=me)
    sel_v8, _ = build_selection_v5(
        cm, rsi, pd.DataFrame(index=cm.index, columns=cm.columns),
        mz, me, use_rev, 0.20, TOP_N)
    tw_base = build_market_gate(gate, idx, close_m.index, me)
    dy_gate = build_dy_gate(dy, me)
    tw = tw_base * dy_gate.reindex(close_m.index).ffill().fillna(1.0)
    eq = run_full(cm, amount, sel_v8, me, tw)
    return eq, tw


# 门控展示名与机制说明（对比报告用）
GATE_DESC = {
    "ma240":     "指数站上 MA240 才持仓（跌破清仓），波动率超 75 分位降档 0.60 —— V3.2 基线",
    "ma120":     "门控缩短为 MA120（更敏感），其余与基线一致",
    "ma60":      "门控缩短为 MA60（最敏感），其余与基线一致",
    "vol_dynamic": "去掉均线硬门控（不再破位清仓），仅按 60 日年化波动率 vs 756日75分位 动态降档 0.60/1.0",
}


def _run_gate_single(gate, close_m, amount, roe, gpm, dy, idx, rsi, me,
                     export_nav, t0):
    """单门控实验：完整 V3.2 链跑指定门控，打印指标（可选导出理论净值）。"""
    print(f"\n[{datetime.now()}] ==== 单门控实验：{gate}（完整 V3.2 链）====")
    eq, tw = _run_full_chain(gate, close_m, amount, roe, gpm, dy, idx, rsi, me)
    mf = compute_metrics(eq)
    mn = compute_metrics(eq.loc[SPLIT_DATE:])
    print(f"  → {gate}: 全期年化={mf['annual_return']*100:.1f}% 夏普={mf['sharpe']:.2f} "
          f"回撤={mf['max_drawdown']*100:.1f}% | 2024-25夏普={mn['sharpe']:.2f} "
          f"回撤={mn['max_drawdown']*100:.1f}% | 耗时 {datetime.now()-t0}")
    if export_nav:
        _export_nav(eq)
    return eq


def _run_gate_compare(close_m, amount, roe, gpm, dy, idx, rsi, me, t0):
    """门控消融对比：对 GATE_CHOICES（ma240 基线 + ma120/ma60/vol_dynamic）
    各跑一次完整 V3.2 链，输出 output/report_gate_compare.html。"""
    gates = list(config.GATE_CHOICES)
    print(f"\n[{datetime.now()}] ==== 门控消融对比（完整 V3.2 链 × {len(gates)} 门控）====")
    eqs, tws, mets = {}, {}, {}
    for g in gates:
        print(f"\n[{datetime.now()}] ---- 门控 {g} ----")
        eq, tw = _run_full_chain(g, close_m, amount, roe, gpm, dy, idx, rsi, me)
        mf = compute_metrics(eq)
        mn = compute_metrics(eq.loc[SPLIT_DATE:])
        eqs[g], tws[g], mets[g] = eq, tw, (mf, mn)
        print(f"  → {g}: 全期年化={mf['annual_return']*100:.1f}% 夏普={mf['sharpe']:.2f} "
              f"回撤={mf['max_drawdown']*100:.1f}% | 2024-25夏普={mn['sharpe']:.2f} "
              f"回撤={mn['max_drawdown']*100:.1f}% | 仓位均值={tw.mean():.0%} "
              f"空仓占比={(tw == 0).mean():.0%} 降档占比={(tw == config.GATE_VOL_REDUCED_WEIGHT).mean():.0%}")
    _write_gate_compare_report(gates, eqs, tws, mets, t0)


def _setup_cn_font() -> bool:
    """注册系统中文字体并写入 rcParams；无可用字体返回 False（图内文字退化英文）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as _fm
    import matplotlib.pyplot as _plt
    for p in [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]:
        if os.path.exists(p):
            try:
                _fm.fontManager.addfont(p)
            except Exception:
                pass
    for name in ["Noto Sans CJK SC", "PingFang SC", "Hiragino Sans GB",
                 "WenQuanYi Zen Hei", "Microsoft YaHei", "SimHei",
                 "Arial Unicode MS"]:
        try:
            if _fm.findfont(_fm.FontProperties(family=name),
                            fallback_to_default=False):
                _plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
                _plt.rcParams["axes.unicode_minus"] = False
                return True
        except Exception:
            continue
    _plt.rcParams["axes.unicode_minus"] = False
    return False


def _plot_gate_compare(gates, eqs, nav_png, dd_png) -> None:
    """净值曲线 + 回撤曲线双图（PNG 落盘，供 HTML 引用）。"""
    cn = _setup_cn_font()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"ma240": "#2471a3", "ma120": "#c0392b", "ma60": "#d35400",
              "vol_dynamic": "#16a085"}
    _zh = (lambda zh, en: zh) if cn else (lambda zh, en: en)

    # 净值曲线
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for g in gates:
        eq = eqs[g]
        ax.plot(eq.index, eq.values, lw=1.3, color=colors.get(g, None),
                label=f"{g}" + ("" if cn else ""))
    ax.set_title(_zh("门控消融对比：全期净值（完整 V3.2 链）",
                     "Gate ablation: full-period NAV (full V3.2 chain)"), fontsize=12)
    ax.set_xlabel(_zh("日期", "Date")); ax.set_ylabel(_zh("净值（期初=1）", "NAV"))
    ax.legend(loc="upper left", fontsize=9, framealpha=0.7)
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(nav_png, dpi=120); plt.close(fig)

    # 回撤曲线
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for g in gates:
        eq = eqs[g]
        dd = eq / eq.cummax() - 1.0
        ax.plot(dd.index, dd.values * 100, lw=1.1, color=colors.get(g, None), label=g)
    ax.set_title(_zh("门控消融对比：回撤曲线（%）", "Gate ablation: drawdown (%)"), fontsize=12)
    ax.set_xlabel(_zh("日期", "Date")); ax.set_ylabel("%")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.7)
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(dd_png, dpi=120); plt.close(fig)


def _write_gate_compare_report(gates, eqs, tws, mets, t0) -> None:
    from pathlib import Path
    base = "ma240"
    base_mf, base_mn = mets[base]
    rows = ""
    for g in gates:
        mf, mn = mets[g]
        tw = tws[g]
        ok = (mf["sharpe"] >= TARGET_V32_FULL and mn["sharpe"] >= TARGET_V32_NEW
              and mf["max_drawdown"] >= TARGET_V32_DD)
        badge = "✅" if ok else "❌"
        d_sh = mf["sharpe"] - base_mf["sharpe"]
        d_dd = (mf["max_drawdown"] - base_mf["max_drawdown"]) * 100
        rows += (f"<tr><td><b>{g}</b>{'（基线）' if g == base else ''}</td>"
                 f"<td style='text-align:left;font-size:12px'>{GATE_DESC[g]}</td>"
                 f"<td>{mf['annual_return']*100:.1f}%</td>"
                 f"<td>{mf['sharpe']:.2f}</td><td>{mf['max_drawdown']*100:.1f}%</td>"
                 f"<td>{mn['sharpe']:.2f}</td><td>{mn['max_drawdown']*100:.1f}%</td>"
                 f"<td>{tw.mean():.0%}</td><td>{(tw == 0).mean():.0%}</td>"
                 f"<td>{badge}</td></tr>")
        print(f"  [报告] {g}: Δ夏普vs基线 {d_sh:+.2f}, Δ回撤 {d_dd:+.1f}pp "
              f"{'✅达标' if ok else '未达标'}(0.66/0.85/-20%)")

    # 结论：达标优先 → 2024-25 夏普最高且回撤≥门槛
    cands = [g for g in gates if mets[g][0]["max_drawdown"] >= TARGET_V32_DD]
    scored = sorted(cands, key=lambda g: (-(1 if (mets[g][0]['sharpe'] >= TARGET_V32_FULL
                                                  and mets[g][1]['sharpe'] >= TARGET_V32_NEW) else 0),
                                           -mets[g][1]["sharpe"]))
    best = scored[0] if scored else max(gates, key=lambda g: mets[g][1]["sharpe"])
    b_mf, b_mn = mets[best]
    b_ok = (b_mf["sharpe"] >= TARGET_V32_FULL and b_mn["sharpe"] >= TARGET_V32_NEW
            and b_mf["max_drawdown"] >= TARGET_V32_DD)
    if best == base:
        b_tag = "基线门控"
    else:
        b_tag = "✅ 达 V3.2 门槛 0.66/0.85/-20%" if b_ok else "未达 V3.2 门槛"
    verdict_html = ("<b>推荐门控：{g}</b> —— 全期夏普 {fs:.2f} / 2024-25夏普 {ns:.2f} "
                    "/ 回撤 {dd:.1f}%（{tag}）").format(
        g=best, fs=b_mf["sharpe"], ns=b_mn["sharpe"],
        dd=b_mf["max_drawdown"] * 100, tag=b_tag)

    config.OUTPUT_DIR.mkdir(exist_ok=True)
    nav_png = config.OUTPUT_DIR / "gate_compare_nav.png"
    dd_png = config.OUTPUT_DIR / "gate_compare_dd.png"
    try:
        _plot_gate_compare(gates, eqs, nav_png, dd_png)
    except Exception as e:      # 图失败不阻断 HTML（工程坑：字体/后端）
        print(f"  [报告] ⚠️ 净值图生成失败（跳过，仅出表）: {e}")
        nav_png = dd_png = None

    imgs = ""
    if nav_png and nav_png.exists():
        imgs += f"<h2>净值曲线</h2><img src='{nav_png.name}' style='max-width:100%'>"
    if dd_png and dd_png.exists():
        imgs += f"<h2>回撤曲线</h2><img src='{dd_png.name}' style='max-width:100%'>"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>门控消融对比报告（ma240/ma120/ma60/vol_dynamic）</title>
<style>body{{font-family:-apple-system,'PingFang SC',sans-serif;max-width:1200px;margin:24px auto;padding:0 16px;color:#222}}
table{{border-collapse:collapse;width:100%;margin:14px 0;font-size:13px}}
th,td{{border:1px solid #ddd;padding:7px 9px;text-align:center}}
th{{background:#f5f5f5}} h2{{border-bottom:2px solid #eee;padding-bottom:6px;margin-top:30px}}
.note{{background:#f2f6fc;border-left:4px solid #4a7abb;padding:10px 14px;font-size:13px;color:#444;margin:12px 0}}
.verdict{{background:#fff7e6;border:1px solid #f0c36d;border-radius:6px;padding:12px 16px;margin:16px 0}}
img{{max-width:100%;margin:6px 0}}</style></head><body>
<h1>门控消融对比：MA窗口 & 波动率动态</h1>
<p>引擎：<b>完整 V3.2 链</b>（质量过滤 ROE≥5%/额≥2000万/上市1年 + 持久性动量 + 股息率门控 σ0.20→仓位20%
+ 行业中性化 cap×1.0），仅切换「市场门控」一个变量。
选股池：V8 指数成分 ∩ 主板 60/00（1004 只）｜月频｜分档滑点｜段1 V8(≤2023)→段2 E等权(≥2024)。</p>
<div class="note"><b>门控机制（config.GATE_* 可配置）：</b>市场门控为日频 target_weight =
MA{'{窗口}'}×波动率：指数站上均线才持仓（跌破清仓 0），站上但 60 日年化波动率 &gt; 历史(756日)75 分位 →
降档 0.60，否则 1.0；再与股息率门控相乘。
<b>ma120 / ma60</b>：仅缩短均线窗口（更早离场/入场）；<b>vol_dynamic</b>：去掉均线硬门
（不再破位清仓，波动率超分位 → 0.60，否则 1.0）。全期回测 2018-01 ~ {eqs[gates[0]].index[-1].date()}。</div>
<h2>指标对比（V3.2 门槛：全期≥0.66 / 2024-25≥0.85 / 回撤≥-20%）</h2>
<table><thead><tr><th>门控</th><th>机制</th><th>全期年化</th><th>全期夏普</th><th>全期回撤</th>
<th>2024-25夏普</th><th>2024-25回撤</th><th>日均仓位</th><th>空仓占比</th><th>达标</th></tr></thead>
<tbody>{rows}</tbody></table>
<div class="verdict">{verdict_html}<br><span style="font-size:12px;color:#777">
基线 ma240：全期夏普 {base_mf['sharpe']:.2f} / 2024-25 {base_mn['sharpe']:.2f} / 回撤 {base_mf['max_drawdown']*100:.1f}%
（应复现 V3.2 主版本 0.76/1.23/-19.5%；数据截止 {eqs[base].index[-1].date()} 与此前快照不同年份，指标以本表为准）。</span></div>
{imgs}
<p style="font-size:12px;color:#999">生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')} 耗时 {datetime.now()-t0}｜
引擎零改动；门控合并入 target_weight，未触碰 backtest_v5/backtest_combo 主逻辑。</p>
</body></html>"""
    out = config.OUTPUT_DIR / "report_gate_compare.html"
    out.write_text(html, encoding="utf-8")
    print(f"\n[{datetime.now()}] 门控对比报告: {out}  耗时 {datetime.now()-t0}")


def _export_nav(eq: pd.Series) -> None:
    """把回测净值序列导出为 data/state/theoretical_nav.csv（date,nav 两列）。
    供 weekly_report.py 周报与模拟盘实际净值对比（理论基准）。"""
    if eq is None or len(eq) == 0:
        print("[export-nav] ⚠️ 无净值序列可导出（eq 为空）")
        return
    out = pd.DataFrame({"date": eq.index, "nav": eq.values.astype(float)})
    out.to_csv(config.THEORETICAL_NAV, index=False, encoding="utf-8")
    print(f"[export-nav] 理论净值已导出: {config.THEORETICAL_NAV} "
          f"({len(out)} 行, {out['date'].iloc[0]} ~ {out['date'].iloc[-1]})")


def _write_report(results, label, eq, mf, mn, st, t0):
    from pathlib import Path
    rows_html = ""
    for s, (m_f, m_n) in results.items():
        nm = {1: "V3-1 仓位门控", 2: "V3-2 +质量过滤", 3: "V3-3 +长动量"}[s]
        rows_html += (f"<tr><td>{nm}</td><td>{m_f['sharpe']:.2f}</td>"
                      f"<td>{m_f['max_drawdown']*100:.2f}%</td>"
                      f"<td>{m_n['sharpe']:.2f}</td></tr>")
    verdict = ("<b>✅ 达标</b>" if (mf["sharpe"] >= TARGET_FULL and mn["sharpe"] >= TARGET_NEW
              and mf["max_drawdown"] >= TARGET_DD) else "<b>❌ 未达标</b>")
    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>主板版 V3 回测报告</title>
<style>body{{font-family:-apple-system,'PingFang SC',sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222}}
table{{border-collapse:collapse;width:100%;margin:14px 0;font-size:13px}}
th,td{{border:1px solid #ddd;padding:7px 9px;text-align:center}}
th{{background:#f5f5f5}} h2{{border-bottom:2px solid #eee;padding-bottom:6px;margin-top:34px}}
.note{{background:#f2f6fc;border-left:4px solid #4a7abb;padding:10px 14px;font-size:13px;color:#444;margin:12px 0}}
.verdict{{background:#fff7e6;border:1px solid #f0c36d;border-radius:6px;padding:12px 16px;margin:16px 0}}
img{{max-width:100%}}</style></head><body>
<h1>主板版 V3 回测报告</h1>
<p>选股池：V8 指数成分 ∩ 主板 60/00（1004 只）｜月频｜分档滑点｜MA240+波动率+股息率门控</p>
<div class="note"><b>V3 机制：</b>股息率从选股因子转为<b>择时门控</b>（每月全市场股息率中位数
&gt;历史36月均值+1σ → 仓位上限降至 70%）；V3-2 叠加质量过滤（ROE≥5%、20日成交额≥2000万、
上市满1年）；V3-3 动量窗口加长（ret_12+ret_24 等权）。选股恒为 ret_12/ROE/毛利率 三因子质量模式。</div>
<h2>Stage 结果（目标：全期≥0.40 / 2024-25≥0.60 / 回撤≤-22%）</h2>
<table><thead><tr><th>Stage</th><th>全期夏普</th><th>全期回撤</th><th>2024-25夏普</th></tr></thead>
<tbody><tr><td>V2 基线</td><td>0.33</td><td>-21.2%</td><td>0.52</td></tr>{rows_html}</tbody></table>
<div class="verdict">{verdict}（生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}）</div>
<p style="font-size:12px;color:#999">引擎零改动；门控合并入 target_weight，未触碰 backtest_v5 主逻辑。</p>
</body></html>"""
    out = config.OUTPUT_DIR / "report_mainboard_v3.html"
    out.write_text(html, encoding="utf-8")
    print(f"\n[{datetime.now()}] 报告: {out}  耗时 {datetime.now()-t0}")


if __name__ == "__main__":
    main()
