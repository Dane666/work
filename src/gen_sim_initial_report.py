# -*- coding: utf-8 -*-
"""
gen_sim_initial_report.py — V7.1 模拟盘初始化对比报告
================================================================
读取 signal_generator 首次初始化产出的信号清单，与 V7.1 回测的最新截面持仓
（output/v7_holdings_2024_2025.csv 的 2025-08-12 行）做对比，验证「逻辑零改动」：
信号器产出的截面应与原回测完全一致。

同时快速重算当日市场状态（MA240 / 波动率降档），输出 HTML 报告。
"""
from __future__ import annotations

import os
for _k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
           "ALL_PROXY", "all_proxy"]:
    os.environ.pop(_k, None)

from datetime import datetime

import numpy as np
import pandas as pd

import config
from factors import compute_rsi, get_month_end_dates
from market_filter import build_ma240_vol_target_weight
from stress_test_v6 import mask_new_listings, month_ends_in


SIGNAL_CSV = config.OUTPUT_DIR / "signals" / "2025-08-14_signal.csv"
BACKTEST_HOLDINGS = config.OUTPUT_DIR / "v7_holdings_2024_2025.csv"
OUT = config.OUTPUT_DIR / "sim_initial_report.html"

MA_BASE = 240
VOL_Q = 0.75
REDUCED_WEIGHT = 0.60
VOL_LOOKBACK = 756


def market_state(as_of):
    close = pd.read_parquet(config.DATA_DIR / "v6_close_panel.parquet")
    idx = pd.read_parquet(config.DATA_DIR / "v6_index.parquet")
    if isinstance(idx, pd.DataFrame):
        idx = idx.iloc[:, 0]
    close_m = mask_new_listings(close, config.NEW_STOCK_MIN_DAYS)
    me = month_ends_in(close_m, config.START_DATE,
                       min(as_of, pd.Timestamp(str(close.index[-1].date()))))
    _, vol_regime = build_ma240_vol_target_weight(
        idx, close_m.index, MA_BASE, month_ends=me,
        vol_q=VOL_Q, reduced_weight=REDUCED_WEIGHT, vol_lookback=VOL_LOOKBACK)
    last_me = me[-1]
    row = vol_regime.loc[last_me]
    return last_me, row, str(close.index[-1].date())


def main():
    sig = pd.read_csv(SIGNAL_CSV, dtype={"code": str})
    bt = pd.read_csv(BACKTEST_HOLDINGS, dtype={"selected_codes": str})
    last_me, mstate, data_end = market_state(pd.Timestamp("2025-08-14"))

    # 信号器选出的截面（BUY+HOLD 即 Top30）
    sel_now = list(sig[sig["action"].isin(["BUY", "HOLD"])]["code"].astype(str))
    # 回测最新截面
    bt_last = bt.iloc[-1]
    bt_codes = [c for c in str(bt_last["selected_codes"]).split(",") if c]

    set_now, set_bt = set(sel_now), set(bt_codes)
    same = (set_now == set_bt)
    only_sig = sorted(set_now - set_bt)
    only_bt = sorted(set_bt - set_now)

    # 市场状态解读
    tw = float(mstate["target_weight"])
    if tw == 0.0:
        mkt = "MA240 跌破 → 空仓（0% 仓位）"
    elif abs(tw - REDUCED_WEIGHT) < 1e-6:
        mkt = "MA240 站上 但 波动率高位(>历史75分位) → 降档至 60%"
    else:
        mkt = "MA240 站上 且 波动率正常 → 满仓 100%"

    fset = sig["factor_set"].iloc[0] if len(sig) else "?"

    # 拟建仓数（受 regime_weight 现金上限）
    eff = int(min(len(sel_now), tw / config.FIXED_WEIGHT)) if tw > 0 else 0

    html = _render(sig, bt_last, sel_now, bt_codes, same, only_sig, only_bt,
                   last_me, mstate, mkt, tw, fset, eff, data_end)
    OUT.write_text(html, encoding="utf-8")
    print(f"[{datetime.now()}] 初始化对比报告已生成: {OUT}")
    print(f"    信号截面 {len(sel_now)} 只 vs 回测截面 {len(bt_codes)} 只  "
          f"完全一致={same}（仅信号独有 {only_sig}，仅回测独有 {only_bt}）")
    print(f"    市场状态: {mkt}  因子集: {fset}")


def _render(sig, bt_last, sel_now, bt_codes, same, only_sig, only_bt,
            last_me, mstate, mkt, tw, fset, eff, data_end):
    sel_rows = ""
    for _, r in sig.iterrows():
        sel_rows += (f"<tr><td>{r['code']}</td><td>{r['name']}</td>"
                     f"<td>{r['action']}</td><td>{float(r['target_weight']):.1f}%</td>"
                     f"<td>{r['factor_set']}</td></tr>\n")
    diff = ("<p style='color:#1a7f37'><b>✅ 完全一致</b>：信号器产出的截面与 V7.1 回测"
            "最新截面（2025-08-12 月末）完全相同 —— 验证了「逻辑零改动」。</p>"
            if same else
            f"<p style='color:#b42318'><b>⚠️ 存在差异</b>：仅信号独有 "
            f"{only_sig}；仅回测独有 {only_bt}。</p>")

    ctx = ("V7.1 回测基准（分档滑点，全样本 2018-2025）：全期夏普 0.43 / "
           "年化 5.82% / 最大回撤 -26.51%；旧区间 2018-2023 夏普 0.56；"
           "新区间 2024-2025 夏普 0.26（较 V6 -0.32 转正）。")

    return f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>V7.1 模拟盘初始化对比报告</title>
<style>
 body{{font-family:-apple-system,'PingFang SC','Microsoft YaHei',sans-serif;
   margin:24px;color:#222;line-height:1.6;background:#fafafa}}
 h1{{font-size:22px;border-left:5px solid #2f6fed;padding-left:12px}}
 h2{{font-size:17px;margin-top:28px;color:#2f6fed}}
 .card{{background:#fff;border:1px solid #e5e7eb;border-radius:10px;
   padding:16px 20px;margin:14px 0;box-shadow:0 1px 3px rgba(0,0,0,.04)}}
 table{{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}}
 th,td{{border:1px solid #e5e7eb;padding:6px 10px;text-align:left}}
 th{{background:#f1f5f9}}
 .kv{{display:flex;gap:12px;flex-wrap:wrap}}
 .pill{{background:#eef2ff;color:#2f6fed;border-radius:999px;padding:4px 12px;
   font-size:13px;font-weight:600}}
 .meta{{color:#666;font-size:13px}}
</style></head><body>
<h1>V7.1 模拟盘初始化对比报告</h1>
<p class="meta">生成时间 {datetime.now().strftime('%Y-%m-%d %H:%M')} ｜ 信号基准日 2025-08-14 ｜
数据截止 {data_end}</p>

<div class="card">
  <h2>① 初始化持仓（今日拟建仓清单 · 来自 signal_generator --init）</h2>
  <div class="kv">
    <span class="pill">因子集 {fset}</span>
    <span class="pill">市场状态 {mkt}</span>
    <span class="pill">目标总仓位 {tw*100:.0f}%</span>
    <span class="pill">拟建仓 {eff} 只（受现金上限）</span>
  </div>
  <p class="meta">共 {len(sel_now)} 只进入 Top30 候选；实际部署受 regime_weight 现金上限约束
  （FIXED_WEIGHT=10%，故至多 {eff} 只填满）。下一交易日开盘价成交（BUY），沿用持仓为 HOLD。</p>
  <table><thead><tr><th>代码</th><th>名称</th><th>动作</th>
  <th>目标仓位</th><th>因子集</th></tr></thead>
  <tbody>{sel_rows}</tbody></table>
</div>

<div class="card">
  <h2>② 与 V7.1 回测最新截面对比（逻辑零改动验证）</h2>
  <p>回测最新截面月末 = <b>{str(last_me.date())}</b>（{bt_last['factor_set']}，
  {int(bt_last['n_selected'])} 只）。{diff}</p>
</div>

<div class="card">
  <h2>③ 当日市场状态（MA240 + 波动率降档）</h2>
  <div class="kv">
    <span>MA240 站上: <b>{bool(mstate['ma_above'])}</b></span>
    <span>CSI300 60日年化波动率: <b>{float(mstate['vol60'])*100:.1f}%</b></span>
    <span>波动率阈值(历史75分位): <b>{float(mstate['vol_thr'])*100:.1f}%</b></span>
    <span>波动率高位: <b>{bool(mstate['elevated'])}</b></span>
    <span>目标仓位: <b>{tw*100:.0f}%</b></span>
  </div>
</div>

<div class="card">
  <h2>④ 上下文：V7.1 回测基准</h2>
  <p>{ctx}</p>
  <p class="meta">下一交易日（下一个交易日开盘）sim_tracker.py 将按本清单以开盘价建仓、
  并以当日收盘价对跌出 Top30 者平仓；每日 15:30 由 scripts/run_daily.sh 自动调度。</p>
</div>
</body></html>"""


if __name__ == "__main__":
    main()
