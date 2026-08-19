# -*- coding: utf-8 -*-
"""
verify_sim_vs_backtest.py — 模拟盘 vs 回测净值一致性验证（V8.1 首次建仓）。

验证口径（诚实披露）：
  模拟盘：2025-08-13 开盘价建仓（含滑点），收盘价标记市值 → NAV_sim（归一化 1.0 起点=08-12）。
  回测理论：V8.1 全期末值 1.907788（08-12 收盘）→ 归一化 1.0；
  08-13 理论 = 08-12 收盘建仓 × (1 + Σ w_i × (close_13/close_12 - 1))（回测口径：close 建仓、含滑点）。
  两者建仓时点差一个隔夜（close_12 vs open_13），属系统性时点差，报告中单独标注。

用法：先运行
  python sim_tracker.py --init --date 2025-08-12
  python sim_tracker.py --date 2025-08-13
再运行本脚本生成对比图与报告。
"""

from __future__ import annotations

import os
import sys
import json
from datetime import datetime

for _k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
           "ALL_PROXY", "all_proxy"]:
    os.environ.pop(_k, None)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from stress_test_v6 import build_slippage_map
from report import _fig_to_b64


def main():
    sig_csv = config.OUTPUT_DIR / "signals" / "2025-08-12_signal.csv"
    state_f = config.OUTPUT_DIR / "sim_nav" / "sim_state.json"
    nav_csv = config.OUTPUT_DIR / "sim_nav" / "sim_nav_history.csv"
    if not sig_csv.exists():
        raise SystemExit("缺少 2025-08-12 信号，先运行 signal_generator.py --init --date 2025-08-12")
    if not state_f.exists():
        raise SystemExit("缺少 sim_state，先运行 sim_tracker.py --init --date 2025-08-12")

    # ---- 1) 模拟盘状态（08-13 执行后的持仓与净值）----
    state = json.loads(state_f.read_text(encoding="utf-8"))
    nav_hist = pd.read_csv(nav_csv) if nav_csv.exists() else pd.DataFrame()
    nav_hist["date"] = pd.to_datetime(nav_hist["date"])
    print("模拟盘净值历史:")
    print(nav_hist[["date", "nav", "cash", "position_value", "note"]].to_string(index=False)
          if "note" in nav_hist.columns else
          nav_hist[["date", "nav", "cash", "position_value"]].to_string(index=False))

    # ---- 2) 回测理论（V8.1 08-12 收盘净值）----
    bt_end = 1.907788e6   # V8.1 全期末值（2025-08-12 收盘，来自 main_v8_1_split 输出）
    sig = pd.read_csv(sig_csv)
    sig = sig[sig["action"] != "SELL"].copy()

    # ---- 3) 08-13 个股行情（open/close）----
    close_panel = pd.read_parquet(config.DATA_DIR / "v8_close_panel.parquet")
    codes = [str(c).zfill(6) for c in sig["code"]]
    # 08-12 收盘（面板）
    p_12 = close_panel.loc["2025-08-12"][codes]
    # 08-13 行情：优先本地 OHLC（v8_ohlcv.pkl），失败则 akshare
    ohlcv = {}
    ohlcv_path = config.DATA_DIR / "v8_ohlcv.pkl"
    if ohlcv_path.exists():
        import pickle
        with open(ohlcv_path, "rb") as f:
            ohlcv = pickle.load(f)

    opens_13, closes_13 = {}, {}
    for c in codes:
        if c in ohlcv and "2025-08-13" in ohlcv[c].index:
            opens_13[c] = float(ohlcv[c].loc["2025-08-13", "open"])
            closes_13[c] = float(ohlcv[c].loc["2025-08-13", "close"])
    if len(opens_13) < len(codes):
        print(f"  [warn] OHLC 本地覆盖 {len(opens_13)}/{len(codes)}，尝试 akshare 补拉...")
        import akshare as ak
        for c in codes:
            if c in opens_13:
                continue
            try:
                df = ak.stock_zh_a_daily(symbol=("sh" + c if c.startswith("6") else "sz" + c),
                                         start_date="20250812", end_date="20250813", adjust="qfq")
                if df is not None and len(df):
                    df = df.set_index("date")
                    df.index = pd.to_datetime(df.index)
                    if "2025-08-13" in df.index:
                        opens_13[c] = float(df.loc["2025-08-13", "open"])
                        closes_13[c] = float(df.loc["2025-08-13", "close"])
            except Exception:
                pass
    have = [c for c in codes if c in opens_13]
    print(f"08-13 行情覆盖: {len(have)}/{len(codes)} 只")

    # ---- 4) 回测理论 08-13 净值（08-12 close 建仓 → 08-13 close）----
    slip_map, _ = build_slippage_map(pd.read_parquet(config.DATA_DIR / "v8_amount_panel.parquet"))
    w_map = {}
    for _, r in sig.iterrows():
        w_map[str(r["code"]).zfill(6)] = float(r["target_weight"]) / 100.0
    r_bt = 0.0
    for c in have:
        w = w_map[c]
        sl = slip_map.get(c, 0.0050)
        ret = closes_13[c] / p_12[c] - 1.0
        r_bt += w * (ret - 2 * sl)          # 回测口径：close 买入/卖出各扣一次滑点
    nav_bt_13_norm = 1.0 * (1 + r_bt)       # 归一化（08-12=1.0）
    print(f"回测理论 08-13 归一化净值: {nav_bt_13_norm:.6f}（08-12 收盘建仓→08-13 收盘，含滑点）")

    # ---- 5) 模拟盘 08-13 净值（若已执行）----
    sim_13 = nav_hist[nav_hist["date"] == "2025-08-13"]
    if len(sim_13):
        nav_sim_13 = float(sim_13["nav"].iloc[0])
    else:
        nav_sim_13 = float(state.get("nav", 1.0))
        print("  [warn] 模拟盘尚未执行 08-13，用 state.nav 近似（可能仍为 1.0 挂起）")

    # 模拟盘口径重算（open 建仓→close，含滑点）：独立核算，与 sim_tracker 交叉验证
    r_sim = 0.0
    for c in have:
        w = w_map[c]
        sl = slip_map.get(c, 0.0050)
        r_sim += w * (closes_13[c] / (opens_13[c] * (1 + sl)) - 1.0)
    nav_sim_calc = 1.0 + r_sim
    print(f"模拟盘口径核算(open建仓): 净值={nav_sim_calc:.6f} | sim_tracker记录={nav_sim_13:.6f}")

    # ---- 6) 偏差 ----
    dev = abs(nav_sim_13 - nav_bt_13_norm)
    dev_pct = dev / nav_bt_13_norm * 100
    print(f"\n偏差: |{nav_sim_13:.6f} - {nav_bt_13_norm:.6f}| = {dev:.6f} = {dev_pct:.3f}%")

    # ---- 7) 图 ----
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot([pd.Timestamp("2025-08-12"), pd.Timestamp("2025-08-13")],
            [1.0, nav_bt_13_norm], marker="o", label="回测理论（08-12 close 建仓）", color="#2980b9")
    ax.plot([pd.Timestamp("2025-08-12"), pd.Timestamp("2025-08-13")],
            [1.0, nav_sim_13], marker="o", label="模拟盘（08-13 open 建仓）", color="#c0392b")
    ax.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax.set_title("V8.1 模拟盘 vs 回测（首次建仓 2025-08-13，归一化）")
    ax.set_ylabel("净值")
    ax.legend()
    ax.grid(alpha=0.3)
    img = _fig_to_b64(fig)
    plt.close(fig)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>V8.1 模拟盘 vs 回测净值对齐验证</title>
<style>body{{font-family:-apple-system,'PingFang SC',sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;color:#222}}
table{{border-collapse:collapse;width:100%;margin:14px 0;font-size:13px}}
th,td{{border:1px solid #ddd;padding:7px 9px;text-align:center}}
th{{background:#f5f5f5}} .verdict{{background:#eefaf0;border:1px solid #a7d7b0;border-radius:6px;padding:12px 16px;margin:16px 0}}
.note{{background:#f2f6fc;border-left:4px solid #4a7abb;padding:10px 14px;font-size:13px;color:#444;margin:12px 0}}
img{{max-width:100%}}</style></head><body>
<h1>V8.1 模拟盘 vs 回测净值对齐验证</h1>
<div class="note"><b>口径说明：</b>回测理论 = 08-12 收盘建仓（V8.1 引擎口径，末值 1,907,788 归一化 1.0）→
08-13 收盘；模拟盘 = 08-13 开盘建仓（sim_tracker，含滑点）→ 08-13 收盘。
两者建仓时点差一个隔夜跳空（close_12 vs open_13），属系统时点差而非实现偏差。</div>
<table><thead><tr><th>指标</th><th>回测理论(08-13)</th><th>模拟盘(08-13)</th><th>偏差</th></tr></thead><tbody>
<tr><td>归一化净值</td><td>{nav_bt_13_norm:.6f}</td><td>{nav_sim_13:.6f}</td><td>{dev_pct:.3f}%</td></tr>
<tr><td>sim_tracker 记录</td><td>—</td><td>{nav_sim_13:.6f}</td><td>独立核算 {nav_sim_calc:.6f}</td></tr>
</tbody></table>
<h2>净值对比图</h2>{img}
<h2>判定</h2>
<div class="verdict">{'✅ 偏差 < 0.5%（口径对齐，首次建仓验证通过）' if dev_pct < 0.5 else '❌ 偏差 ≥ 0.5%，需排查'}
（偏差 {dev_pct:.3f}%，其中含隔夜跳空 + 建仓时点差）</div>
<p style="font-size:12px;color:#999">生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}｜
行情覆盖 {len(have)}/{len(codes)} 只</p>
</body></html>"""
    out = config.OUTPUT_DIR / "report_sim_vs_backtest.html"
    out.write_text(html, encoding="utf-8")
    print(f"\n报告: {out}")


if __name__ == "__main__":
    main()
