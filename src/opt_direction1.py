# -*- coding: utf-8 -*-
"""
opt_direction1.py — 方向1：动量持续性过滤（质量因子增强）验证
====================================================================
对比 V3.1 基线（persistence=False） vs 方向1 新合成（persistence=True）：
  mz_new = (z(ret_12 - ret_3) + z(ret_24) + z(roe) + z(gpm_yoy)) / 4

仅改动 build_mz 合成；MA240 门控 / 股息率门控 / 质量过滤阈值 / 长动量窗口(12/24) 均不变。
运行：cd src && python opt_direction1.py
输出：output/report_opt_direction1.html（前后对比 + 逐年夏普 + 净值曲线）
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd

for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
          "ALL_PROXY", "all_proxy"]:
    os.environ.pop(k, None)

import config
from factors import compute_rsi
from factor_eval import build_selection_v5
from market_filter import build_ma240_vol_target_weight
from stress_test_v6 import build_slippage_map
from report import compute_metrics, yearly_sharpe, chart_multi_equity
import main_mainboard_v3 as mm   # 复用加载 / 质量过滤 / 门控 / run_full

# 方向1 达标线（用户给定）：全期≥0.66 且 2024-25≥0.85
T1_FULL = 0.66
T1_NEW = 0.85
# 总目标：全期≥0.65 / 2024-25≥0.85 / 回撤≤-20%
GOAL_FULL = 0.65
GOAL_NEW = 0.85
GOAL_DD = -0.20


def load_env():
    codes = mm.get_v2_codes()
    close = pd.read_parquet(config.MB_CLOSE).reindex(columns=codes)
    amount = pd.read_parquet(config.MB_AMOUNT).reindex(columns=codes)
    idx = pd.read_parquet(config.DATA_DIR / "v6_index.parquet")
    if isinstance(idx, pd.DataFrame):
        idx = idx.iloc[:, 0]
    roe = pd.read_parquet(config.MB_ROE).reindex(index=close.index, columns=codes).ffill()
    gpm = pd.read_parquet(config.MB_GPM).reindex(index=close.index, columns=codes).ffill()
    dy = pd.read_parquet(config.DATA_DIR / "div_yield_panel_mainboard.parquet").reindex(columns=codes)
    slip_map, _ = build_slippage_map(amount)
    mm.slip_map_ = slip_map

    close_m = mm.mask_new_listings(close, config.NEW_STOCK_MIN_DAYS)
    rsi = compute_rsi(close_m, config.RSI_WINDOW)
    me = mm.month_ends_in(close_m, config.START_DATE, str(close.index[-1].date()))
    tw_base, _ = build_ma240_vol_target_weight(
        idx, close_m.index, mm.MA_BASE, month_ends=me,
        vol_q=0.75, reduced_weight=0.60, vol_lookback=756)
    return dict(close_m=close_m, amount=amount, roe=roe, gpm=gpm, dy=dy,
                rsi=rsi, me=me, tw_base=tw_base)


def run_once(env, persistence: bool):
    close_m, amount, roe, gpm, dy = (env["close_m"], env["amount"], env["roe"],
                                     env["gpm"], env["dy"])
    me, tw_base = env["me"], env["tw_base"]
    cm = mm.apply_quality_mask(close_m.copy(), roe, amount, me)
    mz = mm.build_mz(cm, roe, gpm, me, long_momentum=True, persistence=persistence)
    use_rev = pd.Series(False, index=me)
    empty_rev = pd.DataFrame(index=cm.index, columns=cm.columns)
    sel_v8, _ = build_selection_v5(cm, env["rsi"], empty_rev, mz, me, use_rev, 0.20, mm.TOP_N)
    gate = mm.build_dy_gate(dy, me)
    gate_daily = gate.reindex(close_m.index).ffill().fillna(1.0)
    tw = tw_base * gate_daily
    eq = mm.run_full(cm, amount, sel_v8, me, tw)
    return eq


def main():
    t0 = datetime.now()
    env = load_env()
    print(f"[{datetime.now()}] 数据就绪，开始对比回测（基线 vs 方向1）")

    eq_base = run_once(env, persistence=False)
    mf_b = compute_metrics(eq_base)
    mn_b = compute_metrics(eq_base.loc["2024-01-01":])
    print(f"[{datetime.now()}] 基线完成: 全期{mf_b['sharpe']:.2f}/回撤{mf_b['max_drawdown']*100:.1f}% "
          f"| 2024-25 {mn_b['sharpe']:.2f}")

    eq_new = run_once(env, persistence=True)
    mf_n = compute_metrics(eq_new)
    mn_n = compute_metrics(eq_new.loc["2024-01-01":])
    print(f"[{datetime.now()}] 方向1完成: 全期{mf_n['sharpe']:.2f}/回撤{mf_n['max_drawdown']*100:.1f}% "
          f"| 2024-25 {mn_n['sharpe']:.2f}")

    # 逐年夏普对比（归因用）
    ys_b = yearly_sharpe(eq_base)
    ys_n = yearly_sharpe(eq_new)
    years = sorted(set(ys_b.index) | set(ys_n.index))

    # 判定
    t1_ok = (mf_n["sharpe"] >= T1_FULL and mn_n["sharpe"] >= T1_NEW)
    goal_ok = (mf_n["sharpe"] >= GOAL_FULL and mn_n["sharpe"] >= GOAL_NEW
               and mf_n["max_drawdown"] >= GOAL_DD)

    # 图表（base64 内嵌）
    eq_img = chart_multi_equity({"V3.1 基线": eq_base, "方向1 动量持续性": eq_new},
                                "方向1 前后净值曲线对比（归一化）")

    # 逐年表
    def sval(s, y):
        return s.get(y) if y in s.index else None

    yr_rows = ""
    for y in years:
        b, n = sval(ys_b, y), sval(ys_n, y)
        fb = "—" if b is None or b != b else f"{b:.2f}"
        fn = "—" if n is None or n != n else f"{n:.2f}"
        fd = "—" if (b is None or b != b or n is None or n != n) else f"{n - b:+.2f}"
        yr_rows += f"<tr><td>{y}</td><td>{fb}</td><td>{fn}</td><td>{fd}</td></tr>"

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>方向1：动量持续性过滤 — 回测对比报告</title>
<style>body{{font-family:-apple-system,'PingFang SC',sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222}}
table{{border-collapse:collapse;width:100%;margin:14px 0;font-size:13px}}
th,td{{border:1px solid #ddd;padding:7px 9px;text-align:center}}
th{{background:#f5f5f5}} h2{{border-bottom:2px solid #eee;padding-bottom:6px;margin-top:32px}}
.note{{background:#f2f6fc;border-left:4px solid #4a7abb;padding:10px 14px;font-size:13px;color:#444;margin:12px 0}}
.verdict{{background:#fff7e6;border:1px solid #f0c36d;border-radius:6px;padding:12px 16px;margin:16px 0}}
img{{max-width:100%;margin:8px 0}}</style></head><body>
<h1>方向1：动量持续性过滤（质量因子增强）</h1>
<p>选股池 V8∩主板 1004 只｜月频｜分档滑点｜MA240+波动率+股息率门控（σ=0.20/仓位20%）｜强制质量模式 均不变。</p>
<div class="note"><b>改动：</b>仅替换 <code>build_mz</code> 合成公式——
<b>基线</b>：<code>mz = (z(ret_12+ret_24)/2 + z(roe) + z(gpm_yoy))/3</code><br>
<b>方向1</b>：<code>mz = (z(ret_12−ret_3) + z(ret_24) + z(roe) + z(gpm_yoy))/4</code>
（动量持续性 ret_12−ret_3：长期动量扣除近3月动量，避免追高已衰减的标的；ret_3 = 63 交易日收益）</div>
<h2>前后对比（方向1 达标线：全期≥0.66 且 2024-25≥0.85）</h2>
<table><thead><tr><th>版本</th><th>全期夏普</th><th>全期回撤</th><th>2024-25夏普</th><th>2024-25回撤</th></tr></thead>
<tbody>
<tr><td>V3.1 基线 (persistence=False)</td><td>{mf_b['sharpe']:.2f}</td><td>{mf_b['max_drawdown']*100:.1f}%</td><td>{mn_b['sharpe']:.2f}</td><td>{mn_b['max_drawdown']*100:.1f}%</td></tr>
<tr style="background:#eef7ee"><td><b>方向1 动量持续性</b></td><td><b>{mf_n['sharpe']:.2f}</b></td><td>{mf_n['max_drawdown']*100:.1f}%</td><td><b>{mn_n['sharpe']:.2f}</b></td><td>{mn_n['max_drawdown']*100:.1f}%</td></tr>
<tr><td>边际贡献</td><td>{mf_n['sharpe']-mf_b['sharpe']:+.2f}</td><td>{(mf_n['max_drawdown']-mf_b['max_drawdown'])*100:+.1f}pp</td><td>{mn_n['sharpe']-mn_b['sharpe']:+.2f}</td><td>{(mn_n['max_drawdown']-mn_b['max_drawdown'])*100:+.1f}pp</td></tr>
</tbody></table>
<div class="verdict"><b>方向1 判定：{'✅ 达标' if t1_ok else '❌ 未达标'}</b>
（全期 {mf_n['sharpe']:.2f} {'≥' if mf_n['sharpe']>=T1_FULL else '<'} {T1_FULL}；2024-25 {mn_n['sharpe']:.2f} {'≥' if mn_n['sharpe']>=T1_NEW else '<'} {T1_NEW}）<br>
<b>总目标判定：{'✅ 达标' if goal_ok else '❌ 未达标'}</b>
（全期≥0.65 / 2024-25≥0.85 / 回撤≤-20%）</div>
<h2>逐年夏普分解（归因）</h2>
<table><thead><tr><th>年份</th><th>基线</th><th>方向1</th><th>Δ</th></tr></thead><tbody>{yr_rows}</tbody></table>
<h2>净值曲线</h2>
<img src="data:image/png;base64,{eq_img}">
<p style="font-size:12px;color:#999">生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}，耗时 {datetime.now()-t0}。</p>
</body></html>"""
    out = config.OUTPUT_DIR / "report_opt_direction1.html"
    out.write_text(html, encoding="utf-8")
    print(f"\n[{datetime.now()}] 报告: {out}  耗时 {datetime.now()-t0}")


if __name__ == "__main__":
    main()
