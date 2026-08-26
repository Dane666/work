# -*- coding: utf-8 -*-
"""
opt_direction2.py — 方向2：行业中性化持仓（在方向1 基础上叠加）
====================================================================
四组对比：基线 / 仅方向1（动量持续性）/ 仅方向2（行业中性化）/ 方向1+方向2。
目标：全期夏普 ≥0.66、回撤 ≤-20%、2024-25 夏普 ≥0.85。

行业中性化（config.ENABLE_SECTOR_NEUTRAL）：
  月末对目标持仓做行业上限约束 —— 行业持仓权重 ≤ max(基准市值权重×2, 10%)，
  超额等比例分配至未超限行业持仓（≤10 轮迭代），权重归零移除。
  基准：data/industry_benchmark.parquet（主板池各行业市值权重，市值=注册资金×1e4×收盘价）。

运行：cd src && python opt_direction2.py
输出：output/report_opt_direction2.html + output/industry_concentration_series.csv
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
from report import (compute_metrics, yearly_sharpe, chart_multi_equity,
                    chart_drawdown_compare)
from v3_common import load_sector_data
import main_mainboard_v3 as mm

# 目标线
GOAL_FULL = 0.66
GOAL_NEW = 0.85
GOAL_DD = -0.20

RUNS = [
    ("基线（V3.1）",            False, False),
    ("仅方向1（动量持续性）",    True,  False),
    ("仅方向2（行业中性化）",    False, True),
    ("方向1+方向2（目标）",      True,  True),
]


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


def run_once(env, persistence: bool, enable_sector: bool, sector_map, bench, conc_log):
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
    eq = mm.run_full(cm, amount, sel_v8, me, tw, enable_sector=enable_sector,
                     sector_map=sector_map, bench=bench, conc_log=conc_log)
    return eq


def main():
    t0 = datetime.now()
    env = load_env()
    sector_map, bench = load_sector_data()
    if sector_map is None or bench is None:
        print("!! 行业数据缺失：请先运行 python fetch_mainboard.py --industry-map")
        return

    eqs, ms, conc = {}, {}, []
    for label, pers, sect in RUNS:
        cl = []
        eq = run_once(env, pers, sect, sector_map, bench, cl)
        mf = compute_metrics(eq)
        mn = compute_metrics(eq.loc["2024-01-01":])
        eqs[label] = eq
        ms[label] = (mf, mn)
        for r in cl:
            r["config"] = label
            r["persistence"] = pers
            r["sector"] = sect
        conc.extend(cl)
        print(f"[{datetime.now()}] {label}: 全期{mf['sharpe']:.2f}/回撤{mf['max_drawdown']*100:.1f}% "
              f"| 2024-25 {mn['sharpe']:.2f}")

    # 集中度序列落盘
    conc_df = pd.DataFrame(conc)
    if not conc_df.empty:
        conc_df = conc_df.sort_values(["config", "date"])
        conc_df.to_csv(config.OUTPUT_DIR / "industry_concentration_series.csv",
                       index=False, encoding="utf-8-sig")

    # 判定（方向1+方向2）
    mf_t, mn_t = ms["方向1+方向2（目标）"]
    t_ok = (mf_t["sharpe"] >= GOAL_FULL and mn_t["sharpe"] >= GOAL_NEW
            and mf_t["max_drawdown"] >= GOAL_DD)

    # 逐年夏普
    ys = {label: yearly_sharpe(eq) for label, eq in eqs.items()}
    years = sorted(set().union(*[set(v.index) for v in ys.values()]))

    # 图表
    eq_img = chart_multi_equity({k: v for k, v in eqs.items()}, "四组净值曲线对比（归一化）")
    dd_img = chart_drawdown_compare({k: v for k, v in eqs.items()}, "四组回撤对比")
    conc_img = _conc_chart(conc_df, eqs)

    rows_html = ""
    base_m = ms["基线（V3.1）"][0]
    for label, (mf, mn) in ms.items():
        hl = "background:#eef7ee" if label == "方向1+方向2（目标）" else ""
        rows_html += (f"<tr style='{hl}'><td>{label}</td>"
                      f"<td>{mf['sharpe']:.2f}</td><td>{mf['max_drawdown']*100:.1f}%</td>"
                      f"<td>{mn['sharpe']:.2f}</td><td>{mn['max_drawdown']*100:.1f}%</td>"
                      f"<td>{mf['sharpe']-base_m['sharpe']:+.2f} / "
                      f"{(mf['max_drawdown']-base_m['max_drawdown'])*100:+.1f}pp</td></tr>")

    yr_rows = ""
    for y in years:
        cells = f"<td>{y}</td>"
        for label in [r[0] for r in RUNS]:
            v = ys[label].get(y)
            cells += f"<td>{'—' if v is None or v != v else f'{v:.2f}'}</td>"
        yr_rows += f"<tr>{cells}</tr>"

    verdict = ("✅ <b>达标</b>：全期≥0.66 / 2024-25≥0.85 / 回撤≤-20%"
               if t_ok else "❌ <b>未达标</b>")

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>方向2：行业中性化持仓 — 四组对比报告</title>
<style>body{{font-family:-apple-system,'PingFang SC',sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222}}
table{{border-collapse:collapse;width:100%;margin:14px 0;font-size:13px}}
th,td{{border:1px solid #ddd;padding:7px 9px;text-align:center}}
th{{background:#f5f5f5}} h2{{border-bottom:2px solid #eee;padding-bottom:6px;margin-top:32px}}
.note{{background:#f2f6fc;border-left:4px solid #4a7abb;padding:10px 14px;font-size:13px;color:#444;margin:12px 0}}
.verdict{{background:#fff7e6;border:1px solid #f0c36d;border-radius:6px;padding:12px 16px;margin:16px 0}}
img{{max-width:100%;margin:8px 0}}</style></head><body>
<h1>方向2：行业中性化持仓（方向1 叠加验证）</h1>
<p>选股池 V8∩主板 1004 只｜月频｜分档滑点｜MA240+波动率+股息率门控｜强制质量模式。方向1 = 动量持续性
<code>mz=(z(ret12−ret3)+z(ret24)+z(roe)+z(gpm_yoy))/4</code>；方向2 = 行业上限
<code>max(基准市值权重×2, 10%)</code>，超额等比例分配，≤10 轮迭代。</p>
<div class="note"><b>基准市值权重：</b>主板池各行业 总市值（注册资金×1e4×收盘价）占比，月末计算，
落盘 data/industry_benchmark.parquet；行业映射 data/industry_map.parquet（缺失归"其他"）。
目标：全期夏普 ≥0.66、2024-25 ≥0.85、回撤 ≤-20%。</div>
<h2>四组对比</h2>
<table><thead><tr><th>版本</th><th>全期夏普</th><th>全期回撤</th><th>2024-25夏普</th><th>2024-25回撤</th><th>vs 基线（全期夏普/回撤）</th></tr></thead>
<tbody>{rows_html}</tbody></table>
<div class="verdict">{verdict}（生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}）</div>
<h2>逐年夏普分解</h2>
<table><thead><tr><th>年份</th>{''.join(f'<th>{r[0]}</th>' for r in RUNS)}</tr></thead><tbody>{yr_rows}</tbody></table>
<h2>净值曲线（归一化）</h2><img src="data:image/png;base64,{eq_img}">
<h2>回撤曲线</h2><img src="data:image/png;base64,{dd_img}">
<h2>行业集中度（目标持仓 Top1 行业权重）</h2>
<p>实线=中性化后目标持仓；虚线=原始（未中性化）。约束生效时中性化后应显著低于原始。</p>
<img src="data:image/png;base64,{conc_img}">
<p style="font-size:12px;color:#999">集中度序列：output/industry_concentration_series.csv。
生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}，耗时 {datetime.now()-t0}。</p>
</body></html>"""
    out = config.OUTPUT_DIR / "report_opt_direction2.html"
    out.write_text(html, encoding="utf-8")
    print(f"\n[{datetime.now()}] 报告: {out}  集中度: "
          f"{config.OUTPUT_DIR / 'industry_concentration_series.csv'}  耗时 {datetime.now()-t0}")


def _conc_chart(conc_df: pd.DataFrame, eqs: dict) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io, base64
    if conc_df is None or conc_df.empty:
        return ""
    # 方向1+方向2 的 raw vs neutral（2024+ 主区间）
    fig, ax = plt.subplots(figsize=(11, 4))
    d12 = conc_df[conc_df["config"] == "方向1+方向2（目标）"]
    for mode, ls in [("raw", "--"), ("neutral", "-")]:
        sub = d12[d12["mode"] == mode].sort_values("date")
        if sub.empty:
            continue
        ax.plot(pd.to_datetime(sub["date"]), sub["top1_weight"], ls, label=f"方向1+2 {mode}",
                color="#c0392b" if mode == "neutral" else "#888888")
    # 仅方向1 的 raw 作为对照
    d1 = conc_df[(conc_df["config"] == "仅方向1（动量持续性）") & (conc_df["mode"] == "raw")]
    d1 = d1.sort_values("date")
    if not d1.empty:
        ax.plot(pd.to_datetime(d1["date"]), d1["top1_weight"], "-.", label="仅方向1 raw",
                color="#2980b9", alpha=0.8)
    ax.axhline(0.10, color="orange", lw=0.8, ls=":")
    ax.text(ax.get_xlim()[1], 0.10, " 10% min-cap", fontsize=8, color="orange", va="center")
    ax.set_ylabel("Top1 行业权重")
    ax.set_title("目标持仓行业集中度（Top1 行业权重，月度）")
    ax.legend()
    ax.grid(alpha=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


if __name__ == "__main__":
    main()
