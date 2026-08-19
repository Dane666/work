# -*- coding: utf-8 -*-
"""
V9 报告生成（V9.1 / V9.2 / V9.3 共用）。

相对 V8 报告的关键新增：
  - 「边际贡献」表：当前版本 vs 上一版本（prev_ref）的净变化（夏普/年化/回撤），
    用于逐步叠加模块（分析师因子 → 行业拥挤度 → 周频）时归因每个模块的独立贡献。
  - 逐年夏普图中叠加 当前 / 上一版 / V8基线 三条线，便于观察每个模块对 2021/2024 等
    关键年份的边际影响。

复用 report.py 的图表与格式化助手（chart_multi_equity / chart_drawdown_compare /
chart_yearly_sharpe / chart_vol_regime / _pct / _num）。
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from report import (chart_multi_equity, chart_drawdown_compare,
                    chart_yearly_sharpe, chart_vol_regime, _pct, _num)


def generate_html_v9(eq, eq_clean, idx_eq, slip_map, tier_counts,
                     m_full, m_old, m_new, m_full_c, m_old_c, m_new_c,
                     prev_ref, baseline_ref, m_idx,
                     yearly, yearly_prev, yearly_baseline, yearly_c,
                     vol_regime, switch_log, latest_signal, holdings,
                     n_universe, data_start, data_end, conclusion,
                     version_label, prev_label, modules_text) -> str:
    """生成 V9.x 报告 HTML。

    prev_ref / baseline_ref : dict{'full','old','new','label'}，分别对应
        「上一版本」（边际贡献对照）与「V8 基线」（上下文对照）。
    yearly_prev / yearly_baseline : 上一版 / V8 的逐年夏普 Series。
    """
    eq_img = chart_multi_equity({
        version_label: eq,
        f"{version_label}基准(MA240-only)": eq_clean,
        "CSI300 Index": idx_eq})
    dd_img = chart_drawdown_compare({
        version_label: eq,
        f"{version_label}基准(MA240-only)": eq_clean,
        "CSI300 Index": idx_eq})
    vol_img = chart_vol_regime(vol_regime) if vol_regime is not None else ""
    yearly_df = pd.DataFrame({
        version_label: yearly, prev_label: yearly_prev,
        "V8基线": yearly_baseline})
    yearly_img = chart_yearly_sharpe(yearly_df)

    def row_html(name, m, hl=False):
        cls = " style='background:#eafaf1'" if hl else ""
        return (f"<tr{cls}><td>{name}</td><td>{_pct(m.get('annual_return'))}</td>"
                f"<td>{_pct(m.get('max_drawdown'))}</td><td>{_num(m.get('sharpe'))}</td>"
                f"<td>{_num(m.get('calmar'))}</td></tr>")

    perf_rows = (
        row_html(f"{version_label}（2018-2025，MA240+波动率）", m_full, hl=True)
        + row_html("　└ 旧区间 2018-2023", m_old)
        + row_html("　└ 新区间 2024-2025", m_new)
        + row_html(f"{version_label}基准（MA240-only，无波动过滤）", m_full_c)
        + row_html("　└ 旧区间 2018-2023", m_old_c)
        + row_html("　└ 新区间 2024-2025", m_new_c)
        + row_html(f"{prev_label}（上一版本，MA240+波动率）", prev_ref["full"])
        + row_html("　└ 旧区间 2018-2023", prev_ref["old"])
        + row_html("　└ 新区间 2024-2025", prev_ref["new"])
        + row_html("V8基线（500+创业，MA240+波动率）", baseline_ref["full"])
        + row_html("　└ 旧区间 2018-2023", baseline_ref["old"])
        + row_html("　└ 新区间 2024-2025", baseline_ref["new"])
        + row_html("真实 CSI300 指数买入持有", m_idx)
    )

    tc = tier_counts
    tier_html = (
        f"<li><b>档1 成交额&gt;5亿（单边{_pct(slip_map and 0.001 or 0.0)}）</b>：{tc.get('tier1_>5yi_0.10%',0)} 只</li>"
        f"<li><b>档2 成交额1-5亿（单边0.30%）</b>：{tc.get('tier2_1-5yi_0.30%',0)} 只</li>"
        f"<li><b>档3 成交额&lt;1亿（单边0.50%）</b>：{tc.get('tier3_<1yi_0.50%',0)} 只</li>"
        f"<li>成交额缺失（按档3处理）：{tc.get('missing',0)} 只</li>"
    )

    # ---- 边际贡献表（当前 vs 上一版）----
    def _mc(cur, prev, key, fmt=_num):
        d = cur.get(key, np.nan) - prev.get(key, np.nan)
        color = "green" if d > 0 else ("red" if d < 0 else "")
        return (f"<tr><td>{fmt(cur.get(key))}</td><td>{fmt(prev.get(key))}</td>"
                f"<td class='{color}'>{('+' if d>0 else '') + fmt(d)}</td></tr>")
    mc_rows = (
        _mc(m_full, prev_ref["full"], "sharpe")
        + _mc(m_full, prev_ref["full"], "annual_return", _pct)
        + _mc(m_full, prev_ref["full"], "max_drawdown", _pct)
        + _mc(m_old, prev_ref["old"], "sharpe")
        + _mc(m_new, prev_ref["new"], "sharpe")
        + _mc(m_new, prev_ref["new"], "max_drawdown", _pct)
    )

    # ---- 逐年夏普表 ----
    y_rows = ""
    for y in yearly_df.index:
        cur = yearly_df.loc[y, version_label]
        pv = yearly_df.loc[y, prev_label]
        bs = yearly_df.loc[y, "V8基线"]
        best = "▲" if (not pd.isna(cur) and not pd.isna(pv) and cur > pv) else (
            "▼" if (not pd.isna(cur) and not pd.isna(pv) and cur < pv) else "=")
        y_rows += (f"<tr><td>{y}</td>"
                   f"<td style='font-weight:600'>{_num(cur) if not pd.isna(cur) else '—'}</td>"
                   f"<td>{_num(pv) if not pd.isna(pv) else '—'}</td>"
                   f"<td>{_num(bs) if not pd.isna(bs) else '—'}</td>"
                   f"<td>{best}</td></tr>")

    # ---- 2024-2025 专项 ----
    new_ok = m_new.get("sharpe", 0) >= 0.40
    new_html = (
        f"<li>{version_label} 新区间2024-2025 夏普 <b>{_num(m_new.get('sharpe'))}</b> ｜ "
        f"年化 {_pct(m_new.get('annual_return'))} ｜ 回撤 {_pct(m_new.get('max_drawdown'))}</li>"
        f"<li>{prev_label} 新区间2024-2025 夏普 <b>{_num(prev_ref['new'].get('sharpe'))}</b></li>"
        f"<li>目标 0.26→0.4+ → <b>{'达成' if new_ok else '未达成'}</b>"
        f"（净变化 {_num(m_new.get('sharpe')-prev_ref['new'].get('sharpe'))}）</li>"
    )

    # ---- 最新截面信号 ----
    ls_rows = ""
    if latest_signal is not None and len(latest_signal):
        for _, r in latest_signal.iterrows():
            ls_rows += (f"<tr><td>{r['rank']}</td><td>{r['code']}</td><td>{r['name']}</td>"
                        f"<td>{r['factor_set']}</td><td>{_num(r['regime_weight'])}%</td>"
                        f"<td>{_num(r['target_weight'])}%</td><td>{r['action']}</td></tr>")

    # ---- 2024-2025 持仓 ----
    h_rows = ""
    if holdings is not None and len(holdings):
        for _, r in holdings.iterrows():
            codes = str(r["selected_codes"])
            preview = codes[:60] + ("…" if len(codes) > 60 else "")
            h_rows += (f"<tr><td>{r['month_end'].date() if hasattr(r['month_end'],'date') else r['month_end']}</td>"
                       f"<td>{r['factor_set']}</td><td>{r['n_selected']}</td>"
                       f"<td style='font-size:11px;color:#666'>{preview}</td></tr>")

    goal_sharpe_ok = m_new.get("sharpe", 0) >= 0.40
    goal_full_ok = m_full.get("sharpe", 0) >= baseline_ref["full"].get("sharpe", 0)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>{version_label} 策略 · V9 迭代</title>
<style>
 body{{font-family:-apple-system,'PingFang SC','Microsoft YaHei',sans-serif;margin:32px;color:#222;line-height:1.6}}
 h1{{border-bottom:3px solid #2c7a3d;padding-bottom:8px}}
 h2{{margin-top:32px;color:#2c7a3d}}
 .cards{{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0}}
 .card{{flex:1;min-width:130px;background:#f7f7f7;border:1px solid #ddd;border-radius:8px;padding:12px}}
 .card .k{{font-size:12px;color:#888}}
 .card .v{{font-size:22px;font-weight:700}}
 .green{{color:#27ae60}} .red{{color:#c0392b}}
 table.tbl{{border-collapse:collapse;width:100%;margin:12px 0;font-size:13px}}
 table.tbl th,table.tbl td{{border:1px solid #ddd;padding:6px 8px;text-align:center}}
 table.tbl th{{background:#f0f0f0}}
 table.tbl td.green{{color:#27ae60;font-weight:600}} table.tbl td.red{{color:#c0392b;font-weight:600}}
 img{{max-width:100%;margin:10px 0;border:1px solid #eee;border-radius:6px}}
 .note{{background:#fff8e1;border-left:4px solid #f1c40f;padding:10px 14px;margin:12px 0}}
 .sec{{background:#eaf4fb;border-left:4px solid #2980b9;padding:10px 14px;margin:12px 0}}
 .hl{{background:#fdecea;border-left:4px solid #c0392b;padding:10px 14px;margin:12px 0}}
 .ok{{background:#eafaf1;border-left:4px solid #27ae60;padding:10px 14px;margin:12px 0}}
 .mod{{background:#f3eaff;border-left:4px solid #8e44ad;padding:10px 14px;margin:12px 0}}
</style></head><body>
<h1>V5 多因子 Regime 切换策略 · 迭代 {version_label}</h1>
<p>数据区间 <b>{data_start} ~ {data_end}</b>｜ 标的：<b>中证500 ∪ 创业板指 ∪ 中证1000（{n_universe} 只，剔除ST）</b>｜
市场过滤：<b>MA240 + 波动率降档</b> ｜ 切换框架：反转↔动量/质量（与 V5/V7/V7.1/V8 完全一致，未改动）。</p>
<div class="mod"><b>本轮叠加模块（{version_label} vs {prev_label}）：</b><br>{modules_text}</div>

<h2>一、2024-2025 新区间专项（核心验证）</h2>
<div class="cards">
 <div class="card"><div class="k">{version_label} 新区间夏普</div><div class="v {'green' if m_new.get('sharpe',0)>0 else 'red'}">{_num(m_new.get('sharpe'))}</div></div>
 <div class="card"><div class="k">{prev_label} 新区间夏普</div><div class="v">{_num(prev_ref['new'].get('sharpe'))}</div></div>
 <div class="card"><div class="k">目标</div><div class="v">{'0.4+' if new_ok else '0.26'}</div></div>
 <div class="card"><div class="k">净变化</div><div class="v {'green' if m_new.get('sharpe',0)>=prev_ref['new'].get('sharpe',0) else 'red'}">{_num(m_new.get('sharpe')-prev_ref['new'].get('sharpe'))}</div></div>
</div>
<ul>{new_html}</ul>

<h2>二、模块边际贡献（{version_label} vs {prev_label}）</h2>
<table class="tbl">
<tr><th>指标</th><th>{version_label}</th><th>{prev_label}</th><th>净变化</th></tr>
{mc_rows}
</table>
<div class="note">净变化列绿色=本版更优，红色=本版更差。逐项隔离：本表仅反映「所叠加模块」的边际贡献（其余逻辑与上一版完全一致）。</div>

<h2>三、全区间绩效对比（{version_label} vs {version_label}基准 vs {prev_label} vs V8基线）</h2>
<table class="tbl">
<tr><th>策略 / 区间</th><th>年化收益</th><th>最大回撤</th><th>夏普</th><th>卡玛</th></tr>
{perf_rows}
</table>
<div class="note">公平对照：{version_label} 与 {version_label}基准 使用完全相同的选股信号与分档滑点，唯一差异是市场过滤（是否叠加波动率降档），
差异即波动率过滤净贡献。{prev_label}/V8基线 行由对应净值现场重算（同口径），保证仅差所叠加模块一项。</div>

<h2>四、滑点模型（市值分档，沿用 V6）</h2>
<ul>{tier_html}</ul>

<h2>五、波动率过滤诊断</h2>
<img src="data:image/png;base64,{vol_img}">

<h2>六、逐年夏普分解（{version_label} vs {prev_label} vs V8基线，重点 2021 / 2024）</h2>
<img src="data:image/png;base64,{yearly_img}">
<table class="tbl">
<tr><th>年份</th><th>{version_label}</th><th>{prev_label}</th><th>V8基线</th><th>Δ(本版更优?)</th></tr>
{y_rows}
</table>

<h2>七、最新截面信号清单（对应最近数据日）</h2>
<table class="tbl">
<tr><th>排名</th><th>代码</th><th>名称</th><th>因子集</th><th>regime权重</th><th>目标仓位%</th><th>动作</th></tr>
{ls_rows}
</table>

<h2>八、2024-2025 持仓明细（校验因子集切换）</h2>
<table class="tbl">
<tr><th>月末</th><th>因子集</th><th>持仓数</th><th>选中代码（预览）</th></tr>
{h_rows}
</table>

<h2>九、资金曲线与回撤对比</h2>
<img src="data:image/png;base64,{eq_img}">
<img src="data:image/png;base64,{dd_img}">

<h2>十、诚实结论</h2>
<div class="{'ok' if (goal_sharpe_ok or goal_full_ok) else 'hl'}">{conclusion}</div>

<h2>十一、方法说明与局限</h2>
<ul>
<li>V9 严格不改 V5 的 Regime 切换框架（MA240 + IC门控 + 反转↔动量/质量 + Top30 + 月频/周频）与 V7/V7.1/V8 的修复（滚动重训 + 真实财报 point-in-time）。</li>
<li>所有新增模块均为「附加功能」：分析师因子补充质量评分、行业拥挤度对拥挤行业权重打7折、周频调仓配合滑点上浮。V8 已验证核心参数（MA240/波动率分位/降幅/IC阈值/持仓数/滚动窗口/Regime规则）<b>一律未触碰</b>。</li>
<li>分析师因子数据源：巨潮资讯投资评级（ak.stock_rank_forecast_cninfo）按周频快照、point-in-time 构建；免费 akshare 不提供可回测的历史 EPS 预测修正序列，故以「评级上调」事件等价刻画「分析师预期上调」（rating_change 为用户指定子因子），零未来函数。</li>
<li>中证1000 成分股含幸存者偏差（当前快照而非时点成分），但 V8 同为快照，对比仍干净。</li>
</ul>
<p style="color:#999">生成时间 {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
</body></html>"""
    return html
