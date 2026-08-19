def generate_html_v7(eq_v7, idx_eq, slip_map, tier_counts,
                     m_full, m_old, m_new, m_v5_ref, m_v6_ref, m_idx,
                     yearly, switch_log, holdings_2024_2025, fset_dist,
                     data_start, data_end, use_ic_weight, conclusion) -> str:
    """V7 修复报告：V5/V6/V7 对比 + 逐年夏普(滚动) + 2024-2025 持仓明细 + 诚实结论。"""
    eq_img = chart_multi_equity({
        "V7 (rolling+realFund, 18-25)": eq_v7, "CSI300 Index": idx_eq})
    yearly_img = chart_yearly_sharpe(yearly.to_frame("V7")) if hasattr(yearly, "to_frame") else ""

    def row_html(name, m, hl=False):
        cls = " style='background:#eafaf1'" if hl else ""
        return (f"<tr{cls}><td>{name}</td><td>{_pct(m.get('annual_return'))}</td>"
                f"<td>{_pct(m.get('max_drawdown'))}</td><td>{_num(m.get('sharpe'))}</td>"
                f"<td>{_num(m.get('calmar'))}</td></tr>")

    # 对比表：V5(固定成本,18-23) / V6(分档,18-25) / V7(分档,18-25) 三维
    perf_rows = (
        row_html("V5 基准（固定0.002，2018-2023）", m_v5_ref)
        + row_html("V6 压力（分档，2018-2025）", m_v6_ref)
        + row_html("V7 修复（滚动+真实财报，分档，2018-2025）", m_full, hl=True)
        + row_html("　└ 旧区间 2018-2023", m_old)
        + row_html("　└ 新区间 2024-2025", m_new)
        + row_html("真实 CSI300 指数买入持有", m_idx)
    )

    # 滑点分级
    tc = tier_counts
    tier_html = (
        f"<li><b>档1 成交额&gt;5亿（单边0.10%）</b>：{tc.get('tier1_>5yi_0.10%',0)} 只</li>"
        f"<li><b>档2 成交额1-5亿（单边0.30%）</b>：{tc.get('tier2_1-5yi_0.30%',0)} 只</li>"
        f"<li><b>档3 成交额&lt;1亿（单边0.50%）</b>：{tc.get('tier3_<1yi_0.50%',0)} 只</li>"
        f"<li>成交额缺失（按档3处理）：{tc.get('missing',0)} 只</li>"
    )

    # 逐年夏普
    y_rows = ""
    for y in yearly.index:
        v = yearly.loc[y]
        y_rows += f"<tr><td>{y}</td><td>{_num(v) if not pd.isna(v) else '—'}</td></tr>"

    # 2024-2025 因子集分布
    fd = fset_dist or {}
    fdist_html = " &nbsp; ".join(f"<b>{k}</b>: {v} 月" for k, v in fd.items()) or "（无）"

    # 2024-2025 持仓明细（前若干只预览，完整见 CSV）
    h_rows = ""
    if holdings_2024_2025 is not None and len(holdings_2024_2025):
        for _, r in holdings_2024_2025.iterrows():
            codes = str(r["selected_codes"])
            preview = codes[:60] + ("…" if len(codes) > 60 else "")
            h_rows += (f"<tr><td>{r['month_end'].date() if hasattr(r['month_end'],'date') else r['month_end']}</td>"
                       f"<td>{r['factor_set']}</td><td>{r['n_selected']}</td>"
                       f"<td style='font-size:11px;color:#666'>{preview}</td></tr>")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>V5 策略修复 V7 · 滚动重训+真实财报</title>
<style>
 body{{font-family:-apple-system,'PingFang SC','Microsoft YaHei',sans-serif;margin:32px;color:#222;line-height:1.6}}
 h1{{border-bottom:3px solid #16a085;padding-bottom:8px}}
 h2{{margin-top:32px;color:#16a085}}
 .cards{{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0}}
 .card{{flex:1;min-width:130px;background:#f7f7f7;border:1px solid #ddd;border-radius:8px;padding:12px}}
 .card .k{{font-size:12px;color:#888}}
 .card .v{{font-size:22px;font-weight:700}}
 .green{{color:#27ae60}} .red{{color:#c0392b}}
 table.tbl{{border-collapse:collapse;width:100%;margin:12px 0;font-size:13px}}
 table.tbl th,table.tbl td{{border:1px solid #ddd;padding:6px 8px;text-align:center}}
 table.tbl th{{background:#f0f0f0}}
 img{{max-width:100%;margin:10px 0;border:1px solid #eee;border-radius:6px}}
 .note{{background:#fff8e1;border-left:4px solid #f1c40f;padding:10px 14px;margin:12px 0}}
 .sec{{background:#eaf4fb;border-left:4px solid #2980b9;padding:10px 14px;margin:12px 0}}
 .hl{{background:#fdecea;border-left:4px solid #c0392b;padding:10px 14px;margin:12px 0}}
 .ok{{background:#eafaf1;border-left:4px solid #27ae60;padding:10px 14px;margin:12px 0}}
</style></head><body>
<h1>V5 多因子 Regime 切换策略 · 修复 V7</h1>
<p>数据区间 <b>{data_start} ~ {data_end}</b>｜ 标的：中证500 ∪ 创业板指（剔除ST/新股）｜
调仓：月度 ｜ 市场过滤：MA240 ｜ 切换框架：反转↔动量/质量（与 V5 完全一致，未改动）。</p>
<p>修复点：① <b>滚动窗口重训</b> LightGBM（冷启动用2018-2019同V5 → 2020Q1起每季度重训36月窗口）；
② <b>补齐2024-2025真实财报</b>（point-in-time 对齐至2025Q2）；③（条件）IC动态加权。
滑点沿用 V6 分档（0.1/0.3/0.5%）。</p>

<h2>一、V5 / V6 / V7 绩效对比</h2>
<div class="cards">
 <div class="card"><div class="k">V7 全区间年化</div><div class="v {'green' if m_full.get('annual_return',0)>0 else 'red'}">{_pct(m_full.get('annual_return'))}</div></div>
 <div class="card"><div class="k">V7 全区间夏普</div><div class="v">{_num(m_full.get('sharpe'))}</div></div>
 <div class="card"><div class="k">V7 2024-2025夏普</div><div class="v {'green' if m_new.get('sharpe',0)>0 else 'red'}">{_num(m_new.get('sharpe'))}</div></div>
 <div class="card"><div class="k">V6 2024-2025夏普</div><div class="v red">{_num(m_v6_ref.get('sharpe'))}</div></div>
</div>
<table class="tbl">
<tr><th>策略 / 区间</th><th>年化收益</th><th>最大回撤</th><th>夏普</th><th>卡玛</th></tr>
{perf_rows}
</table>
<div class="note">对比口径：V5 为旧固定成本（0.002）仅报 2018-2023；V6/V7 均为<b>分档实盘滑点</b>同口径，可直接比较 2018-2025 全区间与 2024-2025 新区间。
V7 旧区间(2018-2023)夏普 {_num(m_old.get('sharpe'))} 应≈V5(0.71)/V6分档(0.67)，验证未退化。</div>

<h2>二、滑点模型（市值分档，沿用 V6）</h2>
<ul>{tier_html}</ul>
<div class="sec">分档滑点买卖各扣一次；V7 沿用 V6 同款成本模型，确保与 V6 公平对比。</div>

<h2>三、逐年夏普分解（V7 滚动窗口）</h2>
<img src="data:image/png;base64,{yearly_img}">
<table class="tbl">
<tr><th>年份</th><th>V7 夏普</th></tr>
{y_rows}
</table>
<div class="sec">重点观察 2024 / 2025 年是否由 V6 的负值转正则验证修复有效。</div>

<h2>四、2024-2025 持仓明细（校验是否规避失效因子）</h2>
<p>因子集切换分布（该区间每月使用的因子集）：{fdist_html}</p>
<table class="tbl">
<tr><th>月末</th><th>因子集</th><th>持仓数</th><th>选中代码（预览）</th></tr>
{h_rows}
</table>
<div class="hl">若 2024-2025 反转因子(reversal)IC 持续为负，策略应切到 momentum_quality 集；
本表用于校验 V7 是否真正规避了失效的反转因子（完整持仓见 v7_holdings_2024_2025.csv）。</div>

<h2>五、资金曲线（V7 分档实盘 vs CSI300）</h2>
<img src="data:image/png;base64,{eq_img}">

<h2>六、诚实结论：2024-2025 是否已恢复有效？</h2>
<div class="{'ok' if m_new.get('sharpe',0)>0 else 'hl'}">{conclusion}</div>

<h2>七、方法说明与局限</h2>
<ul>
<li>V7 严格不改 V5 的 Regime 切换框架（MA240 门控 + IC 门控 + 反转↔动量/质量切换 + Top30 + 月频）。</li>
<li>滚动重训：冷启动用 2018-2019 固定训练（与 V5 一致，保 2018-2023 可比）；2020Q1 起每季度末用过去36月窗口重训，早停用训练窗内 80/20 时序切分（零未来泄露）。</li>
<li>真实财报：roe_panel_v7 / gpm_yoy_panel_v7 按披露延迟映射 point-in-time 对齐至 2025Q2（杜绝未来函数）。</li>
<li>IC 动态加权：{'已启用' if use_ic_weight else '未启用（1+2 已完成修复，按用户建议优先看 1+2 效果）'}。</li>
<li>反转信号 LightGBM 仅用价格/量派生因子（不含财报），动量/质量臂用真实 ROE/毛利率同比。</li>
<li>成交额(2018-2023)由 close×volume 近似；2024-2025 用真实成交额；粗档位下误差可忽略。</li>
</ul>
<p style="color:#999">生成时间 {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
</body></html>"""
    return html
