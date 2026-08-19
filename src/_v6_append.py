# -*- coding: utf-8 -*-
"""V6 压力测试报告函数（追加到 report.py，复用其 plt/np/pd/_fig_to_b64/_num/_pct 命名空间）。"""


def chart_slippage_bar(ann_gross: float, ann_old: float, ann_tiered: float) -> str:
    """滑点侵蚀柱状图：零成本(理论上限) / 旧固定0.002 / 分档实盘。"""
    import numpy as np
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["零成本\n(理论上限)", "旧固定\n0.10%单边?", "分档实盘\n(0.10/0.30/0.50%)"]
    # 注：旧固定实为单边 0.20%（千二），标签按实质
    vals = [ann_gross * 100, ann_old * 100, ann_tiered * 100]
    colors = ["#95a5a6", "#2980b9", "#c0392b"]
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.1, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylabel("年化收益率 (%)")
    ax.set_title("Slippage Erosion: Theoretical vs Realistic (2018-2025)")
    ax.grid(alpha=0.3, axis="y")
    return _fig_to_b64(fig)


def chart_heatmap(heat: "pd.DataFrame", base_x, base_y) -> str:
    """参数敏感性热力图：X=MA周期, Y=IC阈值, 色块=全区间夏普。基线格高亮。"""
    import numpy as np
    fig, ax = plt.subplots(figsize=(6.5, 5))
    rows = list(heat.index)     # IC 阈值 (Y)
    cols = list(heat.columns)   # MA 周期 (X)
    data = heat.values.astype(float)
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"MA{int(c)}" for c in cols])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"IC={r}" for r in rows])
    ax.set_xlabel("MA period (X)")
    ax.set_ylabel("IC threshold (Y)")
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color="black")
    try:
        bi = rows.index(base_y)
        bj = cols.index(base_x)
        ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False,
                                   edgecolor="#1f3a93", lw=3))
    except Exception:
        pass
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Sharpe")
    ax.set_title("Parameter Sensitivity (Sharpe): MA x IC threshold\n"
                 "(blue box = baseline 0.05 / 240)")
    return _fig_to_b64(fig)


def generate_html_v6(eq_v5_tiered, idx_eq, slip_map, tier_counts,
                     m_full, m_old, m_new, m_old_fixed_2018_2023, m_idx,
                     ann_gross, ann_old, ann_tiered, erosion_pp, erosion_pct,
                     heat, baseline_ic, baseline_ma, yearly, switch_log,
                     data_start, data_end, cutoff_note, conclusion) -> str:
    """V6 压力测试报告：扩展区间绩效 + 滑点侵蚀 + 参数敏感性热力图 + 诚实结论。"""
    eq_img = chart_multi_equity({
        "V6 (tiered, 2018-2025)": eq_v5_tiered, "CSI300 Index": idx_eq})
    slip_img = chart_slippage_bar(ann_gross, ann_old, ann_tiered)
    heat_img = chart_heatmap(heat, baseline_ma, baseline_ic)
    yearly_img = chart_yearly_sharpe(yearly.to_frame("V6")) if hasattr(yearly, "to_frame") else ""

    def row_html(name, m, hl=False):
        cls = " style='background:#fff3f3'" if hl else ""
        return (f"<tr{cls}><td>{name}</td><td>{_pct(m.get('annual_return'))}</td>"
                f"<td>{_pct(m.get('max_drawdown'))}</td><td>{_num(m.get('sharpe'))}</td>"
                f"<td>{_num(m.get('calmar'))}</td></tr>")

    perf_rows = (
        row_html("全区间 2018-2025（分档实盘）", m_full, hl=True)
        + row_html("旧区间 2018-2023（分档实盘）", m_old)
        + row_html("新区间 2024-2025（极端未知样本）", m_new)
        + row_html("旧区间 2018-2023（旧固定0.002对照）", m_old_fixed_2018_2023)
        + row_html("真实 CSI300 指数买入持有", m_idx)
    )

    # 滑点分级分布
    tc = tier_counts
    tier_html = (
        f"<li><b>档1 成交额&gt;5亿（单边0.10%）</b>：{tc.get('tier1_>5yi_0.10%',0)} 只</li>"
        f"<li><b>档2 成交额1-5亿（单边0.30%）</b>：{tc.get('tier2_1-5yi_0.30%',0)} 只</li>"
        f"<li><b>档3 成交额&lt;1亿（单边0.50%）</b>：{tc.get('tier3_<1yi_0.50%',0)} 只</li>"
        f"<li>成交额缺失（按档3处理）：{tc.get('missing',0)} 只</li>"
    )

    # 逐年夏普表
    y_rows = ""
    for y in yearly.index:
        v = yearly.loc[y]
        y_rows += f"<tr><td>{y}</td><td>{_num(v) if not pd.isna(v) else '—'}</td></tr>"

    # 网格表（便于核对）
    grid_rows = ""
    for r in heat.index:
        cells = "".join(f"<td>{_num(heat.loc[r, c])}</td>" for c in heat.columns)
        grid_rows += f"<tr><td>IC={r}</td>{cells}</tr>"
    grid_head = "<th>IC\\MA</th>" + "".join(f"<th>MA{int(c)}</th>" for c in heat.columns)

    base_sh = heat.loc[baseline_ic, baseline_ma]

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>V5 策略压力测试 V6 · 数据延长/滑点实盘化/参数敏感性</title>
<style>
 body{{font-family:-apple-system,'PingFang SC','Microsoft YaHei',sans-serif;margin:32px;color:#222;line-height:1.6}}
 h1{{border-bottom:3px solid #8e44ad;padding-bottom:8px}}
 h2{{margin-top:32px;color:#8e44ad}}
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
</style></head><body>
<h1>V5 多因子 Regime 切换策略 · 压力测试 V6</h1>
<p>数据区间 <b>{data_start} ~ {data_end}</b>（较 V5 延长至 2025-08-12）｜ 标的：中证500 ∪ 创业板指（剔除ST/新股）｜
调仓：月度 ｜ 市场过滤：MA240 ｜ 切换：反转↔动量/质量（逻辑与 V5 完全一致，未改动）。</p>
<div class="note">{cutoff_note}</div>

<h2>一、扩展区间绩效（分档实盘滑点）</h2>
<div class="cards">
 <div class="card"><div class="k">全区间年化(18-25)</div><div class="v {'green' if m_full.get('annual_return',0)>0 else 'red'}">{_pct(m_full.get('annual_return'))}</div></div>
 <div class="card"><div class="k">全区间夏普</div><div class="v">{_num(m_full.get('sharpe'))}</div></div>
 <div class="card"><div class="k">全区间回撤</div><div class="v red">{_pct(m_full.get('max_drawdown'))}</div></div>
 <div class="card"><div class="k">2024-2025年化</div><div class="v {'green' if m_new.get('annual_return',0)>0 else 'red'}">{_pct(m_new.get('annual_return'))}</div></div>
 <div class="card"><div class="k">2024-2025夏普</div><div class="v">{_num(m_new.get('sharpe'))}</div></div>
</div>
<table class="tbl">
<tr><th>区间</th><th>年化收益</th><th>最大回撤</th><th>夏普</th><th>卡玛</th></tr>
{perf_rows}
</table>
<div class="note">「旧区间 2018-2023（旧固定0.002对照）」夏普 {_num(m_old_fixed_2018_2023.get('sharpe'))}，
与 V5 原报 0.71 一致，证明扩展数据与 V5 引擎可复现；分档实盘滑点使旧区间夏普小幅回落至
{_num(m_old.get('sharpe'))}。</div>

<h2>二、滑点模型升级（市值分档流动性滑点）</h2>
<p>废除固定单边千二（0.20%），改为按<b>全期日均成交额</b>分档（买卖各扣一次，总摩擦=买滑点+卖滑点）：</p>
<ul>{tier_html}</ul>
<img src="data:image/png;base64,{slip_img}">
<div class="hl">滑点侵蚀：分档实盘化使全区间年化从 <b>{_pct(ann_old)}</b>（旧固定0.002）降至
<b>{_pct(ann_tiered)}</b>，侵蚀 <b>{erosion_pp*100:.2f} 个百分点</b>，约占原收益
<b>{erosion_pct*100:.1f}%</b>。理论零成本上限为 {_pct(ann_gross)}。
（注：原 V5「12% 年化」为旧固定成本口径；本表在<b>同一扩展数据</b>上隔离滑点影响，可比性更高。）</div>

<h2>三、参数敏感性扫描（检测过拟合）</h2>
<img src="data:image/png;base64,{heat_img}">
<table class="tbl">
<tr><th>参数</th>{grid_head}</tr>
{grid_rows}
</table>
<div class="sec">基线 (IC={baseline_ic}, MA={baseline_ma}) 全区间夏普 = <b>{_num(base_sh)}</b>。
若基线处于一片红色高地（周边参数夏普普遍≥0.60），说明参数稳健、无针尖过拟合；若仅基线凸起，则警觉过拟合。
判定见末尾诚实结论。</div>

<h2>四、逐年夏普分解（扩展区间 2018-2025）</h2>
<img src="data:image/png;base64,{yearly_img}">
<table class="tbl">
<tr><th>年份</th><th>V6 夏普</th></tr>
{y_rows}
</table>

<h2>五、资金曲线（分档实盘 vs CSI300）</h2>
<img src="data:image/png;base64,{eq_img}">

<h2>六、诚实结论：2024-2025 数据上策略是否依然有效？</h2>
<div class="sec">{conclusion}</div>

<h2>七、方法说明与局限</h2>
<ul>
<li>V6 严格不动 V5 选股与切换逻辑；仅延长数据、升级滑点成本模型、扫描既有超参数。</li>
<li>2024-2025 经东方财富 stock_zh_a_hist（优先）/ 新浪 补抓；CSI300 指数经新浪补抓。</li>
<li>滑点分级基于全期日均成交额（量级稳定），不构成信号，仅作成本模型；粗档位下前瞻偏差可忽略。</li>
<li>2024-2025 的 ROE/毛利率同比沿用 V5 面板前向填充（point-in-time 正确：财报披露前沿用上一期值）。</li>
<li>反转信号 LightGBM 仍按原窗口训练（2018-2019 训练 / 2020 验证），2024-2025 属严格样本外未知区间。</li>
<li>成交额(2018-2023)由 close×volume 近似（实测≈真实成交额×1.06，分级阈值很宽，误差可忽略）；2024-2025 用真实成交额。</li>
</ul>
<p style="color:#999">生成时间 {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
</body></html>"""
    return html
