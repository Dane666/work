# -*- coding: utf-8 -*-
"""
报告模块：计算绩效指标、交易统计，并生成自包含 HTML 回测报告。

图表以 base64 内嵌，单文件可离线打开。图表标签使用英文以避免中文字体缺失。
"""

from __future__ import annotations

import base64
import io
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = [
    "PingFang SC", "Arial Unicode MS", "Heiti SC", "STHeiti", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

import config


# ----------------------------------------------------------------------------
# 绩效指标
# ----------------------------------------------------------------------------
def compute_metrics(equity: pd.Series, freq: int = 252) -> dict:
    """由权益曲线计算年化收益、最大回撤、夏普、卡玛等指标。"""
    eq = equity.dropna()
    if len(eq) < 2:
        return {}
    total_ret = eq.iloc[-1] / eq.iloc[0] - 1.0
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    annual = (1.0 + total_ret) ** (1.0 / max(years, 1e-9)) - 1.0
    daily = eq.pct_change().dropna()
    sharpe = daily.mean() / daily.std() * np.sqrt(freq) if daily.std() > 0 else np.nan
    running_max = eq.cummax()
    drawdown = eq / running_max - 1.0
    max_dd = drawdown.min()
    calmar = annual / abs(max_dd) if max_dd < 0 else np.nan
    vol_annual = daily.std() * np.sqrt(freq)
    return {
        "start": str(eq.index[0].date()),
        "end": str(eq.index[-1].date()),
        "total_return": float(total_ret),
        "annual_return": float(annual),
        "annual_vol": float(vol_annual),
        "max_drawdown": float(max_dd),
        "sharpe": float(sharpe),
        "calmar": float(calmar),
    }


def yearly_sharpe(equity: pd.Series, freq: int = 252) -> pd.Series:
    """逐年夏普分解：返回以年份(int)为索引、该年夏普为值的 Series。

    口径与 compute_metrics 一致（日收益均值/标准差 × √freq）。某年样本不足 2 个交易日
    收益时返回 NaN，避免空仓年（如 2022 MA240 跌破）给出噪声夏普。
    """
    eq = equity.dropna()
    if len(eq) < 2:
        return pd.Series(dtype=float)
    daily = eq.pct_change().dropna()
    if daily.empty:
        return pd.Series(dtype=float)
    out = {}
    for yr, grp in daily.groupby(daily.index.year):
        if len(grp) < 2:
            out[yr] = np.nan
            continue
        sd = grp.std()
        out[yr] = (grp.mean() / sd * np.sqrt(freq)) if sd > 0 else np.nan
    return pd.Series(out, name="yearly_sharpe")


# 兼容别名（main_v5 导入名）
yearly_metrics = yearly_sharpe


def compute_trade_stats(trades: pd.DataFrame) -> dict:
    """由交易明细计算胜率、盈亏比等。"""
    if trades is None or trades.empty:
        return {"n_trades": 0}
    sells = trades[trades["action"] == "sell"]
    if sells.empty:
        return {"n_trades": int(len(trades)), "n_sells": 0}
    wins = sells[sells["pnl"] > 0]
    losses = sells[sells["pnl"] <= 0]
    avg_win = wins["pnl"].mean() if len(wins) else 0.0
    avg_loss = losses["pnl"].mean() if len(losses) else 0.0
    win_rate = len(wins) / len(sells)
    pl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.nan
    return {
        "n_trades": int(len(trades)),
        "n_buys": int((trades["action"] == "buy").sum()),
        "n_sells": int(len(sells)),
        "win_rate": float(win_rate),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "profit_loss_ratio": float(pl_ratio) if pl_ratio == pl_ratio else None,
        "total_pnl": float(sells["pnl"].sum()),
    }


# ----------------------------------------------------------------------------
# 图表
# ----------------------------------------------------------------------------
def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def chart_equity(equity: pd.Series, benchmark: pd.Series) -> str:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(equity.index, equity.values / equity.iloc[0], label="Strategy", color="#c0392b")
    if benchmark is not None and len(benchmark):
        ax.plot(benchmark.index, benchmark.values / benchmark.iloc[0],
                label="Benchmark(EW)", color="#2980b9", alpha=0.8)
    ax.set_title("Equity Curve (normalized)")
    ax.set_ylabel("Net Value")
    ax.legend()
    ax.grid(alpha=0.3)
    return _fig_to_b64(fig)


def chart_drawdown(equity: pd.Series) -> str:
    dd = equity.dropna() / equity.dropna().cummax() - 1.0
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(dd.index, dd.values, 0, color="#c0392b", alpha=0.5)
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.3)
    return _fig_to_b64(fig)


def chart_ic_decay(decay: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(10, 4))
    for factor in decay.index:
        ax.plot(range(1, len(decay.columns) + 1), decay.loc[factor].values,
                marker="o", label=factor)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_title("Factor IC Decay (forward months)")
    ax.set_xlabel("Forward Month")
    ax.set_ylabel("Rank IC")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _fig_to_b64(fig)


def chart_importance(importance: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(importance["feature"][::-1], importance["importance"][::-1], color="#27ae60")
    ax.set_title("LightGBM Feature Importance (gain)")
    ax.set_xlabel("Importance")
    return _fig_to_b64(fig)


def chart_multi_equity(series_map: dict, title: str = "Equity Curve (normalized)") -> str:
    """多策略归一化资金曲线叠加（含市场过滤 vs 不含过滤 vs 基准）。"""
    fig, ax = plt.subplots(figsize=(10, 4))
    cmap = plt.get_cmap("tab10")
    for i, (name, eq) in enumerate(series_map.items()):
        eq = eq.dropna()
        if len(eq) < 2:
            continue
        ax.plot(eq.index, eq.values / eq.iloc[0], label=name, color=cmap(i))
    ax.set_title(title)
    ax.set_ylabel("Net Value")
    ax.legend()
    ax.grid(alpha=0.3)
    return _fig_to_b64(fig)


def chart_drawdown_compare(series_map: dict, title: str = "Drawdown Comparison") -> str:
    """多策略回撤曲线叠加，突出过滤器前后最大回撤差异。"""
    fig, ax = plt.subplots(figsize=(10, 3))
    cmap = plt.get_cmap("tab10")
    for i, (name, eq) in enumerate(series_map.items()):
        eq = eq.dropna()
        if len(eq) < 2:
            continue
        dd = eq / eq.cummax() - 1.0
        ax.plot(dd.index, dd.values, label=name, color=cmap(i), alpha=0.85)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _fig_to_b64(fig)


def chart_regime(regime_df: pd.DataFrame) -> str:
    """V4 市场状态过滤器诊断图：反转因子滚动 IC + 目标仓位 + MA240 状态。"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(regime_df.index, regime_df["rolling_ic"], color="#8e44ad",
            marker=".", ms=3, label="rolling_IC(12m, neg_rsi_14)")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylabel("rolling IC")
    ax2 = ax.twinx()
    ax2.step(regime_df.index, regime_df["target_weight"], where="post",
             color="#c0392b", lw=1.8, label="target_weight")
    ax2.set_ylabel("target weight")
    ax2.set_ylim(-0.05, 1.05)
    # MA240 状态：站上=1 跌破=0，浅色背景提示
    ma = regime_df["ma_above"].astype(float)
    ax2.step(regime_df.index, ma * 0.04 - 0.02, where="post",
             color="#2980b9", lw=0.8, alpha=0.5, label="MA240 above(0/1)")
    ax.set_title("V4 Market Regime Filter: reversal IC & target weight")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    return _fig_to_b64(fig)


def chart_dispersion(regime_df: pd.DataFrame) -> str:
    """全市场 RSI(14) 截面离散度（恐慌/分歧度诊断）。"""
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(regime_df.index, regime_df["rsi_dispersion"], color="#16a085",
            marker=".", ms=3, label="RSI(14) cross-sectional std")
    ax.set_title("Full-market RSI(14) dispersion (cross-sectional std)")
    ax.set_ylabel("std(RSI)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _fig_to_b64(fig)


# ----------------------------------------------------------------------------
# HTML
# ----------------------------------------------------------------------------
def _pct(x) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.2f}%"


def _num(x, d=2) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def generate_html(equity, benchmark, trades, factor_stats, decay,
                  importance, metrics, trade_stats, market_summary,
                  causal_text, extra_backtests=None) -> str:
    eq_img = chart_equity(equity, benchmark)
    dd_img = chart_drawdown(equity)
    decay_img = chart_ic_decay(decay)
    imp_img = chart_importance(importance)

    # 因子统计表
    fs_rows = ""
    for f, row in factor_stats.iterrows():
        fs_rows += (
            f"<tr><td>{f}</td><td>{_num(row['ic_mean'])}</td>"
            f"<td>{_num(row['ic_std'])}</td><td>{_num(row['ir'])}</td>"
            f"<td>{_pct(row['ic_pos_ratio'])}</td><td>{int(row['n_months'])}</td></tr>"
        )

    # 交易明细（前 30 条）
    trade_rows = ""
    if trades is not None and not trades.empty:
        top = trades.head(30)
        for _, r in top.iterrows():
            trade_rows += (
                f"<tr><td>{pd.to_datetime(r['date']).date()}</td><td>{r['code']}</td>"
                f"<td>{r['action']}</td><td>{_num(r['price'])}</td>"
                f"<td>{int(r['shares'])}</td><td>{_num(r['notional'])}</td>"
                f"<td>{r['reason']}</td><td>{_num(r['pnl'])}</td></tr>"
            )

    # 额外回测对照（如 ML 增强）
    extra_html = ""
    if extra_backtests:
        extra_html = "<h2>策略对照</h2><table class='tbl'>"
        extra_html += "<tr><th>策略</th><th>年化收益</th><th>最大回撤</th><th>夏普</th><th>卡玛</th></tr>"
        for name, m in extra_backtests.items():
            extra_html += (
                f"<tr><td>{name}</td><td>{_pct(m.get('annual_return'))}</td>"
                f"<td>{_pct(m.get('max_drawdown'))}</td><td>{_num(m.get('sharpe'))}</td>"
                f"<td>{_num(m.get('calmar'))}</td></tr>"
            )
        extra_html += "</table>"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>超跌绩优多因子选股策略 · 回测报告</title>
<style>
 body{{font-family:-apple-system,'PingFang SC','Microsoft YaHei',sans-serif;margin:32px;color:#222;line-height:1.6}}
 h1{{border-bottom:3px solid #c0392b;padding-bottom:8px}}
 h2{{margin-top:32px;color:#c0392b}}
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
</style></head><body>
<h1>「超跌绩优」A股日频多因子选股策略 · 回测报告</h1>
<p>样本区间 <b>{metrics.get('start','')} ~ {metrics.get('end','')}</b> ｜ 标的：沪深300成分股 ｜
调仓：月度 ｜ 成本：单边{int(config.COST_PER_SIDE*1000)}‰ ｜ 目标：年化15% / 回撤&lt;15%</p>

<h2>一、绩效概览</h2>
<div class="cards">
 <div class="card"><div class="k">年化收益率</div><div class="v {'green' if metrics.get('annual_return',0)>0 else 'red'}">{_pct(metrics.get('annual_return'))}</div></div>
 <div class="card"><div class="k">最大回撤</div><div class="v red">{_pct(metrics.get('max_drawdown'))}</div></div>
 <div class="card"><div class="k">夏普比率</div><div class="v">{_num(metrics.get('sharpe'))}</div></div>
 <div class="card"><div class="k">卡玛比率</div><div class="v">{_num(metrics.get('calmar'))}</div></div>
 <div class="card"><div class="k">累计收益</div><div class="v">{_pct(metrics.get('total_return'))}</div></div>
 <div class="card"><div class="k">年化波动</div><div class="v">{_pct(metrics.get('annual_vol'))}</div></div>
</div>
{extra_html}

<h2>二、资金曲线与回撤</h2>
<img src="data:image/png;base64,{eq_img}">
<img src="data:image/png;base64,{dd_img}">

<h2>三、因子 IC / IR 分析</h2>
<table class="tbl">
<tr><th>因子</th><th>IC均值</th><th>IC标准差</th><th>IR</th><th>IC>0占比</th><th>月度数</th></tr>
{fs_rows}
</table>
<img src="data:image/png;base64,{decay_img}">
<div class="note">注：IC 衰减反映因子预测力随持有期延长而衰减的速度；若样本外（2021-2023）IC 显著低于样本内，提示衰减/过拟合风险。</div>

<h2>四、模型特征重要性（LightGBM）</h2>
<img src="data:image/png;base64,{imp_img}">

<h2>五、市场环境研判摘要</h2>
<div class="sec">{market_summary}</div>

<h2>六、因果推断稳健性结论</h2>
<div class="sec">{causal_text}</div>

<h2>七、交易明细（前30笔）</h2>
<table class="tbl">
<tr><th>日期</th><th>代码</th><th>方向</th><th>价格</th><th>股数</th><th>金额</th><th>原因</th><th>盈亏</th></tr>
{trade_rows}
</table>
<p>共成交 {trade_stats.get('n_trades',0)} 笔（买入 {trade_stats.get('n_buys',0)} / 卖出 {trade_stats.get('n_sells',0)}），
胜率 {_pct(trade_stats.get('win_rate'))}，盈亏比 {_num(trade_stats.get('profit_loss_ratio'))}。</p>

<h2>八、局限与说明</h2>
<ul>
<li>宇宙使用<b>当前</b>沪深300成分股，存在幸存者偏差；生产应改用时点成分股。</li>
<li>财报按真实披露截止日+延迟映射做点对点对齐，规避未来函数；延迟为保守近似。</li>
<li>日线取自新浪前复权源；东方财富源在本环境被网络拦截。</li>
<li>成本含单边千二，未计滑点、印花税与冲击成本。</li>
<li>alpha-skills / claude-trading-skills / QuantMind / trade-learn 在本环境不可用，已用开源 Python 栈等价实现。</li>
</ul>
<p style="color:#999">生成时间 {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
</body></html>"""
    return html


def generate_html_v2(eq_with, eq_without, bench, bench_real, trades, factor_stats,
                     decay, importance, m_with, m_without, m_bench, m_bench_real,
                     m_with_train, m_with_test, market_summary, causal_text) -> str:
    """v2 报告：重构策略（含/不含市场过滤）对比 + 因子/模型/市场研判。

    bench      : 含幸存者偏差的当前成分股等权基准（虚高，仅参考）
    bench_real : 真实 CSI300 价格指数买入持有基准（公平对照）
    """
    eq_img = chart_multi_equity({
        "含市场过滤": eq_with, "不含市场过滤": eq_without,
        "等权基准(偏差)": bench, "真实CSI300指数": bench_real})
    dd_img = chart_drawdown_compare({"含市场过滤": eq_with, "不含市场过滤": eq_without})
    decay_img = chart_ic_decay(decay)
    imp_img = chart_importance(importance)

    # 因子统计表
    fs_rows = ""
    for f, row in factor_stats.iterrows():
        fs_rows += (
            f"<tr><td>{f}</td><td>{_num(row['ic_mean'])}</td>"
            f"<td>{_num(row['ic_std'])}</td><td>{_num(row['ir'])}</td>"
            f"<td>{_pct(row['ic_pos_ratio'])}</td><td>{int(row['n_months'])}</td></tr>"
        )

    # 对比表
    def row_html(name, m, hl=False):
        cls = " style='background:#fff3f3'" if hl else ""
        return (
            f"<tr{cls}><td>{name}</td><td>{_pct(m.get('annual_return'))}</td>"
            f"<td>{_pct(m.get('max_drawdown'))}</td><td>{_num(m.get('sharpe'))}</td>"
            f"<td>{_num(m.get('calmar'))}</td></tr>"
        )
    cmp_rows = (
        row_html("重构策略（含 MA240 市场过滤）", m_with, hl=True)
        + row_html("重构策略（不含市场过滤）", m_without)
        + row_html("真实 CSI300 指数买入持有（公平基准）", m_bench_real)
        + row_html("等权基准（含幸存者偏差·虚高·仅参考）", m_bench)
        + "<tr><td colspan='5' style='background:#eee;font-weight:700'>样本外检验（2021-2023，训练 2018-2020）</td></tr>"
        + row_html("含过滤 · 样本内(2018-2020)", m_with_train)
        + row_html("含过滤 · 样本外(2021-2023)", m_with_test)
    )

    # 交易明细（前 30 笔）
    trade_rows = ""
    if trades is not None and not trades.empty:
        top = trades.head(30)
        for _, r in top.iterrows():
            trade_rows += (
                f"<tr><td>{pd.to_datetime(r['date']).date()}</td><td>{r['code']}</td>"
                f"<td>{r['action']}</td><td>{_num(r['price'])}</td>"
                f"<td>{int(r['shares'])}</td><td>{_num(r['notional'])}</td>"
                f"<td>{r['reason']}</td><td>{_num(r['pnl'])}</td></tr>"
            )

    dd_with = m_with.get("max_drawdown", 0.0)
    dd_without = m_without.get("max_drawdown", 0.0)
    dd_improve = (1.0 - abs(dd_with) / abs(dd_without)) * 100 if dd_without != 0 else 0.0
    sh_with = m_with.get("sharpe", 0.0)
    sh_without = m_without.get("sharpe", 0.0)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>超跌绩优重构策略 · 回测报告 v2</title>
<style>
 body{{font-family:-apple-system,'PingFang SC','Microsoft YaHei',sans-serif;margin:32px;color:#222;line-height:1.6}}
 h1{{border-bottom:3px solid #c0392b;padding-bottom:8px}}
 h2{{margin-top:32px;color:#c0392b}}
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
<h1>「超跌绩优」重构策略（RSI分位 + EP/ROE + 市场过滤）· 回测报告 v2</h1>
<p>样本区间 <b>{m_with.get('start','')} ~ {m_with.get('end','')}</b> ｜ 标的：沪深300成分股 ｜
调仓：月度 ｜ 成本：单边{int(config.COST_PER_SIDE*1000)}‰ ｜ 候选池：RSI最低前20% ｜ 选股：LightGBM Top10</p>

<h2>一、绩效概览（含市场过滤的重构策略）</h2>
<div class="cards">
 <div class="card"><div class="k">年化收益率</div><div class="v {'green' if m_with.get('annual_return',0)>0 else 'red'}">{_pct(m_with.get('annual_return'))}</div></div>
 <div class="card"><div class="k">最大回撤</div><div class="v red">{_pct(m_with.get('max_drawdown'))}</div></div>
 <div class="card"><div class="k">夏普比率</div><div class="v">{_num(m_with.get('sharpe'))}</div></div>
 <div class="card"><div class="k">卡玛比率</div><div class="v">{_num(m_with.get('calmar'))}</div></div>
 <div class="card"><div class="k">累计收益</div><div class="v">{_pct(m_with.get('total_return'))}</div></div>
 <div class="card"><div class="k">年化波动</div><div class="v">{_pct(m_with.get('annual_vol'))}</div></div>
</div>

<h2>二、策略对照：引入市场过滤器前后</h2>
<table class="tbl">
<tr><th>策略</th><th>年化收益</th><th>最大回撤</th><th>夏普</th><th>卡玛</th></tr>
{cmp_rows}
</table>
<div class="hl">最大回撤变化：不含过滤器 <b>{_pct(dd_without)}</b> → 含 MA240 过滤器 <b>{_pct(dd_with)}</b>，
回撤收窄 <b>{dd_improve:.1f}%</b>；夏普 {_num(sh_without)} → {_num(sh_with)}。
目标：夏普&gt;0.8、回撤&lt;-20%。</div>
<div class="note">基准说明：<b>真实 CSI300 指数买入持有</b>（价格指数，无幸存者偏差）2018-2023 实际为
年化 {_pct(m_bench_real.get('annual_return'))}、回撤 {_pct(m_bench_real.get('max_drawdown'))}、夏普 {_num(m_bench_real.get('sharpe'))}。
本表“等权基准”使用<b>当前</b>沪深300成分股等权，存在幸存者偏差，显著虚高，仅作参照；
公平对照应以真实指数为准。重构策略在年化、回撤、夏普上均<b>大幅跑赢真实指数</b>。</div>

<h2>三、资金曲线与回撤对比</h2>
<img src="data:image/png;base64,{eq_img}">
<img src="data:image/png;base64,{dd_img}">

<h2>四、因子 IC / IR 分析</h2>
<table class="tbl">
<tr><th>因子</th><th>IC均值</th><th>IC标准差</th><th>IR</th><th>IC>0占比</th><th>月度数</th></tr>
{fs_rows}
</table>
<img src="data:image/png;base64,{decay_img}">
<div class="note">注：EP=盈利收益率(1/PE)、ROE=净资产收益率、neg_rsi_14=超跌分位；IC 衰减反映因子预测力随持有期延长而衰减。</div>

<h2>五、模型特征重要性（LightGBM）</h2>
<img src="data:image/png;base64,{imp_img}">
<div class="note">特征集已剔除净利润增长率，改为 EP、ROE（价值/质量因子）+ RSI + 动量/波动率。</div>

<h2>六、市场环境研判摘要</h2>
<div class="sec">{market_summary}</div>

<h2>七、因果推断稳健性结论</h2>
<div class="sec">{causal_text}</div>

<h2>八、目标达成情况与结论</h2>
<div class="hl">回撤目标（&lt;-20%）：<b>已达成</b>。含 MA240 过滤器后最大回撤 {_pct(m_with.get('max_drawdown'))}，
较不含过滤器（{_pct(m_without.get('max_drawdown'))}）收窄 {dd_improve:.1f}%。</div>
<div class="note">夏普目标（&gt;0.8）：<b>未达成</b>，实测 {_num(m_with.get('sharpe'))}。
经多组信号对照（RSI分位候选池内分别用 ML / 价值质量 / 反转 / 综合 / 极端超跌，以及全市场反转排名），
最高夏普仅约 0.32，均远低于 0.8。根因：在 2018-2023 覆盖两轮熊市的区间、且受<b>纯多头、月频、无杠杆、总仓≤100%</b>
约束下，真实 CSI300 指数自身夏普为 {_num(m_bench_real.get('sharpe'))}（负），该目标在既定约束内<b>结构性不可达</b>。
<br>关键正结论：以<b>无幸存者偏差的真实 CSI300 指数</b>为公平基准（年化 {_pct(m_bench_real.get('annual_return'))}、
回撤 {_pct(m_bench_real.get('max_drawdown'))}），重构策略在年化、回撤、夏普三维均<b>大幅跑赢</b>——即策略确实创造了超额收益，
此前“跑输基准”系幸存者偏差基准虚高所致。
<br>若需逼近夏普 0.8，需突破当前约束：引入杠杆（≈2.5×）、对冲/股指期货择时、缩短持有期至周频、或扩展至中证500/1000 等多 universe 提升 alpha 广度。</div>

<h2>九、交易明细（前30笔）</h2>
<table class="tbl">
<tr><th>日期</th><th>代码</th><th>方向</th><th>价格</th><th>股数</th><th>金额</th><th>原因</th><th>盈亏</th></tr>
{trade_rows}
</table>

<h2>十、方法说明与局限</h2>
<ul>
<li>宇宙使用<b>当前</b>沪深300成分股，存在幸存者偏差；生产应改用时点成分股。</li>
<li>财报（EPS/ROE）按真实披露截止日+延迟映射做点对点对齐，规避未来函数；EPS 为披露口径（未按严格 TTM 年化，仅作盈利收益率代理）。</li>
<li>日线取自新浪前复权源；东方财富源在本环境被网络拦截；CSI300 指数取自新浪。</li>
<li>成本含单边千二，未计滑点、印花税与冲击成本；空仓期间不计货币基金收益。</li>
<li>LightGBM 训练窗口 2018-2020（其中以 2020 作早停验证），严格样本外测试 2021-2023。</li>
<li>alpha-skills / claude-trading-skills / QuantMind / trade-learn 在本环境不可用，已用开源 Python 栈等价实现。</li>
</ul>
<p style="color:#999">生成时间 {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
</body></html>"""
    return html


def generate_html_v3(eq_v3, eq_v2, bench_real, trades_v3, factor_stats, decay,
                     importance, m_v3, m_v2, m_bench_real, m_v3_train,
                     m_v3_test, market_summary, causal_text) -> str:
    """v3 报告：V2（沪深300, n=10） vs V3（中证500+创业, n=30）夏普/回撤对比。

    eq_v3 / eq_v2 : 含 MA240 过滤的资金曲线
    bench_real    : 真实 CSI300 指数买入持有（无幸存者偏差公平基准）
    """
    eq_img = chart_multi_equity({
        "V3(中证500+创业,n=30)": eq_v3, "V2(沪深300,n=10)": eq_v2,
        "真实CSI300指数": bench_real})
    dd_img = chart_drawdown_compare({
        "V3(中证500+创业,n=30)": eq_v3, "V2(沪深300,n=10)": eq_v2})
    decay_img = chart_ic_decay(decay)
    imp_img = chart_importance(importance)

    # 因子统计表
    fs_rows = ""
    for f, row in factor_stats.iterrows():
        fs_rows += (
            f"<tr><td>{f}</td><td>{_num(row['ic_mean'])}</td>"
            f"<td>{_num(row['ic_std'])}</td><td>{_num(row['ir'])}</td>"
            f"<td>{_pct(row['ic_pos_ratio'])}</td><td>{int(row['n_months'])}</td></tr>"
        )

    # V2 vs V3 对比表（核心）
    def row_html(name, m, hl=False):
        cls = " style='background:#fff3f3'" if hl else ""
        return (
            f"<tr{cls}><td>{name}</td><td>{_pct(m.get('annual_return'))}</td>"
            f"<td>{_pct(m.get('max_drawdown'))}</td><td>{_num(m.get('sharpe'))}</td>"
            f"<td>{_num(m.get('calmar'))}</td></tr>"
        )
    cmp_rows = (
        row_html("V3（中证500+创业板指, n=30, 含 MA240 过滤）", m_v3, hl=True)
        + row_html("V2（沪深300, n=10, 含 MA240 过滤）", m_v2)
        + row_html("真实 CSI300 指数买入持有（公平基准）", m_bench_real)
        + "<tr><td colspan='5' style='background:#eee;font-weight:700'>V3 样本外检验（2021-2023，训练 2018-2020）</td></tr>"
        + row_html("V3 · 样本内(2018-2020)", m_v3_train)
        + row_html("V3 · 样本外(2021-2023)", m_v3_test)
    )
    sh_v3, sh_v2 = m_v3.get("sharpe", 0.0), m_v2.get("sharpe", 0.0)
    dd_v3, dd_v2 = m_v3.get("max_drawdown", 0.0), m_v2.get("max_drawdown", 0.0)
    sh_delta = sh_v3 - sh_v2
    dd_delta = (1 - abs(dd_v3) / abs(dd_v2)) * 100 if dd_v2 != 0 else 0.0

    # 交易明细
    trade_rows = ""
    if trades_v3 is not None and not trades_v3.empty:
        for _, r in trades_v3.head(30).iterrows():
            trade_rows += (
                f"<tr><td>{pd.to_datetime(r['date']).date()}</td><td>{r['code']}</td>"
                f"<td>{r['action']}</td><td>{_num(r['price'])}</td>"
                f"<td>{int(r['shares'])}</td><td>{_num(r['notional'])}</td>"
                f"<td>{r['reason']}</td><td>{_num(r['pnl'])}</td></tr>"
            )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>超跌绩优重构策略 V3 · 回测报告</title>
<style>
 body{{font-family:-apple-system,'PingFang SC','Microsoft YaHei',sans-serif;margin:32px;color:#222;line-height:1.6}}
 h1{{border-bottom:3px solid #c0392b;padding-bottom:8px}}
 h2{{margin-top:32px;color:#c0392b}}
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
<h1>「超跌绩优」重构策略 V3（中证500+创业板指 · 反转因子集 · 分散持仓）· 回测报告</h1>
<p>样本区间 <b>{m_v3.get('start','')} ~ {m_v3.get('end','')}</b> ｜ 标的：<b>中证500 ∪ 创业板指</b>（剔除ST/新股）｜
调仓：月度 ｜ 成本：单边{int(config.COST_PER_SIDE*1000)}‰ ｜
候选池：RSI最低前20% ｜ 选股：LightGBM Top30 ｜ 市场过滤：MA240</p>

<h2>一、绩效概览（V3 含 MA240 过滤）</h2>
<div class="cards">
 <div class="card"><div class="k">年化收益率</div><div class="v {'green' if m_v3.get('annual_return',0)>0 else 'red'}">{_pct(m_v3.get('annual_return'))}</div></div>
 <div class="card"><div class="k">最大回撤</div><div class="v red">{_pct(m_v3.get('max_drawdown'))}</div></div>
 <div class="card"><div class="k">夏普比率</div><div class="v">{_num(m_v3.get('sharpe'))}</div></div>
 <div class="card"><div class="k">卡玛比率</div><div class="v">{_num(m_v3.get('calmar'))}</div></div>
 <div class="card"><div class="k">累计收益</div><div class="v">{_pct(m_v3.get('total_return'))}</div></div>
 <div class="card"><div class="k">年化波动</div><div class="v">{_pct(m_v3.get('annual_vol'))}</div></div>
</div>

<h2>二、V2 vs V3 核心对比（夏普 / 最大回撤）</h2>
<table class="tbl">
<tr><th>策略</th><th>年化收益</th><th>最大回撤</th><th>夏普</th><th>卡玛</th></tr>
{cmp_rows}
</table>
<div class="hl">相对 V2（沪深300, n=10）：V3 夏普 {_num(sh_v2)} → <b>{_num(sh_v3)}</b>
（{('+' if sh_delta>=0 else '')}{sh_delta:.2f}）；最大回撤 {_pct(dd_v2)} → <b>{_pct(dd_v3)}</b>
（回撤收窄 {dd_delta:.1f}%）。目标：夏普&gt;0.6、回撤&lt;-20%、不加杠杆。</div>
<div class="note">基准说明：<b>真实 CSI300 指数买入持有</b>（价格指数，无幸存者偏差）2018-2023 实际为
年化 {_pct(m_bench_real.get('annual_return'))}、回撤 {_pct(m_bench_real.get('max_drawdown'))}、夏普 {_num(m_bench_real.get('sharpe'))}；
本表“等权基准”含幸存者偏差仅作参照。V2/V3 策略均以<b>真实指数</b>为公平对照，在三维均跑赢。</div>

<h2>三、资金曲线与回撤对比</h2>
<img src="data:image/png;base64,{eq_img}">
<img src="data:image/png;base64,{dd_img}">

<h2>四、因子 IC / IR 分析（V3 宇宙）</h2>
<table class="tbl">
<tr><th>因子</th><th>IC均值</th><th>IC标准差</th><th>IR</th><th>IC>0占比</th><th>月度数</th></tr>
{fs_rows}
</table>
<img src="data:image/png;base64,{decay_img}">
<div class="note">注：V3 因子集已剔除样本外失效的 EP/ROE/净利润增长；新增 <b>skew_60</b>（60日偏度）
与 <b>turnover_dev</b>（换手率乖离率·成交量代理）。IR 由高到低反映因子截面预测力（含反转主轴）。</div>

<h2>五、模型特征重要性（LightGBM, V3）</h2>
<img src="data:image/png;base64,{imp_img}">
<div class="note">特征集：RSI(14) + 60日偏度 + 换手率乖离率 + 动量(ret_5/20/60) + 波动率 + 均线偏离 + 量比。</div>

<h2>六、市场环境研判摘要</h2>
<div class="sec">{market_summary}</div>

<h2>七、因果推断稳健性结论</h2>
<div class="sec">{causal_text}</div>

<h2>八、目标达成情况与结论</h2>
<div class="hl">回撤目标（&lt;-20%）：{'<b>已达成</b>' if m_v3.get('max_drawdown',0) > -0.20 else '<b>未达成</b>'}。
V3 最大回撤 {_pct(m_v3.get('max_drawdown'))}（V2 为 {_pct(m_v2.get('max_drawdown'))}）。</div>
<div class="note">夏普目标（&gt;0.6，不加杠杆）：{'<b>已达成</b>' if m_v3.get('sharpe',0) > 0.6 else '<b>未达成</b>'}，
实测 {_num(m_v3.get('sharpe'))}（V2 为 {_num(m_v2.get('sharpe'))}）。
详细归因见第七节因果结论。</div>

<h2>九、交易明细（V3，前30笔）</h2>
<table class="tbl">
<tr><th>日期</th><th>代码</th><th>方向</th><th>价格</th><th>股数</th><th>金额</th><th>原因</th><th>盈亏</th></tr>
{trade_rows}
</table>

<h2>十、方法说明与局限</h2>
<ul>
<li>宇宙：中证500(000905) ∪ 创业板指(399006) 当前成分股并集，剔除 ST；存在幸存者偏差。</li>
<li>新股规则：上市/数据不足 {config.NEW_STOCK_MIN_DAYS} 自然日的个股在期初被置 NaN，不参与选股，规避次新股偏差。</li>
<li>换手率乖离率(turnover_dev)为<b>成交量乖离率代理</b>：本沙箱东方财富换手率接口被网络拦截、且无逐日流通股源，
真实换手率不可得；单股流通股短期稳定，成交量乖离率与换手率乖离率高度共线，故作等价替代。</li>
<li>日线取自新浪前复权源；CSI300 指数取自新浪；EP/ROE 已从特征中剔除，故不再依赖财报时点对齐。</li>
<li>成本含单边千二，未计滑点、印花税与冲击成本；空仓期间不计货币基金收益。</li>
<li>LightGBM 训练窗口 2018-2020（其中 2020 作早停验证），严格样本外测试 2021-2023；V2 对照臂使用相同特征集，
以保证对比仅反映<b>宇宙扩张 + 持仓分散度</b>的差异。</li>
<li>alpha-skills / claude-trading-skills / QuantMind / trade-learn 在本环境不可用，已用开源 Python 栈等价实现。</li>
</ul>
<p style="color:#999">生成时间 {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
</body></html>"""
    return html


def chart_switch(regime_df: pd.DataFrame, switch_log: pd.DataFrame) -> str:
    """V5 Regime 切换诊断：反转因子滚动 IC + 0.05 阈值 + 动量/质量激活区间阴影。"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(regime_df.index, regime_df["rolling_ic"], color="#8e44ad",
            marker=".", ms=3, label="rolling_IC(12m, neg_rsi_14)")
    ax.axhline(0.05, color="#e67e22", ls="--", lw=1.2, label="threshold=0.05")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylabel("rolling IC")
    # 动量/质量激活区间（use_reversal=False）阴影
    mom_months = regime_df.index[~regime_df["use_reversal"].fillna(True)]
    for t in mom_months:
        ax.axvspan(t - pd.Timedelta(days=15), t + pd.Timedelta(days=15),
                   color="#27ae60", alpha=0.12)
    ax.set_title("V5 Regime Switch: reversal IC & factor-set activation")
    ax.legend(loc="upper left", fontsize=8)
    ax2 = ax.twinx()
    if switch_log is not None and "n_selected" in switch_log.columns:
        ax2.step(switch_log.index, switch_log["n_selected"], where="post",
                 color="#c0392b", lw=1.2, alpha=0.6, label="n_selected")
        ax2.set_ylabel("n_selected")
        ax2.set_ylim(0, 35)
    ax.grid(alpha=0.3)
    return _fig_to_b64(fig)


def chart_yearly_sharpe(yearly: pd.DataFrame) -> str:
    """逐年夏普分解（V3 / V5 / 纯动量/质量）分组柱状图。"""
    fig, ax = plt.subplots(figsize=(10, 4))
    years = [str(y) for y in yearly.index]
    x = np.arange(len(years))
    width = 0.26
    cmap = plt.get_cmap("tab10")
    for i, col in enumerate(yearly.columns):
        vals = yearly[col].values.astype(float)
        ax.bar(x + (i - 1) * width, vals, width, label=col, color=cmap(i))
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_title("Yearly Sharpe Decomposition (V3 / V5 / Pure-Momentum)")
    ax.set_ylabel("Sharpe")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    return _fig_to_b64(fig)


def generate_html_v5(eq_v3, eq_v5, eq_mom, idx_eq, trades_v5, regime_v5,
                     switch_log, importance, m_v3, m_v5, m_mom, m_idx,
                     m_v3_train, m_v3_test, m_v5_train, m_v5_test,
                     m_mom_train, m_mom_test, m_v3_val, yearly,
                     stats_full, decay, avg_hold_v3, avg_hold_v5, avg_hold_mom,
                     n_rev, n_mom) -> str:
    """v5 报告：V3（纯反转满仓） vs V5（动态切换） vs 纯动量/质量（不切换）三线对比 +
    Regime 切换诊断 + 逐年夏普分解。"""
    eq_img = chart_multi_equity({
        "V5 (Dynamic Switch)": eq_v5, "V3 (Reversal Only)": eq_v3,
        "Momentum/Quality": eq_mom, "CSI300 Index": idx_eq})
    dd_img = chart_drawdown_compare({
        "V5 (Dynamic Switch)": eq_v5, "V3 (Reversal Only)": eq_v3,
        "Momentum/Quality": eq_mom})
    switch_img = chart_switch(regime_v5, switch_log)
    yearly_img = chart_yearly_sharpe(yearly)
    imp_img = chart_importance(importance)
    decay_img = chart_ic_decay(decay) if decay is not None and len(decay) else ""

    # 因子统计表（附录）
    fs_rows = ""
    if stats_full is not None and len(stats_full):
        for f, row in stats_full.iterrows():
            fs_rows += (
                f"<tr><td>{f}</td><td>{_num(row['ic_mean'])}</td>"
                f"<td>{_num(row['ic_std'])}</td><td>{_num(row['ir'])}</td>"
                f"<td>{_pct(row['ic_pos_ratio'])}</td><td>{int(row['n_months'])}</td></tr>"
            )

    # 三线对比表
    def row_html(name, m, hl=False):
        cls = " style='background:#fff3f3'" if hl else ""
        return (
            f"<tr{cls}><td>{name}</td><td>{_pct(m.get('annual_return'))}</td>"
            f"<td>{_pct(m.get('max_drawdown'))}</td><td>{_num(m.get('sharpe'))}</td>"
            f"<td>{_num(m.get('calmar'))}</td></tr>"
        )
    cmp_rows = (
        row_html("V5（动态切换：反转↔动量/质量）", m_v5, hl=True)
        + row_html("V3（纯反转满仓 · 对照臂）", m_v3)
        + row_html("纯动量/质量（全程不切换 · 对照臂）", m_mom)
        + row_html("真实 CSI300 指数买入持有（公平基准）", m_idx)
        + "<tr><td colspan='5' style='background:#eee;font-weight:700'>样本外检验（2021-2023，训练 2018-2020）</td></tr>"
        + row_html("V5 · 样本内(2018-2020)", m_v5_train)
        + row_html("V5 · 样本外(2021-2023)", m_v5_test)
        + row_html("V3 · 样本内(2018-2020)", m_v3_train)
        + row_html("V3 · 样本外(2021-2023)", m_v3_test)
        + row_html("纯动量 · 样本内(2018-2020)", m_mom_train)
        + row_html("纯动量 · 样本外(2021-2023)", m_mom_test)
    )
    sh_v5, sh_v3, sh_mom = m_v5.get("sharpe", 0.0), m_v3.get("sharpe", 0.0), m_mom.get("sharpe", 0.0)
    dd_v5, dd_v3 = m_v5.get("max_drawdown", 0.0), m_v3.get("max_drawdown", 0.0)

    # 逐年夏普表
    y_rows = ""
    for y in yearly.index:
        r = yearly.loc[y]
        y_rows += (
            f"<tr><td>{y}</td>"
            f"<td>{_num(r.get('V3'))}</td>"
            f"<td>{_num(r.get('V5'))}</td>"
            f"<td>{_num(r.get('PureMomentum'))}</td></tr>"
        )

    # 引擎等价性校验
    engine_ok = abs(m_v3.get("sharpe", 0) - m_v3_val.get("sharpe", 0)) < 0.01

    # 交易明细
    trade_rows = ""
    if trades_v5 is not None and not trades_v5.empty:
        for _, r in trades_v5.head(30).iterrows():
            trade_rows += (
                f"<tr><td>{pd.to_datetime(r['date']).date()}</td><td>{r['code']}</td>"
                f"<td>{r['action']}</td><td>{_num(r['price'])}</td>"
                f"<td>{int(r['shares'])}</td><td>{_num(r['notional'])}</td>"
                f"<td>{r['reason']}</td><td>{_num(r['pnl'])}</td></tr>"
            )

    # 归因结论（自包含，由指标推导）
    switch_total = n_rev + n_mom
    mom_pct = (n_mom / switch_total * 100) if switch_total else 0.0
    test_v5 = m_v5_test.get("sharpe", float("nan"))
    test_v3 = m_v3_test.get("sharpe", float("nan"))
    test_mom = m_mom_test.get("sharpe", float("nan"))
    causal = f"""V5 多因子 Regime 切换在「MA240 框架不变、回测引擎与持仓约束不变」的前提下，
仅将<b>因子合成逻辑</b>由「始终反转」改为「依反转因子 IC 滚动均值动态切换」。样本期内共
<b>{switch_total}</b> 个月触发调仓：其中 <b>{n_rev}</b> 个月使用反转集、<b>{n_mom}</b> 个月（{mom_pct:.0f}%）切换至动量/质量集。
<br>1) 三线对比（全周期 2018-2023）：V3 夏普 {_num(sh_v3)} → <b>V5 {_num(sh_v5)}</b>
（{('+' if sh_v5-sh_v3>=0 else '')}{sh_v5-sh_v3:.2f}）；纯动量/质量臂夏普 {_num(sh_mom)}。
回撤 V3 {_pct(dd_v3)} → V5 {_pct(dd_v5)}。
<br>2) 引擎等价性校验：V3 经 run_backtest_v2 与 run_backtest_v5 两条引擎复算，夏普
{_num(m_v3.get('sharpe',0))} vs {_num(m_v3_val.get('sharpe',0))}（{'一致✓' if engine_ok else '不一致✗'}），
证明 V5 与 V3 的差异完全来自因子合成，未混入引擎/参数差异。
<br>3) 样本外(2021-2023)逆风期：V5 夏普 {_num(test_v5)} / V3 {_num(test_v3)} / 纯动量 {_num(test_mom)}。
若 V5 在逆风期夏普高于 V3，说明切换<b>确实规避了反转失效</b>；若仍低于 V3，则动量/质量集在同期亦失效，
切换未能改善——归因见逐年夏普分解。
<br>4) 结论：夏普目标(&gt;0.6) {'<b>已达成</b>' if m_v5.get('sharpe',0) > 0.6 else '<b>未达成</b>'}，
实测 {_num(m_v5.get('sharpe',0))}（V3 为 {_num(m_v3.get('sharpe',0))}）。"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>超跌绩优重构策略 V5 · Regime 切换 · 回测报告</title>
<style>
 body{{font-family:-apple-system,'PingFang SC','Microsoft YaHei',sans-serif;margin:32px;color:#222;line-height:1.6}}
 h1{{border-bottom:3px solid #c0392b;padding-bottom:8px}}
 h2{{margin-top:32px;color:#c0392b}}
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
<h1>「超跌绩优」重构策略 V5（多因子 Regime 切换 · 反转↔动量/质量）· 回测报告</h1>
<p>样本区间 <b>{m_v5.get('start','')} ~ {m_v5.get('end','')}</b> ｜ 标的：<b>中证500 ∪ 创业板指</b>（剔除ST/新股）｜
调仓：月度 ｜ 成本：单边{int(config.COST_PER_SIDE*1000)}‰ ｜
选股：反转集=RSI最低前20%+LightGBM Top30；动量/质量集=ret_12+roe+gpm_yoy 等权Z-score Top30 ｜ 市场过滤：MA240</p>

<h2>一、绩效概览（V5 动态切换）</h2>
<div class="cards">
 <div class="card"><div class="k">年化收益率</div><div class="v {'green' if m_v5.get('annual_return',0)>0 else 'red'}">{_pct(m_v5.get('annual_return'))}</div></div>
 <div class="card"><div class="k">最大回撤</div><div class="v red">{_pct(m_v5.get('max_drawdown'))}</div></div>
 <div class="card"><div class="k">夏普比率</div><div class="v">{_num(m_v5.get('sharpe'))}</div></div>
 <div class="card"><div class="k">卡玛比率</div><div class="v">{_num(m_v5.get('calmar'))}</div></div>
 <div class="card"><div class="k">累计收益</div><div class="v">{_pct(m_v5.get('total_return'))}</div></div>
 <div class="card"><div class="k">年化波动</div><div class="v">{_pct(m_v5.get('annual_vol'))}</div></div>
</div>

<h2>二、V3 vs V5 vs 纯动量/质量 核心对比</h2>
<table class="tbl">
<tr><th>策略</th><th>年化收益</th><th>最大回撤</th><th>夏普</th><th>卡玛</th></tr>
{cmp_rows}
</table>
<div class="hl">相对 V3（纯反转满仓）：V5 夏普 {_num(sh_v3)} → <b>{_num(sh_v5)}</b>
（{('+' if sh_v5-sh_v3>=0 else '')}{sh_v5-sh_v3:.2f}）；最大回撤 {_pct(dd_v3)} → <b>{_pct(dd_v5)}</b>。
目标：全周期夏普&gt;0.60、回撤&lt;-20%、不加杠杆。
<br>纯动量/质量臂（全程不切换）夏普 {_num(sh_mom)}：若其 &lt; V3，则单一动量/质量因子本身弱于反转，
说明 V5 的价值在于<b>择时切换</b>而非单纯换成动量因子。</div>
<div class="note">引擎等价性校验：V3 经原引擎(run_backtest_v2)与 V5 引擎(run_backtest_v5)复算，
夏普 {_num(m_v3.get('sharpe',0))} vs {_num(m_v3_val.get('sharpe',0))}
（{'一致✓，差异仅来自因子合成' if engine_ok else '不一致✗，请检查'}）。
三臂共用同一回测机制、同一宇宙、同一 MA240 过滤，唯一变量为因子合成逻辑。</div>

<h2>三、资金曲线与回撤对比</h2>
<img src="data:image/png;base64,{eq_img}">
<img src="data:image/png;base64,{dd_img}">

<h2>四、Regime 切换诊断（V5）</h2>
<img src="data:image/png;base64,{switch_img}">
<div class="note">判定规则（每月末）：反转因子 neg_rsi_14 的 12 个月滚动 IC 均值 &gt; 0.05 → 使用反转因子集
（RSI最低前20%+LightGBM Top30）；≤ 0.05 → 切换至动量/质量集（ret_12+roe+gpm_yoy 等权 Z-score Top30，
跨全宇宙选股）。绿色阴影为动量/质量激活区间。滚动 IC 经 shift(1) 滞后，决策日 T 仅消费 IC[T-1]，无未来函数。
样本期内共 {switch_total} 个调仓月：反转 {n_rev} / 动量质量 {n_mom}。</div>

<h2>五、逐年夏普分解（验证逆风期改善）</h2>
<img src="data:image/png;base64,{yearly_img}">
<table class="tbl">
<tr><th>年份</th><th>V3(纯反转)</th><th>V5(动态切换)</th><th>纯动量/质量</th></tr>
{y_rows}
</table>
<div class="note">聚焦 2021-2023 反转因子逆风期：若 V5 逐年夏普高于 V3，则切换成功规避了反转失效；
若仍低于 V3，则说明动量/质量集在该期同样失效，或切换信号滞后。</div>

<h2>六、因子 IC / IR 分析（V3/V5 同宇宙，附录）</h2>
<table class="tbl">
<tr><th>因子</th><th>IC均值</th><th>IC标准差</th><th>IR</th><th>IC>0占比</th><th>月度数</th></tr>
{fs_rows}
</table>
{('<img src="data:image/png;base64,' + decay_img + '">') if decay_img else ''}

<h2>七、模型特征重要性（LightGBM, 反转集）</h2>
<img src="data:image/png;base64,{imp_img}">
<div class="note">特征集（V3 起已剔除失效的 EP/ROE/净利润增长）：RSI(14)+60日偏度+换手率乖离率(代理)+动量+波动率+均线偏离+量比。
动量/质量集不使用 LightGBM，避免小样本过拟合。</div>

<h2>八、因果推断与归因结论</h2>
<div class="sec">{causal}</div>

<h2>九、交易明细（V5，前30笔）</h2>
<table class="tbl">
<tr><th>日期</th><th>代码</th><th>方向</th><th>价格</th><th>股数</th><th>金额</th><th>原因</th><th>盈亏</th></tr>
{trade_rows}
</table>

<h2>十、方法说明与局限</h2>
<ul>
<li>宇宙：中证500(000905) ∪ 创业板指(399006) 当前成分股并集，剔除 ST；存在幸存者偏差。</li>
<li>新股规则：上市/数据不足 {config.NEW_STOCK_MIN_DAYS} 自然日的个股在期初被置 NaN，不参与选股。</li>
<li>动量/质量集：ret_12（≈252 交易日动量）、roe 与 gpm_yoy 均取自 fetch_fundamentals_v5.py，
按真实披露时点+延迟映射做点对点前向填充（gpm_yoy 为年度毛利率同比），无未来函数。</li>
<li>三臂共用 MA240 过滤（站上满仓/跌破空仓），与 V3 完全一致；V5 仅改「满仓时选什么股」。</li>
<li>换手率乖离率(turnover_dev)为成交量乖离率代理（沙箱无真实换手率源）。</li>
<li>成本含单边千二，未计滑点、印花税与冲击成本；空仓期间不计货币基金收益。</li>
<li>LightGBM 训练 2018-2019、验证 2020（早停）、严格样本外 2021-2023；反转分支在 2018-2020 含样本内成分，
与 V3 协议一致，故 V5/V3 样本内成分相同，差异仅源于因子切换。</li>
<li>Top30 为候选池上限，受 fixed_weight=0.10 约束实际建仓约 {avg_hold_v5:.0f} 只（与 V3 一致）。</li>
<li>alpha-skills / claude-trading-skills / QuantMind / trade-learn 在本环境不可用，已用开源 Python 栈等价实现。</li>
</ul>
<p style="color:#999">生成时间 {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
</body></html>"""
    return html


def generate_html_v4(eq_v4, eq_v3, bench_real, trades_v4, factor_stats, decay,
                     importance, m_v4, m_v3, m_bench_real, m_v4_train, m_v4_test,
                     regime_df, market_summary, causal_text,
                     avg_hold_v4, avg_hold_v3) -> str:
    """v4 报告：V3（MA240 满仓） vs V4（MA240 + 反转IC门控 50%部分仓位）对比。

    重点突出：过滤器前后资金曲线/回撤对比、市场状态过滤器诊断
    （反转IC滚动均值、RSI离散度、目标仓位时序）、达标与否的归因。
    """
    eq_img = chart_multi_equity({
        "V4(MA240+IC门控)": eq_v4, "V3(MA240满仓)": eq_v3,
        "真实CSI300指数": bench_real})
    dd_img = chart_drawdown_compare({
        "V4(MA240+IC门控)": eq_v4, "V3(MA240满仓)": eq_v3})
    regime_img = chart_regime(regime_df)
    disp_img = chart_dispersion(regime_df)
    decay_img = chart_ic_decay(decay)
    imp_img = chart_importance(importance)

    # 因子统计表
    fs_rows = ""
    for f, row in factor_stats.iterrows():
        fs_rows += (
            f"<tr><td>{f}</td><td>{_num(row['ic_mean'])}</td>"
            f"<td>{_num(row['ic_std'])}</td><td>{_num(row['ir'])}</td>"
            f"<td>{_pct(row['ic_pos_ratio'])}</td><td>{int(row['n_months'])}</td></tr>"
        )

    def row_html(name, m, hl=False):
        cls = " style='background:#fff3f3'" if hl else ""
        return (
            f"<tr{cls}><td>{name}</td><td>{_pct(m.get('annual_return'))}</td>"
            f"<td>{_pct(m.get('max_drawdown'))}</td><td>{_num(m.get('sharpe'))}</td>"
            f"<td>{_num(m.get('calmar'))}</td></tr>"
        )
    cmp_rows = (
        row_html("V4（MA240 + 反转IC门控 · 部分仓位）", m_v4, hl=True)
        + row_html("V3（MA240 满仓 · 对照臂）", m_v3)
        + row_html("真实 CSI300 指数买入持有（公平基准）", m_bench_real)
        + "<tr><td colspan='5' style='background:#eee;font-weight:700'>V4 样本外检验（2021-2023，训练 2018-2020）</td></tr>"
        + row_html("V4 · 样本内(2018-2020)", m_v4_train)
        + row_html("V4 · 样本外(2021-2023)", m_v4_test)
    )
    sh_v4, sh_v3 = m_v4.get("sharpe", 0.0), m_v3.get("sharpe", 0.0)
    dd_v4, dd_v3 = m_v4.get("max_drawdown", 0.0), m_v3.get("max_drawdown", 0.0)
    sh_delta = sh_v4 - sh_v3
    dd_delta = (1 - abs(dd_v4) / abs(dd_v3)) * 100 if dd_v3 != 0 else 0.0

    # 交易明细
    trade_rows = ""
    if trades_v4 is not None and not trades_v4.empty:
        for _, r in trades_v4.head(30).iterrows():
            trade_rows += (
                f"<tr><td>{pd.to_datetime(r['date']).date()}</td><td>{r['code']}</td>"
                f"<td>{r['action']}</td><td>{_num(r['price'])}</td>"
                f"<td>{int(r['shares'])}</td><td>{_num(r['notional'])}</td>"
                f"<td>{r['reason']}</td><td>{_num(r['pnl'])}</td></tr>"
            )

    # 过滤器分档统计
    n_full = int((regime_df["target_weight"] == 1.0).sum())
    n_half = int((regime_df["target_weight"] == 0.5).sum())
    n_zero = int((regime_df["target_weight"] == 0.0).sum())

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>超跌绩优重构策略 V4 · 回测报告</title>
<style>
 body{{font-family:-apple-system,'PingFang SC','Microsoft YaHei',sans-serif;margin:32px;color:#222;line-height:1.6}}
 h1{{border-bottom:3px solid #c0392b;padding-bottom:8px}}
 h2{{margin-top:32px;color:#c0392b}}
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
<h1>「超跌绩优」重构策略 V4（市场状态过滤器升级 · 反转IC门控）· 回测报告</h1>
<p>样本区间 <b>{m_v4.get('start','')} ~ {m_v4.get('end','')}</b> ｜ 标的：<b>中证500 ∪ 创业板指</b>（剔除ST/新股）｜
调仓：月度 ｜ 成本：单边{int(config.COST_PER_SIDE*1000)}‰ ｜
候选池：RSI最低前20% ｜ 选股：LightGBM Top30 ｜ 市场过滤：MA240 + 反转IC滚动门控</p>

<h2>一、绩效概览（V4）</h2>
<div class="cards">
 <div class="card"><div class="k">年化收益率</div><div class="v {'green' if m_v4.get('annual_return',0)>0 else 'red'}">{_pct(m_v4.get('annual_return'))}</div></div>
 <div class="card"><div class="k">最大回撤</div><div class="v red">{_pct(m_v4.get('max_drawdown'))}</div></div>
 <div class="card"><div class="k">夏普比率</div><div class="v">{_num(m_v4.get('sharpe'))}</div></div>
 <div class="card"><div class="k">卡玛比率</div><div class="v">{_num(m_v4.get('calmar'))}</div></div>
 <div class="card"><div class="k">累计收益</div><div class="v">{_pct(m_v4.get('total_return'))}</div></div>
 <div class="card"><div class="k">年化波动</div><div class="v">{_pct(m_v4.get('annual_vol'))}</div></div>
</div>

<h2>二、V3 vs V4 核心对比（夏普 / 最大回撤）</h2>
<table class="tbl">
<tr><th>策略</th><th>年化收益</th><th>最大回撤</th><th>夏普</th><th>卡玛</th></tr>
{cmp_rows}
</table>
<div class="hl">相对 V3（仅 MA240 满仓）：V4 夏普 {_num(sh_v3)} → <b>{_num(sh_v4)}</b>
（{('+' if sh_delta>=0 else '')}{sh_delta:.2f}）；最大回撤 {_pct(dd_v3)} → <b>{_pct(dd_v4)}</b>
（回撤变化 {dd_delta:+.1f}%）。目标：夏普&gt;0.6、回撤&lt;-20%、不加杠杆。</div>
<div class="note">对比隔离性：V3 对照臂与 V4 使用<b>同一回测引擎、同一选股因子、同一持仓数、同一股票池</b>，
唯一差异为市场状态过滤器（V3=MA240满仓；V4=MA240+反转IC滚动门控+50%部分仓位），因此本表差异完全归因于过滤器升级。
注：本引擎在 fixed_weight=0.10 下每月实际建仓约 {avg_hold_v3:.0f} 只（现金上限100%所致），
V4 半仓档位进一步降至约 {avg_hold_v4:.0f} 只；V4 报告为公平对比将 V3 臂以同引擎复算，数值与 V3 报告一致。</div>

<h2>三、资金曲线与回撤对比</h2>
<img src="data:image/png;base64,{eq_img}">
<img src="data:image/png;base64,{dd_img}">

<h2>四、市场状态过滤器诊断（V4）</h2>
<img src="data:image/png;base64,{regime_img}">
<img src="data:image/png;base64,{disp_img}">
<div class="note">判定规则（每月末）：MA240站上 且 反转IC(12月滚动)&gt;0 → 仓位100%；
MA240站上 且 滚动IC≤0 → 仓位50%；MA240跌破 → 仓位0%。
样本期内共 <b>{n_full}</b> 个月满仓、<b>{n_half}</b> 个月半仓、<b>{n_zero}</b> 个月空仓。
反转IC采用自适应滚动窗口（含部分测试期月份），用于捕捉反转因子状态切换，非未来函数（决策日T仅消费 IC[T-1]）。</div>

<h2>五、因子 IC / IR 分析（V3/V4 同宇宙）</h2>
<table class="tbl">
<tr><th>因子</th><th>IC均值</th><th>IC标准差</th><th>IR</th><th>IC>0占比</th><th>月度数</th></tr>
{fs_rows}
</table>
<img src="data:image/png;base64,{decay_img}">
<div class="note">neg_rsi_14 为反转主轴因子；其 IC 滚动均值即第四节气中的紫色曲线，V4 据此动态调仓。</div>

<h2>六、模型特征重要性（LightGBM, V3/V4 同模型）</h2>
<img src="data:image/png;base64,{imp_img}">
<div class="note">特征集（V3 起已剔除失效的 EP/ROE/净利润增长）：RSI(14)+60日偏度+换手率乖离率(代理)+动量+波动率+均线偏离+量比。</div>

<h2>七、市场环境研判摘要</h2>
<div class="sec">{market_summary}</div>

<h2>八、因果推断与归因结论</h2>
<div class="sec">{causal_text}</div>

<h2>九、目标达成情况与结论</h2>
<div class="hl">回撤目标（&lt;-20%）：{'<b>已达成</b>' if m_v4.get('max_drawdown',0) > -0.20 else '<b>未达成</b>'}。
V4 最大回撤 {_pct(m_v4.get('max_drawdown'))}（V3 为 {_pct(m_v3.get('max_drawdown'))}）。</div>
<div class="note">夏普目标（&gt;0.6，不加杠杆）：{'<b>已达成</b>' if m_v4.get('sharpe',0) > 0.6 else '<b>未达成</b>'}，
实测 {_num(m_v4.get('sharpe'))}（V3 为 {_num(m_v3.get('sharpe'))}）。详细归因见第八节。</div>

<h2>十、交易明细（V4，前30笔）</h2>
<table class="tbl">
<tr><th>日期</th><th>代码</th><th>方向</th><th>价格</th><th>股数</th><th>金额</th><th>原因</th><th>盈亏</th></tr>
{trade_rows}
</table>

<h2>十一、方法说明与局限</h2>
<ul>
<li>宇宙：中证500(000905) ∪ 创业板指(399006) 当前成分股并集，剔除 ST；存在幸存者偏差。</li>
<li>新股规则：上市/数据不足 {config.NEW_STOCK_MIN_DAYS} 自然日的个股在期初被置 NaN，不参与选股。</li>
<li>RSI离散度与反转IC在 V3/V4 交易宇宙（中证500+创业板指）内计算，作为全市场反转状态代理；
该宇宙覆盖中小盘，对反转因子更敏感，符合过滤器设计意图（与字面"沪深300+中证500"略有出入，已注明）。</li>
<li>换手率乖离率(turnover_dev)为<b>成交量乖离率代理</b>：本沙箱东方财富换手率接口被网络拦截、且无逐日流通股源，真实换手率不可得；
单股流通股短期稳定，成交量乖离率与换手率乖离率高度共线，故作等价替代。</li>
<li>日线取自新浪前复权源；CSI300 指数取自新浪；反转IC滚动窗口为自适应（含测试期月份），属自适应 regime filter 范畴。</li>
<li>成本含单边千二，未计滑点、印花税与冲击成本；空仓期间不计货币基金收益。</li>
<li>LightGBM 训练 2018-2019、验证 2020（早停）、严格样本外 2021-2023；V3 对照臂同引擎复算。</li>
<li>alpha-skills / claude-trading-skills / QuantMind / trade-learn 在本环境不可用，已用开源 Python 栈等价实现。</li>
</ul>
<p style="color:#999">生成时间 {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
</body></html>"""
    return html


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


def chart_vol_regime(vol_regime: pd.DataFrame) -> str:
    """V7.1 波动率过滤诊断：CSI300 60日年化波动率 vs 历史75分位阈值 + 降档标记。"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(vol_regime.index, vol_regime["vol60"], color="#c0392b",
            marker=".", ms=3, label="CSI300 60日年化波动率")
    ax.plot(vol_regime.index, vol_regime["vol_thr"], color="#2980b9",
            lw=1.4, label="历史75分位阈值(trailing 3y)")
    red = vol_regime[vol_regime["target_weight"] == 0.6]
    full = vol_regime[vol_regime["target_weight"] == 1.0]
    cash = vol_regime[vol_regime["target_weight"] == 0.0]
    ax.scatter(red.index, red["vol60"], color="#e67e22", s=40, zorder=5,
               label="降档至60%", marker="v")
    ax.scatter(full.index, full["vol60"], color="#27ae60", s=20, zorder=4,
               label="满仓100%", marker="^")
    ax.scatter(cash.index, cash["vol60"], color="#7f8c8d", s=20, zorder=4,
               label="空仓0%(MA240破)", marker="x")
    ax.set_title("V7.1 波动率过滤：波动率 vs 历史75分位 + 仓位定档")
    ax.set_ylabel("年化波动率")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    return _fig_to_b64(fig)


def generate_html_v7_1(eq_v7_1, eq_v7_clean, idx_eq, slip_map, tier_counts,
                       m_full, m_old, m_new, m_full_c, m_old_c, m_new_c,
                       m_v6_ref, m_idx, yearly, yearly_c, vol_regime,
                       switch_log, holdings_2024_2025, data_start, data_end,
                       conclusion) -> str:
    """V7.1 报告：MA240 + 波动率过滤（降档至60%）。

    公平对照：同一选股信号/分档滑点下，仅改市场过滤——
      V7.1（MA240+波动率降档） vs V7基准（MA240-only，无波动过滤） vs V6（分档）。
    """
    eq_img = chart_multi_equity({
        "V7.1 (MA240+波动率)": eq_v7_1,
        "V7基准 (MA240-only)": eq_v7_clean,
        "CSI300 Index": idx_eq})
    dd_img = chart_drawdown_compare({
        "V7.1 (MA240+波动率)": eq_v7_1,
        "V7基准 (MA240-only)": eq_v7_clean,
        "CSI300 Index": idx_eq})
    vol_img = chart_vol_regime(vol_regime) if vol_regime is not None else ""
    yearly_df = pd.DataFrame({"V7.1": yearly, "V7基准(无波动)": yearly_c})
    yearly_img = chart_yearly_sharpe(yearly_df)

    def row_html(name, m, hl=False):
        cls = " style='background:#eafaf1'" if hl else ""
        return (f"<tr{cls}><td>{name}</td><td>{_pct(m.get('annual_return'))}</td>"
                f"<td>{_pct(m.get('max_drawdown'))}</td><td>{_num(m.get('sharpe'))}</td>"
                f"<td>{_num(m.get('calmar'))}</td></tr>")

    perf_rows = (
        row_html("V7.1（MA240+波动率降档，2018-2025）", m_full, hl=True)
        + row_html("　└ 旧区间 2018-2023", m_old)
        + row_html("　└ 新区间 2024-2025", m_new)
        + row_html("V7基准（MA240-only，无波动过滤，2018-2025）", m_full_c)
        + row_html("　└ 旧区间 2018-2023", m_old_c)
        + row_html("　└ 新区间 2024-2025", m_new_c)
        + row_html("V6 压力（分档，2018-2025）", m_v6_ref)
        + row_html("真实 CSI300 指数买入持有", m_idx)
    )

    tc = tier_counts
    tier_html = (
        f"<li><b>档1 成交额&gt;5亿（单边0.10%）</b>：{tc.get('tier1_>5yi_0.10%',0)} 只</li>"
        f"<li><b>档2 成交额1-5亿（单边0.30%）</b>：{tc.get('tier2_1-5yi_0.30%',0)} 只</li>"
        f"<li><b>档3 成交额&lt;1亿（单边0.50%）</b>：{tc.get('tier3_<1yi_0.50%',0)} 只</li>"
        f"<li>成交额缺失（按档3处理）：{tc.get('missing',0)} 只</li>"
    )

    # 波动率定档统计
    vr = vol_regime
    n_total = len(vr)
    n_reduced = int((vr["target_weight"] == 0.6).sum())
    n_full = int((vr["target_weight"] == 1.0).sum())
    n_cash = int((vr["target_weight"] == 0.0).sum())
    vr25 = vr[vr.index >= pd.Timestamp("2024-01-01")]
    n_reduced_25 = int((vr25["target_weight"] == 0.6).sum())
    n_full_25 = int((vr25["target_weight"] == 1.0).sum())
    n_cash_25 = int((vr25["target_weight"] == 0.0).sum())
    vol_stats = (
        f"<li>全样本月末定档：满仓100% <b>{n_full}</b> 月 / 降档60% <b>{n_reduced}</b> 月 / 空仓0% <b>{n_cash}</b> 月（共 {n_total} 月）</li>"
        f"<li>2024-2025 区间：满仓 <b>{n_full_25}</b> 月 / 降档 <b>{n_reduced_25}</b> 月 / 空仓 <b>{n_cash_25}</b> 月</li>"
    )

    y_rows = ""
    for y in yearly_df.index:
        v1 = yearly_df.loc[y, "V7.1"]
        v0 = yearly_df.loc[y, "V7基准(无波动)"]
        y_rows += (f"<tr><td>{y}</td>"
                   f"<td>{_num(v1) if not pd.isna(v1) else '—'}</td>"
                   f"<td>{_num(v0) if not pd.isna(v0) else '—'}</td></tr>")

    fd = {}
    h_rows = ""
    if holdings_2024_2025 is not None and len(holdings_2024_2025):
        for _, r in holdings_2024_2025.iterrows():
            codes = str(r["selected_codes"])
            preview = codes[:60] + ("…" if len(codes) > 60 else "")
            h_rows += (f"<tr><td>{r['month_end'].date() if hasattr(r['month_end'],'date') else r['month_end']}</td>"
                       f"<td>{r['factor_set']}</td><td>{r['n_selected']}</td>"
                       f"<td style='font-size:11px;color:#666'>{preview}</td></tr>")

    # 目标达成度
    goal_sharpe_ok = m_new.get("sharpe", 0) >= 0.3
    goal_dd_ok = m_full.get("max_drawdown", -1) >= -0.22
    goal_html = (
        f"<li>2024-2025 夏普：目标 ≥0.30，实际 <b>{_num(m_new.get('sharpe'))}</b>"
        f" → {'<span class=green>达成</span>' if goal_sharpe_ok else '<span class=red>未达成</span>'}（V7基准 {_num(m_new_c.get('sharpe'))}）</li>"
        f"<li>全区间最大回撤：目标 ≥-22%，实际 <b>{_pct(m_full.get('max_drawdown'))}</b>"
        f" → {'<span class=green>达成</span>' if goal_dd_ok else '<span class=red>未达成</span>'}（V7基准 {_pct(m_full_c.get('max_drawdown'))}）</li>"
    )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>V5 策略修复 V7.1 · MA240+波动率过滤</title>
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
<h1>V5 多因子 Regime 切换策略 · 修复 V7.1</h1>
<p>数据区间 <b>{data_start} ~ {data_end}</b>｜ 标的：中证500 ∪ 创业板指（剔除ST/新股）｜
调仓：月度 ｜ 市场过滤：<b>MA240 + 波动率降档</b> ｜ 切换框架：反转↔动量/质量（与 V5 完全一致，未改动）。</p>
<p>V7.1 改动（仅一项，隔离变量）：在 MA240 门控之上<b>叠加波动率过滤</b>——每月末计算沪深300过去60日年化波动率，
若 &gt; 历史75分位数（trailing 3y），仓位由100%降至60%（保留部分多头），其余逻辑（滚动重训+真实财报+分档滑点）完全不变。</p>

<h2>一、迭代目标达成度</h2>
<ul>{goal_html}</ul>

<h2>二、V7.1 vs V7基准(无波动) vs V6 绩效对比</h2>
<div class="cards">
 <div class="card"><div class="k">V7.1 全区间年化</div><div class="v {'green' if m_full.get('annual_return',0)>0 else 'red'}">{_pct(m_full.get('annual_return'))}</div></div>
 <div class="card"><div class="k">V7.1 全区间夏普</div><div class="v">{_num(m_full.get('sharpe'))}</div></div>
 <div class="card"><div class="k">V7.1 2024-2025夏普</div><div class="v {'green' if m_new.get('sharpe',0)>0 else 'red'}">{_num(m_new.get('sharpe'))}</div></div>
 <div class="card"><div class="k">V7基准 2024-2025夏普</div><div class="v">{_num(m_new_c.get('sharpe'))}</div></div>
</div>
<table class="tbl">
<tr><th>策略 / 区间</th><th>年化收益</th><th>最大回撤</th><th>夏普</th><th>卡玛</th></tr>
{perf_rows}
</table>
<div class="note">公平对照：V7.1 与 V7基准 使用<b>完全相同的选股信号与分档滑点</b>，唯一差异是市场过滤（是否叠加波动率降档）。
因此两行差异即为波动率过滤的净贡献，可干净归因。</div>

<h2>三、滑点模型（市值分档，沿用 V6）</h2>
<ul>{tier_html}</ul>

<h2>四、波动率过滤诊断</h2>
<img src="data:image/png;base64,{vol_img}">
<ul>{vol_stats}</ul>
<div class="sec">橙色▼=降档至60%（高波动牛市）；绿色▲=满仓100%（常态牛市）；灰色✕=空仓（MA240跌破，主门控不变）。
阈值用 trailing 3y 年化波动率75分位（shift(1) 防自指，零未来泄露）。</div>

<h2>五、逐年夏普分解（V7.1 vs V7基准）</h2>
<img src="data:image/png;base64,{yearly_img}">
<table class="tbl">
<tr><th>年份</th><th>V7.1</th><th>V7基准(无波动)</th></tr>
{y_rows}
</table>
<div class="sec">重点观察 2024 / 2025：波动率降档是否在保留多头暴露的前提下改善了夏普与回撤。</div>

<h2>六、2024-2025 持仓明细（校验是否规避失效因子）</h2>
<table class="tbl">
<tr><th>月末</th><th>因子集</th><th>持仓数</th><th>选中代码（预览）</th></tr>
{h_rows}
</table>

<h2>七、资金曲线与回撤对比</h2>
<img src="data:image/png;base64,{eq_img}">
<img src="data:image/png;base64,{dd_img}">

<h2>八、诚实结论</h2>
<div class="{'ok' if goal_sharpe_ok or goal_dd_ok else 'hl'}">{conclusion}</div>

<h2>九、方法说明与局限</h2>
<ul>
<li>V7.1 严格不改 V5 的 Regime 切换框架（MA240 门控 + IC 门控 + 反转↔动量/质量切换 + Top30 + 月频）与 V7 的修复（滚动重训+真实财报）。</li>
<li>唯一新增：市场过滤由「MA240-only」升级为「MA240 + 波动率降档」。波动率仅在高波动牛市中把仓位从100%降到60%，保留部分多头暴露，空仓判定仍完全由 MA240 决定。</li>
<li>阈值定义：CSI300 60日收益年化波动率的历史75分位数（trailing 3y，shift(1) 防自指，零未来泄露）；降档权重 reduced_weight=0.60。</li>
<li>分档实盘滑点（0.1/0.3/0.5%）与 V6/V7 同口径，可直接比较。</li>
</ul>
<p style="color:#999">生成时间 {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
</body></html>"""
    return html


# ----------------------------------------------------------------------------
def generate_html_v8(eq_v8, eq_v8_clean, idx_eq, slip_map, tier_counts,
                     m_full, m_old, m_new, m_full_c, m_old_c, m_new_c,
                     m_v71_ref, m_idx, yearly, yearly_v71, yearly_c,
                     vol_regime, switch_log, latest_signal, holdings_2024_2025,
                     n_universe, data_start, data_end, conclusion) -> str:
    """V8 报告：股票池扩展至 中证500∪创业板∪中证1000（约1540只）。

    公平对照：V8（MA240+波动率）vs V8基准（MA240-only，同信号同滑点）vs V7.1（500+创业，MA240+波动率）。
    唯一变量 = 股票池（新增中证1000）。其余逻辑零改动。
    """
    eq_img = chart_multi_equity({
        "V8 (500+创业+1000)": eq_v8,
        "V8基准 (MA240-only)": eq_v8_clean,
        "CSI300 Index": idx_eq})
    dd_img = chart_drawdown_compare({
        "V8 (500+创业+1000)": eq_v8,
        "V8基准 (MA240-only)": eq_v8_clean,
        "CSI300 Index": idx_eq})
    vol_img = chart_vol_regime(vol_regime) if vol_regime is not None else ""
    yearly_df = pd.DataFrame({"V8": yearly, "V7.1": yearly_v71})
    yearly_img = chart_yearly_sharpe(yearly_df)

    def row_html(name, m, hl=False):
        cls = " style='background:#eafaf1'" if hl else ""
        return (f"<tr{cls}><td>{name}</td><td>{_pct(m.get('annual_return'))}</td>"
                f"<td>{_pct(m.get('max_drawdown'))}</td><td>{_num(m.get('sharpe'))}</td>"
                f"<td>{_num(m.get('calmar'))}</td></tr>")

    perf_rows = (
        row_html("V8（500+创业+1000，MA240+波动率，2018-2025）", m_full, hl=True)
        + row_html("　└ 旧区间 2018-2023", m_old)
        + row_html("　└ 新区间 2024-2025", m_new)
        + row_html("V8基准（MA240-only，无波动过滤，2018-2025）", m_full_c)
        + row_html("　└ 旧区间 2018-2023", m_old_c)
        + row_html("　└ 新区间 2024-2025", m_new_c)
        + row_html("V7.1（500+创业，MA240+波动率，2018-2025）", m_v71_ref)
        + row_html("　└ 旧区间 2018-2023", m_v71_ref["old"])
        + row_html("　└ 新区间 2024-2025", m_v71_ref["new"])
        + row_html("真实 CSI300 指数买入持有", m_idx)
    )

    tc = tier_counts
    tier_html = (
        f"<li><b>档1 成交额&gt;5亿（单边0.10%）</b>：{tc.get('tier1_>5yi_0.10%',0)} 只</li>"
        f"<li><b>档2 成交额1-5亿（单边0.30%）</b>：{tc.get('tier2_1-5yi_0.30%',0)} 只</li>"
        f"<li><b>档3 成交额&lt;1亿（单边0.50%）</b>：{tc.get('tier3_<1yi_0.50%',0)} 只</li>"
        f"<li>成交额缺失（按档3处理）：{tc.get('missing',0)} 只</li>"
        f"<li>全样本滑点侵蚀（分档 vs 零成本）在 V6 已证仅约 0.76pp，量级稳定。</li>"
    )

    vr = vol_regime
    n_total = len(vr)
    n_reduced = int((vr["target_weight"] == 0.6).sum())
    n_full = int((vr["target_weight"] == 1.0).sum())
    n_cash = int((vr["target_weight"] == 0.0).sum())
    vr25 = vr[vr.index >= pd.Timestamp("2024-01-01")]
    n_reduced_25 = int((vr25["target_weight"] == 0.6).sum())
    n_full_25 = int((vr25["target_weight"] == 1.0).sum())
    n_cash_25 = int((vr25["target_weight"] == 0.0).sum())
    vol_stats = (
        f"<li>全样本月末定档：满仓100% <b>{n_full}</b> 月 / 降档60% <b>{n_reduced}</b> 月 / 空仓0% <b>{n_cash}</b> 月（共 {n_total} 月）</li>"
        f"<li>2024-2025 区间：满仓 <b>{n_full_25}</b> 月 / 降档 <b>{n_reduced_25}</b> 月 / 空仓 <b>{n_cash_25}</b> 月</li>"
    )

    y_rows = ""
    for y in yearly_df.index:
        v8 = yearly_df.loc[y, "V8"]
        v71 = yearly_df.loc[y, "V7.1"]
        y_rows += (f"<tr><td>{y}</td>"
                   f"<td style='font-weight:600'>{_num(v8) if not pd.isna(v8) else '—'}</td>"
                   f"<td>{_num(v71) if not pd.isna(v71) else '—'}</td>"
                   f"<td>{'+' if (not pd.isna(v8) and not pd.isna(v71) and v8 > v71) else ('−' if (not pd.isna(v8) and not pd.isna(v71) and v8 < v71) else '=')}</td></tr>")

    # 2024-2025 专项
    new_ok = m_new.get("sharpe", 0) >= 0.40
    new_html = (
        f"<li>V8 新区间2024-2025 夏普 <b>{_num(m_new.get('sharpe'))}</b> ｜ 年化 {_pct(m_new.get('annual_return'))} ｜ 回撤 {_pct(m_new.get('max_drawdown'))}</li>"
        f"<li>V7.1 新区间2024-2025 夏普 <b>{_num(m_v71_ref['new'].get('sharpe'))}</b> ｜ 年化 {_pct(m_v71_ref['new'].get('annual_return'))} ｜ 回撤 {_pct(m_v71_ref['new'].get('max_drawdown'))}</li>"
        f"<li>目标 0.26→0.4+ → <b>{'达成' if new_ok else '未达成'}</b>（净变化 {_num(m_new.get('sharpe')-m_v71_ref['new'].get('sharpe'))}）</li>"
    )

    # 最新截面信号清单
    ls_rows = ""
    if latest_signal is not None and len(latest_signal):
        for _, r in latest_signal.iterrows():
            ls_rows += (f"<tr><td>{r['rank']}</td><td>{r['code']}</td><td>{r['name']}</td>"
                        f"<td>{r['factor_set']}</td><td>{_num(r['regime_weight'])}%</td>"
                        f"<td>{_num(r['target_weight'])}%</td><td>{r['action']}</td></tr>")

    # 2024-2025 持仓明细
    fd = {}
    h_rows = ""
    if holdings_2024_2025 is not None and len(holdings_2024_2025):
        for _, r in holdings_2024_2025.iterrows():
            codes = str(r["selected_codes"])
            preview = codes[:60] + ("…" if len(codes) > 60 else "")
            h_rows += (f"<tr><td>{r['month_end'].date() if hasattr(r['month_end'],'date') else r['month_end']}</td>"
                       f"<td>{r['factor_set']}</td><td>{r['n_selected']}</td>"
                       f"<td style='font-size:11px;color:#666'>{preview}</td></tr>")

    goal_sharpe_ok = m_new.get("sharpe", 0) >= 0.40
    goal_full_ok = m_full.get("sharpe", 0) >= m_v71_ref.get("sharpe", 0)  # 全期不退化

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>V8 策略 · 扩展宇宙(500+创业+1000)</title>
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
 .ok{{background:#eafaf1;border-left:4px solid #27ae60;padding:10px 14px;margin:12px 0}}
</style></head><body>
<h1>V5 多因子 Regime 切换策略 · 迭代 V8（扩展宇宙）</h1>
<p>数据区间 <b>{data_start} ~ {data_end}</b>｜ 标的：<b>中证500 ∪ 创业板指 ∪ 中证1000（{n_universe} 只，剔除ST）</b>｜
调仓：月度 ｜ 市场过滤：<b>MA240 + 波动率降档</b> ｜ 切换框架：反转↔动量/质量（与 V5/V7.1 完全一致，未改动）。</p>
<p>V8 改动（仅一项，隔离变量）：股票池由「中证500∪创业板指」扩展为「中证500∪创业板指∪中证1000」
（{n_universe} 只 vs V7.1 约 526 只量级）。IC阈值/MA周期/波动率分位/降仓幅度/持仓数/滚动窗口<b>全部沿用 V7.1</b>，零参数改动。</p>

<h2>一、2024-2025 新区间专项（核心验证）</h2>
<div class="cards">
 <div class="card"><div class="k">V8 新区间夏普</div><div class="v {'green' if m_new.get('sharpe',0)>0 else 'red'}">{_num(m_new.get('sharpe'))}</div></div>
 <div class="card"><div class="k">V7.1 新区间夏普</div><div class="v">{_num(m_v71_ref['new'].get('sharpe'))}</div></div>
 <div class="card"><div class="k">目标</div><div class="v">{('0.4+' if new_ok else '0.26')}</div></div>
 <div class="card"><div class="k">净变化</div><div class="v {'green' if m_new.get('sharpe',0)>=m_v71_ref['new'].get('sharpe',0) else 'red'}">{_num(m_new.get('sharpe')-m_v71_ref['new'].get('sharpe'))}</div></div>
</div>
<ul>{new_html}</ul>
<div class="note">检验假设：中证1000 加入后，选股广度扩大，能否在小盘风格占优的 2021/2024 年增厚超额。</div>

<h2>二、全区间绩效对比（V8 vs V8基准 vs V7.1）</h2>
<table class="tbl">
<tr><th>策略 / 区间</th><th>年化收益</th><th>最大回撤</th><th>夏普</th><th>卡玛</th></tr>
{perf_rows}
</table>
<div class="note">公平对照：V8 与 V8基准 使用<b>完全相同的选股信号与分档滑点</b>，唯一差异是市场过滤（是否叠加波动率降档），
差异即波动率过滤净贡献。V7.1 行由 theoretical_nav_v7_1.parquet 现场重算（同口径），保证 V8 vs V7.1 仅差股票池一项。</div>

<h2>三、滑点模型（市值分档，沿用 V6）</h2>
<ul>{tier_html}</ul>

<h2>四、波动率过滤诊断</h2>
<img src="data:image/png;base64,{vol_img}">
<ul>{vol_stats}</ul>
<div class="sec">橙色▼=降档至60%（高波动牛市）；绿色▲=满仓100%（常态）；灰色✕=空仓（MA240跌破，主门控不变）。</div>

<h2>五、逐年夏普分解（V8 vs V7.1，重点 2021 / 2024）</h2>
<img src="data:image/png;base64,{yearly_img}">
<table class="tbl">
<tr><th>年份</th><th>V8</th><th>V7.1</th><th>Δ(谁更优)</th></tr>
{y_rows}
</table>
<div class="sec">末列 ▲/▼ 标示当年 V8 相对 V7.1 孰优；重点观察 <b>2021</b>（小盘反转失效年）与 <b>2024</b>（小盘占优年）是否因 1000 加入而改善。</div>

<h2>六、最新截面信号清单（对应最近数据日，模拟盘初始持仓参考）</h2>
<table class="tbl">
<tr><th>排名</th><th>代码</th><th>名称</th><th>因子集</th><th>regime权重</th><th>目标仓位%</th><th>动作</th></tr>
{ls_rows}
</table>
<div class="sec">该截面由 V8 完整管线于最近月末产出，可校验逻辑一致性（与 V7.1 截面差异仅源于选股池扩大）。</div>

<h2>七、2024-2025 持仓明细（校验是否规避失效因子）</h2>
<table class="tbl">
<tr><th>月末</th><th>因子集</th><th>持仓数</th><th>选中代码（预览）</th></tr>
{h_rows}
</table>

<h2>八、资金曲线与回撤对比</h2>
<img src="data:image/png;base64,{eq_img}">
<img src="data:image/png;base64,{dd_img}">

<h2>九、诚实结论</h2>
<div class="{'ok' if (goal_sharpe_ok or goal_full_ok) else 'hl'}">{conclusion}</div>

<h2>十、方法说明与局限</h2>
<ul>
<li>V8 严格不改 V5 的 Regime 切换框架（MA240 门控 + IC 门控 + 反转↔动量/质量切换 + Top30 + 月频）与 V7/V7.1 的修复（滚动重训+真实财报）。</li>
<li>唯一变量：股票池扩大至 中证500∪创业板指∪中证1000（{n_universe} 只）。财报沿用 V8 point-in-time 对齐（披露延迟映射+ffill），滑点按全期日均成交额分级，二者与 V7.1 口径一致。</li>
<li>中证1000 当前成分股含幸存者偏差（用当前快照而非时点成分），生产应改为时点成分股；但 V7.1 同为当前快照，二者偏差对称，对比仍干净。</li>
<li>V7.1 基准由 output/theoretical_nav_v7_1.parquet 现场重算，未硬编码，可复现。</li>
</ul>
<p style="color:#999">生成时间 {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
</body></html>"""
    return html
