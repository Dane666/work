# -*- coding: utf-8 -*-
"""
chart_utils.py — 仪表盘增强图表（行业暴露 / 因子暴露 / 回撤归因）
====================================================================

供 generate_dashboard.py 调用，生成三张增强图并嵌入 docs/index.html：
  - docs/assets/industry_exposure.png   持仓行业权重 vs 最新一期行业基准（水平柱状对比）
  - docs/assets/factor_exposure.png     四因子暴露（动量/价值/质量/小市值，组合平均 Z-score）
  - docs/assets/drawdown_attr.png       净值回撤归因（>5% 回撤区间，主要贡献行业）

数据源（data 分支持久化 / 本地面板）：
  - data/state/sim_state.json           当前持仓（code -> shares/cost/entry_date）
  - data/industry_map.parquet           code -> 申万一级行业
  - data/industry_benchmark.parquet     月频行业市值权重基准（index=月末，columns=31 行业）
  - data/mainboard_close_panel.parquet  前复权收盘价（日频，全主板池）
  - data/roe_panel_mainboard.parquet    质量：ROE
  - data/gpm_yoy_panel_mainboard.parquet 质量：毛利率同比
  - data/div_yield_panel_mainboard.parquet 价值：股息率（月频，主板选股池 1004 只）
  - data/industry_capital.parquet       注册资金（万元，静态股本代理）
  - data/state/sim_nav_history.csv      净值历史（回撤识别与归因）

约定：
  - 因子 Z-score 相对「全市场主板池」截面计算（同一日期所有可用股票）。
  - 组合因子暴露 = 持仓市值权重 × 个股 Z-score 加权平均。
  - 图表沿用 A 股惯例：红=超配/正向，绿=低配/负向。
  - 持仓为空 / 数据不足时返回 False，调用方降级（不渲染该图）。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
           "ALL_PROXY", "all_proxy"]:
    os.environ.pop(_k, None)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import config

# matplotlib 后端必须在导入 pyplot 前指定（无 GUI 环境）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# A股惯例配色（红涨绿跌；超配/正偏=红，低配/负偏=绿）
UP_COLOR = "#c0392b"
DOWN_COLOR = "#1e8449"
BENCH_COLOR = "#2471a3"
NEUTRAL = "#95a5a6"

# 图表默认尺寸
FIG_W = 9.5


# ---------------------------------------------------------------------------
# 通用：中文字体可用性（复用 generate_dashboard 的全局设置；此处仅兜底）
# ---------------------------------------------------------------------------
def _cn_ok() -> bool:
    return "sans-serif" in plt.rcParams and any(
        "PingFang" in str(f) or "Noto" in str(f) or "WenQuanYi" in str(f)
        for f in plt.rcParams.get("font.sans-serif", []))


def T(zh: str, en: str = "") -> str:
    return zh if _cn_ok() else (en or "")


# ---------------------------------------------------------------------------
# 持仓市值权重
# ---------------------------------------------------------------------------
def compute_position_weights(positions: dict, close_panel: pd.DataFrame) -> dict:
    """{code: 市值占组合权重}。价格 = close_panel 各 code 最后有效收盘价；
    无价格时回退成本价（cost）。"""
    out: dict = {}
    total = 0.0
    for code, h in positions.items():
        shares = float(h.get("shares", 0.0))
        if shares <= 0:
            continue
        c = str(code)
        price = float("nan")
        if c in close_panel.columns:
            s = close_panel[c].dropna()
            if len(s):
                price = float(s.iloc[-1])
        if not (price == price):
            price = float(h.get("cost", 0.0))
        if price <= 0:
            continue
        mv = shares * price
        out[c] = mv
        total += mv
    if total <= 0:
        return {}
    return {c: mv / total for c, mv in out.items()}


# ---------------------------------------------------------------------------
# 图表1：行业暴露（持仓 vs 基准）
# ---------------------------------------------------------------------------
def plot_industry_exposure(positions: dict, industry_map: pd.DataFrame,
                           bench_df: pd.DataFrame, out_path: Path,
                           top_n: int = 15) -> bool:
    """持仓行业权重 vs 基准最新一期，水平柱状对比图。

    - 持仓行业权重：持仓市值按行业聚合。
    - 基准：industry_benchmark.parquet 最后一行（最新月末行业市值权重）。
    - 按「超配幅度」降序取 top_n 显示；无持仓或数据缺失返回 False。
    """
    if not positions:
        print("[chart_utils] 无持仓，跳过行业暴露图")
        return False
    if bench_df is None or len(bench_df) == 0:
        print("[chart_utils] 无行业基准数据，跳过行业暴露图")
        return False
    imap = {}
    if industry_map is not None and len(industry_map):
        imap = dict(zip(industry_map["code"].astype(str),
                        industry_map["industry"]))
    close = _load_close_panel()
    if close is None:
        return False

    w = compute_position_weights(positions, close)
    if not w:
        print("[chart_utils] 持仓市值权重为空，跳过行业暴露图")
        return False

    # 持仓行业权重
    pos_ind: dict = {}
    for code, wt in w.items():
        ind = imap.get(code, "其他")
        pos_ind[ind] = pos_ind.get(ind, 0.0) + wt

    bench = bench_df.iloc[-1].dropna()          # 最新一期基准
    bench_map = {str(k): float(v) for k, v in bench.items()}
    all_ind = sorted(set(pos_ind) | set(bench_map))
    # 取「持仓涉及行业 ∪ 基准 Top 行业」中按超配幅度排序的 top_n
    rows = []
    for ind in all_ind:
        rows.append((ind, pos_ind.get(ind, 0.0), bench_map.get(ind, 0.0)))
    rows.sort(key=lambda r: (r[1] - r[2]), reverse=True)   # 按超配降序
    rows = rows[:top_n]
    rows = rows[::-1]                            # 画图自下而上

    labels = [r[0] for r in rows]
    pos_w = [r[1] * 100 for r in rows]
    ben_w = [r[2] * 100 for r in rows]
    excess = [(p - b) for p, b in zip(pos_w, ben_w)]

    fig, ax = plt.subplots(figsize=(FIG_W, max(4.5, 0.34 * len(rows))), dpi=92)
    y = np.arange(len(rows))
    ax.barh(y - 0.19, pos_w, height=0.34, color=UP_COLOR,
            label=T("持仓", "Portfolio"))
    ax.barh(y + 0.19, ben_w, height=0.34, color=BENCH_COLOR,
            label=T("基准(最新月末)", "Benchmark"))
    # 超配/低配标注
    for i, (p, b) in enumerate(zip(pos_w, ben_w)):
        ax.annotate(f"{p - b:+.1f}pp", xy=(max(p, b), y[i]),
                    xytext=(2, 0), textcoords="offset points",
                    fontsize=8, va="center",
                    color=UP_COLOR if p - b >= 0 else DOWN_COLOR)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("%", fontsize=9)
    ax.set_title(T("行业暴露：持仓 vs 基准（超配/低配 pp）",
                   "Industry Exposure: Portfolio vs Benchmark"),
                 fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.6)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[chart_utils] ✓ 行业暴露图 → {out_path}（{len(rows)} 行业）")
    return True


# ---------------------------------------------------------------------------
# 图表2：因子暴露（动量/价值/质量/小市值）
# ---------------------------------------------------------------------------
def _load_close_panel() -> pd.DataFrame | None:
    p = config.MB_CLOSE if getattr(config, "USE_MAINBOARD", False) else None
    p = p or config.DATA_DIR / "mainboard_close_panel.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df.columns = df.columns.astype(str)
    return df


def _load_factor_data() -> dict:
    """加载因子原始数据（主板全池截面，末日/最新月）。

    返回 {factor: pd.Series(index=code, value)}，其中 value 为原始值：
      - momentum : ret_12 = close_last / close_252d_ago - 1
      - value    : 股息率最新月（div_yield_panel_mainboard，1004 只）
      - quality  : (z(roe) + z(gpm_yoy)) / 2 —— 此处先返回 roe 与 gpm 原始值
      - size     : -log(市值)（市值 = 注册资金万 × 1e4 × 最新收盘价）
    """
    close = _load_close_panel()
    if close is None:
        return {}
    last = close.iloc[-1].dropna()
    if len(last) >= 253:
        prev = close.iloc[-253].dropna()
        ret12 = (last / prev - 1.0).dropna()
    else:
        ret12 = last * np.nan
    out = {"momentum": ret12}

    def _panel_series(name: str, pick: str = "last"):
        p = config.DATA_DIR / name
        if not p.exists():
            return None
        df = pd.read_parquet(p)
        df.columns = df.columns.astype(str)
        if pick == "last":
            s = df.iloc[-1].dropna()
        else:
            s = df.dropna(how="all").iloc[-1].dropna()
        return s.astype(float)

    # 质量：roe + gpm_yoy（各取最新有效值）
    roe = _panel_series("roe_panel_mainboard.parquet")
    gpm = _panel_series("gpm_yoy_panel_mainboard.parquet")
    out["roe"] = roe
    out["gpm"] = gpm

    # 价值：股息率最新月
    divy = _panel_series("div_yield_panel_mainboard.parquet")
    out["value"] = divy

    # 小市值：-log(注册资金万×1e4×最新收盘价)
    cap_p = config.DATA_DIR / "industry_capital.parquet"
    size = None
    if cap_p.exists():
        cap = pd.read_parquet(cap_p)
        cap["code"] = cap["code"].astype(str)
        mv = cap.set_index("code")["capital_wan"].astype(float) * 1e4
        mv = mv.reindex(last.index).dropna()
        mv = mv * last.reindex(mv.index).dropna()
        mv = mv.dropna()
        size = -np.log(mv.clip(lower=1.0))
    out["size"] = size
    return out


def compute_factor_exposure(positions: dict, w: dict) -> dict | None:
    """组合四因子暴露：{因子: 组合市值加权平均 Z}。全池截面 Z 后加权。

    因子定义（相对全市场主板池）：
      - 动量 momentum : z(ret_12)
      - 价值 value    : z(股息率)（无估值面板，以股息率作价值代理）
      - 质量 quality  : (z(roe) + z(gpm_yoy)) / 2
      - 小市值 size   : z(-log(市值))
    """
    if not w:
        return None
    fd = _load_factor_data()
    if not fd:
        return None
    weights = pd.Series(w)

    def _z(s: pd.Series | None) -> pd.Series | None:
        if s is None or len(s.dropna()) < 20:
            return None
        s = s.dropna()
        mu, sd = s.mean(), s.std(ddof=0)
        if not (sd > 0):
            return None
        return (s - mu) / sd

    z_mom = _z(fd.get("momentum"))
    z_val = _z(fd.get("value"))
    z_roe = _z(fd.get("roe"))
    z_gpm = _z(fd.get("gpm"))
    z_size = _z(fd.get("size"))

    quality = None
    if z_roe is not None and z_gpm is not None:
        common = z_roe.index.intersection(z_gpm.index)
        if len(common):
            quality = (z_roe.reindex(common) + z_gpm.reindex(common)) / 2

    def _port_z(z: pd.Series | None) -> float | None:
        if z is None:
            return None
        common = z.index.intersection(weights.index)
        if not len(common):
            return None
        ww = weights.reindex(common)
        ww = ww / ww.sum()          # 仅在有效子集内重新归一
        return float((z.reindex(common) * ww).sum())

    return {
        "momentum": _port_z(z_mom),
        "value": _port_z(z_val),
        "quality": _port_z(quality),
        "size": _port_z(z_size),
    }


def plot_factor_exposure(positions: dict, out_path: Path) -> bool:
    """四因子暴露条形图：组合平均 Z-score vs 0（全市场主板池中性线）。"""
    close = _load_close_panel()
    if close is None:
        return False
    w = compute_position_weights(positions, close)
    expo = compute_factor_exposure(positions, w)
    if not expo:
        print("[chart_utils] 因子数据不足，跳过因子暴露图")
        return False
    labels = [T("动量", "Momentum"), T("价值", "Value"),
              T("质量", "Quality"), T("小市值", "Small-cap")]
    keys = ["momentum", "value", "quality", "size"]
    vals = [expo.get(k) for k in keys]
    if all(v is None for v in vals):
        print("[chart_utils] 因子 Z 全部缺失，跳过因子暴露图")
        return False
    show_v = [0.0 if v is None else v for v in vals]
    colors = [UP_COLOR if v >= 0 else DOWN_COLOR for v in show_v]
    fig, ax = plt.subplots(figsize=(FIG_W, 4.2), dpi=92)
    bars = ax.bar(labels, show_v, color=colors, width=0.55,
                  edgecolor="white", linewidth=0.8)
    ax.axhline(0, color="#7f8c8d", linewidth=1.0)
    for b, v in zip(bars, show_v):
        ax.annotate(f"{v:+.2f}σ", xy=(b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 4 if v >= 0 else -12),
                    textcoords="offset points", ha="center", fontsize=10,
                    fontweight="bold",
                    color=UP_COLOR if v >= 0 else DOWN_COLOR)
    ax.set_ylim(min(show_v) - 0.5, max(show_v) + 0.6)
    ax.set_ylabel(T("组合平均 Z-score（vs 全市场主板池）", "Avg Z vs mainboard pool"),
                  fontsize=9)
    ax.set_title(T("因子暴露分解（动量 / 价值 / 质量 / 小市值）",
                   "Factor Exposure (Momentum / Value / Quality / Size)"),
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[chart_utils] ✓ 因子暴露图 → {out_path}")
    return True


# ---------------------------------------------------------------------------
# 图表3：回撤归因（>5% 回撤区间，行业归因）
# ---------------------------------------------------------------------------
def find_drawdowns(nav: pd.Series, threshold: float = -0.05) -> list:
    """识别净值回撤区间（峰→谷，深度低于 threshold）。

    返回 [(peak_date, trough_date, depth_pct, peak_nav, trough_nav), ...]，
    按深度降序；同一回撤期内多次探底取最深谷。
    """
    if nav is None or len(nav) < 2:
        return []
    nav = nav.dropna().sort_index()
    peak_date, peak_nav = nav.index[0], float(nav.iloc[0])
    trough_date, trough_nav = None, None
    dd_list = []
    for dt, v in nav.items():
        v = float(v)
        if v >= peak_nav:
            if trough_nav is not None and (trough_nav / peak_nav - 1) < threshold:
                dd_list.append((peak_date, trough_date,
                                (trough_nav / peak_nav - 1) * 100,
                                peak_nav, trough_nav))
            peak_date, peak_nav = dt, v
            trough_date, trough_nav = None, None
        else:
            if trough_nav is None or v < trough_nav:
                trough_date, trough_nav = dt, v
    if trough_nav is not None and (trough_nav / peak_nav - 1) < threshold:
        dd_list.append((peak_date, trough_date,
                        (trough_nav / peak_nav - 1) * 100,
                        peak_nav, trough_nav))
    # 合并相邻（同峰区段只保留最深）
    dd_list.sort(key=lambda r: r[2])   # 深度升序
    return dd_list


def _industry_returns(close: pd.DataFrame, peak_dt, trough_dt,
                      industry_map: pd.DataFrame) -> pd.Series:
    """回撤区间内行业等权收益（close 面板，取区间 [peak, trough] 首尾收盘）。"""
    if close is None or len(close) == 0:
        return pd.Series(dtype=float)
    sub = close.loc[(close.index >= peak_dt) & (close.index <= trough_dt)]
    if len(sub) < 2:
        return pd.Series(dtype=float)
    first = sub.iloc[0]
    last = sub.iloc[-1]
    ret = (last / first - 1.0).dropna()
    imap = dict(zip(industry_map["code"].astype(str),
                    industry_map["industry"])) if industry_map is not None else {}
    ret.index = ret.index.astype(str)
    grp = pd.Series(dtype=float)
    for code, r in ret.items():
        ind = imap.get(code, "其他")
        grp[ind] = grp.get(ind, 0.0) + r
    cnt = pd.Series(0, index=grp.index, dtype=float)
    for code in ret.index:
        cnt[imap.get(code, "其他")] += 1
    return grp / cnt.clip(lower=1)


def plot_drawdown_attribution(nav_history: pd.DataFrame, positions: dict,
                              industry_map: pd.DataFrame, out_path: Path,
                              threshold: float = -0.05) -> bool:
    """回撤归因：对每个 >5% 回撤区间，用「持仓行业权重 × 行业区间等权收益」归因。

    归因口径（数据可得性约束下的合理近似）：
      行业贡献 = 持仓行业权重 × 该行业成分股在回撤区间的等权收益。
    实际回撤 = nav_trough/nav_peak - 1。展示每个回撤区间的 Top 负贡献行业。
    """
    if nav_history is None or len(nav_history) < 2:
        print("[chart_utils] 净值历史不足，跳过回撤归因图")
        return False
    nh = nav_history.copy()
    nh["date"] = pd.to_datetime(nh["date"])
    nav = nh.set_index("date")["nav"].dropna().astype(float)
    dds = find_drawdowns(nav, threshold)
    if not dds:
        print("[chart_utils] 无 >5% 回撤区间，跳过回撤归因图")
        return False

    close = _load_close_panel()
    imap = industry_map
    # 持仓行业权重（用最新持仓近似整个回撤期的持仓结构）
    w = compute_position_weights(positions, close) if positions else {}
    pos_ind: dict = {}
    if w and imap is not None and len(imap):
        im = dict(zip(imap["code"].astype(str), imap["industry"]))
        for code, wt in w.items():
            ind = im.get(code, "其他")
            pos_ind[ind] = pos_ind.get(ind, 0.0) + wt

    # 展示最近 3 个最深回撤，每个回撤一个横向子图
    dds = dds[:3]
    fig, axes = plt.subplots(len(dds), 1,
                             figsize=(FIG_W, 3.4 * len(dds)), dpi=92)
    if len(dds) == 1:
        axes = [axes]
    for ax, (pk, tr, depth, pkn, trn) in zip(axes, dds):
        ind_ret = _industry_returns(close, pk, tr, imap) if close is not None else pd.Series(dtype=float)
        # 归因贡献 = 持仓行业权重 × 行业区间收益；未持仓行业贡献 0
        contrib = {}
        for ind, r in ind_ret.items():
            contrib[ind] = pos_ind.get(ind, 0.0) * r
        contrib = {k: v for k, v in contrib.items() if abs(v) > 1e-9}
        if contrib:
            items = sorted(contrib.items(), key=lambda kv: kv[1])[:6]   # 负贡献 Top6
        else:
            items = []
        label = (f"{pk.date()} → {tr.date()}  {depth:+.1f}%"
                 + ("（{:.0f} 个交易日）".format(len(nav.loc[pk:tr]))))
        ax.set_title(T("回撤归因：", "Drawdown attribution: ") + label,
                     fontsize=11, fontweight="bold", loc="left")
        if not items:
            ax.text(0.5, 0.5, T("该回撤无持仓行业可归因（持仓为空/行业缺失）",
                                "no attributable industry"),
                    ha="center", va="center", transform=ax.transAxes,
                    color=NEUTRAL, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        names = [k for k, _ in items][::-1]
        vals = [v * 100 for _, v in items][::-1]
        yy = np.arange(len(items))
        ax.barh(yy, vals, color=[DOWN_COLOR if v < 0 else UP_COLOR for v in vals],
                height=0.62)
        ax.axvline(0, color="#7f8c8d", linewidth=0.9)
        ax.set_yticks(yy)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel(T("对组合回撤的贡献 pp（行业收益×持仓权重）", "contrib pp"),
                      fontsize=8)
        ax.grid(axis="x", alpha=0.25, linestyle="--")
        for i, v in enumerate(vals):
            ax.annotate(f"{v:+.2f}", xy=(v, yy[i]),
                        xytext=(3, 0), textcoords="offset points",
                        fontsize=8, va="center")
    fig.suptitle(T("净值回撤归因（>5% 回撤区间，持仓行业贡献）",
                   "Drawdown Attribution (>5%, by industry)"),
                 fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[chart_utils] ✓ 回撤归因图 → {out_path}（{len(dds)} 个回撤区间）")
    return True
