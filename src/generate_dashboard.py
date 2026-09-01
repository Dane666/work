#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_dashboard.py — V8.1 模拟盘持仓走势图 + GitHub Pages 总览生成
====================================================================

在 signal_generator / sim_tracker 之后运行，读取最新持仓与净值历史，生成：
  - docs/assets/charts/{code}.png      每只持仓股票的 K 线走势图（蜡烛图 + MA20/MA60 +
                                       买入点绿色箭头）
  - docs/assets/nav_curve.png          模拟盘 vs 沪深300 净值曲线
  - docs/assets/pnl_bar.png            持仓盈亏柱状图（A股惯例：红涨绿跌）
  - docs/assets/industry_exposure.png  行业暴露：持仓 vs 基准（方向B增强，chart_utils）
  - docs/assets/factor_exposure.png    四因子暴露（动量/价值/质量/小市值，方向B增强）
  - docs/assets/drawdown_attr.png      净值回撤归因（>5% 区间行业贡献，方向B增强）
  - docs/index.html                    总览页（概览卡片 + 净值曲线 + 盈亏柱 + 增强图 + 持仓表格）
  - docs/holdings.html                 详情页（全部持仓股票图表网格）

数据来源：
  - 持仓：      data/state/sim_state.json（优先；回退 output/sim_nav/sim_state.json）
  - 净值：      data/state/sim_nav_history.csv（优先；回退 output/sim_nav/ 缓存）
  - 行业/因子： data/industry_map.parquet / industry_benchmark.parquet /
                roe_panel_mainboard.parquet / gpm_yoy_panel_mainboard.parquet /
                div_yield_panel_mainboard.parquet / industry_capital.parquet
  - 买入日期：  output/signals/*_signal.csv（BUY 信号日；执行日=其后第一个交易日）
  - K 线：      data/v8_ohlcv.pkl（缺省回退 data/_v8_ohlcv_ckpt.pkl，{code: DataFrame(index=date,
                open/high/low/close)}，前复权与 akshare qfq 对齐）

用法（从 src/ 目录运行）：
  python generate_dashboard.py                # 离线模式（GitHub Actions 用，仅用本地 pkl）
  python generate_dashboard.py --live         # 联网补充持仓股票最新行情（仅限持仓，akshare）
  python generate_dashboard.py --days 250     # 单图显示最近 N 个交易日（默认 120）
  python generate_dashboard.py --max-charts 5 # 调试：仅渲染前 5 只

部署：GitHub Pages 指向 docs/ 目录（Settings -> Pages -> Deploy from a branch -> docs）。
"""

from __future__ import annotations

import argparse
import glob
import html
import json
import os
import pickle
import sys
from datetime import datetime
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
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

DOCS_DIR = config.DOCS_DIR
CHARTS_DIR = DOCS_DIR / "assets" / "charts"
NAV_DIR = config.OUTPUT_DIR / "sim_nav"
SIGNAL_DIR = config.OUTPUT_DIR / "signals"
# 状态优先读持久化主路径 data/state/（跨运行共享，与 sim_tracker 一致）；
# 缺失时回退本地缓存 output/sim_nav/（旧路径产物）。
STATE_FILE = config.SIM_STATE_DIR / "sim_state.json"
STATE_FILE_CACHE = NAV_DIR / "sim_state.json"
NAV_HIST_FILE = config.SIM_STATE_DIR / "sim_nav_history.csv"
NAV_HIST_FILE_CACHE = NAV_DIR / "sim_nav_history.csv"

# 图表窗口（交易日）
DEFAULT_LAST_DAYS = 120

# 图表配色：A股惯例（涨=红，跌=绿）
UP_COLOR = "#c0392b"
DOWN_COLOR = "#1e8449"
MA20_COLOR = "#2471a3"
MA60_COLOR = "#d35400"
BUY_COLOR = "#16a085"

CN_OK = False  # 是否有可用中文字体（无则图内文字退化为英文）


# ---------------------------------------------------------------------------
# 中文字体
# ---------------------------------------------------------------------------
def setup_font() -> bool:
    """注册系统字体并选择可用中文字体；返回是否有中文字体。"""
    global CN_OK
    for p in [
        "/System/Library/Fonts/PingFang.ttc",                      # macOS
        "/System/Library/Fonts/STHeiti Light.ttc",                 # macOS
        "/System/Library/Fonts/Hiragino Sans GB.ttc",              # macOS
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Ubuntu (noto-cjk)
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",  # Debian
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",            # wqy-zenhei
    ]:
        if os.path.exists(p):
            try:
                fm.fontManager.addfont(p)
            except Exception:
                pass
    for name in ["Noto Sans CJK SC", "PingFang SC", "Hiragino Sans GB",
                 "WenQuanYi Zen Hei", "Microsoft YaHei", "SimHei",
                 "Arial Unicode MS"]:
        try:
            if fm.findfont(fm.FontProperties(family=name),
                           fallback_to_default=False):
                plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
                plt.rcParams["axes.unicode_minus"] = False
                CN_OK = True
                return True
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False
    return False


def T(zh: str, en: str = "") -> str:
    """无中文字体时图内文字退化为英文。"""
    return zh if CN_OK else (en or "")


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def find_ohlcv_path() -> Path | None:
    """定位 K 线 pkl：主板版优先 mainboard_ohlcv.pkl；否则 v8_ohlcv.pkl 回退 _v8_ohlcv_ckpt.pkl。"""
    if getattr(config, "USE_MAINBOARD", False) and config.MB_OHLCV.exists():
        return config.MB_OHLCV
    for p in [config.V8_OHLCV, config.V8_OHLCV_CKPT]:
        if p.exists():
            return p
    return None


def load_ohlcv() -> dict:
    p = find_ohlcv_path()
    if p is None:
        print("⚠️  未找到 K 线数据（data/v8_ohlcv.pkl 或 _v8_ohlcv_ckpt.pkl）")
        return {}
    with open(p, "rb") as f:
        d = pickle.load(f)
    print(f"✓ 加载 K 线 {p.name}：{len(d)} 只")
    return d


def load_state() -> dict:
    for p in (STATE_FILE, STATE_FILE_CACHE):
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
    return {}


def load_nav_history() -> pd.DataFrame:
    for p in (NAV_HIST_FILE, NAV_HIST_FILE_CACHE):
        if p.exists():
            try:
                df = pd.read_csv(p)
                df["date"] = pd.to_datetime(df["date"])
                return df
            except Exception:
                continue
    return pd.DataFrame()


def load_stock_names() -> dict:
    p = config.DATA_DIR / "stock_names.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def collect_buy_dates() -> dict:
    """{code: [信号日, ...]} —— 来自全部 *_signal.csv 中的 BUY 记录。"""
    buys: dict = {}
    files = sorted(glob.glob(str(SIGNAL_DIR / "*_signal.csv")))
    for f in files:
        sig_date = pd.Timestamp(Path(f).stem.replace("_signal", ""))
        try:
            df = pd.read_csv(f, dtype={"code": str})
        except Exception:
            continue
        for c in df.loc[df["action"] == "BUY", "code"]:
            buys.setdefault(str(c), []).append(sig_date)
    return buys


# ---------------------------------------------------------------------------
# 最新行情补充（--live，仅限持仓股票）
# ---------------------------------------------------------------------------
def fetch_latest_quotes(codes, ohlcv: dict) -> dict:
    """用 akshare 拉取持仓股票最近行情，前复权锚点对齐后 append 到本地 K 线。

    返回 {code: merged_df}，仅含拉取到新数据的股票；失败/无新数据则不含该 code。
    注意：不写回 data/ 下原始 pkl（原始文件不可修改约束）。
    """
    import time
    try:
        import akshare as ak
    except Exception as e:
        print(f"⚠️  akshare 不可用（{e}），跳过最新行情补充")
        return {}

    out: dict = {}
    d1 = datetime.now().strftime("%Y%m%d")
    for i, code in enumerate(codes, 1):
        df = ohlcv.get(code)
        if df is None or len(df) == 0:
            continue
        last_date = df.index[-1]
        d0 = (last_date - pd.Timedelta(days=25)).strftime("%Y%m%d")
        ak_df = None
        for attempt in range(3):
            try:
                ak_df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                           start_date=d0, end_date=d1, adjust="qfq")
                if ak_df is not None and len(ak_df):
                    break
            except Exception:
                ak_df = None
            time.sleep(0.4)
        if ak_df is None or len(ak_df) == 0:
            continue
        ak_df = ak_df.rename(columns={"日期": "date", "开盘": "open",
                                      "最高": "high", "最低": "low", "收盘": "close"})
        if "date" not in ak_df.columns:          # 新浪源兜底
            ak_df = ak_df.reset_index().rename(columns={"index": "date"})
        ak_df["date"] = pd.to_datetime(ak_df["date"])
        ak_df = ak_df.set_index("date")[["open", "high", "low", "close"]]
        ak_df = ak_df[~ak_df.index.duplicated(keep="last")].sort_index()

        # 前复权锚点对齐：用重叠期 close 比值中位数把新数据缩放到本地价格基
        overlap = df.index.intersection(ak_df.index)
        if len(overlap) >= 5:
            ratio = float((df.loc[overlap, "close"] /
                           ak_df.loc[overlap, "close"]).median())
            if 0.2 < ratio < 5.0:               # 防御异常缩放
                ak_df[["open", "high", "low", "close"]] = ak_df[["open", "high", "low", "close"]] * ratio
        new_rows = ak_df[ak_df.index > df.index[-1]]
        if len(new_rows) == 0:
            continue
        merged = pd.concat([df, new_rows])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        out[code] = merged
        print(f"   {code}: 本地末日 {last_date.date()} → 补充至 {merged.index[-1].date()} "
              f"（+{len(new_rows)} 根）")
    return out


# ---------------------------------------------------------------------------
# 单只股票 K 线图（蜡烛图 + MA20/MA60 + 买入点绿色箭头）
# ---------------------------------------------------------------------------
def plot_stock(code: str, name: str, df: pd.DataFrame,
               buy_dates: list, out_path: Path, last_days: int) -> None:
    d = df.tail(last_days)
    if len(d) < 5:
        print(f"⚠️  跳过 {code}：K 线不足 5 根")
        return

    x = np.arange(len(d))
    dates = d.index

    fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=92)
    # 蜡烛影线
    ax.vlines(x, d["low"], d["high"], color=[UP_COLOR if c >= o else DOWN_COLOR
                                             for c, o in zip(d["close"], d["open"])],
              linewidth=0.8)
    # 蜡烛实体（红涨绿跌，A股惯例）
    body = d["close"] - d["open"]
    colors = [UP_COLOR if v >= 0 else DOWN_COLOR for v in body]
    ax.bar(x, body.abs(), bottom=np.minimum(d["open"], d["close"]),
           color=colors, width=0.6, edgecolor=colors, linewidth=0.5)

    # 均线
    ma20 = d["close"].rolling(20).mean()
    ma60 = d["close"].rolling(60).mean()
    ax.plot(x, ma20, color=MA20_COLOR, linewidth=1.1, label="MA20")
    ax.plot(x, ma60, color=MA60_COLOR, linewidth=1.1, label="MA60")

    # 买入点（BUY 信号日之后第一个交易日）
    for bd in sorted(buy_dates):
        later = dates[dates > bd]
        if len(later) == 0:
            continue
        exec_date = later[0]
        xi = int(np.where(dates == exec_date)[0][0])
        y_low = d["low"].iloc[xi]
        y_mark = y_low * 0.985
        ax.scatter([xi], [y_mark], marker="^", s=130, color=BUY_COLOR,
                   zorder=5, edgecolors="white", linewidths=0.8)
        ax.annotate(T("买入", "BUY") + f" {exec_date.strftime('%m-%d')}",
                    xy=(xi, y_mark), xytext=(xi + 1, y_mark),
                    fontsize=8, color=BUY_COLOR, va="center",
                    arrowprops=dict(arrowstyle="-", color=BUY_COLOR, lw=0.7))

    # 坐标轴
    step = max(1, len(x) // 8)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([dt.strftime("%Y-%m-%d") for dt in dates[::step]],
                       rotation=30, ha="right", fontsize=8)
    ax.set_xlim(-1, len(x) + 1)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.6)

    title = f"{code} {name}" if CN_OK else code
    sub = T(f"近 {len(d)} 个交易日 · 数据截至 {dates[-1].strftime('%Y-%m-%d')}",
            f"last {len(d)} days, data to {dates[-1].strftime('%Y-%m-%d')}")
    ax.set_title(f"{title}\n{sub}", fontsize=12, fontweight="bold")
    ax.set_ylabel(T("价格", "price"), fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 净值曲线
# ---------------------------------------------------------------------------
def plot_nav_curve(hist: pd.DataFrame, out_path: Path) -> None:
    if hist.empty:
        return
    fig, ax = plt.subplots(figsize=(9.5, 4.2), dpi=92)
    ax.plot(hist["date"], hist["nav"], color=UP_COLOR, linewidth=1.6,
            label=T("模拟盘 NAV", "Sim NAV"))
    if "benchmark_nav" in hist.columns:
        b = hist.dropna(subset=["benchmark_nav"])
        if len(b):
            ax.plot(b["date"], b["benchmark_nav"], color="#34495e",
                    linewidth=1.3, linestyle="--", label=T("沪深300", "CSI300"))
    ax.axhline(1.0, color="#95a5a6", linewidth=0.8, linestyle=":")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="upper left", fontsize=10)
    ax.set_title(T("模拟盘净值曲线", "Sim NAV Curve"), fontsize=12, fontweight="bold")
    ax.set_xlabel(T("日期", "date"), fontsize=9)
    ax.set_ylabel("NAV", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 持仓盈亏柱状图（红涨绿跌）
# ---------------------------------------------------------------------------
def plot_pnl_bar(rows: list, out_path: Path) -> None:
    """rows: [{code, name, pnl_pct}]，按 pnl_pct 升序。"""
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r["pnl_pct"])
    labels = [f"{r['code']} {r['name']}"[:16] for r in rows]
    vals = [r["pnl_pct"] * 100 for r in rows]
    colors = [UP_COLOR if v >= 0 else DOWN_COLOR for v in vals]
    fig, ax = plt.subplots(figsize=(9.5, max(4.0, 0.32 * len(rows))), dpi=92)
    ax.barh(labels, vals, color=colors, height=0.65)
    ax.axvline(0, color="#7f8c8d", linewidth=0.9)
    ax.set_xlabel("%", fontsize=9)
    ax.set_title(T("持仓盈亏（最新价 vs 成本）", "Position PnL (last vs cost)"),
                 fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# HTML 页面
# ---------------------------------------------------------------------------
def esc(s) -> str:
    return html.escape(str(s), quote=True)


def build_index_html(ctx: dict) -> str:
    rows = ctx["rows"]
    nav_rows = ""
    for r in rows:
        cls = "pos" if r["pnl_pct"] >= 0 else "neg"
        nav_rows += (
            f"<tr>"
            f"<td class='code'>{esc(r['code'])}</td>"
            f"<td>{esc(r['name'])}</td>"
            f"<td>{esc(r['buy_date'] or '—')}</td>"
            f"<td>{r['cost']:.2f}</td>"
            f"<td>{r['last']:.2f}</td>"
            f"<td>{r['weight_pct']:.2f}%</td>"
            f"<td class='{cls}'>{r['pnl_pct'] * 100:+.2f}%</td>"
            f"<td><a href='assets/charts/{esc(r['code'])}.png'>走势</a></td>"
            f"</tr>"
        )
    total_pnl = sum(r["pnl_amt"] for r in rows)
    gainers = sum(1 for r in rows if r["pnl_pct"] >= 0)
    o = ctx["overview"]
    enh = ctx.get("enh", {})

    def _blk(key: str, title: str, img: str, alt: str) -> str:
        return "" if not enh.get(key) else (
            f'<div class="chart-block">\n    <h2>{title}</h2>\n'
            f'    <img src="{img}" alt="{alt}">\n  </div>')

    ind_block = _blk("industry", "行业暴露（持仓 vs 基准）",
                     "assets/industry_exposure.png", "industry exposure")
    fac_block = _blk("factor", "因子暴露分解（动量 / 价值 / 质量 / 小市值）",
                     "assets/factor_exposure.png", "factor exposure")
    dd_block = _blk("drawdown", "净值回撤归因（>5% 回撤区间，行业贡献）",
                    "assets/drawdown_attr.png", "drawdown attribution")
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>V8.1 模拟盘 Dashboard</title>
<style>
  body {{ font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
         margin: 0; background: #f4f6f8; color: #2c3e50; }}
  .wrap {{ max-width: 1100px; margin: 0 auto; padding: 24px 16px 60px; }}
  header {{ display: flex; justify-content: space-between; align-items: baseline;
           border-bottom: 2px solid #c0392b; padding-bottom: 12px; margin-bottom: 20px; }}
  header h1 {{ font-size: 22px; margin: 0; }}
  header .meta {{ color: #7f8c8d; font-size: 13px; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
           gap: 12px; margin-bottom: 24px; }}
  .card {{ background: #fff; border-radius: 10px; padding: 14px 16px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
  .card .k {{ font-size: 12px; color: #7f8c8d; }}
  .card .v {{ font-size: 20px; font-weight: 700; margin-top: 4px; }}
  .pos {{ color: #c0392b; }} .neg {{ color: #1e8449; }}
  .chart-block {{ background: #fff; border-radius: 10px; padding: 14px;
                 box-shadow: 0 1px 3px rgba(0,0,0,.08); margin-bottom: 24px; }}
  .chart-block h2 {{ font-size: 16px; margin: 0 0 10px; }}
  .chart-block img {{ width: 100%; height: auto; border-radius: 6px; }}
  table {{ width: 100%; border-collapse: collapse; background: #fff; border-radius: 10px;
          overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
  th, td {{ padding: 8px 10px; text-align: right; font-size: 13px; }}
  th {{ background: #2c3e50; color: #fff; font-weight: 600; }}
  td:first-child, th:first-child {{ text-align: left; }}
  td.code {{ font-family: Menlo, Consolas, monospace; }}
  tr:nth-child(even) td {{ background: #f8fafb; }}
  a {{ color: #2471a3; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .links {{ margin-top: 16px; font-size: 14px; }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>V8.1 模拟盘 Dashboard</h1>
    <div class="meta">生成时间 {ctx["generated_at"]} · 数据截至 {ctx["data_as_of"]}</div>
  </header>

  <div class="cards">
    <div class="card"><div class="k">NAV</div><div class="v">{o["nav"]:.4f}</div></div>
    <div class="card"><div class="k">累计收益</div>
      <div class="v {'pos' if o['cum_ret'] >= 0 else 'neg'}">{o["cum_ret"] * 100:+.2f}%</div></div>
    <div class="card"><div class="k">沪深300 NAV</div><div class="v">{o["bench"]:.4f}</div></div>
    <div class="card"><div class="k">超额收益</div>
      <div class="v {'pos' if o['excess'] >= 0 else 'neg'}">{o["excess"] * 100:+.2f}%</div></div>
    <div class="card"><div class="k">持仓数</div><div class="v">{len(rows)}</div></div>
    <div class="card"><div class="k">现金比例</div><div class="v">{o["cash"] * 100:.1f}%</div></div>
    <div class="card"><div class="k">持仓盈亏合计</div>
      <div class="v {'pos' if total_pnl >= 0 else 'neg'}">{total_pnl * 100:+.2f}%</div></div>
    <div class="card"><div class="k">盈利/亏损</div><div class="v">{gainers}/{len(rows) - gainers}</div></div>
  </div>

  <div class="chart-block">
    <h2>净值曲线（模拟盘 vs 沪深300）</h2>
    <img src="assets/nav_curve.png" alt="nav curve">
  </div>

  <div class="chart-block">
    <h2>持仓盈亏（最新价 vs 成本）</h2>
    <img src="assets/pnl_bar.png" alt="pnl bar">
  </div>

  {ind_block}

  {fac_block}

  {dd_block}

  <div class="chart-block">
    <h2>持仓明细（{len(rows)} 只）</h2>
    <table>
      <tr><th>代码</th><th>名称</th><th>买入日期</th><th>成本价</th><th>最新价</th>
          <th>仓位</th><th>盈亏</th><th></th></tr>
      {nav_rows}
    </table>
  </div>

  <div class="links">
    <a href="holdings.html">→ 查看全部持仓走势图（{len(rows)} 张）</a>
  </div>
</div>
</body>
</html>"""


def build_holdings_html(ctx: dict) -> str:
    cards = ""
    for r in ctx["rows"]:
        cls = "pos" if r["pnl_pct"] >= 0 else "neg"
        img = f"assets/charts/{esc(r['code'])}.png"
        cards += f"""
  <div class="card">
    <div class="head"><span class="t">{esc(r['code'])} {esc(r['name'])}</span>
      <span class="{cls}">{r['pnl_pct'] * 100:+.2f}%</span></div>
    <div class="sub">买入 {esc(r['buy_date'] or '—')} · 成本 {r['cost']:.2f} · 最新 {r['last']:.2f}</div>
    <a href="{img}" target="_blank"><img src="{img}" alt="{esc(r['code'])}"></a>
  </div>"""
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>持仓走势图 · V8.1 模拟盘</title>
<style>
  body {{ font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
         margin: 0; background: #f4f6f8; color: #2c3e50; }}
  .wrap {{ max-width: 1280px; margin: 0 auto; padding: 24px 16px 60px; }}
  header {{ display: flex; justify-content: space-between; align-items: baseline;
           border-bottom: 2px solid #c0392b; padding-bottom: 12px; margin-bottom: 20px; }}
  header h1 {{ font-size: 22px; margin: 0; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
          gap: 18px; }}
  .card {{ background: #fff; border-radius: 10px; padding: 14px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
  .head {{ display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 4px; }}
  .t {{ font-size: 15px; font-weight: 700; }}
  .sub {{ font-size: 12px; color: #7f8c8d; margin-bottom: 8px; }}
  .card img {{ width: 100%; height: auto; border-radius: 6px; }}
  .pos {{ color: #c0392b; }} .neg {{ color: #1e8449; }}
  .links {{ margin-bottom: 16px; font-size: 14px; }}
  a {{ color: #2471a3; text-decoration: none; }}
  .meta {{ color: #7f8c8d; font-size: 13px; }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>持仓走势图（{len(ctx["rows"])} 只）</h1>
    <div class="meta">数据截至 {ctx["data_as_of"]}</div>
  </header>
  <div class="links"><a href="index.html">← 返回总览</a></div>
  <div class="grid">{cards}
  </div>
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="V8.1 模拟盘持仓走势图生成器")
    ap.add_argument("--live", action="store_true",
                    help="联网补充持仓股票最新行情（akshare，仅限持仓）")
    ap.add_argument("--days", type=int, default=DEFAULT_LAST_DAYS,
                    help="单图显示最近 N 个交易日（默认 120）")
    ap.add_argument("--max-charts", type=int, default=0,
                    help="调试：仅渲染前 N 只（0=全部）")
    args = ap.parse_args()

    setup_font()
    print(f"{'✓ 中文字体可用' if CN_OK else '⚠️  无中文字体，图内文字使用英文'}")

    # ---- 读取数据 ----
    state = load_state()
    positions = state.get("positions", {})
    if not positions:
        print("⚠️  sim_state.json 无持仓（可能尚未初始化），仅生成空总览。")
    hist = load_nav_history()
    buy_dates = collect_buy_dates()
    names = load_stock_names()
    ohlcv = load_ohlcv()

    codes = list(positions.keys())
    print(f"✓ 持仓 {len(codes)} 只，买入记录 {sum(len(v) for v in buy_dates.values())} 条")

    # ---- 最新行情补充（--live）----
    fetch_codes = codes[:args.max_charts] if args.max_charts else codes
    if args.live and fetch_codes:
        print("┌─ 补充最新行情（仅持仓股票）")
        merged = fetch_latest_quotes(fetch_codes, ohlcv)
        ohlcv.update(merged)
        print("└─ 补充完成")

    # ---- 计算每只持仓指标 ----
    # 起始日期过滤：只保留信号日 >= SIM_START_DATE 的买入点（与 sim_tracker 消费口径一致）
    start = getattr(config, "SIM_START_DATE", None)
    start_ts = pd.Timestamp(start).normalize() if start else None
    rows = []
    for code in codes:
        h = positions[code]
        shares = float(h.get("shares", 0.0))
        cost = float(h.get("cost", 0.0))
        df = ohlcv.get(code)
        if df is not None and len(df):
            last = float(df["close"].iloc[-1])
            data_as_of = df.index[-1]
        else:
            last = cost          # 无 K 线时以成本价近似
            data_as_of = None
        pnl_amt = shares * (last - cost)
        pnl_pct = (last / cost - 1.0) if cost > 0 else 0.0
        name = names.get(code, code)
        bd = [sd for sd in buy_dates.get(code, [])
              if start_ts is None or sd >= start_ts]   # 只保留起始日之后的买入信号
        buy_str = ""
        if bd:
            # 买入执行日 = 信号日之后第一个交易日；无 K 线时直接显示信号日
            exec_dates = []
            for sd in bd:
                later = df.index[df.index > sd] if (df is not None and len(df)) else []
                exec_dates.append(later[0].strftime("%Y-%m-%d") if len(later) else sd.strftime("%Y-%m-%d"))
            buy_str = ", ".join(sorted(set(exec_dates)))
        rows.append({
            "code": code, "name": name, "shares": shares, "cost": cost,
            "last": last, "pnl_pct": pnl_pct, "pnl_amt": pnl_amt,
            "buy_date": buy_str, "df": df, "buy_dates": bd,
        })

    nav = float(hist["nav"].iloc[-1]) if len(hist) else 1.0
    for r in rows:
        r["weight_pct"] = (r["shares"] * r["last"]) / nav * 100.0 if nav > 0 else 0.0

    # ---- 绘制 ----
    n_plot = 0
    limit = args.max_charts or len(rows)
    for r in rows[:limit]:
        if r["df"] is None or len(r["df"]) == 0:
            continue
        plot_stock(r["code"], r["name"], r["df"], r["buy_dates"],
                   CHARTS_DIR / f"{r['code']}.png", args.days)
        n_plot += 1
    # 清理不在当前持仓中的历史图表（避免 docs 目录与 GitHub 仓库无限膨胀）
    if not args.max_charts:      # 全量生成时才清理（调试 --max-charts 不动历史文件）
        valid = {r["code"] for r in rows}
        removed = 0
        for f in CHARTS_DIR.glob("*.png"):
            if f.stem not in valid:
                f.unlink()
                removed += 1
        if removed:
            print(f"✓ 清理 {removed} 张已不在持仓中的历史图表")
    print(f"✓ 生成 {n_plot} 张持仓走势图 → docs/assets/charts/")

    plot_nav_curve(hist, DOCS_DIR / "assets" / "nav_curve.png")
    plot_pnl_bar([{k: r[k] for k in ("code", "name", "pnl_pct")} for r in rows],
                 DOCS_DIR / "assets" / "pnl_bar.png")
    print("✓ 生成净值曲线与盈亏柱状图 → docs/assets/")

    # ---- 方向B：仪表盘增强（行业暴露 / 因子暴露 / 回撤归因）----
    import chart_utils
    industry_map = None
    _p = config.DATA_DIR / "industry_map.parquet"
    if _p.exists():
        industry_map = pd.read_parquet(_p)
    bench_df = None
    _p = config.DATA_DIR / "industry_benchmark.parquet"
    if _p.exists():
        bench_df = pd.read_parquet(_p)
    enh = {
        "industry": chart_utils.plot_industry_exposure(
            positions, industry_map, bench_df,
            DOCS_DIR / "assets" / "industry_exposure.png"),
        "factor": chart_utils.plot_factor_exposure(
            positions, DOCS_DIR / "assets" / "factor_exposure.png"),
        "drawdown": chart_utils.plot_drawdown_attribution(
            hist, positions, industry_map,
            DOCS_DIR / "assets" / "drawdown_attr.png"),
    }
    print("✓ 仪表盘增强图：行业暴露={} 因子暴露={} 回撤归因={}".format(
        enh["industry"], enh["factor"], enh["drawdown"]))

    # ---- 概览 & 页面 ----
    bench = float(hist["benchmark_nav"].iloc[-1]) if (
        len(hist) and "benchmark_nav" in hist.columns and
        pd.notna(hist["benchmark_nav"].iloc[-1])) else 1.0
    overview = {
        "nav": nav,
        "cum_ret": float(hist["cum_return"].iloc[-1]) if len(hist) and "cum_return" in hist.columns else 0.0,
        "bench": bench,
        "excess": nav / bench - 1.0 if bench > 0 else 0.0,
        "cash": float(hist["cash"].iloc[-1]) if len(hist) and "cash" in hist.columns else 0.0,
    }
    data_as_of = max((r["df"].index[-1] for r in rows if r["df"] is not None and len(r["df"])),
                     default=pd.Timestamp(datetime.now().date()))
    ctx = {
        "rows": rows,
        "overview": overview,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_as_of": data_as_of.strftime("%Y-%m-%d"),
        "enh": enh,
    }
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "index.html").write_text(build_index_html(ctx), encoding="utf-8")
    (DOCS_DIR / "holdings.html").write_text(build_holdings_html(ctx), encoding="utf-8")
    print(f"✓ 生成 docs/index.html（NAV={overview['nav']:.4f}，持仓 {len(rows)} 只）")
    print(f"✓ 生成 docs/holdings.html")
    print(f"✓ 全部完成 → {DOCS_DIR}")


if __name__ == "__main__":
    main()
