# -*- coding: utf-8 -*-
"""
monitor.py — 持仓监控预警（V3.2 风控辅助工具，方向E）
====================================================================

每日收盘后扫描当前模拟盘持仓，触发以下预警（仅提示，不自动执行卖出）：
  ⚠️ 止损预警   盈亏率 <  MONITOR_STOP_LOSS        （默认 -8%）
  💰 止盈预警   盈亏率 >  MONITOR_PROFIT_TARGET     （默认 +20%）
  📉 移动止损   自持仓期间最高价回撤 > MONITOR_TRAILING_STOP（默认 6%）
  📊 估值偏高   市盈率 >  MONITOR_PE_RATIO          （默认 50；需 PE 面板，缺失自动跳过）

数据来源：
  - 持仓：data/state/sim_state.json（持久化主路径；本地缺失回退 output/sim_nav/sim_state.json）
  - 收盘价：data/mainboard_close_panel.parquet（date 索引 × code 列，取每只持仓最后有效收盘价）
  - 名称：data/stock_names.json
  - 市盈率：data/pe_panel_mainboard.parquet（可选；当前仓库无此文件 → 估值预警自动跳过）

推送策略（与方向D order_push.py 一致）：
  - LIVE_MODE=true（环境变量，run_daily.sh 实盘分支注入）或 --push → Bark 推送预警摘要（最多前 10 只）
  - 默认（本地 / LIVE_MODE=false）→ 仅打印到日志，不推送
  - 无预警时（推送模式）→ 推送「✅ 今日无预警」，作为每日持仓健康度报告

用法：
  cd src && python monitor.py                 # 仅打印（不推送）
  python monitor.py --push                    # 打印 + Bark 推送
  LIVE_MODE=true python monitor.py            # 等同 --push（run_daily.sh 实盘分支）
  python monitor.py --state /tmp/mock.json    # 指定持仓状态文件（模拟验证，不碰真实状态）
  python monitor.py --preview                 # 打印推送正文预览（不发请求）
退出码恒为 0：预警 ≠ 错误；任何异常内部降级，不中断 run_daily.sh 主流程。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# 清代理（与其余 src 模块一致；新浪/同花顺源在代理下可能被拦截，本模块只读本地文件，防御性保留）
for _k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
           "ALL_PROXY", "all_proxy"]:
    os.environ.pop(_k, None)

# 兼容两种调用方式：python monitor.py（sys.path[0]=src）与 python -m src.monitor（sys.path[0]=项目根）
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import config

# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def load_sim_state(state_path: str | Path | None = None):
    """读取持仓状态。返回 (state, 来源描述)。

    优先级：显式 --state > data/state/sim_state.json（Actions 持久化主路径）
            > output/sim_nav/sim_state.json（本地缓存副本）。
    文件不存在 / 解析失败 → 返回空状态（不抛异常）。
    """
    candidates = []
    if state_path:
        candidates.append((Path(state_path), f"显式指定 {state_path}"))
    candidates.append((config.SIM_STATE_DIR / "sim_state.json", "data/state（持久化主路径）"))
    candidates.append((config.OUTPUT_DIR / "sim_nav" / "sim_state.json", "output/sim_nav（本地缓存）"))

    for p, desc in candidates:
        if p.exists():
            try:
                st = json.loads(p.read_text(encoding="utf-8"))
                print(f"[monitor] 读取持仓: {p}（{desc}）")
                return st, desc
            except Exception as e:
                print(f"[monitor] 状态文件解析失败（{p}）: {e}，尝试下一来源")
    print("[monitor] 未找到 sim_state.json（data/state 与 output/sim_nav 均缺失），按空仓处理")
    return {"positions": {}}, "空仓（无状态文件）"


def load_close_panel() -> pd.DataFrame | None:
    """加载主板收盘价面板；缺失/解析失败返回 None（调用方降级跳过该股）。"""
    p = config.MB_CLOSE
    if not p.exists():
        print(f"[monitor] 收盘价面板缺失: {p}")
        return None
    try:
        df = pd.read_parquet(p)
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.sort_index()
        return df
    except Exception as e:
        print(f"[monitor] 收盘价面板读取失败: {e}")
        return None


def load_names() -> dict:
    """加载 code→name 映射（data/stock_names.json）；缺失返回空 dict。"""
    p = config.DATA_DIR / "stock_names.json"
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def load_pe_panel() -> pd.DataFrame | None:
    """加载 PE 面板（可选）。当前仓库无 pe_panel_mainboard.parquet → 返回 None，估值预警跳过。"""
    p = config.MONITOR_PE_PANEL
    if not p.exists():
        print(f"[monitor] 未找到 PE 面板（{p.name}），估值预警跳过（可放入 data/ 后自动启用）")
        return None
    try:
        df = pd.read_parquet(p)
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.sort_index()
        return df
    except Exception as e:
        print(f"[monitor] PE 面板读取失败（{e}），估值预警跳过")
        return None


# ---------------------------------------------------------------------------
# 指标计算与预警判定
# ---------------------------------------------------------------------------
def _latest_value(series: pd.Series):
    """取序列最后有效值（面板已按日期升序）。"""
    s = series.dropna()
    return float(s.iloc[-1]) if len(s) else None


def compute_position_metrics(positions: dict, close: pd.DataFrame | None,
                             pe: pd.DataFrame | None) -> list[dict]:
    """对每只持仓计算监控指标。

    返回 [{code, name, cost, shares, latest, pnl, high, drawdown, pe}]。
    价格缺失的持仓跳过（cost 无法对比）。
    """
    names = load_names()
    metrics = []
    for code, h in positions.items():
        code = str(code)
        try:
            cost = float(h.get("cost", 0.0))
            shares = float(h.get("shares", 0.0))
        except (TypeError, ValueError):
            continue
        if cost <= 0 or shares <= 0:
            continue

        latest = None
        if close is not None and code in close.columns:
            s = close[code].dropna()
            if len(s):
                latest = float(s.iloc[-1])
        if latest is None or latest <= 0:
            print(f"[monitor] 持仓 {code} 在收盘价面板中无有效价格，跳过")
            continue

        # 持仓期间最高价：entry_date（含当日）之后的面板最高收盘；缺省回退全历史
        entry_date = h.get("entry_date")
        s = close[code].dropna()
        if entry_date:
            try:
                s = s[s.index >= pd.Timestamp(entry_date)]
            except Exception:
                pass
        high = float(s.max()) if len(s) else latest
        high = max(high, latest)

        pnl = (latest - cost) / cost
        drawdown = (high - latest) / high if high > 0 else 0.0

        pe_val = None
        if pe is not None and code in pe.columns:
            pe_val = _latest_value(pe[code])

        metrics.append({
            "code": code,
            "name": names.get(code, code),
            "cost": cost,
            "shares": shares,
            "latest": latest,
            "pnl": pnl,
            "high": high,
            "drawdown": drawdown,
            "pe": pe_val,
        })
    return metrics


def build_alerts(metrics: list[dict]) -> list[dict]:
    """按监控规则生成预警清单，并按盈亏率升序（亏损最重在前）排序。

    返回 [{code, name, pnl, drawdown, pe, tags: [str]}]，tags 为预警标签（含 emoji）。
    """
    alerts = []
    for m in metrics:
        tags = []
        if m["pnl"] < config.MONITOR_STOP_LOSS:
            tags.append("⚠️止损")
        if m["pnl"] > config.MONITOR_PROFIT_TARGET:
            tags.append("💰止盈")
        if m["drawdown"] > config.MONITOR_TRAILING_STOP:
            tags.append("📉移动止损")
        if m["pe"] is not None and m["pe"] > config.MONITOR_PE_RATIO:
            tags.append("📊估值")
        if tags:
            alerts.append({
                "code": m["code"],
                "name": m["name"],
                "pnl": m["pnl"],
                "drawdown": m["drawdown"],
                "pe": m["pe"],
                "tags": tags,
            })
    alerts.sort(key=lambda a: a["pnl"])
    return alerts


# ---------------------------------------------------------------------------
# 输出与推送
# ---------------------------------------------------------------------------
def _fmt_pct(v, signed: bool = False) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    if f != f:
        return "—"
    sign = "+" if signed and f > 0 else ""
    return f"{sign}{f * 100:.2f}%"


def print_alerts(alerts: list[dict], n_positions: int, as_of: str) -> None:
    """控制台输出预警清单（全量，不截断）。"""
    if not alerts:
        print(f"✅ 今日无预警（持仓 {n_positions} 只，行情截至 {as_of}）")
        return
    print(f"[monitor] ⚠️ 触发预警 {len(alerts)} 只（持仓 {n_positions} 只，行情截至 {as_of}）:")
    for a in alerts:
        line = (f"  {a['tags'][0]} {a['code']} {a['name']} "
                f"{_fmt_pct(a['pnl'], signed=True)} "
                f"(回撤 {_fmt_pct(a['drawdown'])}"
                + (f"，PE {a['pe']:.1f}" if a["pe"] is not None else "")
                + ")")
        if len(a["tags"]) > 1:
            line += f"  [{','.join(a['tags'])}]"
        print(line)


def build_push_text(alerts: list[dict], n_positions: int, as_of: str,
                    max_display: int) -> tuple[str, str]:
    """构造 Bark 推送 (title, body)。无预警时输出健康度报告。"""
    date_str = as_of or datetime.now().strftime("%Y-%m-%d")
    if not alerts:
        if n_positions == 0:
            return f"✅ 持仓健康 {date_str}", "当前无持仓，无需监控"
        return f"✅ 持仓健康 {date_str}", f"今日无预警（持仓 {n_positions} 只，行情截至 {as_of}）"

    title = f"⚠️ 持仓预警 {date_str}（{len(alerts)}只）"
    lines = []
    for a in alerts[:max_display]:
        tag_str = "/".join(a["tags"])
        lines.append(f"▪ {a['code']} {a['name']} {_fmt_pct(a['pnl'], signed=True)} {tag_str}")
    if len(alerts) > max_display:
        lines.append(f"… 共 {len(alerts)} 只（仅显示前 {max_display}）")
    return title, "\n".join(lines)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="持仓监控预警（仅提示，不自动卖出）")
    ap.add_argument("--state", type=str, default=None,
                    help="指定 sim_state.json 路径（默认 data/state → output/sim_nav 顺序回退）")
    ap.add_argument("--push", action="store_true",
                    help="强制 Bark 推送（默认仅打印；LIVE_MODE=true 环境变量等效 --push）")
    ap.add_argument("--no-push", action="store_true", help="强制不推送（覆盖 LIVE_MODE）")
    ap.add_argument("--preview", action="store_true",
                    help="仅打印推送正文预览，不发请求")
    ap.add_argument("--max-display", type=int, default=config.MONITOR_MAX_DISPLAY,
                    help="推送最多显示条数（默认 %(default)s）")
    args = ap.parse_args()

    # 推送开关：--no-push > --push > 环境变量 LIVE_MODE
    env_live = os.environ.get("LIVE_MODE", "").strip().lower() == "true"
    do_push = env_live or args.push
    if args.no_push:
        do_push = False

    # 1. 读取数据
    state, _src = load_sim_state(args.state)
    positions = state.get("positions", {}) or {}
    close = load_close_panel()
    pe = load_pe_panel()

    # 2. 计算指标 + 生成预警
    metrics = compute_position_metrics(positions, close, pe)
    alerts = build_alerts(metrics)
    as_of = ""
    if close is not None and len(close):
        as_of = str(pd.Timestamp(close.index[-1]).date())

    # 3. 控制台输出（始终打印；含预览时先输出正文）
    print_alerts(alerts, len(metrics), as_of or "—")
    title, body = build_push_text(alerts, len(metrics), as_of, args.max_display)
    if args.preview:
        print(f"[monitor] 推送预览（不发送）:\n标题: {title}\n正文:\n{body}")
        return 0

    # 4. 推送（条件触发）
    if do_push:
        from push_utils import push_to_bark
        ok = push_to_bark(title, body)
        print(f"[monitor] Bark 推送{'成功' if ok else '未发送（失败或未配置 key）'}: {title}")
    else:
        print("[monitor] 未启用推送（LIVE_MODE=false 或未加 --push；仅打印）")

    return 0


if __name__ == "__main__":
    sys.exit(main())
