# -*- coding: utf-8 -*-
"""
drawdown_watcher.py — 净值回撤预警（V3.2 风控辅助 · 净值事后审计）
====================================================================

每日收盘后（sim_tracker 更新净值后）对比「模拟盘净值」与「历史峰值」：
    回撤 = 当前净值 / 历史峰值 - 1（≤ 0）。
本模块只做净值的"事后审计"：只读 sim_nav_history.csv，不修改策略逻辑、不触发调仓。

预警分级（阈值见 config.DD_*_THRESHOLD）：
    -10% < 回撤 ≤ -5%    → ⚠️ 净值回撤关注
    -15% < 回撤 ≤ -10%   → 🚨 净值回撤警戒
     回撤 ≤ -15%         → 🔴 净值回撤严重（建议人工评估是否启动风控）
     回撤 > -5%          → ✅ 健康（不推送；自预警状态恢复时推送一次 ✅ 修复）

幂等（避免每天同样的消息骚扰）：data/state/last_dd_level.txt 记录「上次推送级别」
（0=健康 / 1=关注 / 2=警戒 / 3=严重）：
    同级            → 不重复推送
    级别加深(↑)     → 每次推送（首次破关注线、以及逐级加深）
    恢复至健康(→0)  → 推送一次修复通知
    级别收窄但未回健康(3→2/1、2→1) → 不推送（避免阈值边界震荡刷屏），仅更新记录

数据来源：
  - 净值：data/state/sim_nav_history.csv（date,nav 列；本地回退
    output/sim_nav/sim_nav_history.csv）
  - 幂等文件：data/state/last_dd_level.txt（随 data 分支持久化；--level-file 可覆盖）

用法：
  cd src && python drawdown_watcher.py            # 仅打印（不推送）
  python drawdown_watcher.py --push               # 打印 + Bark 推送（LIVE_MODE=true 等效）
  python drawdown_watcher.py --preview            # 打印推送正文预览（不发请求）
  python drawdown_watcher.py --sim-nav <csv>      # 指定净值文件（验证用）
  python drawdown_watcher.py --level-file <path>  # 覆盖幂等文件（验证用）
退出码恒为 0：回撤预警 ≠ 错误；净值缺失/解析异常内部降级，不中断 run_daily.sh 主流程。
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# 清代理（与其余 src 模块一致；防御性保留，本模块只读本地文件）
for _k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
           "ALL_PROXY", "all_proxy"]:
    os.environ.pop(_k, None)

# 兼容两种调用方式：python drawdown_watcher.py（sys.path[0]=src）
# 与 python -m src.drawdown_watcher（sys.path[0]=项目根）
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import config

# 级别元信息：level -> (emoji, 名称)
LEVEL_META = {
    1: ("⚠️", "关注"),
    2: ("🚨", "警戒"),
    3: ("🔴", "严重"),
}


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def load_nav(sim_path: str | Path | None = None) -> pd.Series:
    """读取模拟盘净值序列（date 索引 → nav，升序去重）。

    优先级：显式 --sim-nav > data/state/sim_nav_history.csv（持久化主路径）
            > output/sim_nav/sim_nav_history.csv（本地缓存）。
    解析失败/空 → 返回空 Series（不抛异常）。
    """
    candidates = []
    if sim_path:
        candidates.append((Path(sim_path), f"显式指定 {sim_path}"))
    candidates.append((config.SIM_STATE_DIR / "sim_nav_history.csv",
                       "data/state（持久化主路径）"))
    candidates.append((config.OUTPUT_DIR / "sim_nav" / "sim_nav_history.csv",
                       "output/sim_nav（本地缓存）"))

    for p, desc in candidates:
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
            if "date" not in df.columns or "nav" not in df.columns:
                print(f"[dd] {p} 缺少 date/nav 列，尝试下一来源")
                continue
            df["date"] = pd.to_datetime(df["date"])
            s = df.set_index("date")["nav"].astype(float).dropna().sort_index()
            s = s[~s.index.duplicated(keep="last")]
            if len(s) == 0:
                print(f"[dd] {p} 无有效 nav 记录，尝试下一来源")
                continue
            print(f"[dd] 读取净值记录: {p}（{desc}，{len(s)} 条, "
                  f"{s.index[0].date()} ~ {s.index[-1].date()}）")
            return s
        except Exception as e:
            print(f"[dd] 净值读取失败（{p}）: {e}，尝试下一来源")
    print("[dd] 未找到 sim_nav_history.csv（模拟盘尚无净值记录），按健康处理不推送")
    return pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# 回撤计算与分级
# ---------------------------------------------------------------------------
def compute_drawdown(nav: pd.Series) -> dict:
    """计算当前净值相对历史峰值的回撤指标。

    返回 {cur, peak, peak_date, dd, start_date, days}：
      cur       当前净值（最后一条）
      peak      截至当前日期的历史峰值 max(nav)
      peak_date 峰值日期
      dd        回撤 = cur/peak - 1（≤0；peak<=0 时视为 0）
      start_date 回撤起始日（峰值之后首个净值 < 峰值的交易日；无回撤为 None）
      days      回撤持续记录数（start_date 至最新的净值记录数；无回撤为 0）
    """
    if len(nav) == 0:
        return {"cur": None, "peak": None, "peak_date": None,
                "dd": 0.0, "start_date": None, "days": 0}
    cur = float(nav.iloc[-1])
    peak = float(nav.max())
    peak_date = nav.idxmax()          # 首次达到峰值的位置
    dd = (cur / peak - 1.0) if peak > 0 else 0.0

    start_date, days = None, 0
    if dd < 0:
        after_peak = nav[nav.index > peak_date]
        below = after_peak[after_peak < peak]
        if len(below):
            start_date = below.index[0]
            days = int((nav.index >= start_date).sum())
    return {"cur": cur, "peak": peak, "peak_date": peak_date,
            "dd": float(dd), "start_date": start_date, "days": days}


def level_of(dd: float) -> int:
    """回撤分级：0 健康 / 1 关注 / 2 警戒 / 3 严重。"""
    if dd > config.DD_WATCH_THRESHOLD:      # 回撤 > -5%
        return 0
    if dd > config.DD_ALERT_THRESHOLD:      # -10% < 回撤 ≤ -5%
        return 1
    if dd > config.DD_CRITICAL_THRESHOLD:   # -15% < 回撤 ≤ -10%
        return 2
    return 3                                # 回撤 ≤ -15%


def read_last_level(level_file: Path) -> int | None:
    """读取上次推送级别；文件缺失/损坏返回 None。"""
    if not level_file.exists():
        return None
    try:
        return int(level_file.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def write_last_level(level_file: Path, level: int) -> None:
    """写幂等标记。写入失败仅 warn（不阻塞主流程）。"""
    try:
        level_file.parent.mkdir(parents=True, exist_ok=True)
        level_file.write_text(f"{level}\n", encoding="utf-8")
    except Exception as e:
        print(f"[dd] 幂等文件写入失败（{level_file}）: {e}")


def decide_action(new_level: int, stored_level: int | None) -> str:
    """推送决策。

    返回：'healthy'（健康不推送）/ 'skip'（同级不重复）
          / 'first'（首次破关注线）/ 'escalate'（级别加深）
          / 'recover'（自预警恢复健康）
    """
    if new_level == 0:
        return "recover" if stored_level in (1, 2, 3) else "healthy"
    if stored_level is None:
        return "first"
    if new_level > stored_level:
        return "escalate"
    if new_level == stored_level:
        return "skip"
    return "improve"          # 0 < new < stored：级别收窄但未回健康 → 不推送


# ---------------------------------------------------------------------------
# 输出与推送
# ---------------------------------------------------------------------------
def _fmt_pct(v, signed: bool = False) -> str:
    if v is None or v != v:  # None 或 NaN
        return "—"
    sign = "+" if signed and v > 0 else ""
    return f"{sign}{v * 100:.2f}%"


def build_push_text(m: dict, level: int, action: str) -> tuple[str, str]:
    """构造 (title, body)。level=0 仅出现在 recover 场景。"""
    cur = m["cur"]
    peak = m["peak"]
    peak_s = m["peak_date"].strftime("%Y-%m-%d") if m["peak_date"] is not None else "—"
    start_s = m["start_date"].strftime("%Y-%m-%d") if m["start_date"] is not None else "—"

    if level == 0:   # 恢复健康
        title = "✅ 净值回撤修复"
        body = (f"当前净值: {cur:.4f}\n历史峰值: {peak:.4f} ({peak_s})\n"
                f"回撤 {_fmt_pct(m['dd'])}，已收窄至 -5% 关注线以内，恢复健康 ✅")
        return title, body

    emoji, name = LEVEL_META[level]
    title = f"{emoji} 净值回撤{name} ({_fmt_pct(m['dd'])})"
    lines = [
        f"当前净值: {cur:.4f}",
        f"历史峰值: {peak:.4f} ({peak_s})",
        f"起始日: {start_s}",
        f"持续天数: {m['days']}",
    ]
    if action == "first":
        lines.append("（首次触发，请关注模拟盘净值走势）")
    elif action == "escalate":
        lines.append("（回撤进一步加深 ⚠️→🚨→🔴，建议人工评估）")
    return title, "\n".join(lines)


def print_summary(m: dict, level: int, action: str) -> None:
    """控制台摘要（始终打印，不依赖推送开关）。"""
    if m["cur"] is None:
        print("[dd] ✅ 净值无异常回撤（无净值记录）")
        return
    cur, peak = m["cur"], m["peak"]
    peak_s = m["peak_date"].strftime("%Y-%m-%d") if m["peak_date"] is not None else "—"
    print(f"[dd] 当前净值 {cur:.4f} | 历史峰值 {peak:.4f} ({peak_s}) | "
          f"回撤 {_fmt_pct(m['dd'])}"
          + (f" | 起始日 {m['start_date'].date()} 持续 {m['days']} 天"
             if m["start_date"] is not None else ""))
    if level == 0:
        print("[dd] ✅ 净值无异常回撤（未破 -5% 关注线）"
              + ("，已自预警恢复" if action == "recover" else ""))
    else:
        emoji, name = LEVEL_META[level]
        hint = {"first": "首次触发", "escalate": "级别加深",
                "skip": "同级已推送，今日跳过（幂等）",
                "improve": "级别收窄未回健康，今日不推送"}.get(action, "")
        print(f"[dd] {emoji} 触发净值回撤{name}（{_fmt_pct(m['dd'])}）"
              + (f" —— {hint}" if hint else ""))


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="净值回撤预警（事后审计，仅提示不调仓）")
    ap.add_argument("--sim-nav", type=str, default=None,
                    help="指定 sim_nav_history.csv 路径（默认 data/state → output/sim_nav）")
    ap.add_argument("--level-file", type=str, default=None,
                    help="指定幂等级别文件路径（默认 data/state/last_dd_level.txt）")
    ap.add_argument("--push", action="store_true",
                    help="强制 Bark 推送（默认仅打印；LIVE_MODE=true 环境变量等效 --push）")
    ap.add_argument("--no-push", action="store_true", help="强制不推送（覆盖 LIVE_MODE）")
    ap.add_argument("--preview", action="store_true",
                    help="仅打印推送正文预览，不发请求")
    args = ap.parse_args()

    # 推送开关：--no-push > --push > 环境变量 LIVE_MODE
    env_live = os.environ.get("LIVE_MODE", "").strip().lower() == "true"
    do_push = env_live or args.push
    if args.no_push:
        do_push = False

    # 1. 读取净值 + 幂等标记
    nav = load_nav(args.sim_nav)
    level_file = Path(args.level_file) if args.level_file else config.DD_LEVEL_FILE
    stored = read_last_level(level_file)

    # 2. 计算回撤与级别
    m = compute_drawdown(nav)
    if m["cur"] is None:
        print("[dd] ✅ 净值无异常回撤（无净值记录，跳过推送）")
        return 0
    level = level_of(m["dd"])
    action = decide_action(level, stored)

    # 3. 输出
    print_summary(m, level, action)
    title, body = build_push_text(m, level, action)

    # 4. 是否推送（healthy / skip / improve 不推）
    push_now = action in ("first", "escalate", "recover")
    if args.preview:
        if push_now:
            print(f"[dd] 推送预览（不发送）:\n标题: {title}\n正文:\n{body}")
        else:
            print("[dd] 本次无推送（预览）: 健康/同级幂等/级别收窄")
        return 0

    if push_now and do_push:
        from push_utils import push_to_bark
        ok = push_to_bark(title, body)
        print(f"[dd] Bark 推送{'成功' if ok else '未发送（失败或未配置 key）'}: {title}")
    elif push_now:
        print("[dd] 触发推送但未启用推送（LIVE_MODE=false 或未加 --push；仅打印）")
    else:
        print("[dd] 无需推送（健康 / 同级幂等 / 级别收窄未回健康）")

    # 5. 更新幂等级别（无论是否推送都记录当前级别）
    write_last_level(level_file, level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
