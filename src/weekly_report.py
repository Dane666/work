# -*- coding: utf-8 -*-
"""
weekly_report.py — 模拟盘 vs 回测理论净值 周报对比（V3.2 风控辅助，方向G）
====================================================================

每周最后一个交易日对比「模拟盘实际净值」与「V3.2 回测理论净值」，
生成偏差报告并通过 Bark 推送摘要（偏差率超 ±5% 附加预警）。

数据来源：
  - 模拟盘净值：data/state/sim_nav_history.csv（date, nav 列；本地回退
    output/sim_nav/sim_nav_history.csv）
  - 回测理论净值：data/state/theoretical_nav.csv（date, nav 列）。若缺失，
    自动触发一次回测导出：main_mainboard_v3.py --stage 3 --export-nav
    （V3.2 完整 stage=3 的净值序列；约需几分钟，失败则打印友好提示不推送）。

对比口径（两序列起点基准不同，先统一锚点再比）：
  模拟盘 NAV 基准 = 1.0（sim_state 现金基准）；回测 eq 是自 1.0 累计的绝对净值。
  处理：取两序列共同交易日（sim 记录日在回测日期区间内），把回测净值在
  "首个共同交易日"归一为 1.0（theo_norm = theo / theo[t0]），再与 sim 对齐。
  —— 这样比较的是「同期相对收益」，与基准货币单位无关。

指标：
  最新偏差率        = sim_nav / theo_norm - 1（最新共同日）
  本周偏差变化      = 最新偏差率 - 上周五（最后一个周五或 5 个交易日前）偏差率
  20日偏差标准差    = 近 20 个共同交易日的偏差率 std（数据不足显示 —）
  ⚠️ 标记           = |最新偏差率| > WEEKLY_DEV_THRESHOLD（默认 5%）

用法：
  cd src && python weekly_report.py            # 仅打印
  python weekly_report.py --push               # 打印 + Bark 推送
  python weekly_report.py --preview            # 打印推送正文预览（不发请求）
  python weekly_report.py --sim-nav <csv>      # 指定模拟盘净值文件（验证用）
  python weekly_report.py --theo-nav <csv>     # 指定理论净值文件（不自动导出）
  python weekly_report.py --no-export          # 理论文件缺失时禁止自动导出（仅提示）
退出码恒为 0：对比结果 ≠ 错误；异常内部降级，不中断 run_daily.sh 主流程。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# 清代理（与其余 src 模块一致；防御性保留，本模块只读本地文件）
for _k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
           "ALL_PROXY", "all_proxy"]:
    os.environ.pop(_k, None)

# 兼容两种调用方式：python weekly_report.py（sys.path[0]=src）
# 与 python -m src.weekly_report（sys.path[0]=项目根）
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import config

# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def load_sim_nav(sim_path: str | Path | None = None) -> tuple[pd.Series, str]:
    """读取模拟盘净值序列（date 索引 → nav）。

    优先级：显式 --sim-nav > data/state/sim_nav_history.csv（持久化主路径）
            > output/sim_nav/sim_nav_history.csv（本地缓存）。
    解析失败/空 → 返回空 Series。
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
            df = pd.read_csv(p, dtype={"code": str})
            if "date" not in df.columns or "nav" not in df.columns:
                print(f"[weekly] {p} 缺少 date/nav 列，尝试下一来源")
                continue
            df["date"] = pd.to_datetime(df["date"])
            s = df.set_index("date")["nav"].astype(float).dropna().sort_index()
            s = s[~s.index.duplicated(keep="last")]
            if len(s) == 0:
                print(f"[weekly] {p} 无有效 nav 记录，尝试下一来源")
                continue
            print(f"[weekly] 读取模拟盘净值: {p}（{desc}，{len(s)} 条, "
                  f"{s.index[0].date()} ~ {s.index[-1].date()}）")
            return s, desc
        except Exception as e:
            print(f"[weekly] 模拟盘净值读取失败（{p}）: {e}，尝试下一来源")
    print("[weekly] 未找到 sim_nav_history.csv（模拟盘尚无净值记录）")
    return pd.Series(dtype=float), "无模拟盘净值"


def export_theoretical_nav(timeout_sec: int = 1800) -> bool:
    """缺失理论净值时自动触发回测导出（V3.2 stage=3 完整净值序列）。"""
    print(f"[weekly] ⚠️ 理论净值缺失 → 自动触发回测导出 "
          f"(main_mainboard_v3.py --stage 3 --export-nav，最长 {timeout_sec}s)...")
    try:
        r = subprocess.run(
            [sys.executable, "main_mainboard_v3.py", "--stage", "3", "--export-nav"],
            cwd=str(Path(__file__).resolve().parent),
            capture_output=True, text=True, timeout=timeout_sec)
        if r.returncode == 0 and config.THEORETICAL_NAV.exists():
            print("[weekly] ✅ 回测导出成功，理论净值已生成")
            return True
        print(f"[weekly] ❌ 回测导出失败（exit={r.returncode}）:\n"
              f"{r.stdout[-800:]}\n{r.stderr[-800:]}")
    except subprocess.TimeoutExpired:
        print(f"[weekly] ❌ 回测导出超时（>{timeout_sec}s），本次跳过对比")
    except Exception as e:
        print(f"[weekly] ❌ 回测导出异常: {e}")
    return False


def load_theoretical_nav(theo_path: str | Path | None = None,
                         no_export: bool = False) -> tuple[pd.Series, str]:
    """读取回测理论净值序列。文件缺失且未禁止自动导出时触发一次回测。"""
    p = Path(theo_path) if theo_path else config.THEORETICAL_NAV
    if p.exists():
        try:
            df = pd.read_csv(p)
            if "date" not in df.columns or "nav" not in df.columns:
                print(f"[weekly] ⚠️ {p} 缺少 date/nav 列（格式: date,nav），按缺失处理")
            else:
                df["date"] = pd.to_datetime(df["date"])
                s = (df.set_index("date")["nav"].astype(float).dropna()
                     .sort_index())
                s = s[~s.index.duplicated(keep="last")]
                if len(s) == 0:
                    raise ValueError("空序列")
                print(f"[weekly] 读取回测理论净值: {p}（{len(s)} 条, "
                      f"{s.index[0].date()} ~ {s.index[-1].date()}）")
                return s, str(p)
        except Exception as e:
            print(f"[weekly] 理论净值读取失败（{p}）: {e}")

    if no_export or theo_path:
        print(f"[weekly] 理论净值文件不存在: {p}"
              + ("（--no-export 已禁止自动导出）" if no_export else ""))
        return pd.Series(dtype=float), "无理论净值"
    if export_theoretical_nav():
        return load_theoretical_nav(p, no_export=True)  # 递归读（不再次导出）
    return pd.Series(dtype=float), "无理论净值（导出失败）"


# ---------------------------------------------------------------------------
# 对齐与指标计算
# ---------------------------------------------------------------------------
def _fmt_pct(v, signed: bool = False) -> str:
    if v is None or v != v:  # None 或 NaN
        return "—"
    sign = "+" if signed and v > 0 else ""
    return f"{sign}{v * 100:.2f}%"


def compute_deviation_series(sim: pd.Series, theo: pd.Series) -> pd.Series:
    """对齐共同交易日，返回偏差率序列（sim_nav / theo_norm - 1）。

    theo 在首个共同交易日归一为 1.0，消除两序列基准差异（模拟盘 1.0 基准
    vs 回测绝对净值累计），使偏差反映"同期相对收益差"。
    """
    common = sim.index.intersection(theo.index)
    if len(common) == 0:
        return pd.Series(dtype=float)
    common = common.sort_values()
    t0 = common[0]
    theo_norm = theo.loc[common] / float(theo.loc[t0])
    dev = sim.loc[common] / theo_norm - 1.0
    return dev


def last_weekday_of_week(dates: pd.DatetimeIndex) -> pd.Timestamp:
    """返回 dates 中每个自然周最后一个交易日（用于"上周五"口径）。"""
    if len(dates) == 0:
        return None
    d = dates[-1]
    # 回退找该周最后一个 ≤ 最新日 且属于 dates 的交易日：从最新日往前
    # 直到跨周（简化：最近一次周一~周五内的最后记录即本周收尾日）
    week_last = dates[dates <= d]
    # 取最新记录所在 ISO 周内的最后一条
    iso = d.isocalendar()
    in_same_week = [x for x in week_last
                    if x.isocalendar()[:2] == iso[:2]]
    if in_same_week:
        return in_same_week[-1]
    return d


def compute_weekly_metrics(dev: pd.Series,
                           threshold: float) -> dict:
    """从偏差序列计算周报指标。"""
    if len(dev) == 0:
        return {"latest_dev": None, "prev_dev": None,
                "weekly_change": None, "dev_std20": None,
                "n_points": 0, "as_of": None, "flag": False}
    latest_dev = float(dev.iloc[-1])
    as_of = dev.index[-1]
    n = len(dev)

    # 本周 vs 上周五：取最近两个"周收尾交易日"（或 5 个交易日前）
    # 简化稳健口径：最新日 vs 最近一个 ≥5 交易日前的记录（≈上周五）
    prev = None
    if n >= 6:
        cutoff = dev.index[-1] - pd.Timedelta(days=7)
        older = dev[dev.index <= cutoff]
        if len(older):
            prev = float(older.iloc[-1])
    elif n >= 2:
        prev = float(dev.iloc[-2])   # 数据太少：用前一记录近似
    weekly_change = (latest_dev - prev) if prev is not None else None

    # 近 20 个共同交易日偏差率标准差
    dev_std = None
    if n >= 2:
        dev_std = float(dev.tail(20).std())

    flag = abs(latest_dev) > threshold
    return {"latest_dev": latest_dev, "prev_dev": prev,
            "weekly_change": weekly_change, "dev_std20": dev_std,
            "n_points": n, "as_of": as_of, "flag": flag}


# ---------------------------------------------------------------------------
# 输出与推送
# ---------------------------------------------------------------------------
def build_report_text(m: dict, sim_latest: float | None,
                      theo_latest: float | None) -> tuple[str, str]:
    """构造周报 (title, body)。m 为 compute_weekly_metrics 结果。"""
    date_str = m["as_of"].strftime("%Y-%m-%d") if m["as_of"] is not None \
        else datetime.now().strftime("%Y-%m-%d")

    if m["n_points"] == 0 or m["latest_dev"] is None:
        return (f"📋 净值周报 {date_str}",
                "暂无共同交易日，无法对比（模拟盘尚未产生净值记录）")

    flag_str = " ⚠️ 超±5%关注线" if m["flag"] else ""
    title = f"📊 模拟盘周报 {date_str}{flag_str}"

    lines = [
        f"模拟盘 NAV {sim_latest:.4f} vs 理论 {theo_latest:.4f}",
        f"▪ 最新偏差率 {_fmt_pct(m['latest_dev'], signed=True)}"
        + (" ⚠️" if m["flag"] else ""),
    ]
    if m["weekly_change"] is not None:
        lines.append(f"▪ 本周变化 {_fmt_pct(m['weekly_change'], signed=True)}")
    else:
        lines.append("▪ 本周变化 —（数据不足）")
    if m["dev_std20"] is not None:
        lines.append(f"▪ 20日偏差σ {_fmt_pct(m['dev_std20'])}")
    else:
        lines.append("▪ 20日偏差σ —（需≥2个共同交易日）")
    lines.append(f"▪ 对比样本 {m['n_points']} 个共同交易日")
    if m["flag"]:
        lines.append("⚠️ 偏差率超出关注线(±5%)，请核查模拟盘与回测的持仓差异")
    return title, "\n".join(lines)


def print_report(m: dict, sim_latest: float | None,
                 theo_latest: float | None) -> None:
    if m["n_points"] == 0 or m["latest_dev"] is None:
        print("[weekly] 无可对比数据：模拟盘与回测理论净值无共同交易日")
        return
    print(f"[weekly] 周报对比（截至 {m['as_of'].date()}，"
          f"{m['n_points']} 个共同交易日）:")
    print(f"  模拟盘 NAV = {sim_latest:.4f} | 回测理论(归一) = {theo_latest:.4f}")
    print(f"  最新偏差率 = {_fmt_pct(m['latest_dev'], signed=True)}"
          + ("  ⚠️ 超关注线" if m["flag"] else "  ✅ 正常"))
    if m["weekly_change"] is not None:
        print(f"  本周偏差变化 = {_fmt_pct(m['weekly_change'], signed=True)}")
    else:
        print("  本周偏差变化 = —（数据不足）")
    if m["dev_std20"] is not None:
        print(f"  20日偏差标准差 = {_fmt_pct(m['dev_std20'])}")
    else:
        print("  20日偏差标准差 = —（需≥2个共同交易日）")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="模拟盘 vs 回测理论净值周报对比")
    ap.add_argument("--sim-nav", type=str, default=None,
                    help="指定 sim_nav_history.csv 路径（默认 data/state → output/sim_nav）")
    ap.add_argument("--theo-nav", type=str, default=None,
                    help="指定 theoretical_nav.csv 路径（默认 data/state/theoretical_nav.csv）")
    ap.add_argument("--no-export", action="store_true",
                    help="理论净值缺失时禁止自动回测导出（仅友好提示）")
    ap.add_argument("--push", action="store_true",
                    help="强制 Bark 推送（默认仅打印；LIVE_MODE=true 环境变量等效 --push）")
    ap.add_argument("--no-push", action="store_true", help="强制不推送（覆盖 LIVE_MODE）")
    ap.add_argument("--preview", action="store_true",
                    help="仅打印推送正文预览，不发请求")
    ap.add_argument("--threshold", type=float, default=config.WEEKLY_DEV_THRESHOLD,
                    help="偏差关注阈值（默认 ±%(default)s）")
    args = ap.parse_args()

    env_live = os.environ.get("LIVE_MODE", "").strip().lower() == "true"
    do_push = env_live or args.push
    if args.no_push:
        do_push = False

    # 1. 读取数据
    sim, _s = load_sim_nav(args.sim_nav)
    theo, _t = load_theoretical_nav(args.theo_nav, no_export=args.no_export)
    if len(sim) == 0 or len(theo) == 0:
        title, body = build_report_text(
            {"n_points": 0, "latest_dev": None, "as_of": None,
             "flag": False, "weekly_change": None, "dev_std20": None,
             "prev_dev": None}, None, None)
        print(body)
        if args.preview:
            print(f"[weekly] 推送预览（不发送）:\n标题: {title}\n正文:\n{body}")
            return 0
        if do_push:
            from push_utils import push_to_bark
            ok = push_to_bark(title, body)
            print(f"[weekly] Bark 推送{'成功' if ok else '未发送（失败或未配置 key）'}: {title}")
        return 0

    # 2. 对齐并计算
    dev = compute_deviation_series(sim, theo)
    m = compute_weekly_metrics(dev, args.threshold)

    # 归一化理论最新值（用于展示）
    theo_latest = None
    if m["as_of"] is not None and m["as_of"] in theo.index:
        t0 = dev.index[0]
        theo_latest = float(theo.loc[m["as_of"]] / theo.loc[t0])
    sim_latest = float(sim.loc[dev.index[-1]]) if len(dev) else None

    # 3. 输出
    print_report(m, sim_latest, theo_latest)
    title, body = build_report_text(m, sim_latest, theo_latest)
    if args.preview:
        print(f"[weekly] 推送预览（不发送）:\n标题: {title}\n正文:\n{body}")
        return 0

    # 4. 推送（条件触发；超阈值附加预警已含在正文）
    if do_push:
        from push_utils import push_to_bark
        ok = push_to_bark(title, body)
        print(f"[weekly] Bark 推送{'成功' if ok else '未发送（失败或未配置 key）'}: {title}")
    else:
        print("[weekly] 未启用推送（LIVE_MODE=false 或未加 --push；仅打印）")

    return 0


if __name__ == "__main__":
    sys.exit(main())
