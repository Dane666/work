# -*- coding: utf-8 -*-
"""V8.1 每日结果 Bark 推送（参考 Momentum/notify 模块）。

读取最新信号 CSV 与模拟盘 NAV，生成一份简明摘要并推送至手机。
BARK_DEVICE_KEY 通过环境变量注入（GitHub Actions secrets）；未配置时仅本地预览、不推送。

运行方式（从 src/ 目录）：
    python notify_push_daily.py
"""
import os
import sys
import glob
import logging
from pathlib import Path

import pandas as pd

# 允许从 src/ 直接运行（cwd=src 时 config 可导入）
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OUTPUT_DIR
from notify.bark import send_bark

# 让 bark.py 内的 logger 输出到控制台（Actions 日志可见真实 HTTP 状态）
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def latest_signal_csv():
    """按修改时间取最新信号 CSV。"""
    fs = sorted(glob.glob(str(OUTPUT_DIR / "signals" / "*_signal.csv")), key=os.path.getmtime)
    return fs[-1] if fs else None


def latest_nav_row():
    """取模拟盘净值历史最后一行。"""
    p = OUTPUT_DIR / "sim_nav" / "sim_nav_history.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        if df.empty:
            return None
        return df.iloc[-1].to_dict()
    except Exception:
        return None


def build_message():
    """构建 (title, body)。"""
    sig = latest_signal_csv()
    title = "V8.1 每日策略信号"
    if sig is None:
        return title, "⚠️ 未找到信号文件，请检查 signal_generator 输出。"

    df = pd.read_csv(sig, dtype={"code": str})
    sig_date = Path(sig).stem.replace("_signal", "")

    n_buy = int((df["action"] == "BUY").sum())
    n_sell = int((df["action"] == "SELL").sum())
    n_hold = int((df["action"] == "HOLD").sum())

    lines = [f"📅 信号截面: {sig_date}",
             f"▪ 指令: BUY {n_buy} / SELL {n_sell} / HOLD {n_hold}"]

    # 模拟盘净值
    row = latest_nav_row()
    if row is not None:
        nav = float(row.get("nav", 0) or 0)
        cum = float(row.get("cum_return", 0) or 0)
        cash = float(row.get("cash", 0) or 0)
        bench = float(row.get("benchmark_nav", 0) or 0)
        lines.append(f"▪ 模拟盘 NAV: {nav:.4f}  累计 {cum:+.2%}  现金 {cash:.1%}")
        if bench:
            excess = nav / bench - 1
            lines.append(f"▪ 基准(沪深300) NAV: {bench:.4f}  超额 {excess:+.2%}")

    # Top 买入清单
    buys = df[df["action"] == "BUY"].head(15)
    if len(buys):
        lines.append("")
        lines.append("Top 买入:")
        for i, (_, r) in enumerate(buys.iterrows(), 1):
            w = r.get("target_weight", "")
            wtxt = f" {float(w):.2f}%" if isinstance(w, (int, float)) and pd.notna(w) else f" {w}%"
            lines.append(f"{i}. {r['name']}({r['code']}){wtxt}")

    # 数据基准提示（Actions 用 data 分支面板，可能滞后）
    lines.append("")
    lines.append("⚠️ Actions 数据基准 2025-08-12，截面取最近月末；本地更新面板后 git push origin data 即生效。")
    return title, "\n".join(lines)


def main():
    title, body = build_message()
    print("=== 推送内容预览 ===")
    print(f"[{title}]\n{body}")

    key = os.environ.get("BARK_DEVICE_KEY", "").strip()
    if not key:
        print("\n⚠️ 未配置 BARK_DEVICE_KEY，仅本地预览，未推送。")
        return
    ok = send_bark(title, body, device_key=key)
    if ok:
        print("\n✅ Bark 推送成功（HTTP 200），手机应已收到。")
    else:
        print("\n❌ Bark 推送失败，请检查 BARK_DEVICE_KEY 是否正确 / Actions 网络。")


if __name__ == "__main__":
    main()
