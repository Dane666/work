# -*- coding: utf-8 -*-
"""
order_push.py — 委托单 Bark 推送（方向D，实盘模式收尾）
====================================================================

读取 sim_tracker.py --live 生成的委托单 CSV（output/orders/YYYY-MM-DD_orders.csv），
格式化为 Bark 摘要推送至手机：
  - 每条：{action} {code} {name} × {shares}股 ≈ {amount}元
    （action 映射：BUY→买入，SELL→卖出）
  - 超过 max_display 只时截断，末尾标注 "… 共 N 只"
  - 委托单为空 / 无 CSV 时推送 "今日无操作"

设计约束（与 push_utils 一致）：
  - 推送失败静默降级（返回 False、打印 warning，不抛异常、不中断 run_daily.sh）
  - Key 来自 config.BARK_DEVICE_KEY（环境变量 / 项目根 .env / Actions secret）
  - --preview 仅打印不推送（本地验证用，不发请求）

用法：
  cd src
  python order_push.py                    # 推送最新委托单摘要
  python order_push.py --preview          # 只打印预览，不推送
  python order_push.py --csv path.csv     # 指定委托单文件
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

for _k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
           "ALL_PROXY", "all_proxy"]:
    os.environ.pop(_k, None)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import config

ACTION_CN = {"BUY": "买入", "SELL": "卖出"}


def format_orders_for_bark(orders_csv_path: str | Path,
                           max_display: int = 15) -> str:
    """读取委托单 CSV，格式化为 Bark 文本。

    - 文件不存在 / 无有效行 → 返回 "今日无操作"
    - 每条：{action} {code} {name} × {shares}股 ≈ {amount}元
    - 超过 max_display 只 → 截断并在末尾标注 "… 共 N 只"
    """
    p = Path(orders_csv_path)
    if not p.exists():
        return "今日无操作"
    try:
        df = pd.read_csv(p, dtype={"code": str})
    except Exception as e:
        print(f"[order_push] 读取委托单失败（{e}），按空单处理")
        return "今日无操作"
    if df is None or len(df) == 0:
        return "今日无操作"

    lines = []
    for _, r in df.head(max_display).iterrows():
        action = ACTION_CN.get(str(r.get("direction", "")).upper(),
                               str(r.get("direction", "")))
        code = str(r.get("code", ""))
        name = str(r.get("name", code))
        try:
            shares = int(float(r.get("shares", 0)))
        except (TypeError, ValueError):
            shares = 0
        try:
            amount = float(r.get("amount", 0.0))
        except (TypeError, ValueError):
            amount = 0.0
        lines.append(f"{action} {code} {name} × {shares}股 ≈ {amount:,.0f}元")

    if len(df) > max_display:
        lines.append(f"… 共 {len(df)} 只")
    return "\n".join(lines)


def find_latest_orders(dir_path: Path | None = None) -> Path | None:
    """定位 output/orders/ 下最新一份 *_orders.csv；无则 None。"""
    d = dir_path or config.RISK_ORDERS_DIR
    if not d.exists():
        return None
    files = sorted(d.glob("*_orders.csv"))
    return files[-1] if files else None


def main() -> int:
    ap = argparse.ArgumentParser(description="委托单 Bark 推送")
    ap.add_argument("--csv", type=str, default=None,
                    help="指定委托单 CSV（默认 output/orders/ 下最新一份）")
    ap.add_argument("--max-display", type=int, default=15,
                    help="最多显示条数（默认 15）")
    ap.add_argument("--preview", action="store_true",
                    help="仅打印预览，不推送（本地验证用）")
    args = ap.parse_args()

    path = Path(args.csv) if args.csv else find_latest_orders()
    if path is None or not path.exists():
        text = "今日无操作"
        date_str = datetime.now().strftime("%Y-%m-%d")
        print(f"[order_push] 未找到委托单 CSV → 推送「今日无操作」")
    else:
        text = format_orders_for_bark(path, args.max_display)
        date_str = path.stem.replace("_orders", "")
        print(f"[order_push] 读取委托单: {path}")

    title = f"📈 实盘委托单 {date_str}"

    if args.preview:
        print(f"[order_push] 预览（不推送）:\n标题: {title}\n正文:\n{text}")
        return 0

    from push_utils import push_to_bark
    ok = push_to_bark(title, text)
    print(f"[order_push] 推送{'成功' if ok else '未发送（失败或未配置 key）'}: {title}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
