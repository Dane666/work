# -*- coding: utf-8 -*-
"""
risk_manager.py — 实盘风控与委托单生成（方向C）
====================================================================

将 signal_generator 的目标权重转换为「可执行委托单」，叠加 4 道风控：
  - 单只持仓上限 MAX_SINGLE_POSITION_PCT（NAV 占比，硬性截断）
  - 总仓位上限   MAX_TOTAL_POSITION_PCT （Σ(持仓+新买入) ≤ 比例）
  - 单日累计买入上限 MAX_DAILY_BUY_AMOUNT（元）
  - 单日跌幅熔断 MAX_DAILY_LOSS_PCT（日 NAV 跌幅低于阈值则禁止新 BUY）

委托单格式（output/orders/YYYY-MM-DD_orders.csv）：
  code, name, direction(BUY/SELL), price_type(limit/market),
  price(限价), shares, amount, reason

价格口径（与 V8.1 EXECUTION_PRICE="next_open" 对齐）：
  - BUY  限价 = 次日开盘价 × (1 + 0.05%)  // 小溢价抢成交
  - SELL 限价 = 当日收盘价 × (1 - 0.10%)  // 滑点保护
  - 行情缺失时退化为市价单

约束（实盘必须遵守）：
  - LIVE_MODE 默认 false；任何"实际下单"必须在券商端人工或可信 SDK 中执行
  - 风控参数从 config 加载；激活前请先在 config.py 复核券商账户约束
  - 本模块输出 CSV 不与券商 API 直连（券商 SDK 集成见 README "实盘接入指南"）
"""

from __future__ import annotations

import os
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

ORDER_COLS = ["code", "name", "direction", "price_type", "price",
              "shares", "amount", "reason"]


# ---------------------------------------------------------------------------
# 风控过滤与委托单生成
# ---------------------------------------------------------------------------
def build_orders(signal_df: pd.DataFrame,
                 sim_state: dict,
                 prices: dict,
                 as_of: pd.Timestamp,
                 nav: float | None = None,
                 nav_unit: float = 1.0,
                 slip_buy: float = 0.0005,
                 slip_sell: float = 0.0010) -> pd.DataFrame:
    """生成本日「可执行委托单」。

    Parameters
    ----------
    signal_df : signal_generator 输出（code, name, action, target_weight[%]）
    sim_state : 当前 sim_state（positions, nav, last_nav_date, daily_return?）
    prices    : {code: {'open':, 'close':}} 当日行情（fetch_day_prices 输出）
    as_of     : 执行日（限价基准 / 文件名）
    nav       : NAV 分数（如 1.331 表示 +33.1%）；与 nav_unit 相乘得金额（元）
    nav_unit  : NAV 分数单位（元/分数），默认 1.0（即 nav 直接是元）。通常传入
                config.INIT_CAPITAL（如 1_000_000），将 nav=1.331 → 1_331_000 元
    slip_buy  : BUY 限价溢价（默认 0.05%）
    slip_sell : SELL 限价贴水（默认 0.10%）

    Returns
    -------
    pd.DataFrame（列 = ORDER_COLS）；可能为空（无成交）。
    """
    # ---- 风控参数 ----
    max_daily_buy = config.RISK_MAX_DAILY_BUY_AMOUNT
    max_single_pct = config.RISK_MAX_SINGLE_POSITION_PCT
    max_total_pct = config.RISK_MAX_TOTAL_POSITION_PCT
    max_daily_loss = config.RISK_MAX_DAILY_LOSS_PCT

    nav_score = float(sim_state.get("nav", 1.0)) if nav is None else float(nav)
    nav_amount = nav_score * nav_unit
    if nav_amount <= 0:
        nav_amount = 1.0

    positions = sim_state.get("positions", {}) or {}
    daily_ret = float(sim_state.get("daily_return", 0.0))
    buy_disabled = daily_ret < max_daily_loss           # 跌幅熔断

    # 当前持仓市值（用价格表中的 close，无则用 cost）
    cur_mv: dict = {}
    for code, h in positions.items():
        sh = float(h.get("shares", 0.0))
        if sh <= 0:
            continue
        px = float(prices.get(str(code), {}).get("close") or h.get("cost", 0.0))
        if px <= 0:
            continue
        cur_mv[str(code)] = sh * px
    total_pv = sum(cur_mv.values())

    orders: list = []
    cum_buy = 0.0

    for _, row in signal_df.iterrows():
        code = str(row["code"])
        name = str(row.get("name", code))
        action = str(row.get("action", "")).upper()
        try:
            target_w = float(row.get("target_weight", 0.0)) / 100.0      # % → ratio
        except (TypeError, ValueError):
            target_w = 0.0

        # 1) SELL：清仓跌出目标组合的标的
        if action == "SELL" and code in positions:
            px = float(prices.get(code, {}).get("close")
                       or positions[code].get("cost", 0.0))
            sh = float(positions[code].get("shares", 0.0))
            if sh <= 0 or px <= 0:
                continue
            limit_px = px * (1 - slip_sell)
            amt = sh * limit_px
            orders.append({
                "code": code, "name": name, "direction": "SELL",
                "price_type": "limit", "price": round(limit_px, 4),
                "shares": int(sh), "amount": round(amt, 2),
                "reason": f"信号 SELL（跌出目标组合）",
            })
            continue

        # 2) BUY / HOLD 中的建仓增量
        if action in ("BUY", "HOLD") and code not in positions:
            if buy_disabled:
                continue                                            # 熔断：新开仓禁止
            target_amt = target_w * nav_amount
            cur_amt = cur_mv.get(code, 0.0)
            add_amt = max(0.0, target_amt - cur_amt)
            if add_amt <= 0:
                continue
            # 单只上限
            cap = max_single_pct * nav_amount
            add_amt = min(add_amt, cap - cur_amt)
            if add_amt <= 100.0:
                continue
            # 单日累计买入上限
            if cum_buy + add_amt > max_daily_buy:
                add_amt = max(0.0, max_daily_buy - cum_buy)
                if add_amt <= 100.0:
                    continue
            px = float(prices.get(code, {}).get("open") or cur_mv.get(code, 0.0) / max(1.0, float(positions.get(code, {}).get("shares", 1.0))) or 0.0)
            if px <= 0:
                px = float(positions.get(code, {}).get("cost", 0.0))
            if px <= 0:
                continue
            limit_px = px * (1 + slip_buy) if px > 0 else 0.0
            # 按手取整（A股 100 股一手）
            shares = int(int(add_amt / limit_px) // 100 * 100)
            if shares <= 0:
                continue
            amt = shares * limit_px
            cum_buy += amt
            orders.append({
                "code": code, "name": name, "direction": "BUY",
                "price_type": "limit", "price": round(limit_px, 4),
                "shares": int(shares), "amount": round(amt, 2),
                "reason": (f"信号 BUY，目标 {target_w*100:.2f}%；"
                           f"风控{cap/1000:.0f}千/单日{max_daily_buy/1000:.0f}千"),
            })

    # 3) 总仓位上限过滤（持仓 + 累计买入 ≤ 比例）
    total_buy = sum(o["amount"] for o in orders if o["direction"] == "BUY")
    target_total = max_total_pct * nav_amount
    excess = (total_pv + total_buy) - target_total
    if excess > 0 and total_buy > 0:
        scale = max(0.0, (target_total - total_pv) / max(total_buy, 1e-9))
        if scale <= 0:
            orders = [o for o in orders if o["direction"] != "BUY"]
        else:
            kept = []
            for o in orders:
                if o["direction"] != "BUY":
                    kept.append(o)
                    continue
                new_sh = int(int(o["shares"]) * scale // 100 * 100)
                if new_sh <= 0:
                    continue
                o["shares"] = new_sh
                o["amount"] = round(new_sh * o["price"], 2)
                o["reason"] += f" / 总仓位上限缩量至 {scale*100:.0f}%"
                kept.append(o)
            orders = kept

    df = pd.DataFrame(orders, columns=ORDER_COLS)
    return df


def write_orders(orders_df: pd.DataFrame, as_of: pd.Timestamp,
                 out_dir: Path | None = None) -> Path:
    """写委托单到 {out_dir}/YYYY-MM-DD_orders.csv，返回写入路径。"""
    if out_dir is None:
        out_dir = config.RISK_ORDERS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{as_of.strftime('%Y-%m-%d')}_orders.csv"
    orders_df.to_csv(p, index=False, encoding="utf-8")
    return p