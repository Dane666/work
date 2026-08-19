"""
模拟盘偏离预警模块（B）
================================================================
- 每日自动计算模拟净值 vs V7.1 回测理论净值的偏差
- 超阈值（绝对偏差 > ±5% 或 年化跟踪误差 > 8%）则非零退出（便于 cron 捕获告警）

依赖：main_v7.py 已实现(C)并导出 output/theoretical_nav_v7_1.parquet
      sim_tracker.py 已累积 output/sim_nav/sim_nav_history.csv

阈值（推荐值）：
  ABS_DEVIATION_THRESHOLD = 0.05   绝对偏差 ±5%
  TRACKING_ERROR_THRESHOLD = 0.08  年化跟踪误差 8%

退出码：
  0  -> 正常 / 数据不足暂不判定
  1  -> 预警触发（偏差超阈值）
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 导入项目配置
sys.path.append(str(Path(__file__).parent))
from config import OUTPUT_DIR


# ================== 阈值配置 ==================
ABS_DEVIATION_THRESHOLD = 0.05      # 绝对偏差 ±5%
TRACKING_ERROR_THRESHOLD = 0.08     # 年化跟踪误差 8%


# ================== 加载理论净值 ==================
def load_theoretical_nav():
    """从 V7.1 导出的 parquet 读取理论净值（C 已实现）。"""
    theo_path = OUTPUT_DIR / "theoretical_nav_v7_1.parquet"
    if not theo_path.exists():
        print(f"[预警] ❌ 理论净值文件不存在: {theo_path}")
        print("[预警] 请先运行 main_v7.py 导出理论净值（C 已实现）")
        sys.exit(1)

    df = pd.read_parquet(theo_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    if "nav" not in df.columns:
        print("[预警] ❌ 理论净值文件缺少 'nav' 列")
        sys.exit(1)
    return df["nav"]


# ================== 加载最新模拟净值 ==================
def load_latest_sim_nav():
    """从 sim_tracker 的输出中读取净值序列。

    优先读取 sim_nav_history.csv（累积完整序列，便于计算跟踪误差）；
    若不存在则回退到最新一份 YYYY-MM-DD_nav.csv。
    """
    sim_dir = OUTPUT_DIR / "sim_nav"
    if not sim_dir.exists():
        print(f"[预警] ❌ 模拟净值目录不存在: {sim_dir}")
        return None

    hist = sim_dir / "sim_nav_history.csv"
    if hist.exists():
        df = pd.read_csv(hist)
        src = hist.name
    else:
        nav_files = sorted(sim_dir.glob("*_nav.csv"))
        if not nav_files:
            print("[预警] ❌ 未找到模拟净值 CSV 文件")
            return None
        df = pd.read_csv(nav_files[-1])
        src = nav_files[-1].name

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    if "nav" not in df.columns:
        print("[预警] ❌ 模拟净值文件缺少 'nav' 列")
        return None
    return df["nav"], src


# ================== 核心检查逻辑 ==================
def check_deviation(sim_nav, theo_nav):
    """计算偏差并判定是否触发预警。返回 dict。

    对齐策略：
      - 若两序列存在 ≥5 个日历重叠日（生产常态：理论净值随实时数据延伸后覆盖模拟盘日期），
        按日历日期交集对齐（mode='calendar'），结果直接解释为「同日偏差」。
      - 若日历无重叠且模拟盘已累积 ≥5 个交易日（沙箱/部署初期：理论净值数据截止早于模拟盘起始），
        退而按「交易日序号」对齐（mode='sequence'），即 sim 第 k 日 vs 理论第 k 日，
        用于监测执行轨迹是否与历史回测形态一致（不同市场区段，仅供参考）。
      - 模拟盘不足 5 日：返回 insufficient_data，暂不判定。
    """
    common = sim_nav.index.intersection(theo_nav.index)
    if len(common) >= 5:
        sim_a = sim_nav.loc[common]
        theo_a = theo_nav.loc[common]
        mode = "calendar"
        latest_date = common[-1]
    elif len(sim_nav) >= 5:
        k = min(len(sim_nav), len(theo_nav))
        sim_a = sim_nav.iloc[:k]
        theo_a = theo_nav.iloc[:k]
        mode = "sequence"
        latest_date = sim_nav.index[-1]
    else:
        return {
            "status": "insufficient_data",
            "message": f"模拟盘仅 {len(sim_nav)} 个交易日，不足5天，暂不判定",
            "latest_date": None,
            "mode": None,
            "latest_abs_dev": 0.0,
            "tracking_error_annual": 0.0,
        }

    # 绝对偏差（最新对齐点，二者均重定基到各自起点=1.0，避免基数差异误导）
    sim_rb = sim_a / sim_a.iloc[0]
    theo_rb = theo_a / theo_a.iloc[0]
    latest_abs_dev = sim_rb.iloc[-1] / theo_rb.iloc[-1] - 1

    # 年化跟踪误差：日收益差标准差 × √252
    # 用 .to_numpy() 按「位置」对齐，避免序列模式日期标签(2025 vs 2018)不对齐导致全 NaN
    r_sim = sim_a.pct_change().fillna(0).to_numpy()
    r_theo = theo_a.pct_change().fillna(0).to_numpy()
    daily_diff = r_sim - r_theo
    tracking_error = float(np.nanstd(daily_diff)) * np.sqrt(252)

    # 判定
    abs_trigger = abs(latest_abs_dev) > ABS_DEVIATION_THRESHOLD
    te_trigger = tracking_error > TRACKING_ERROR_THRESHOLD

    return {
        "status": "alert" if (abs_trigger or te_trigger) else "ok",
        "mode": mode,
        "latest_date": latest_date.strftime("%Y-%m-%d") if latest_date is not None else None,
        "latest_abs_dev": latest_abs_dev,
        "tracking_error_annual": tracking_error,
        "abs_trigger": abs_trigger,
        "te_trigger": te_trigger,
        "sim_days": len(sim_nav),
        "overlap_days": len(common),
        "thresholds": {
            "abs_dev": ABS_DEVIATION_THRESHOLD,
            "tracking_error": TRACKING_ERROR_THRESHOLD,
        },
    }


# ================== 主入口 ==================
if __name__ == "__main__":
    print("[预警] 开始模拟盘偏离检查...")

    # 1. 加载理论净值
    theo_nav = load_theoretical_nav()
    print(f"[预警] 理论净值加载成功，共 {len(theo_nav)} 天")

    # 2. 加载模拟净值
    sim_result = load_latest_sim_nav()
    if sim_result is None:
        print("[预警] ❌ 模拟净值加载失败，退出码1")
        sys.exit(1)
    sim_nav, sim_file = sim_result
    print(f"[预警] 模拟净值加载成功: {sim_file}，共 {len(sim_nav)} 天")

    # 3. 执行检查
    result = check_deviation(sim_nav, theo_nav)

    # 4. 输出结果
    print(f"[预警] 最新日期: {result.get('latest_date', 'N/A')}")
    if result["status"] == "insufficient_data":
        print(f"[预警] ⚠️ {result['message']}")
        sys.exit(0)  # 数据不足不算预警

    print(f"[预警] 最新绝对偏差: {result['latest_abs_dev']:.2%}")
    print(f"[预警] 年化跟踪误差: {result['tracking_error_annual']:.2%}")
    print(f"[预警] 阈值: 绝对偏差 ±{result['thresholds']['abs_dev']:.0%}, "
          f"跟踪误差 {result['thresholds']['tracking_error']:.0%}")
    _mode = result.get("mode", "-")
    _mode_desc = {"calendar": "同日对比（理论净值随实时数据已覆盖模拟盘日期）",
                  "sequence": "交易日序号对齐（无日历重叠，不同市场区段，仅供形态参考）"}.get(_mode, _mode)
    print(f"[预警] 对齐方式: {_mode} —— {_mode_desc}")

    if result["status"] == "alert":
        reasons = []
        if result["abs_trigger"]:
            reasons.append(f"绝对偏差 {result['latest_abs_dev']:.2%} 超阈值")
        if result["te_trigger"]:
            reasons.append(f"跟踪误差 {result['tracking_error_annual']:.2%} 超阈值")
        print(f"[预警] ⚠️ 预警触发！原因: {', '.join(reasons)}")
        sys.exit(1)  # 非零退出码，便于 cron / 监控捕获
    else:
        print("[预警] ✅ 偏差在正常范围内")
        sys.exit(0)
