#!/bin/bash
# ============================================================================
# V7.1 每日模拟盘自动任务
# ----------------------------------------------------------------------------
# 功能：每个交易日收盘后(15:30) 依次执行
#        1) signal_generator.py  —— 生成次日买卖信号清单
#        2) sim_tracker.py       —— 按信号模拟成交并记录当日净值
#
# 仅工作日运行。cron 配置（编辑: crontab -e）：
#       30 15 * * 1-5 /Users/admin/WorkBuddy/stategy/scripts/run_daily.sh >> /Users/admin/WorkBuddy/stategy/output/cron.log 2>&1
#   （周一~周五 15:30；非交易日 akshare 无数据，脚本自动记录挂起状态，不会报错）
#
# 说明：
#   - 依赖 Python 须含 akshare / lightgbm / pandas / numpy，即本项目的托管 venv。
#     若换环境，请把 PYTHON 改为该环境 python 的绝对路径。
#   - 显式 unset 代理变量，避免 akshare 东方财富源在代理下被拦截。
# ============================================================================
set -e

# ---- 依赖 Python（默认 python3；可用环境变量 PYTHON=/path/to/python 覆盖）----
# 本地/云服务器：先激活 venv（source .venv/bin/activate）再运行本脚本；
# 或直接 PYTHON=/opt/stategy/.venv/bin/python bash scripts/run_daily.sh
PYTHON="${PYTHON:-python3}"

# ---- 定位 src 目录（本脚本位于 project/scripts/）----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$SCRIPT_DIR/../src"

# ---- 日志文件（cron 重定向之外，预警模块也写入此文件）----
LOG_FILE="$SCRIPT_DIR/../output/cron.log"

# ---- 清代理（与回测脚本一致）----
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy

# ---- Bark 推送密钥（可选）----
# 传递 BARK_DEVICE_KEY 给 signal_generator.py / sim_tracker.py（脚本内未配置时自动跳过推送，不中断）。
# 本地可 export BARK_DEVICE_KEY=xxx 或写项目根目录 .env（BARK_DEVICE_KEY=xxx）；Actions 由 secrets 注入同名变量。
export BARK_DEVICE_KEY="${BARK_DEVICE_KEY:-}"
cd "$SRC_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === V7.1 每日任务开始 ==="

# 1) 信号生成（--live 会先拉取最新行情；无网则回退本地最新数据）
"$PYTHON" signal_generator.py --live

# ---- 实盘模式开关（方向C）：LIVE_MODE=true 时 sim_tracker 仅生成委托单，不模拟成交 ----
# 委托单写到 output/orders/YYYY-MM-DD_orders.csv，由券商端人工或 SDK 执行；
# 不会修改 sim_state，不进入净值/持仓跟踪（与本地"模拟盘"是两套独立流程）。
LIVE_MODE="${LIVE_MODE:-false}"
if [ "$LIVE_MODE" = "true" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 实盘模式 LIVE_MODE=true → sim_tracker --live 生成委托单"
    "$PYTHON" sim_tracker.py --live
    # 委托单生成后，通过 Bark 推送摘要（无委托单时推送「今日无操作」）
    # 推送失败/未配置 key 时静默跳过（|| true），不影响已生成的委托单（set -e 保护）。
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 推送委托单摘要（order_push.py）"
    "$PYTHON" order_push.py || echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️ 委托单推送失败（不影响委托单生成）"
    # 持仓监控预警（方向E）：LIVE_MODE=true 时预警摘要随委托单同时推送；仅提示，不自动执行
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 持仓监控预警（monitor.py）"
    "$PYTHON" monitor.py || echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️ 持仓监控执行异常（不影响委托单生成）"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 委托单生成完成（券商端执行需人工/SDK）"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === 实盘模式任务完成 ==="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === 实盘模式任务完成 ===" >> "$LOG_FILE"
    exit 0
fi

# 2) 净值跟踪（读取上一步信号，按次日开盘/当日收盘模拟成交）
"$PYTHON" sim_tracker.py

# ---- 持仓监控预警（方向E，风控辅助）：收盘净值更新后扫描持仓，触发卖出条件则预警 ----
# 仅提示不自动执行；LIVE_MODE=false（默认）时仅打印到日志，不推送。
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 持仓监控预警（monitor.py）"
"$PYTHON" monitor.py || echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️ 持仓监控执行异常（不影响主流程）"

# ==================== 3. 偏离预警检查（B）====================
# 非零退出码(=1) 表示预警触发；用 || 捕获避免 set -e 直接中断任务。
WATCHER_EXIT=0
"$PYTHON" deviation_watcher.py || WATCHER_EXIT=$?

if [ "$WATCHER_EXIT" -ne 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️ 偏离预警触发！请检查模拟盘与回测理论曲线的偏差。"
    # （可选）如需邮件通知，取消下面注释并配置收件邮箱：
    # echo "量化策略模拟盘偏离预警触发，请登录服务器查看详情。" | mail -s "[量化预警] 模拟盘偏离" your-email@example.com
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 偏离预警检查通过"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 所有任务完成，偏离预警退出码: $WATCHER_EXIT" >> "$LOG_FILE"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === V7.1 每日任务完成 ==="
