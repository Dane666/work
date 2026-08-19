# V8.1 A股多因子策略 · 模拟盘（回测验证 → 云端自动运行）

> 版本演进：V1 超跌绩优 → V5 多因子 Regime 切换 → V8 扩中证1000（0.57）→
> **V8.1 分区间部署（≤2023 V8 / ≥2024 E等权组合 V8+Trend+Breakout）**，全期夏普 0.60、
> 2024-25 夏普 0.79、回撤 -19.40%；执行口径已对齐实盘（T+1 开盘价 + 冲击成本），
> 正式预期 0.50 / 0.58 / -20.4%。

## 一、目录结构

```
stategy/
├── src/                      # 全部 Python 代码
│   ├── signal_generator.py   # V8.1 每日信号生成（日期门控：≤2023 V8 / ≥2024 E组合）
│   ├── sim_tracker.py        # 模拟盘净值跟踪（按 target_weight 建仓，T+1 开盘成交）
│   ├── strategies/           # 并行策略模块（trend_ema / mean_reversion / vol_breakout）
│   ├── backtest_v5.py        # 回测引擎（run_backtest_v5 close 口径 / run_backtest_v5_ne next_open+冲击）
│   ├── fetch_all_v8.py       # 历史数据抓取（1539 只日线+财报，首次约 1-2 小时）
│   ├── fetch_ohlc_v8.py      # OHLC 抓取（ATR/next_open 用，约 12 分钟）
│   └── config.py             # 参数与开关（EXECUTION_PRICE / ENABLE_IMPACT_COST 等）
├── scripts/run_daily.sh      # 每日任务入口（信号 → 净值 → 偏离预警）
├── data/                     # 数据资产（308MB，gitignore，不入库）
├── output/                   # 报告/信号/净值（gitignore）
├── .github/workflows/daily_run.yml   # GitHub Actions 每日调度
├── Dockerfile                # 容器化部署（可选）
└── requirements.txt          # 依赖锁定
```

## 二、本地运行

### 1. 安装依赖

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.10+
pip install --upgrade pip
pip install -r requirements.txt
```

> Linux/macOS 注意：LightGBM 依赖 OpenMP。macOS 若缺 `libomp` 请 `brew install libomp`；
> Linux（Debian/Ubuntu）`apt-get install -y libgomp1`；也可参考 `src/setup_lightgbm.sh`。

### 2. 首次运行（抓取历史数据）

```bash
cd src
python fetch_all_v8.py      # 8 年日线+财报（1539 只，约 1-2 小时，可断点续跑）
python fetch_ohlc_v8.py     # OHLC 面板（next_open 执行口径必需，约 12 分钟）
```

### 3. 初始化信号与模拟盘（用本地数据最新截面）

```bash
python signal_generator.py --init --date 2025-08-12   # 生成信号 CSV
python sim_tracker.py --init --date 2025-08-12        # 记录初始净值 1.0
python sim_tracker.py --date 2025-08-13               # 次日开盘建仓（需联网拉当日行情）
```

### 4. 每日自动运行

```bash
bash scripts/run_daily.sh
```

或配置 cron（Linux/Mac，交易日 15:30 盘后）：

```
30 15 * * 1-5 cd /path/to/stategy && bash scripts/run_daily.sh >> logs/cron.log 2>&1
```

每日产出：`output/signals/YYYY-MM-DD_signal.csv`（次日买卖清单）、
`output/sim_nav/YYYY-MM-DD_nav.csv`（当日净值）。

## 三、GitHub Actions 部署

1. 将本仓库推送到 GitHub：
   ```bash
   git init && git add . && git commit -m "V8.1 strategy" && git push -u origin main
   ```
   （`data/` 与 `output/` 已被 .gitignore 排除，不入库。）

2. 工作流 `.github/workflows/daily_run.yml`：
   - **调度**：`cron '0 16 * * 1-5'` = UTC 16:00 = 北京时间次日 00:00（数据已收盘）；
     需盘后更早运行改为 `'0 8 * * 1-5'`（北京 16:00）。
   - **数据**：GitHub Actions 无持久化磁盘。工作流用 `actions/cache` 缓存 `data/`
     （每日重建）；首次运行会自动执行 `fetch_all_v8.py + fetch_ohlc_v8.py`（约 1-2 小时，
     在 6 小时免费限制内）。
   - **产出**：信号与净值作为 artifact 上传（Actions 页面可下载）。
   - 手动触发：Actions 页 → Run workflow。

> ⚠️ 局限：GitHub Actions 每次运行都要重放数据（缓存策略受 7 天限制），且需联网抓取。
> **生产环境建议用云服务器方案（下节）**，可本地持久化数据、每日运行 < 5 分钟。

## 四、云服务器部署（推荐，数据持久化）

```bash
# 首次
git clone <repo-url> /opt/stategy && cd /opt/stategy
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd src && python fetch_all_v8.py && python fetch_ohlc_v8.py   # 一次性抓取历史数据

# 每日（crontab -e）
30 15 * * 1-5 cd /opt/stategy && git pull && bash scripts/run_daily.sh >> logs/cron.log 2>&1
```

数据策略：`data/` 在服务器本地持久化，每日 `--live` 只追加当日行情（增量更新，
无需重抓 8 年）；每日运行约 3-5 分钟。

## 五、Docker 部署（可选）

```bash
docker build -t v81-strategy .
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  v81-strategy
```

数据与输出用卷挂载持久化；容器内不保存状态。首次运行前宿主机需先抓取历史数据。

## 六、执行口径（V8.1 与实盘对齐）

- **信号**：月末收盘生成（`signal_date`），**T+1 开盘价成交**（`execution_date`，停牌顺延）。
- **成本**：分档滑点（0.10/0.30/0.50%）+ 冲击成本（按 20 日均成交额：>5亿 0.05% /
  1-5亿 0.15% / <1亿 0.30%），单边、买卖各一次。
- **开关**：`config.py` 的 `EXECUTION_PRICE`（"close"/"next_open"）、`ENABLE_IMPACT_COST`。
- **分区间**：`SPLIT_DATE=2024-01-01`；≤2023 V8 原样（每只 10%）；≥2024 E 等权组合
  （V8+Trend+Breakout 各 1/3，单只 cap 10%）。
- **审计**：每日日志含 `AUDIT signal_date / execution_date`；组合权重明细
  `output/v8_combo_weights.csv`。

## 七、关键研究结论（供回测复现）

- V9 三模块（分析师因子 / 行业拥挤度 / 周频调仓）均回测证伪，维持 V8。
- 卖出规则：任何日内止损/止盈（固定 -10%+RSI70、ATR 动态止损）都是损耗，月末轮换为最优。
- Breakout（接近20日高点+放量）2024-25 夏普 0.88 极强但全期回撤 -35.8%，未独立上线；
  并入 E 等权组合后 2024-25 显著增强（0.79 vs 0.67）。
- 详细报告见 `output/report_v8_1_split.html`、`report_v8_1_next_open.html`。
