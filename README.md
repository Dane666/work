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
│   ├── sim_tracker.py        # 模拟盘/实盘执行（--init / --offline / --live 三模式）
│   ├── chart_utils.py        # 仪表盘增强图（行业暴露/因子暴露/回撤归因，方向B）
│   ├── risk_manager.py       # 实盘风控与委托单生成（方向C）
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
   - **数据**：通过 data 分支持久化（`data/` 被 main .gitignore 排除，单独以 data 分支管理）。
     每次运行先 checkout data 分支取面板 → **`Fetch latest daily panel` 步骤增量拉取最新行情**
     （见下方第 4 点）→ 跑信号/模拟盘/监控 → Persist 阶段把状态、信号与更新后的面板推回 data 分支。
   - **产出**：信号与净值作为 artifact 上传（Actions 页面可下载）。
   - 手动触发：Actions 页 → Run workflow（可传 `fetch_limit` 快速验证，如 `20`）。

3. **Bark 手机推送（可选）**：每日跑出的信号与模拟盘净值会自动推送至手机（参考 Momentum/notify）。
   - 在仓库 **Settings → Secrets and variables → Actions** 新增仓库密钥 `BARK_DEVICE_KEY`，
     值为你的 Bark 设备 Key（或完整推送 URL，如 `https://api.day.app/<你的key>`，脚本会自动提取末尾 key）。
   - 配置后工作流末尾的 `Bark push (每日结果)` 步骤自动执行；未配置则跳过、不影响主流程。
   - 推送内容：信号截面日期、BUY/SELL/HOLD 指令数、模拟盘 NAV/累计收益/超额、Top15 买入清单。
   - 本地预览（不推送）：`cd src && python notify_push_daily.py`。

> ⚠️ 局限：GitHub Actions 每次运行都要 checkout data 分支重放面板（约 20s），增量拉取依赖
> 新浪源可达性（美国 runner 实测可访问，前置 curl 12s 探测不可达则快速跳过沿用旧面板）。
> **生产环境建议用云服务器方案（下节）**，可本地持久化数据、每日运行 < 5 分钟。

4. **数据自动更新（每日增量拉取面板，方向F）**
   - 面板（`mainboard_close_panel.parquet` / `mainboard_amount_panel.parquet`）不再需要手工拉取，
     每日 cron 运行前由 `Fetch latest daily panel` 步骤自动增量更新：
     - `src/update_daily_panel.py`（自包含，不依赖 `src/config.py`，data 分支上可直接运行）
     - 逐只从新浪源拉增量（qfq 前复权）；检测到复权因子变化自动重拉 365 天替换列尾；
       盘中未收盘行丢弃；已最新则幂等跳过
     - 失败兜底：连通性探测失败 / 超时（45min）→ `::warning::` 跳过，沿用现有面板继续，不影响信号与监控
   - 手动快速验证：`gh workflow run daily_run.yml -f fetch_limit=20`（只拉前 20 只）；
     cron 默认全量（3046 只，约 30-45 分钟）
   - 本地手动更新：`cd src && python update_daily_panel.py --limit 20`（试跑）或全量（去掉 --limit）

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

## 八、实盘接入指南（方向C · LIVE_MODE）

> **⚠️ 安全提醒**：本节描述如何将策略输出的**委托单 CSV** 喂给券商账户执行。**模拟盘（默认 `LIVE_MODE=false`）与实盘是两套独立流程**，互不影响；
> 实盘下单必须经过人工复核或可信券商 SDK，**严禁**未经审核的自动下单。

### 1. 开启方式

`LIVE_MODE=true` 是总开关，默认关闭。激活方式（三选一）：

```bash
# (a) 单次执行（推荐先验证）
LIVE_MODE=true python src/sim_tracker.py --live
# 或
LIVE_MODE=true bash scripts/run_daily.sh

# (b) 写入项目根目录 .env
echo "LIVE_MODE=true" >> .env

# (c) 改 src/config.py（不推荐，需代码改动）
RISK_LIVE_MODE = True
```

开启后 `sim_tracker.py --live` 仅调用 `risk_manager.build_orders(...)` 生成委托单 CSV（`output/orders/YYYY-MM-DD_orders.csv`），**不模拟成交、不修改 `sim_state`**。

### 2. 风控参数（`src/config.py`）

| 参数 | 默认值 | 含义 |
|---|---|---|
| `RISK_MAX_DAILY_BUY_AMOUNT` | 100_000 | 单日累计买入上限（元），防止一次性打满仓 |
| `RISK_MAX_SINGLE_POSITION_PCT` | 0.10 | 单只持仓上限（NAV 占比），与 V8.1 `FIXED_WEIGHT` 对齐 |
| `RISK_MAX_TOTAL_POSITION_PCT` | 0.90 | 总仓位上限，留 10% 现金应急 |
| `RISK_MAX_DAILY_LOSS_PCT` | -0.05 | 单日 NAV 跌幅低于此阈值禁止新开仓（熔断） |

### 3. 委托单格式

```csv
code,name,direction,price_type,price,shares,amount,reason
600519,贵州茅台,BUY,limit,1700.50,100,170050.00,信号 BUY，目标 10.00%；风控100千/单日100千
000001,平安银行,SELL,limit,11.85,500,5925.00,信号 SELL（跌出目标组合）
```

价格口径：`BUY`=`open × 1.0005`（限价+0.05%）、`SELL`=`close × 0.999`（限价-0.10%），均为次日开盘前报送的限价单；行情缺失时回退为市价单。

### 4. 主流券商 Python SDK 对比与接入示例

| 券商 | 接入方式 | 优 | 缺 | 推荐场景 |
|---|---|---|---|---|
| **华泰证券**（htsc-quant） | 社区 SDK（非官方） | 协议稳定；量化权限申请门槛中等 | 需自维护认证 token；官方 SDK 不开放 | 个人量化账户、长期部署 |
| **国信证券**（iQuant） | `gxquant-py-sdk` 社区 + iQuant 平台 | 与 iQuant 平台打通，可回测+实盘 | iQuant 平台部分功能付费 | 已在国信开户、想用统一平台 |
| **东方财富**（Choice 量化） | 东财终端 + Choice API | 数据源最全（与本策略 akshare 同源） | 自动化下单接口受限；机构通道为主 | 数据研究为主、下单人工 |
| **同花顺**（iFinD/Hedge） | `thstrader` 社区 SDK | 协议广、社区活跃 | 部分券商需 VPN；下单延迟较高 | 已有同花顺账户、快速接入 |

#### 示例模板（伪代码）

```python
# 通用委托单执行框架（券商 SDK 由用户接入；以下示例供参考，不绑定任何券商）
import pandas as pd
import risk_manager as rm

def place_orders(orders_csv: str, broker_client):
    """读委托单并通过 broker_client 下单；下单结果写日志（人工监控）。"""
    orders = pd.read_csv(orders_csv, dtype={"code": str})
    for _, o in orders.iterrows():
        if o["direction"] == "BUY":
            broker_client.buy(code=o["code"], shares=int(o["shares"]),
                             price_type=o["price_type"], price=float(o["price"]))
        elif o["direction"] == "SELL":
            broker_client.sell(code=o["code"], shares=int(o["shares"]),
                               price_type=o["price_type"], price=float(o["price"]))
        # 务必先在券商端开启"密码独立 + 资金复核"等安全约束

# 华泰示例（htsc-quant 风格）
# from htsc_quant import Client
# client = Client(account=os.environ["HTSC_ACCOUNT"], token=os.environ["HTSC_TOKEN"])
# place_orders("output/orders/2026-08-28_orders.csv", client)

# 同花顺示例（thstrader 风格）
# import thstrader
# trader = thstrader.THS(account=..., pwd=..., exe_path="C:/ths/同花顺/xiadan.exe")
# place_orders("output/orders/2026-08-28_orders.csv", trader)
```

### 5. 风险提示

- **先模拟盘，后实盘**：本仓库的 `sim_tracker.py`（默认 `--offline`）已长期回测达标后再考虑实盘。
- **资金隔离**：实盘账户资金建议 ≤ 总可投金额的 20%~30%，单策略 ≤ 50 万。
- **人工复核**：建议实盘第一天至少手动核对前 3 笔成交是否与 CSV 一致；任何"全自动化无人值守"都是禁止配置。
- **风控二次校验**：券商端 App 仍需设置单笔/单日金额上限（与 `config.RISK_*` 对齐），防止 SDK 异常时失控。

## 九、推送配置（Bark 手机通知）

模拟盘每天在三个节点推送 Bark 通知（**纯附加功能，不影响任何策略逻辑**）：

| 节点 | 触发脚本 | 标题 | 内容 |
|---|---|---|---|
| 今日信号 | `signal_generator.py` | 📈 今日信号 {日期} | 目标持仓前 10（代码/名称/权重）、共 N 只；调仓日标注 🔄 次日开盘执行 |
| 今日执行 | `sim_tracker.py` | 💰 今日执行 {日期} | 实际买入清单（代码/名称/成交价/数量）与卖出清单（代码/名称/成交价），过长截断 |
| 收盘净值 | `sim_tracker.py` | 📊 收盘净值 {日期} | NAV、当日/累计收益、持仓数、现金占比、CSI300 基准对比 |
| 持仓预警 | `monitor.py` | ⚠️ 持仓预警 / ✅ 持仓健康 {日期} | 触发卖出条件的持仓清单（止损/止盈/移动止损/估值），最多前 10 只；无预警推「✅ 今日无预警」 |

> **持仓监控预警（方向E · 风控辅助，仅提示不自动执行）**：每日收盘净值更新后 `monitor.py`
> 自动扫描当前持仓，按 `src/config.py` 阈值生成预警清单：
> ⚠️ 止损（盈亏率 < -8%）、💰 止盈（> +20%）、📉 移动止损（自持仓期间最高点回撤 > 6%）、
> 📊 估值偏高（PE > 50，需 `data/pe_panel_mainboard.parquet`，缺失自动跳过）。
> `LIVE_MODE=false`（默认）仅打印到日志；`LIVE_MODE=true` 时预警随委托单同时推送。

### 1. 获取 Bark Key

- iOS 安装 [Bark](https://github.com/Finb/Bark) App，打开后复制你的设备 Key（形如 `xxxxxxxxxxxxxxxx`，或完整推送 URL `https://api.day.app/<key>`，脚本会自动提取末尾 key）。

### 2. 配置方式（三选一）

**A. GitHub Actions（推荐云端）**：仓库 **Settings → Secrets and variables → Actions → New repository secret**，新增 `BARK_DEVICE_KEY`，值为你的 Bark Key。工作流已为 `signal_generator.py` / `sim_tracker.py` 注入 `env: BARK_DEVICE_KEY: ${{ secrets.BARK_DEVICE_KEY }}`，无需改 workflow。未配置时脚本自动跳过推送，主流程不受影响。

**B. 本地/云服务器**：直接 export 后运行 `scripts/run_daily.sh`（脚本已透传该变量）：
```bash
export BARK_DEVICE_KEY=xxxxxxxxxxxxxxxx
bash scripts/run_daily.sh
```
**C. 项目根目录 .env 文件**（脚本启动时自动读取，无需 export）：
```bash
echo "BARK_DEVICE_KEY=xxxxxxxxxxxxxxxx" >> .env
```

### 3. 兼容与失败处理

- 变量名统一为 `BARK_DEVICE_KEY`（兼容旧 `BARK_KEY` 仍可被脚本识别）。
- 推送失败一律**静默跳过**（打印 warning、不抛异常），不会中断信号生成 / 模拟盘 / CI。
- 本地预览推送效果：`cd src && BARK_DEVICE_KEY=你的key python signal_generator.py`。
- 推送实现：`src/push_utils.py`（`push_to_bark` / `format_stock_list` / `format_percent`）。
