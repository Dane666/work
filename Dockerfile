# V8.1 策略容器化部署（可选：本地 / 云服务器 / 容器平台）
# 用法：
#   docker build -t v81-strategy .
#   docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output v81-strategy
# 说明：data/ 与 output/ 以卷挂载（容器内不保存状态），数据由宿主机持久化。
FROM python:3.10-slim

# LightGBM 在 Debian slim 上需要 libgomp（OpenMP 运行时）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY scripts/ scripts/
COPY output/ output/

# 数据目录（宿主机挂载；首次运行需先抓取历史数据，见 README）
RUN mkdir -p /app/data /app/output/signals /app/output/sim_nav

CMD ["bash", "scripts/run_daily.sh"]
