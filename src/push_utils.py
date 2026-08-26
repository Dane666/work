# -*- coding: utf-8 -*-
"""
push_utils.py — Bark 推送工具（模拟盘通知专用，自包含版）
================================================================
提供三个函数：
  - push_to_bark(title, body, key=None)   发送 Bark 通知（GET 路径格式，URL 编码）
  - format_stock_list(items, max_display) 股票列表格式化（限显示数量）
  - format_percent(value, signed)         百分比格式化

设计约束：
  - 不改变任何策略逻辑；推送失败一律静默降级（打印 warning，返回 False，不抛异常），
    避免中断 signal_generator / sim_tracker / CI 主流程。
  - Key 解析优先级：显式参数 > 环境变量 BARK_DEVICE_KEY（兼容旧 BARK_KEY 仍可用）。
  - 支持完整 URL 形式传入 key（自动提取末尾 device key）。
  - 本地支持 .env 文件（项目根目录）注入 BARK_DEVICE_KEY，GitHub Actions 用 secrets 注入同名变量。

用法：
  from push_utils import push_to_bark, format_stock_list, format_percent
  push_to_bark("标题", "正文")
"""

from __future__ import annotations

import os
from urllib.parse import quote

import requests

import config

BARK_BASE = "https://api.day.app"
BARK_POST_URL = "https://api.day.app/push"
MAX_BODY = 3000           # 正文上限（Bark 服务端建议）
MAX_GET_URL = 2000        # GET 路径长度上限（超过则回退 POST，避免 431）
GROUP = "V3.1模拟盘"


def _ensure_env() -> None:
    """读取项目根目录 .env（KEY=VALUE），仅填充缺失的环境变量（不覆盖已有 env）。"""
    env_path = getattr(config, "BASE_DIR", None) / ".env" if getattr(config, "BASE_DIR", None) else None
    if env_path is None or not env_path.exists():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k:
                os.environ.setdefault(k, v)
    except Exception:
        pass


_ensure_env()   # 模块导入时装载 .env（BARK_DEVICE_KEY 等）


def _resolve_key(key):
    key = (key or os.environ.get("BARK_DEVICE_KEY") or os.environ.get("BARK_KEY") or "").strip()
    if not key:
        return ""
    # URL 格式（https://api.day.app/<xxx>）→ 提取末尾 key
    if key.startswith("http"):
        parts = key.rstrip("/").split("/")
        key = parts[-1] if parts[-1] else (parts[-2] if len(parts) > 1 else key)
    return key.strip()


def push_to_bark(title: str, body: str, key: str = None) -> bool:
    """发送 Bark 推送。失败静默降级：返回 False 并打印 warning，不抛异常。"""
    try:
        k = _resolve_key(key)
        if not k:
            print("[bark] 未配置 BARK_DEVICE_KEY，跳过推送")
            return False
        title = str(title)[:200]
        body = str(body)[:MAX_BODY]

        # 主路径：GET https://api.day.app/{key}/{title}/{body}（URL 编码）
        url = f"{BARK_BASE}/{k}/{quote(title)}/{quote(body)}"
        if len(url) <= MAX_GET_URL:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                return True
            # 非 200：回退 POST /push（Bark 官方推荐方式，避免 GET 超长）
            print(f"[bark] GET 返回 {r.status_code}，回退 POST")
            payload = {"device_key": k, "title": title, "body": body, "group": GROUP}
            r2 = requests.post(BARK_POST_URL, json=payload, timeout=10)
            ok = r2.status_code == 200
            if not ok:
                print(f"[bark] 推送失败 (HTTP {r2.status_code}): {r2.text[:200]}")
            return ok

        # GET 过长 → 直接 POST
        payload = {"device_key": k, "title": title, "body": body, "group": GROUP}
        r3 = requests.post(BARK_POST_URL, json=payload, timeout=10)
        ok = r3.status_code == 200
        if not ok:
            print(f"[bark] 推送失败 (HTTP {r3.status_code}): {r3.text[:200]}")
        return ok
    except Exception as e:
        print(f"[bark] 推送异常（静默跳过）: {e}")
        return False


def format_stock_list(items, max_display: int = 10) -> str:
    """将股票列表格式化为可读多行文本，限制显示数量避免推送过长。

    items 元素可为：
      - 字符串："000603"
      - 二元组："000603 盛达资源"
      - 多元组："000603 盛达资源 3.33%"（各字段以空格连接）
    超出 max_display 时末尾追加省略行。
    """
    items = list(items)
    if not items:
        return "(空)"
    lines = []
    for it in items[:max_display]:
        if isinstance(it, (tuple, list)):
            lines.append("▪ " + " ".join(str(x) for x in it if str(x) != ""))
        else:
            lines.append("▪ " + str(it))
    if len(items) > max_display:
        lines.append(f"… 共 {len(items)} 只（仅显示前 {max_display}）")
    return "\n".join(lines)


def format_percent(value, signed: bool = False, digits: int = 2) -> str:
    """百分比格式化。value 为小数（0.0123 → "+1.23%" / "1.23%"）。

    signed=True 时输出带 + 号（用于涨跌幅）；value 为 None/NaN 时返回 "—"。
    """
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if v != v:      # NaN
        return "—"
    pct = v * 100.0
    sign = "+" if signed and pct > 0 else ""
    return f"{sign}{pct:.{digits}f}%"
