# -*- coding: utf-8 -*-
"""Bark 推送通知模块（参考 Momentum/notify/bark.py，自包含版）。

与 Momentum 版本的差异：
  - Momentum 版从 momentum.config.BARK_DEVICE_KEY 读 key；本版直接读环境变量
    BARK_DEVICE_KEY（GitHub Actions 通过 secrets 注入），无需依赖本项目外的 config 包。
  - 其余行为保持一致：统一用 POST 避免 GET URL 超长 431；body 截断 3800 字符；
    支持以完整 URL 形式传入 device key（自动提取末尾 key）。
"""
import os
import logging
import requests

logger = logging.getLogger("v81")

BARK_URL = "https://api.day.app/push"


def send_bark(title: str, content: str, device_key: str = None, icon: str = None):
    """发送 Bark 推送，统一用 POST 避免 GET URL 超长 431。

    device_key 优先级：显式参数 > 环境变量 BARK_DEVICE_KEY。
    未配置 key 时仅记录 warning 并静默返回（不抛异常，避免中断 CI）。
    """
    key = (device_key or os.environ.get("BARK_DEVICE_KEY", "")).strip()
    if not key:
        logger.warning("Bark device_key not configured")
        return

    body = content[:3800]

    # URL 格式 → 提取末尾 device key
    if key.startswith("http"):
        parts = key.rstrip("/").split("/")
        key = parts[-1] if parts[-1] else (parts[-2] if len(parts) > 1 else key)

    payload = {"device_key": key, "title": title, "body": body, "group": "V8.1策略"}
    if icon:
        payload["icon"] = icon
    try:
        r = requests.post(BARK_URL, json=payload, timeout=10)
        if r.status_code == 200:
            logger.info("Bark sent")
        else:
            logger.error(f"Bark failed: {r.text}")
    except Exception as e:
        logger.error(f"Bark error: {e}")


def send_msg(title: str, content: str):
    """统一入口（兼容 Momentum 的 send_feishu_msg 重定向习惯）。"""
    send_bark(title, content)


def send_card(title: str, fields: list):
    """统一入口（兼容 Momentum 的 send_feishu_card）。fields 为 [{title, value}, ...]。"""
    text = "\n".join(f"▪ {f.get('title', '')}: {f.get('value', '')}" for f in fields)
    send_bark(title, text)
