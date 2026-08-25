# -*- coding: utf-8 -*-
"""V8.1 通知模块（参考 Momentum notify）。

提供 Bark 推送能力，供每日运行结果推送至手机。
"""
from .bark import send_bark, send_msg, send_card

__all__ = ["send_bark", "send_msg", "send_card"]
