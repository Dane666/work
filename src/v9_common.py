# -*- coding: utf-8 -*-
"""V9 各版本间净值复用的小工具（避免重复回测，保证边际贡献归因干净）。"""

from __future__ import annotations

import pandas as pd

from report import compute_metrics, yearly_sharpe


def load_nav(path) -> pd.Series:
    df = pd.read_parquet(path)
    df = df.rename(columns={df.columns[0]: "date", df.columns[1]: "nav"})
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["nav"].sort_index()


def nav_metrics(eq: pd.Series):
    return (compute_metrics(eq),
            compute_metrics(eq.loc[:"2023-12-31"]),
            compute_metrics(eq.loc["2024-01-01":]),
            yearly_sharpe(eq))


def ref_from_nav(path, label: str) -> dict:
    """由已保存净值 parquet 构造 report 所需的 ref dict（full/old/new/label）。"""
    eq = load_nav(path)
    m_full, m_old, m_new, _ = nav_metrics(eq)
    return {"full": m_full, "old": m_old, "new": m_new, "label": label}
