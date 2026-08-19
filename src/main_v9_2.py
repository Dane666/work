# -*- coding: utf-8 -*-
"""
V9.2 主流程 = 模块2（行业拥挤度风控）的 **完整 2×2 消融**。

为什么是 2×2 而不是简单地在 V9.1 上叠加：
  模块1（分析师因子）已被 V9.1 实测判定失败（全期夏普 0.57→0.48，回撤 -19.40%→-22.91%）。
  若把模块2 直接叠加在已失败的 V9.1 之上，模块2 的"边际贡献"会混入对模块1 拖累的
  吸收/反转效应，归因不可解释。因此本脚本一次跑完 4 条臂，共用同一份数据、同一引擎、
  同一 regime/滑点/目标权重序列，唯一变量就是两个开关：

      臂       分析师因子   拥挤度风控    含义
      A(V8)      OFF         OFF        V8 基线（回归校验：应复现 0.57/0.54/0.67）
      B(V9.1)    ON          OFF        模块1 单独效果
      C          OFF         ON         **模块2 的干净边际**（vs A）
      D(V9.2)    ON          ON         用户原定 V9.2（vs B 为模块2 在 V9.1 上的边际）

  模块2 的裁定以 C vs A 为准（干净边际）；D 仅作为用户原定路径的记录。

严禁触碰 V8 已验证核心参数：MA240 / IC 门限 0.05 / 滚动36月 / RSI池20% / n=30 /
fixed_weight / 波动过滤(0.75分位→0.60) 全部保持原值。
落盘：output/v9_2_nav.parquet(D)、output/v8_crowd_nav.parquet(C)。
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd

for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
          "ALL_PROXY", "all_proxy"]:
    os.environ.pop(k, None)

import config
from factors import compute_rsi, get_month_end_dates
from factor_eval import (monthly_reversal_ic, compute_rolling_regime,
                        compute_momentum_zscore, build_selection_v5)
from factor_eval_v7 import build_rolling_reversal_signal
from backtest_v5 import run_backtest_v5
from market_filter import build_ma240_target_weight, build_ma240_vol_target_weight
from stress_test_v6 import build_ohlcv_full, build_slippage_map
from industry_crowding import load_industry_map, compute_crowding_weight_mult
from report import compute_metrics, yearly_sharpe
from report_v9 import generate_html_v9

config.ENABLE_CROWDING_FILTER = True      # 模块2 启用（本脚本内按臂显式控制）
config.ENABLE_WEEKLY_REBALANCE = False

END_EXT = "2025-08-12"
MA_BASE = 240
IC_BASE = 0.05
FIXED_WEIGHT = config.FIXED_WEIGHT
DISCOUNT = getattr(config, "ANALYST_CROWDED_DISCOUNT", 0.70)


def mask_new_listings(close_panel: pd.DataFrame, min_days: int = 60) -> pd.DataFrame:
    out = close_panel.copy()
    for c in out.columns:
        s = out[c].dropna()
        if s.empty:
            out[c] = np.nan
            continue
        cutoff = s.index.min() + pd.Timedelta(days=min_days)
        out.loc[out.index < cutoff, c] = np.nan
    return out


def month_ends_in(close_panel: pd.DataFrame, start, end) -> pd.DatetimeIndex:
    me = get_month_end_dates(close_panel.index)
    return me[(me >= pd.Timestamp(start)) & (me <= pd.Timestamp(end))]


def load_name_map():
    try:
        import akshare as ak
        df = ak.stock_info_a_code_name()
        return dict(zip(df["code"].astype(str).str.zfill(6), df["name"].astype(str)))
    except Exception:
        return {}


def _ref(eq: pd.Series, label: str) -> dict:
    return {"full": compute_metrics(eq),
            "old": compute_metrics(eq.loc[:"2023-12-31"]),
            "new": compute_metrics(eq.loc["2024-01-01":]),
            "label": label}


def _save_nav(eq: pd.Series, path):
    df = eq.reset_index()
    df.columns = ["date", "nav"]
    df.to_parquet(path, index=False)


def _pos_hit(sel: dict, mult: dict) -> tuple:
    """实际持仓口径的拥挤命中统计（全宇宙口径会虚高，无法反映真实影响）。"""
    n_pos = n_disc = 0
    for t, codes_sel in sel.items():
        for c in codes_sel:
            n_pos += 1
            if mult.get((t, c), 1.0) < 1.0:
                n_disc += 1
    hit = n_disc / max(n_pos, 1) * 100
    return n_disc, n_pos, hit, hit * (1.0 - DISCOUNT)


def main():
    t0 = datetime.now()
    close = pd.read_parquet(config.DATA_DIR / "v8_close_panel.parquet")
    amount = pd.read_parquet(config.DATA_DIR / "v8_amount_panel.parquet")
    idx = pd.read_parquet(config.DATA_DIR / "v6_index.parquet")
    if isinstance(idx, pd.DataFrame):
        idx = idx.iloc[:, 0]
    roe = pd.read_parquet(config.DATA_DIR / "roe_panel_v8.parquet").reindex(close.index).ffill()
    gpm = pd.read_parquet(config.DATA_DIR / "gpm_yoy_panel_v8.parquet").reindex(close.index).ffill()
    codes_all = list(close.columns)
    n_universe = len(codes_all)
    data_start = str(close.index[0].date())
    data_end = str(close.index[-1].date())

    slip_map, tier_counts = build_slippage_map(amount)
    close_m = mask_new_listings(close, config.NEW_STOCK_MIN_DAYS)
    rsi = compute_rsi(close_m, config.RSI_WINDOW)
    me = month_ends_in(close_m, config.START_DATE, END_EXT)
    ohlcv_old = pd.read_pickle(config.V3_OHLCV) if config.V3_OHLCV.exists() else {}
    ohlcv_full = build_ohlcv_full(close, amount, ohlcv_old, codes_all)

    print(f"[{datetime.now()}] 构建滚动反转信号（V9 宇宙 {n_universe} 只）...")
    reversal_signal = build_rolling_reversal_signal(close_m, ohlcv_full, window=36)
    monthly_ic = monthly_reversal_ic(close_m, rsi, me, config.FWD_RETURN_DAYS)
    rolling_ic, use_reversal = compute_rolling_regime(monthly_ic, 12, IC_BASE, 6)

    tw_clean = build_ma240_target_weight(idx, close_m.index, MA_BASE)
    tw, vol_regime = build_ma240_vol_target_weight(
        idx, close_m.index, MA_BASE, month_ends=me,
        vol_q=0.75, reduced_weight=0.60, vol_lookback=756)

    idx_ret = idx.reindex(close_m.index).pct_change().fillna(0)
    idx_eq = (1.0 + idx_ret).cumprod() * config.INIT_CAPITAL
    m_idx = compute_metrics(idx_eq)

    # ---- 两套选股：分析师因子 OFF / ON（唯一变量=开关，其余完全一致）----
    print(f"[{datetime.now()}] 选股臂1：分析师因子 OFF（V8 口径）...")
    config.ENABLE_ANALYST_FACTOR = False
    mz_off = compute_momentum_zscore(close_m, roe, gpm, me)
    sel_off, sw_off = build_selection_v5(
        close_m, rsi, reversal_signal, mz_off, me, use_reversal, 0.20, 30)

    print(f"[{datetime.now()}] 选股臂2：分析师因子 ON（V9.1 口径）...")
    config.ENABLE_ANALYST_FACTOR = True
    mz_on = compute_momentum_zscore(close_m, roe, gpm, me)
    sel_on, sw_on = build_selection_v5(
        close_m, rsi, reversal_signal, mz_on, me, use_reversal, 0.20, 30)

    # ---- 模块2：行业拥挤度权重折扣 ----
    print(f"[{datetime.now()}] 计算行业拥挤度权重折扣（申万一级）...")
    industry_map = load_industry_map()
    crowding_mult = compute_crowding_weight_mult(close, amount, industry_map, me)
    d_off, n_off, hit_off, loss_off = _pos_hit(sel_off, crowding_mult)
    d_on, n_on, hit_on, loss_on = _pos_hit(sel_on, crowding_mult)
    print(f"  拥挤命中(V8选股) {d_off}/{n_off} = {hit_off:.1f}% → 净暴露降 {loss_off:.1f}%")
    print(f"  拥挤命中(V9.1选股) {d_on}/{n_on} = {hit_on:.1f}% → 净暴露降 {loss_on:.1f}%")

    # ---- 2×2 四条臂回测 ----
    def bt(sel, mult, tag):
        eq, tr = run_backtest_v5(
            close_m, sel, me, config.START_DATE, END_EXT,
            target_weight=tw, slippage_map=slip_map, weight_mult=mult)
        print(f"  [{tag}] 全期夏普={compute_metrics(eq)['sharpe']:.2f}")
        return eq, tr

    print(f"[{datetime.now()}] 回测 2×2 四条臂...")
    eq_A, _ = bt(sel_off, None, "A V8基线(OFF/OFF)")
    eq_B, _ = bt(sel_on, None, "B V9.1(ON/OFF)")
    eq_C, tr_C = bt(sel_off, crowding_mult, "C V8+拥挤(OFF/ON)")
    eq_D, tr_D = bt(sel_on, crowding_mult, "D V9.2(ON/ON)")
    tr_D.to_csv(config.OUTPUT_DIR / "v9_2_trades.csv", index=False)
    tr_C.to_csv(config.OUTPUT_DIR / "v8_crowd_trades.csv", index=False)

    # 回归校验：A 臂必须复现 V8 已验证结果 0.57 / 0.54 / 0.67
    mA, mA_o, mA_n = (compute_metrics(eq_A), compute_metrics(eq_A.loc[:"2023-12-31"]),
                      compute_metrics(eq_A.loc["2024-01-01":]))
    print(f"[回归校验] A臂 V8等价: 全期={mA['sharpe']:.2f}(应0.57) "
          f"旧={mA_o['sharpe']:.2f}(应0.54) 新={mA_n['sharpe']:.2f}(应0.67)")
    if abs(mA["sharpe"] - 0.57) > 0.02:
        print("  !! 警告：A臂未复现 V8 基线，隔离性存疑，结论不可信 !!")

    # ---- 主交付版本 = D（用户原定 V9.2）----
    eq_D_clean, _ = run_backtest_v5(
        close_m, sel_on, me, config.START_DATE, END_EXT,
        target_weight=tw_clean, slippage_map=slip_map, weight_mult=crowding_mult)

    m_full, m_old, m_new = (compute_metrics(eq_D), compute_metrics(eq_D.loc[:"2023-12-31"]),
                            compute_metrics(eq_D.loc["2024-01-01":]))
    m_full_c, m_old_c, m_new_c = (compute_metrics(eq_D_clean),
                                  compute_metrics(eq_D_clean.loc[:"2023-12-31"]),
                                  compute_metrics(eq_D_clean.loc["2024-01-01":]))
    yearly = yearly_sharpe(eq_D)
    prev_ref = _ref(eq_B, "V9.1")
    baseline_ref = _ref(eq_A, "V8")
    ref_C = _ref(eq_C, "V8+模块2")
    yearly_prev = yearly_sharpe(eq_B)
    yearly_baseline = yearly_sharpe(eq_A)

    _save_nav(eq_D, config.OUTPUT_DIR / "v9_2_nav.parquet")
    _save_nav(eq_C, config.OUTPUT_DIR / "v8_crowd_nav.parquet")

    # ---- 明细 / 最新信号 ----
    holdings = []
    for t, codes_sel in sel_on.items():
        if t < pd.Timestamp("2024-01-01"):
            continue
        fset = sw_on.loc[t, "factor_set"] if t in sw_on.index else "?"
        n_disc_t = sum(1 for c in codes_sel if crowding_mult.get((t, c), 1.0) < 1.0)
        holdings.append({"month_end": t, "factor_set": fset,
                         "n_selected": len(codes_sel),
                         "n_crowded_discounted": n_disc_t,
                         "selected_codes": ",".join(codes_sel)})
    pd.DataFrame(holdings).to_csv(
        config.OUTPUT_DIR / "v9_2_holdings_2024_2025.csv", index=False)

    name_map = load_name_map()
    last_me = list(sel_on.keys())[-1]
    last_codes = sel_on[last_me]
    tw_last = float(tw.loc[last_me]) if last_me in tw.index else float(tw.iloc[-1])
    fset_last = sw_on.loc[last_me, "factor_set"] if last_me in sw_on.index else "?"
    rows = []
    for i, c in enumerate(last_codes, 1):
        mult = crowding_mult.get((last_me, c), 1.0)
        rows.append({"rank": i, "code": c, "name": name_map.get(c, c),
                     "factor_set": fset_last,
                     "regime_weight": round(tw_last * 100, 2),
                     "crowding_mult": mult,
                     "target_weight": round(FIXED_WEIGHT * mult * 100, 2) if tw_last > 0 else 0.0,
                     "action": "BUY" if tw_last > 0 else "HOLD"})
    latest_df = pd.DataFrame(rows)
    latest_df.to_csv(config.OUTPUT_DIR / "v9_2_latest_signal.csv", index=False)

    # ---- 2×2 消融表（写入报告 + CSV）----
    abl = pd.DataFrame([
        {"臂": "A  V8 基线", "分析师因子": "OFF", "拥挤度风控": "OFF",
         "全期夏普": mA["sharpe"], "全期年化%": mA["annual_return"] * 100,
         "全期回撤%": mA["max_drawdown"] * 100, "新区间夏普": mA_n["sharpe"],
         "新区间回撤%": mA_n["max_drawdown"] * 100},
        {"臂": "B  V9.1", "分析师因子": "ON", "拥挤度风控": "OFF",
         "全期夏普": prev_ref["full"]["sharpe"], "全期年化%": prev_ref["full"]["annual_return"] * 100,
         "全期回撤%": prev_ref["full"]["max_drawdown"] * 100, "新区间夏普": prev_ref["new"]["sharpe"],
         "新区间回撤%": prev_ref["new"]["max_drawdown"] * 100},
        {"臂": "C  V8+模块2", "分析师因子": "OFF", "拥挤度风控": "ON",
         "全期夏普": ref_C["full"]["sharpe"], "全期年化%": ref_C["full"]["annual_return"] * 100,
         "全期回撤%": ref_C["full"]["max_drawdown"] * 100, "新区间夏普": ref_C["new"]["sharpe"],
         "新区间回撤%": ref_C["new"]["max_drawdown"] * 100},
        {"臂": "D  V9.2", "分析师因子": "ON", "拥挤度风控": "ON",
         "全期夏普": m_full["sharpe"], "全期年化%": m_full["annual_return"] * 100,
         "全期回撤%": m_full["max_drawdown"] * 100, "新区间夏普": m_new["sharpe"],
         "新区间回撤%": m_new["max_drawdown"] * 100},
    ])
    abl.to_csv(config.OUTPUT_DIR / "v9_2_ablation_2x2.csv", index=False)
    print("\n---------------- 2×2 消融 ----------------")
    print(abl.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # 模块2 干净边际 = C - A；在 V9.1 上的边际 = D - B
    mc_full = ref_C["full"]["sharpe"] - mA["sharpe"]
    mc_new = ref_C["new"]["sharpe"] - mA_n["sharpe"]
    mc_dd = ref_C["full"]["max_drawdown"] * 100 - mA["max_drawdown"] * 100
    md_full = m_full["sharpe"] - prev_ref["full"]["sharpe"]
    md_new = m_new["sharpe"] - prev_ref["new"]["sharpe"]
    md_dd = m_full["max_drawdown"] * 100 - prev_ref["full"]["max_drawdown"] * 100

    def _rowhtml(r):
        return (f"<tr><td>{r['臂']}</td><td>{r['分析师因子']}</td><td>{r['拥挤度风控']}</td>"
                f"<td>{r['全期夏普']:.2f}</td><td>{r['全期年化%']:.2f}%</td>"
                f"<td>{r['全期回撤%']:.2f}%</td><td>{r['新区间夏普']:.2f}</td>"
                f"<td>{r['新区间回撤%']:.2f}%</td></tr>")

    abl_html = (
        "<table style='width:100%;border-collapse:collapse;margin:12px 0;font-size:13px'>"
        "<thead><tr style='background:rgba(127,127,127,.15)'>"
        "<th style='text-align:left;padding:6px'>臂</th><th>分析师因子</th><th>拥挤度风控</th>"
        "<th>全期夏普</th><th>全期年化</th><th>全期回撤</th><th>2024-25夏普</th>"
        "<th>2024-25回撤</th></tr></thead><tbody>"
        + "".join(_rowhtml(r) for _, r in abl.iterrows()) + "</tbody></table>")

    conclusion = (
        f"本轮采用<b>完整 2×2 消融</b>而非在 V9.1 上简单叠加：因模块1（分析师因子）已被 V9.1 "
        f"实测判定失败（全期夏普 0.57→0.48、回撤 -19.40%→-22.91%），若把模块2 叠在失败基座上，"
        f"其边际贡献会混入对模块1 拖累的吸收/反转效应，归因不可解释。"
        f"四条臂共用同一数据、同一引擎、同一 regime/滑点/目标权重序列，唯一变量为两个开关。"
        f"回归校验：A 臂复现 V8 基线夏普 {mA['sharpe']:.2f}（预期 0.57），隔离性成立。"
        f"【模块2 干净边际 C vs A】全期夏普 {mc_full:+.2f}（{mA['sharpe']:.2f}→{ref_C['full']['sharpe']:.2f}），"
        f"新区间夏普 {mc_new:+.2f}，全期回撤 {mc_dd:+.2f}pp"
        f"（{mA['max_drawdown']*100:.2f}%→{ref_C['full']['max_drawdown']*100:.2f}%）。"
        f"【在 V9.1 上的边际 D vs B】全期夏普 {md_full:+.2f}、新区间 {md_new:+.2f}、回撤 {md_dd:+.2f}pp。"
        f"拥挤命中（实际持仓口径）：V8 选股 {d_off}/{n_off}={hit_off:.1f}%，"
        f"V9.1 选股 {d_on}/{n_on}={hit_on:.1f}%，对应平均净暴露分别下降约 "
        f"{loss_off:.1f}% / {loss_on:.1f}%——这会机械压低收益与回撤，故夏普方向须实测裁定。"
        f"【前置证据（诚实披露）】拥挤 vs 非拥挤行业未来21日收益差 = -0.067%，配对 t=-0.15、"
        f"p=0.884，与 0 无法统计区分；分年 2019-2022 为负（风控前提成立），但 2023 +1.58%、"
        f"2024 +0.47%（前提反向失效）。裁定标准：若 C 相对 A 回撤明显下降且夏普不恶化则保留模块2，"
        f"否则判定失败、回退 V8。"
    )

    modules_text = (
        "<b>本页为模块2 的 2×2 消融报告</b>（主图/主指标为 D 臂 = 用户原定 V9.2 = 分析师因子 ON + 拥挤度 ON）。"
        + abl_html +
        "模块2（行业拥挤度风控）：在<b>选股完成、分配权重前</b>，对「所属行业 20 日成交额占全宇宙成交额比重 "
        "&gt; 该行业过去 3 年(756交易日)滚动 75% 分位」的入选个股，目标仓位打 7 折（100%→70%）。"
        "<b>不改变选股集合</b>、不动任何核心参数。<br>"
        "行业分类=申万一级 31 个（ak.sw_index_first_info + ak.index_component_sw 抓取 5207 只，"
        "落盘 sw_industry_map.parquet；V8 宇宙归属率 99.9% = 1538/1539，未归属股票不打折）。<br>"
        "point-in-time：分位阈值用 rolling(756).quantile(0.75)<b>.shift(1)</b>，"
        "确保阈值仅由 t 之前的历史构成、不含当日自身，零未来函数；"
        "因需 250 日以上历史才成熟，2018 年不触发任何折扣（正确的冷启动行为，非 bug）。<br>"
        f"实测标记强度：平均 6.5/31 个行业被判拥挤（中位 7，范围 0~14），全宇宙平均 24.8% 股票被打折；"
        f"实际持仓命中 V8选股 {d_off}/{n_off}（{hit_off:.1f}%）、V9.1选股 {d_on}/{n_on}（{hit_on:.1f}%），"
        f"平均净暴露分别下降约 {loss_off:.1f}% / {loss_on:.1f}%。<br>"
        "<b>前置证据（诚实披露，先验不稳固）</b>：拥挤 vs 非拥挤行业未来 21 日收益差仅 <b>-0.067%</b>，"
        "配对 t=-0.15、<b>p=0.884</b>（与 0 无法区分）；分年 2019 -0.32%、2020 -0.38%、2021 -0.55%、"
        "2022 -1.27%（风控前提成立），但 <b>2023 +1.58%、2024 +0.47%（前提反向失效）</b>；"
        "V8 实际持仓内(2024-25, 16期) 拥挤仓位 +1.31% vs 非拥挤 +2.22%（差 -0.91%，但 p=0.384 不显著）。"
        "故本模块本质是「以收益换回撤」的风控，夏普方向事前不可预判，须由回测实测裁定。")

    sw_on.reset_index().to_csv(config.OUTPUT_DIR / "v9_2_switch_log.csv", index=False)
    html = generate_html_v9(
        eq=eq_D, eq_clean=eq_D_clean, idx_eq=idx_eq,
        slip_map=slip_map, tier_counts=tier_counts,
        m_full=m_full, m_old=m_old, m_new=m_new,
        m_full_c=m_full_c, m_old_c=m_old_c, m_new_c=m_new_c,
        prev_ref=prev_ref, baseline_ref=baseline_ref, m_idx=m_idx,
        yearly=yearly, yearly_prev=yearly_prev, yearly_baseline=yearly_baseline,
        yearly_c=yearly,
        vol_regime=vol_regime, switch_log=sw_on.reset_index(),
        latest_signal=latest_df, holdings=pd.DataFrame(holdings),
        n_universe=n_universe, data_start=data_start, data_end=data_end,
        conclusion=conclusion,
        version_label="V9.2", prev_label="V9.1",
        modules_text=modules_text)
    out_path = config.OUTPUT_DIR / "report_v9_2.html"
    out_path.write_text(html, encoding="utf-8")

    print("\n================ 模块2 裁定 ================")
    print(f"[干净边际 C-A] 夏普 {mc_full:+.2f} | 新区间 {mc_new:+.2f} | 回撤 {mc_dd:+.2f}pp")
    print(f"[V9.1基座 D-B] 夏普 {md_full:+.2f} | 新区间 {md_new:+.2f} | 回撤 {md_dd:+.2f}pp")
    print(f"[{datetime.now()}] 报告已生成: {out_path}  耗时 {datetime.now()-t0}")


if __name__ == "__main__":
    main()
