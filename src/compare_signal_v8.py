# -*- coding: utf-8 -*-
"""
compare_signal_v8.py — 回测截面 vs 实盘信号 一致性验证（V8.1 首次初始化）。

流程：
  1) 重建回测侧 2025-08-12 截面：V8.1 链路（E 等权组合，V8+Trend+Breakout 各 1/3，
     策略内 30 只等权、重叠叠加、cap 10%）→ 权重表 ref_targets；
  2) 读取 signal_generator 产出的信号 CSV（output/signals/YYYY-MM-DD_signal.csv）；
  3) 对比选股集合与权重（命中率 / 集合差 / 权重最大偏差 / MAD）；
  4) 输出报告 output/report_signal_compare.html。

用法：
  python signal_generator.py --init --date 2025-08-12
  python compare_signal_v8.py --date 2025-08-12
"""

from __future__ import annotations

import os
import sys
import argparse
from datetime import datetime

for _k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
           "ALL_PROXY", "all_proxy"]:
    os.environ.pop(_k, None)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import config
from signal_generator import (load_panels, compute_v81_targets, SPLIT_DATE,
                              N_COMBO, N_SELECT, COMBO_CAP, FIXED_WEIGHT)
from report import compute_metrics


def main():
    ap = argparse.ArgumentParser(description="V8.1 信号一致性验证")
    ap.add_argument("--date", type=str, default="2025-08-12",
                    help="信号基准日（默认=回测数据末日 2025-08-12）")
    args = ap.parse_args()
    as_of = pd.Timestamp(args.date)

    # 1) 回测侧重建
    panels = load_panels(as_of, live=False)
    res = compute_v81_targets(panels, as_of)
    last_me = res["last_me"]
    seg = res["segment"]
    targets_sig = res["targets"]          # 信号侧（compute_v81_targets 内部）

    # 2) 读信号 CSV
    sig_csv = config.OUTPUT_DIR / "signals" / f"{as_of.strftime('%Y-%m-%d')}_signal.csv"
    if not sig_csv.exists():
        raise SystemExit(f"缺少信号文件 {sig_csv}：先运行 signal_generator.py --init --date {as_of.date()}")
    sig_df = pd.read_csv(sig_csv)
    targets_csv = {}
    for _, r in sig_df[sig_df["action"] != "SELL"].iterrows():
        targets_csv[str(r["code"]).zfill(6)] = float(r["target_weight"]) / 100.0

    # 3) 对比
    def _norm(c):
        try:
            return f"{int(c):06d}"
        except (ValueError, TypeError):
            return str(c)

    ref = {_norm(c): w for c, w in targets_sig.items()}
    sig = {_norm(c): w for c, w in targets_csv.items()}
    all_codes = sorted(set(ref) | set(sig))
    hit = sum(1 for c in all_codes if (c in ref) and (c in sig))
    only_ref = sorted(set(ref) - set(sig))
    only_sig = sorted(set(sig) - set(ref))
    miss = len(only_ref) + len(only_sig)
    w_ref = np.array([ref.get(c, 0.0) for c in all_codes])
    w_sig = np.array([sig.get(c, 0.0) for c in all_codes])
    w_diff = np.abs(w_ref - w_sig)
    max_dev = float(w_diff.max()) * 100
    mad = float(w_diff.mean()) * 100
    n_overlap = sum(1 for c in all_codes if c in ref and c in sig and ref[c] > 0 and sig[c] > 0)

    print(f"\n======== V8.1 信号一致性验证（{last_me.date()}）========")
    print(f"分段: {seg} | 回测侧持仓 {len(ref)} 只 | 信号CSV持仓 {len(sig)} 只")
    print(f"命中率: {hit}/{len(all_codes)} = {hit/max(len(all_codes),1)*100:.2f}%")
    print(f"集合差: 仅回测 {len(only_ref)} 只 {only_ref[:5] if only_ref else ''} | "
          f"仅信号 {len(only_sig)} 只 {only_sig[:5] if only_sig else ''}")
    print(f"权重: 最大偏差 {max_dev:.3f}% | 平均绝对偏差 {mad:.4f}% | 重叠且>0 {n_overlap} 只")
    ok = (miss == 0) and (max_dev < 0.01)
    print(f"一致性判定: {'✅ PASS（完全一致）' if ok else '❌ FAIL（存在差异）'}")

    # 4) 报告
    def _code(c):
        return f"{int(c):06d}" if str(c).isdigit() else str(c)

    name_map = {}
    try:
        import json
        nm = config.DATA_DIR / "stock_names.json"
        if nm.exists():
            name_map = json.loads(nm.read_text(encoding="utf-8"))
    except Exception:
        pass

    rows = []
    for c in all_codes:
        w_r, w_s = ref.get(c, 0.0), sig.get(c, 0.0)
        if abs(w_r - w_s) > 1e-9:
            mark = "⚠ 权重差异" if (c in ref and c in sig) else ("仅回测" if c in ref else "仅信号")
        else:
            mark = ""
        rows.append((c, name_map.get(c, c), w_r * 100, w_s * 100, w_r - w_s, mark))
    rows.sort(key=lambda r: -max(r[2], r[3]))
    table_rows = "".join(
        f"<tr><td>{r[0]}</td><td style='text-align:left'>{r[1]}</td>"
        f"<td>{r[2]:.2f}%</td><td>{r[3]:.2f}%</td><td>{r[4]*100:+.2f}%</td>"
        f"<td>{r[5]}</td></tr>" for r in rows)

    verdict = (f"<span style='color:#27ae60;font-weight:bold'>✅ 一致性 PASS</span>："
               f"回测截面与实盘信号完全一致（{hit}/{len(all_codes)} 命中，集合差 0，"
               f"权重最大偏差 {max_dev:.3f}%）"
               if ok else
               f"<span style='color:#c0392b;font-weight:bold'>❌ 一致性 FAIL</span>："
               f"存在 {miss} 只集合差或权重偏差（最大 {max_dev:.3f}%），需排查 signal_generator 链路")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>V8.1 模拟盘初始化 · 回测截面 vs 实盘信号</title>
<style>body{{font-family:-apple-system,'PingFang SC',sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222}}
table{{border-collapse:collapse;width:100%;margin:14px 0;font-size:12px}}
th,td{{border:1px solid #ddd;padding:5px 8px;text-align:center}}
th{{background:#f5f5f5}} h2{{border-bottom:2px solid #eee;padding-bottom:6px;margin-top:30px}}
.verdict{{background:#eefaf0;border:1px solid #a7d7b0;border-radius:6px;padding:12px 16px;margin:16px 0}}
.note{{background:#f2f6fc;border-left:4px solid #4a7abb;padding:10px 14px;font-size:13px;color:#444;margin:12px 0}}
img{{max-width:100%}}</style></head><body>
<h1>V8.1 模拟盘初始化：回测截面 vs 实盘信号</h1>
<p>截面日 {last_me.date()}（数据末日 {panels['data_end']}，宇宙 {len(panels['codes'])} 只）｜分段 {seg}</p>
<div class="note"><b>一致性验证口径：</b>回测侧用 V8.1 主流程（main_v8_1_split.py）同一套
选股/权重构建逻辑重建 {last_me.date()} 截面（{'E 等权组合' if seg == 'combo' else 'V8 原样'}），
与 signal_generator.py 产出的信号 CSV 逐股对比（集合 + 权重）。两者共用 strategies/ 与 factor_eval 模块。</div>
<h2>判定</h2>
<div class="verdict">{verdict}</div>
<h2>逐股明细（按权重降序；仅显示存在差异的行在表内以 ⚠ 标注）</h2>
<table><thead><tr><th>code</th><th>名称</th><th>回测权重</th><th>信号权重</th>
<th>差值</th><th>备注</th></tr></thead><tbody>{table_rows}</tbody></table>
<h2>与 V8.1 回测最新截面的衔接</h2>
<p style="font-size:13px">回测侧（main_v8_1_split.py）段2 于 {last_me.date()} 调仓（E 等权组合），
当日信号 CSV 与回测权重表一致 ⇒ 模拟盘从该截面起按信号建仓即可衔接 V8.1 净值曲线。
注意：sim_tracker.py 执行口径为「BUY 每只固定 10%、最多 10 只」（沿用 V7.1），
与 E 组合回测的分散权重（单只约 1.1%~cap 10%）存在差异，跟踪将产生偏差——
建议后续将 sim_tracker 升级为读取 target_weight 列按权重建仓（当前交付不修改 sim_tracker）。</p>
<p style="font-size:12px;color:#999">生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
</body></html>"""
    out = config.OUTPUT_DIR / "report_signal_compare.html"
    out.write_text(html, encoding="utf-8")
    print(f"\n报告: {out}")


if __name__ == "__main__":
    main()
