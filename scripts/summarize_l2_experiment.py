"""Render an existing paired research report without tuning or re-training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def render(report: dict) -> str:
    cfg = report["config"]
    lines = [
        "# L2 日级特征对照实验",
        "",
        "**研究结果，不是可执行交易回测，不会自动替换生产模型。**",
        "",
        "## 实验范围",
        "",
        f"- 训练信息日期：{cfg['train_start']} 至 {cfg['validation_start']} 之前。",
        f"- 验证信息日期：{cfg['validation_start']} 至 {cfg['test_start']} 之前。",
        f"- 测试信息日期：{cfg['test_start']} 至 {cfg['end']}；每个周期另剔除跨界标签。",
        f"- 固定股票数：{len(report['universe']['symbols'])}。",
        f"- 共同有效特征样本：{report['coverage']['common_feature_rows']} 个股票日。",
        "- 基线为现有StockFormer算法的15项量价特征，不是线上完整辅助特征模型。",
        "- 对照组仅增加13项L2特征，使用相同训练样本、岭回归参数、标签及成本代理。",
        "- 日t信息于下一交易日开盘前使用，标签为下一交易日开盘至h日后开盘价格变化。",
        "",
        "## 样本外测试",
        "",
        "| 周期 | 股票日/日期数 | 基线Rank IC | 加L2 Rank IC | 基线方向准确率 | 加L2方向准确率 | Top-K均值差 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    def number(value, pct=False):
        return (
            "不可计算"
            if value is None
            else (f"{value * 100:.2f}%" if pct else f"{value:.4f}")
        )

    for horizon, result in report["horizons"].items():
        split = result["splits"]["test"]
        base, plus = split["baseline"], split["baseline_plus_l2"]
        delta = split["plus_l2_minus_baseline"]["top_k_forward_return"]
        lines.append(
            f"| {horizon}日 | {base['rows']}/{base['dates']} | {number(base['rank_ic'])} | "
            f"{number(plus['rank_ic'])} | {number(base['direction_accuracy'], True)} | "
            f"{number(plus['direction_accuracy'], True)} | {number(delta, True)} |"
        )
    lines += [
        "",
        "Top-K均值差为加L2减基线，百分数表示收益差的百分点；不是组合累计收益。",
        "每日Top-K标签可能重叠，未模拟涨跌停、停牌、整手成交和资金占用，不能直接作为实盘收益。",
        "",
        "## 诊断与限制",
        "",
        "本轮只验证是否存在值得继续研究的增益，不据此宣称L2有效或适合实盘。",
        "股票池较小、历史市场阶段有限，样本间也并非独立；必须保留负结果，不按测试结果重新调参。",
        "",
    ]
    lines += [f"- {text}" for text in report["diagnostics"]]
    lines += [
        "",
        "完整口径、每日指标、特征权重及归一化参数见同目录report.json，逐样本预测见paired_predictions.parquet。",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    output = args.report.parent / "RESULTS.md"
    with output.open("x", encoding="utf-8") as handle:
        handle.write(render(report))
    print(output)


if __name__ == "__main__":
    main()
