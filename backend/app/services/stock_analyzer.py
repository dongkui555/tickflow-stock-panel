"""AI 个股分析服务 — 技术面 / 基本面 / 财务面 / 消息面 四维综合分析。

职责:
  组合一只股票的 K 线(含已算好的技术指标)+ 财务表 + 关键价位 →
  拼装客观技术分析系统提示词 → 流式调用 LLM → 逐 chunk 吐给前端。

与 financial_analyzer.py 的区别(刻意区分,非复用):
  - 角色:客观技术分析师(非 CFA 财务分析师)
  - 数据源:K 线 + 技术指标为主,财务表为辅(财务分析以财务表为主)
  - 输出框架:技术面→基本面→财务面→消息面(四维),落点是客观技术状态与风险提示
    (财务分析的落点是财务质量评级)。注意:本服务不输出买卖建议、操作指令。

不知道: HTTP、前端、配置持久化。
"""
from __future__ import annotations

import datetime
import json
import logging
import math
import numbers
from decimal import Decimal
from pathlib import Path
from typing import Any
from collections.abc import AsyncIterator
from datetime import timedelta

import polars as pl

from app.indicators.levels import compute_levels, summarize_levels
from app.market_time import cn_today
from app.services.financial_sync import get_financial_df

logger = logging.getLogger(__name__)

# 注入最近多少根日 K(技术面分析样本)
_KLINE_WINDOW = 90
# 注入财务表的最近期数
_MAX_PERIODS = 4


def _json_safe(value: Any) -> Any:
    """递归转换为 JSON 安全值。"""
    if value is None:
        return None
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        value = float(value)
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        number = float(value)
        if not math.isfinite(number):
            return None
        if isinstance(value, numbers.Integral):
            return int(value)
        return round(number, 4)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:  # noqa: BLE001
            return str(value)
    return value


# ================================================================
# 数据加载
# ================================================================

def _load_kline(repo, symbol: str) -> pl.DataFrame:
    """读取该标的最近 N 根日 K(已含技术指标 / 信号)。

    repo: KlineRepository;走内存缓存,性能可控。
    """
    end = cn_today()
    start = end - timedelta(days=_KLINE_WINDOW * 2)  # 多取一些保证交易日够
    # 按资产类型分流: ETF/指数走独立 enriched 存储 (无财务数据, 提示词已有兜底)
    df = repo.get_daily_asset(repo.resolve_asset_type(symbol), symbol, start, end)
    if df.is_empty():
        return df
    return df.tail(_KLINE_WINDOW)


def _clean_rows(df: pl.DataFrame, keep_cols: list[str]) -> list[dict]:
    """把 DataFrame 转成 JSON 安全的 dict 列表(只保留关键列 + 清洗 NaN/Inf + date→字符串)。

    polars 的 date 列会变成 Python datetime.date,json.dumps 无法直接序列化,
    必须转成 ISO 字符串,否则 json.dumps 抛 TypeError 让整个流静默失败。
    """
    cols = [c for c in keep_cols if c in df.columns]
    sub = df.select(cols)
    return _json_safe(sub.to_dicts())


def _load_financials(data_dir: Path, symbol: str) -> dict[str, list[dict]]:
    """读取该标的核心财务指标 + 利润表(只取最有信息量的两张表)。

    财务面只需要关键指标(ROE / 增速 / 毛利率 等),不需要把 4 张表全塞进上下文
    (那是 financial_analyzer 的职责)。这里取轻量,留给技术面更多 token。
    """
    out: dict[str, list[dict]] = {}
    for table in ("metrics", "income"):
        df = get_financial_df(data_dir, table)
        if df.is_empty():
            out[table] = []
            continue
        df = df.filter(pl.col("symbol") == symbol)
        if df.is_empty():
            out[table] = []
            continue
        if "period_end" in df.columns:
            df = df.sort("period_end", descending=True).head(2)  # 只取最近 2 期
        rows = []
        for rec in df.to_dicts():
            clean = {}
            for k, v in rec.items():
                if k == "symbol":
                    continue
                clean[k] = _json_safe(v)
            rows.append(clean)
        out[table] = rows
    return out


# ================================================================
# 系统提示词 —— 客观技术分析四维框架(与财务分析明确区分)
# 重要原则:只描述"指标/价位处于什么客观状态",不输出"应该怎么操作"。
# 本报告定位为客观行情分析,不提供任何买卖建议、仓位建议或交易指令。
# ================================================================

_SYSTEM_PROMPT = """你是一位拥有 15 年 A 股一线研究经验的技术分析分析师,擅长从 K 线、量价、关键价位与基本面交叉验证中客观解读个股的技术状态。你的任务是:基于提供的个股数据,产出一份**客观、中立、不包含任何买卖或操作建议**的技术分析报告。

## 核心红线(务必遵守)

- **绝对不输出**"买入/卖出/加仓/减仓/观望/轻仓/重仓/止损/止盈/建议买入区间/操作建议"等任何交易指令或倾向性措辞
- **绝对不**按"激进型/稳健型/保守型"给用户分别的操作建议
- 你的角色是**客观陈述**该股当前的技术状态、价位结构、量价特征与潜在风险,让读者自行判断
- 换成"一个中立财经记者能不能写出来"——能写就保留,不能写就删除

## 输出规范

用 **Markdown** 格式输出,严格遵循以下结构。不要输出任何 JSON 或代码块,直接输出 Markdown 正文。

### 1. 🎯 一句话定调(1-2 句)
用一句话概括该股当前的**技术状态**(如"近期高位放量滞涨,量能持续性存疑"/"价格在 60 日均线上方运行,均线呈多头排列")。结尾用【当前状态:企稳 / 反弹 / 震荡 / 调整 / 走弱】客观描述技术形态,**不评价好坏、不下操作结论**。

### 2. 📈 技术面分析(核心维度)
这是你的主战场,务必深入,只陈述客观事实:
- **趋势判断**:均线多头/空头排列、20/60 日均线方向、价格在均线之上/下
- **形态结构**:近期是否出现突破/破位/双底/双顶/旗形等关键形态
- **指标信号**:MACD 金叉/死叉/背离、KDJ 超买超卖、RSI 强弱、布林通道位置
- **量价配合**:放量上涨/缩量回调/量价背离/换手率异动
每条结论必须引用具体数值(如"MACD 在 6/12 出现金叉,DIF 0.32 上穿 DEA 0.18"),客观陈述,不下买卖定性。

### 3. 💰 关键价位(客观价位结构)
基于提供的关键价位数据,客观列出价位结构:
- **上方压力位**(逐档列出,标注来源):第一压力、第二压力
- **下方支撑位**(逐档列出,标注来源):第一支撑、第二支撑
- 用数据说话,引用提供的压力/支撑(成交密集区)/枢轴点数值
- **注意:只客观列出价位及其技术含义(如"此处为前期成交密集区"),不输出"建议买入区间""止损位""止盈位"等操作指令**

### 4. 🏭 基本面与财务面(辅助验证)
简要点评(2-4 句,不展开长篇):
- 盈利质量(ROE / 毛利率水平)、成长性(营收/利润增速)的客观水平
- 与技术面的**客观对照**:好公司 + 技术面走坏 → 客观陈述两者背离;差公司 + 技术面强势 → 客观提示炒作可能性
- 不下"逢低吸纳""规避"等结论

**当用户消息中标注了"该标的暂无财务数据"时**,本节请输出:
> 📌 财务面分析能力正在接入中。当前未同步该标的的财务报表,基本面维度暂无法评估。
> 技术面分析不依赖财务数据,以下结论依然有效;待财务数据同步后可补充本维度。

**当用户消息中标注了该标的为 ETF 时**,本节请输出:
> 📌 该标的为场内 ETF,无上市公司财务报表,基本面/财务面不适用公司盈利质量框架。
> 请仅基于价量与关键价位做技术分析,不要编造 ROE/营收等数字,也不要写成「财务接入中」。

**绝对不要**在无数据时编造 ROE / 增速等数字。

### 5. 📰 消息面(价量异动推断)
**注意:本期无直接新闻数据输入。** 请基于 K 线的**异动信号**进行客观推断(如:
- 涨停/连板/炸板 → 可能存在利好或资金关注
- 放量暴跌 → 可能存在未公开扰动
- 突破放量 → 可能存在催化剂
明确标注"[推断]",告诉读者这是基于价量的客观推测,真实消息面数据待接入。若无明显异动,直说"近期价量平稳,无明显异动信号"。

### 6. ⚖️ 综合研判与风险提示
2-3 段,只做客观描述,不下操作结论:
- 客观描述该股当前所处的技术阶段(底部企稳 / 上升途中 / 高位震荡 / 下跌趋势)
- 客观评估当前价位的"空间不对称性"(距上方压力位与下方支撑位的距离),不评价好坏
- 客观列出后续值得关注的量价信号(如量能能否维持、某均线得失、是否放量突破压力位),**不附任何操作结论**

## 分析准则(务必遵守)

1. **技术面优先**:技术面和量价是主要分析对象,基本面是交叉验证手段,主次分明
2. **数据说话**:每个判断引用具体数值,严禁空泛套话("走势良好"必须改成"连续 3 日站稳 20 日均线且放量")
3. **客观中立**:看多就客观陈述多头特征,看空就客观陈述空头特征,不下"该买/该卖"结论;数据不支持时直言无法判断
4. **价位精确**:压力位/支撑位必须落到具体价格,基于提供的关键价位数据陈述
5. **不输出操作指令**:不写"买入/卖出/止损/加仓/减仓/仓位建议/操作建议"等任何交易指令;提示潜在风险但不下操作结论
6. **简明客观**:用读者能扫读的密度输出,总字数 1000-1800 字,重在客观信息密度

## 重要免责
报告末尾附一行:"> ⚠️ 本内容由 AI 基于公开行情与财务数据生成,仅客观陈述技术状态,不构成任何投资建议或买卖指令。交易有风险,入市需谨慎。"

现在请基于下方数据进行分析。"""


# ================================================================
# 用户消息构建
# ================================================================

def _paper_position_part(positions: list[dict], close: float | None, raw_close: float | None) -> str:
    """虚拟持仓上下文 (V3): 模拟盘持有该标的时注入提示词, 让 AI 结合仓位给建议。

    positions 为多账户列表 [{account_id, qty, avg_cost}, ...] (V2 账户隔离),
    只收 qty>0 的账户, 逐账户一行; 全空则返回空串 (不注入"空仓"之类误导性表述)。
    浮盈用 raw 口径: 持仓 avg_cost 是不复权 raw 成本, 与前复权 close 混算会在
    除权后产生虚假盈亏, 故优先用 raw_close (缺列时退化用 close 并接受口径误差)。
    """
    lines: list[str] = []
    for item in positions or []:
        qty = item.get("qty") or 0
        avg_cost = item.get("avg_cost") or 0
        if qty <= 0 or avg_cost <= 0:
            continue
        line = f"账户 {item.get('account_id') or 'default'}: 持仓 {qty} 股, 平均成本 {avg_cost:.3f} 元 (不复权)"
        ref = raw_close if (raw_close is not None and raw_close > 0) else close
        if ref is not None and ref > 0:
            line += f", 现价 {ref:.2f} 元, 浮动盈亏 {(ref / avg_cost - 1) * 100:+.2f}%"
        lines.append(line)
    if not lines:
        return ""
    return (
        "\n以下是该标的在模拟盘 (虚拟账户) 的当前持仓:\n"
        + "\n".join(lines)
        + "\n请在结论中结合该虚拟持仓状态给出持有/减仓/加仓的可执行参考,"
        "并明确这是基于模拟盘虚拟持仓的建议。\n"
    )


def _build_user_prompt(
    kline_tail: list[dict],
    fins: dict[str, list[dict]],
    levels: dict[str, list[dict]],
    close: float | None,
    symbol: str,
    focus: str,
    asset_type: str = "stock",
    paper_positions: list[dict] | None = None,
    raw_close: float | None = None,
) -> str:
    """构建用户消息:标的 + 价位摘要 + 技术指标 JSON + 财务摘要 + 关注点 (+ 虚拟持仓)。

    asset_type 用于区分无财务数据时的文案:指数/ETF 无财务是常态,不走 Free 文案。
    """
    parts: list[str] = [
        f"标的标准代码: {symbol}",
        f"关键价位概览: {summarize_levels(levels, close)}",
        "",
        "以下是该标的最近日 K 数据(JSON,含 OHLCV 与已计算的技术指标。"
        f"最近 {_KLINE_WINDOW} 个交易日,升序):",
        "```json",
        json.dumps(_json_safe(kline_tail), ensure_ascii=False, allow_nan=False),
        "```",
    ]

    has_fin = any(fins.values())
    if has_fin:
        parts.extend([
            "",
            "以下是该标的最新财务数据(JSON,核心指标 + 利润表,金额单位为元):",
            "```json",
            json.dumps(_json_safe(fins), ensure_ascii=False, allow_nan=False),
            "```",
        ])
    elif asset_type == "index":
        parts.extend([
            "",
            "(该标的为指数: 无财务、股本与涨跌停数据。请按系统提示词第 4 节的说明,"
            "在基本面/财务面维度给出\"接入中\"的友好提示,不要编造数据;"
            "消息面维度基于价量异动推断即可。)",
        ])
    elif asset_type == "etf":
        parts.extend([
            "",
            "(该标的为 ETF: 场内基金无上市公司财务报表、股本与涨跌停口径。"
            "请按系统提示词第 4 节的说明,在基本面/财务面维度明确这是 ETF 常态,"
            "不要编造公司财务数字,也不要写成「财务接入中」;"
            "消息面维度基于价量异动推断即可。)",
        ])
    else:
        parts.extend([
            "",
            "(该标的暂无财务数据:当前为 Free 模式或尚未同步财务报表。"
            "请按系统提示词第 4 节的说明,在基本面/财务面维度给出\"接入中\"的友好提示,不要编造数据。)",
        ])

    from app.services.ai_provider import build_focus_instruction
    focus_instruction = build_focus_instruction(focus, report_name="个股分析报告")
    if focus_instruction:
        parts.extend(["", focus_instruction])
    paper_part = _paper_position_part(paper_positions or [], close, raw_close)
    if paper_part:
        parts.extend(["", paper_part])
    return "\n".join(parts)


# ================================================================
# 关键列筛选(控制上下文体积)
# ================================================================

_KLINE_KEEP_COLS = [
    "date", "open", "high", "low", "close", "volume", "change_pct",
    "ma5", "ma10", "ma20", "ma60",
    "macd_dif", "macd_dea", "macd_hist",
    "kdj_k", "kdj_d", "kdj_j",
    "rsi_6", "rsi_14", "rsi_24",
    "boll_upper", "boll_mid", "boll_lower",
    "atr_14", "vol_ratio_5d", "turnover_rate",
    "consecutive_limit_ups",
    # 信号类(布尔)——只挑对消息面推断有用的几个
    "signal_limit_up", "signal_broken_limit_up", "signal_macd_golden",
    "signal_macd_dead", "signal_ma_golden_5_20", "signal_volume_surge",
    "signal_boll_breakout_upper", "signal_boll_breakdown_lower",
]


_AGENT_SPECS = [
    {
        "key": "technical",
        "name": "技术分析师",
        "mission": "只看趋势、均线、形态、MACD/KDJ/RSI/BOLL、量价配合与支撑阻力，判断当前技术位置和信号有效性。",
    },
    {
        "key": "fund_flow",
        "name": "资金分析师",
        "mission": "重点分析主力/大单资金、近5/20/60日资金趋势、放量来源、换手变化与资金持续性，不能把单日流入当作充分理由。",
    },
    {
        "key": "fundamental",
        "name": "基本面分析师",
        "mission": "分析公司基本信息、行业位置、估值、财务质量、业绩变化和基本面风险；数据缺失时明确降级，不编造指标。",
    },
    {
        "key": "sentiment",
        "name": "新闻情绪分析师",
        "mission": "分析国内新闻、海外市场、美股/汇率/大宗商品等外部扰动、题材热度和情绪催化；新闻缺失时只能依据已知行情异动做推断。",
    },
    {
        "key": "risk",
        "name": "风险管理师",
        "mission": "识别涨跌停、炸板、连续异动、流动性、波动率、回撤空间、数据覆盖缺口和事件风险，给出风险等级与失效条件。",
    },
    {
        "key": "trader",
        "name": "短线交易分析师",
        "mission": "把前面维度转成1-5个交易日的条件化交易参考，包括触发买点、卖点/止盈、止损、仓位上限和不交易条件。",
    },
]


_AGENT_SYSTEM_PROMPT = """你是 A 股短线研究系统中的{agent_name}。

你的职责: {mission}

要求:
- 只基于用户提供的数据和可验证推断，不编造财务、资金或新闻数据。
- 所有判断必须引用关键数据或说明数据缺失。
- 输出必须紧凑，面向后续总评审使用。
- 只输出 JSON，不要 Markdown，不要代码块。

JSON 结构:
{{
  "stance": "偏多/中性/偏空/数据不足",
  "score": 0-100,
  "confidence": 0-100,
  "summary": "一句话结论",
  "evidence": ["证据1", "证据2"],
  "risks": ["风险1", "风险2"],
  "watch": ["后续观察点1", "后续观察点2"]
}}
"""


_CHIEF_REVIEWER_SYSTEM_PROMPT = """你是 A 股短线多 Agent 研究系统的总评审。

你会收到同一只股票的行情、关键价位、财务、资金、新闻/海外上下文，以及 6 位分析师的独立结论。你的任务是做交叉验证，不是简单平均分。

输出要求:
- 用 Markdown 输出，直接给正文，不要 JSON，不要代码块。
- 必须把“数据可靠性”和“数据缺失”写清楚。
- 可以给短线交易参考，但必须是条件化表达，例如“若放量站上某价位，则...；若跌破某价位，则...”，不能写成确定性承诺。
- 不得承诺收益，不得暗示一定上涨。
- 总字数控制在 1200-2200 字。

固定结构:
### 1. 总评结论
给出综合评级、短线状态、确定性等级和核心矛盾。

### 2. 六位分析师交叉验证
用表格列出技术、资金、基本面、新闻情绪、风险、短线交易六个维度的结论、证据、风险。

### 3. 关键价位与触发条件
列出上方压力、下方支撑、触发买点、卖点/止盈参考、止损/失效参考。价位必须来自提供的关键价位或最近 K 线。

### 4. 1-5个交易日短线计划参考
分“强势确认”“震荡等待”“走弱处理”三种情景写，不输出无条件买卖指令。

### 5. 数据质量与风险提示
说明资金流、新闻、财务、海外数据是否可用，以及这些缺失如何影响置信度。

末尾附一行:
> ⚠️ 本内容由 AI 基于本地行情、资金、财务和新闻上下文生成，仅供研究参考，不构成投资建议或买卖指令。交易有风险，入市需谨慎。
"""


def _event_json(payload: dict) -> str:
    return json.dumps(_json_safe(payload), ensure_ascii=False, allow_nan=False)


def _brief_text(text: str, limit: int = 90) -> str:
    compact = " ".join(str(text or "").replace("\n", " ").split())
    return compact[:limit] + ("…" if len(compact) > limit else "")


def _safe_asset_type(repo, symbol: str) -> str:
    try:
        return repo.resolve_asset_type(symbol)
    except Exception:  # noqa: BLE001
        return "stock"


def _compact_aiagents_context(context: dict | None) -> dict:
    if not isinstance(context, dict):
        return {}
    fund = context.get("fund_flow") or {}
    news = context.get("news") or {}
    news_data = news.get("news_data") or {}
    overseas = context.get("overseas") or {}
    overseas_market = overseas.get("market") or {}
    overseas_news = overseas.get("global_news") or {}
    return _json_safe({
        "generated_at": context.get("generated_at"),
        "quote": context.get("quote") or {},
        "technical": context.get("technical") or {},
        "basic": context.get("basic") or {},
        "fund_flow": {
            "data_success": fund.get("data_success"),
            "source": fund.get("source"),
            "days": fund.get("days"),
            "date_start": fund.get("date_start"),
            "date_end": fund.get("date_end"),
            "latest_trade_date": fund.get("latest_trade_date"),
            "freshness_status": fund.get("freshness_status"),
            "lag_days": fund.get("lag_days"),
            "freshness_note": fund.get("freshness_note"),
            "latest": fund.get("latest"),
            "main_net_total": fund.get("main_net_total"),
            "main_net_5d": fund.get("main_net_5d"),
            "main_net_20d": fund.get("main_net_20d"),
            "main_net_60d": fund.get("main_net_60d"),
            "positive_days_20": fund.get("positive_days_20"),
            "positive_days_60": fund.get("positive_days_60"),
            "rows": (fund.get("rows") or [])[-12:],
            "error": fund.get("error"),
        },
        "news": {
            "data_success": news.get("data_success"),
            "source": news.get("source"),
            "count": news_data.get("count"),
            "items": (news_data.get("items") or [])[:12],
            "error": news.get("error"),
        },
        "overseas": {
            "data_success": overseas.get("data_success"),
            "impact": overseas.get("impact") or {},
            "market": {
                "data_success": overseas_market.get("data_success"),
                "source": overseas_market.get("source"),
                "quotes": (overseas_market.get("quotes") or [])[:12],
                "error": overseas_market.get("error"),
            },
            "global_news": {
                "items": (overseas_news.get("items") or [])[:12],
                "count": overseas_news.get("count"),
                "error": overseas_news.get("error"),
            },
        },
    })


def _build_multi_agent_payload(
    *,
    kline_tail: list[dict],
    fins: dict[str, list[dict]],
    levels: dict[str, list[dict]],
    close: float | None,
    symbol: str,
    focus: str,
    asset_type: str,
    aiagents_context: dict | None,
) -> dict:
    from app.services.ai_provider import sanitize_focus

    return _json_safe({
        "symbol": symbol,
        "asset_type": asset_type,
        "close": close,
        "level_summary": summarize_levels(levels, close),
        "levels": levels,
        "recent_kline": kline_tail[-60:],
        "financials": fins,
        "enhanced_context": _compact_aiagents_context(aiagents_context),
        "user_focus": sanitize_focus(focus),
    })


def _agent_messages(agent: dict, payload: dict) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": _AGENT_SYSTEM_PROMPT.format(
                agent_name=agent["name"],
                mission=agent["mission"],
            ),
        },
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, allow_nan=False),
        },
    ]


def _reviewer_messages(payload: dict, agent_outputs: dict[str, dict]) -> list[dict[str, str]]:
    review_payload = _json_safe({
        **payload,
        "agent_outputs": agent_outputs,
    })
    return [
        {"role": "system", "content": _CHIEF_REVIEWER_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(review_payload, ensure_ascii=False, allow_nan=False)},
    ]


# ================================================================
# 流式分析入口
# ================================================================

async def analyze_stock_stream(
    repo,
    data_dir: Path,
    symbol: str,
    focus: str = "",
    mode: str = "multi_agent",
) -> AsyncIterator[str]:
    """流式个股分析:yield 出每个 NDJSON 事件。

    协议(与 financial_analyzer 一致,前端解析无差异):
      {"type":"meta","symbol","summary","levels"}  数据 + 价位摘要
      {"type":"delta","content":"..."}             逐 chunk 文本
      {"type":"error","message":"..."}
      {"type":"done"}
    """
    # 1. 加载 K 线
    df = _load_kline(repo, symbol)
    if df.is_empty():
        yield json.dumps({
            "type": "error",
            "message": f"标的 {symbol} 暂无日 K 数据,请先同步",
        }, ensure_ascii=False)
        return

    # 2. 价位计算(基于 K 线)
    levels = compute_levels(df)
    close = float(df.tail(1)["close"][0]) if "close" in df.columns else None

    # 3. 财务(辅助)。ETF/指数无公司报表, 跳过扫描, 由提示词走对应常态文案。
    asset_type = repo.resolve_asset_type(symbol)
    fins = {} if asset_type in {"etf", "index"} else _load_financials(data_dir, symbol)

    analysis_mode = (mode or "multi_agent").strip().lower()
    if analysis_mode not in {"quick", "multi_agent"}:
        analysis_mode = "multi_agent"

    # 4. meta
    yield _event_json({
        "type": "meta",
        "symbol": symbol,
        "summary": summarize_levels(levels, close),
        "levels": levels,
        "close": close,
        "analysis_mode": analysis_mode,
        "agents": (
            [{"key": "data_context", "name": "数据上下文", "status": "pending", "message": ""}]
            + [
                {"key": a["key"], "name": a["name"], "status": "pending", "message": ""}
                for a in _AGENT_SPECS
            ]
            + [{"key": "chief_reviewer", "name": "总评审", "status": "pending", "message": ""}]
        ) if analysis_mode == "multi_agent" else [],
    })

    # 5+6. 构建提示词 + 流式调用 LLM(整体 try-except,任何异常都 yield error,避免前端卡死)
    try:
        from app.services.ai_provider import generate_ai_text, stream_ai_text

        # 虚拟持仓上下文 (V3): 逐账户 (V2 多账户) 收集持有该标的的持仓注入
        # (任何异常静默跳过, 不影响分析)。
        paper_positions: list[dict] = []
        try:
            from app.strategy import paper as paper_trading
            for acc_id in paper_trading.list_account_ids(data_dir):
                pos = paper_trading.load_positions(data_dir, account_id=acc_id).get(symbol)
                if pos and (pos.get("qty") or 0) > 0:
                    paper_positions.append({"account_id": acc_id, **pos})
        except Exception as e:  # noqa: BLE001
            logger.warning("读取模拟盘持仓失败 (跳过注入): %s", e)
        # 浮盈口径用 raw: 持仓成本不复权, 与前复权 close 混算会在除权后失真
        raw_close = float(df.tail(1)["raw_close"][0]) if "raw_close" in df.columns else None

        kline_tail = _clean_rows(df, _KLINE_KEEP_COLS)
        asset_type = _safe_asset_type(repo, symbol)

        if analysis_mode == "quick":
            user_prompt = _build_user_prompt(kline_tail, fins, levels, close, symbol, focus,
                                             asset_type=asset_type, paper_positions=paper_positions,
                                             raw_close=raw_close)
            got_content = False
            async for delta in stream_ai_text(
                [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.5,
                # 不限制输出: 推理模型(deepseek reasoner 系)思考 token 计入 max_tokens
                # 预算, 固定上限会把正文挤光(实测 4500 全被推理吃掉 → 正文 0 字)。
                max_tokens=None,
            ):
                got_content = True
                yield _event_json({"type": "delta", "content": delta})

            if not got_content:
                logger.warning("AI stock quick analysis ended with empty content for %s", symbol)
                yield _event_json({"type": "error", "message": "AI 未返回正文(输出被截断), 请重试"})
                return
            yield _event_json({"type": "done"})
            return

        aiagents_context: dict | None = None
        yield _event_json({
            "type": "agent",
            "agent_key": "data_context",
            "agent_name": "数据上下文",
            "status": "running",
            "message": "正在读取资金、新闻、海外和旧 6Agent 上下文",
        })
        try:
            from app.services.aiagents_bridge import market_context

            aiagents_context = market_context(
                symbol,
                include_fund_flow=True,
                include_news=True,
                include_overseas=True,
                fund_flow_days=60,
            )
            compact = _compact_aiagents_context(aiagents_context)
            fund = compact.get("fund_flow") or {}
            fund_status = "可用" if fund.get("data_success") else "降级"
            if fund.get("freshness_status") == "lagging_1d":
                fund_status += "(滞后1日)"
            elif fund.get("freshness_status") == "stale":
                fund_status += f"(滞后{fund.get('lag_days') or '?'}日)"
            data_bits = [
                f"资金{fund_status}",
                f"新闻{(compact.get('news') or {}).get('count') or 0}条",
                f"海外{'可用' if (compact.get('overseas') or {}).get('data_success') else '降级'}",
            ]
            yield _event_json({
                "type": "agent",
                "agent_key": "data_context",
                "agent_name": "数据上下文",
                "status": "done",
                "message": " / ".join(data_bits),
            })
        except Exception as exc:  # noqa: BLE001
            logger.warning("AI agents context failed for %s: %s", symbol, exc)
            yield _event_json({
                "type": "agent",
                "agent_key": "data_context",
                "agent_name": "数据上下文",
                "status": "error",
                "message": f"增强上下文读取失败: {exc}",
            })

        payload = _build_multi_agent_payload(
            kline_tail=kline_tail,
            fins=fins,
            levels=levels,
            close=close,
            symbol=symbol,
            focus=focus,
            asset_type=asset_type,
            aiagents_context=aiagents_context,
        )
        payload["paper_positions"] = paper_positions
        payload["raw_close"] = raw_close
        agent_outputs: dict[str, dict] = {}
        for agent in _AGENT_SPECS:
            yield _event_json({
                "type": "agent",
                "agent_key": agent["key"],
                "agent_name": agent["name"],
                "status": "running",
                "message": "分析中",
            })
            try:
                text = await generate_ai_text(
                    _agent_messages(agent, payload),
                    temperature=0.25,
                    max_tokens=None,
                    timeout=160,
                )
                agent_outputs[agent["key"]] = {
                    "name": agent["name"],
                    "raw": text,
                }
                yield _event_json({
                    "type": "agent",
                    "agent_key": agent["key"],
                    "agent_name": agent["name"],
                    "status": "done",
                    "message": _brief_text(text),
                })
            except Exception as exc:  # noqa: BLE001
                logger.exception("AI stock agent %s failed for %s: %s", agent["key"], symbol, exc)
                agent_outputs[agent["key"]] = {
                    "name": agent["name"],
                    "error": str(exc),
                }
                yield _event_json({
                    "type": "agent",
                    "agent_key": agent["key"],
                    "agent_name": agent["name"],
                    "status": "error",
                    "message": str(exc),
                })

        yield _event_json({
            "type": "agent",
            "agent_key": "chief_reviewer",
            "agent_name": "总评审",
            "status": "running",
            "message": "正在综合六位分析师结论",
        })
        got_content = False
        async for delta in stream_ai_text(
            _reviewer_messages(payload, agent_outputs),
            temperature=0.5,
            max_tokens=None,
            prefer_final_answer=True,
        ):
            got_content = True
            yield _event_json({"type": "delta", "content": delta})

    except Exception as e:
        logger.exception("AI stock analysis failed for %s: %s", symbol, e)
        yield _event_json({"type": "error", "message": f"AI 分析失败: {e}"})
        return

    if not got_content:
        # 流正常结束但一个正文块都没有(典型: 输出上限被思考吃光后静默截断)
        logger.warning("AI stock analysis ended with empty content for %s", symbol)
        yield _event_json({"type": "error", "message": "AI 未返回正文(输出被截断), 请重试"})
        return
    yield _event_json({
        "type": "agent",
        "agent_key": "chief_reviewer",
        "agent_name": "总评审",
        "status": "done",
        "message": "综合报告已生成",
    })
    yield _event_json({"type": "done"})
