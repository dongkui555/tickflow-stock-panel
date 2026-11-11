"""Allow-listed Tushare gateway API catalog.

The catalog mirrors the 80 endpoints documented by the configured 15000-point
gateway.  It deliberately contains metadata only: callers still have to supply
the parameters required by each upstream API.  Keeping the allow-list in code
prevents the generic gateway endpoint from becoming an arbitrary URL proxy.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TushareCatalogItem:
    api: str
    label: str
    category: str
    http_supported: bool = True
    note: str = ""


_ROWS = (
    ("daily", "日线行情", "股票"),
    ("weekly", "周线行情", "股票"),
    ("monthly", "月线行情", "股票"),
    ("pro_bar", "复权行情", "股票"),
    ("daily_basic", "每日指标", "股票"),
    ("new_share", "IPO新股列表", "股票"),
    ("top_list", "龙虎榜每日明细", "股票"),
    ("top_inst", "龙虎榜机构明细", "股票"),
    ("pledge_detail", "股权质押明细", "股票"),
    ("pledge_stat", "股权质押统计", "股票"),
    ("margin", "融资融券交易汇总", "股票"),
    ("margin_detail", "融资融券交易明细", "股票"),
    ("repurchase", "股票回购", "股票"),
    ("share_float", "限售股解禁", "股票"),
    ("block_trade", "大宗交易", "股票"),
    ("stk_holdernumber", "股东人数", "股票"),
    ("moneyflow", "个股资金流向", "股票"),
    ("stk_holdertrade", "股东增减持", "股票"),
    ("stk_limit", "每日涨跌停价格", "股票"),
    ("hk_hold", "沪深股通持股明细", "股票"),
    ("income", "利润表", "财务"),
    ("balancesheet", "资产负债表", "财务"),
    ("cashflow", "现金流量表", "财务"),
    ("forecast", "业绩预告", "财务"),
    ("express", "业绩快报", "财务"),
    ("dividend", "分红送股", "财务"),
    ("fina_indicator", "财务指标数据", "财务"),
    ("fina_audit", "财务审计意见", "财务"),
    ("fina_mainbz", "主营业务构成", "财务"),
    ("disclosure_date", "财报披露计划", "财务"),
    ("fund_basic", "公募基金列表", "基金"),
    ("fund_company", "公募基金公司", "基金"),
    ("fund_nav", "公募基金净值", "基金"),
    ("fund_daily", "场内基金日线行情", "基金"),
    ("fund_div", "公募基金分红", "基金"),
    ("fund_portfolio", "公募基金持仓", "基金"),
    ("fund_adj", "基金复权因子", "基金"),
    ("fut_basic", "期货合约列表", "期货"),
    ("trade_cal", "期货交易日历", "期货"),
    ("fut_daily", "期货日线行情", "期货"),
    ("fut_holding", "每日成交持仓排名", "期货"),
    ("fut_wsr", "仓单日报", "期货"),
    ("fut_settle", "结算参数", "期货"),
    ("index_daily", "指数日线行情", "指数"),
    ("opt_basic", "期权合约列表", "期权"),
    ("opt_daily", "期权日线行情", "期权"),
    ("cb_basic", "可转债基础信息", "债券"),
    ("cb_issue", "可转债发行数据", "债券"),
    ("cb_daily", "可转债日线数据", "债券"),
    ("fx_obasic", "外汇基础信息", "外汇"),
    ("fx_daily", "外汇日线行情", "外汇"),
    ("index_basic", "指数基本信息", "指数"),
    ("index_weekly", "指数周线行情", "指数"),
    ("index_monthly", "指数月线行情", "指数"),
    ("index_weight", "指数成分和权重", "指数"),
    ("index_dailybasic", "大盘指数每日指标", "指数"),
    ("index_classify", "申万行业分类", "指数"),
    ("index_member_all", "申万行业成分", "指数"),
    ("hk_basic", "港股列表", "港股"),
    ("shibor", "SHIBOR利率数据", "宏观"),
    ("shibor_quote", "SHIBOR报价数据", "宏观"),
    ("shibor_lpr", "LPR贷款基础利率", "宏观"),
    ("libor", "LIBOR拆借利率", "宏观"),
    ("hibor", "HIBOR拆借利率", "宏观"),
    ("wz_index", "温州民间借贷利率", "宏观"),
    ("gz_index", "广州民间借贷利率", "宏观"),
    ("tmt_twincome", "台湾电子产业月营收", "行业特色"),
    ("tmt_twincomedetail", "台湾电子产业月营收明细", "行业特色"),
    ("bo_monthly", "电影月度票房", "行业特色"),
    ("bo_weekly", "电影周度票房", "行业特色"),
    ("bo_daily", "电影日度票房", "行业特色"),
    ("bo_cinema", "影院每日票房", "行业特色"),
    ("film_record", "全国电影剧本备案", "行业特色"),
    ("teleplay_record", "全国电视剧本备案", "行业特色"),
    ("report_rc", "券商盈利预测", "特色"),
    ("cyq_perf", "每日筹码成本和胜率", "特色"),
    ("cyq_chips", "每日筹码分布", "特色"),
    ("stk_rewards", "券商金股", "特色"),
    ("stk_factor_pro", "技术面因子", "特色"),
    ("stk_nineturn", "神奇九转", "特色"),
)

TUSHARE_API_CATALOG: dict[str, TushareCatalogItem] = {
    api: TushareCatalogItem(
        api=api,
        label=label,
        category=category,
        http_supported=api != "pro_bar",
        note="该接口按上游文档不支持 HTTP 调用" if api == "pro_bar" else "",
    )
    for api, label, category in _ROWS
}


def catalog_payload() -> list[dict]:
    return [asdict(item) for item in TUSHARE_API_CATALOG.values()]


assert len(TUSHARE_API_CATALOG) == 80
