# Tushare 80 接口接入

系统现在登记并开放了 `F:\qq\15000积分基础功能.txt` 中的 80 个接口。

## 配置

```dotenv
TUSHARE_MODE=promax
TUSHARE_PROMAX_URL=https://pcd.mobcvb.cn/tushare/pro
TUSHARE_PROMAX_KEY=你的Promax密钥
TUSHARE_PROMAX_VERIFY_TLS=0

TUSHARE_DATAHUB_ENABLED=1
TUSHARE_DATAHUB_URL=https://datahubco.com/app-api/openapi/v1/tushare
# 空值时复用 TUSHARE_PROMAX_KEY；也可以单独填 datahubco 密钥
TUSHARE_DATAHUB_KEY=
TUSHARE_DATAHUB_VERIFY_TLS=0
TUSHARE_EXT_SOURCE_CHAIN=promax,datahub
```

新版默认 Datahub 地址使用 HTTPS。2026-10-10 的核验中，旧 HTTP 地址返回
301，同一主机和路径的 HTTPS 日线请求正常。已有 `.env` 显式配置不会自动改变；
切换前应核对该配置。日线可用不代表分钟接口有权限，分钟 403/400 必须保留为源错误。

## 使用

- `GET /api/tushare/catalog`：查看 80 个接口、分类、是否可 HTTP 调用、是否已有结构化落盘和自动同步状态。
- `POST /api/tushare/query`：按接口名和参数试调。请求体示例：

```json
{
  "api": "moneyflow",
  "source": "auto",
  "params": {
    "ts_code": "000001.SZ",
    "start_date": "20260701",
    "end_date": "20260804"
  },
  "max_rows": 100
}
```

`source=auto` 先走 Promax，只有请求抛出错误时才切换 datahubco；空结果是合法的“查询成功但无记录”，不会误切换。

## 落盘策略

日线、资金流、涨跌停、龙虎榜、板块、筹码、财务等已有业务表继续走原来的结构化 Parquet 和增量任务。80 个接口中的其余低频表先通过目录/试调接口接入，按需查询，避免把基金、期货、宏观和特色接口在每天盘后无参数全量拉取。

`pro_bar` 按网关文档标记为 SDK 限定，HTTP 端点不会伪造调用；集合竞价也仍使用现有 Promax 专用接口。
