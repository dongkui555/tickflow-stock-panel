# TickFlow + AI Agents 集成部署

本目录以 TickFlow Stock Panel 为底座，保留 TickFlow 的看板、选股、指标流水、回测、监控和数据扩展，同时把 AI Agents 能力直接整合进本地服务：

- `aiagents` 数据源插件：A 股日线、分钟线、实时行情的多源降级。
- `/api/aiagents/*` 增强接口：资金流、海外行情与新闻、结构化多 Agent 分析、持仓优化、模拟盘。
- 前端 `/aiagents` 页面：统一查看增强研究、持仓优化和模拟盘。

## 本地启动

Windows:

```powershell
cd D:\chatgpt\aiagents-stock\tickflow-stock-panel-integrated
.\dev.ps1
```

访问：

- 前端开发服务：http://localhost:3011
- 后端 API：http://localhost:3018

## Docker 部署

```bash
cd /data/tickflow-stock-panel-integrated
cp .env.example .env
```

编辑 `.env`：

```env
PORT=3018
TICKFLOW_API_KEY=
AI_API_KEY=
AUTH_PASSWORD=你的访问密码
AIAGENTS_ENABLE=1
```

启动：

```bash
docker compose up -d --build
```

访问：

```text
http://服务器IP:3018
```

## 数据源切换

启动后进入：

```text
设置 -> 数据源
```

若 `AI Agents 多源降级（行情/资金/海外）` 显示可用，可把这些数据源切到 `aiagents`：

- 日 K 数据源
- 分钟 K 数据源
- 实时行情数据源

如果插件显示不可用，先确认：

- `DATA_DIR` 可写，TickFlow 能正常创建本地缓存。
- 机器可访问公开数据源；若配置了 `TUSHARE_TOKEN`，资金流和历史数据会更稳。
- Docker 能访问外部数据接口。

## 增强 API

```text
GET  /api/aiagents/health
POST /api/aiagents/context
POST /api/aiagents/analyze
GET  /api/aiagents/holdings
GET  /api/aiagents/paper
POST /api/aiagents/paper/order
```

所有增强结论仅用于研究和模拟，不构成投资建议。
