# 通达信交易日盘后同步

Windows 任务计划程序中的任务名为 `TickStockPanel-TDX-DailySync`。北京时间工作日 17:00 触发，Ubuntu 交易日历确认开市后才同步。计划任务使用当前 Windows 用户和既有 SSH 密钥，Windows 需要保持开机且用户已登录。通达信的盘后下载把沪深北 1 分钟文件准备到 `vipdoc`。

任务只处理当日 LC1 和通达信最新完整报告期的财务增量。先核对股票身份、原始 240 分钟时间轴、日K 量价和文件指纹，再复用现有追加导入器。冲突走现有跨源核验和缺口队列。财务包按通达信 HTTPS 目录的大小及摘要校验，使用已验收的 41 个字段，按股票和报告期去重。已有同报告期字段差异留在核验回执。

数据或盘后写入窗口未就绪时，每 5 分钟重试。新写入最晚在 18:35 前启动，以便在晚间影子任务前恢复面板。正式写入时暂停相关写入进程，结束后恢复服务、复核分钟覆盖和接口。服务恢复和阶段状态保存在 Ubuntu 每日回执目录 `runtime/data/import_reports/tdx_daily_sync/date=YYYY-MM-DD/`。

## 安装与验收

将 `server.py`、`worker.py`、`source.py` 和 `financial-fields.json` 放入 Ubuntu 项目 `scripts/tdx_daily_sync/`，同时部署仓库的 `scripts/audit_market_minute_coverage.py` 到同目录。验收器与导入器使用相同的权威日K选择规则。Windows 以实际路径执行 `install_task.ps1 -Python <pythonw.exe> -Vipdoc <vipdoc> -Logs <logs>`。计划任务使用现有 `tickpanel-ubuntu` SSH 别名。

使用 `client.py --tdx <vipdoc> --logs <logs> --day YYYY-MM-DD --dry-run` 只读验证历史输入。用 `Get-ScheduledTaskInfo -TaskName TickStockPanel-TDX-DailySync` 查看下一次计划时间和退出码。暂停可执行 `Disable-ScheduledTask -TaskName TickStockPanel-TDX-DailySync`。

[通达信专业财务数据说明](https://www.tdx.com.cn/article/stockfin.html)
