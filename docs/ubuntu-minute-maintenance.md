# Ubuntu 分钟导入后的服务恢复

离线写入仍须暂停 app 和所有正在运行的影子 worker，使用既有分钟写入事务。
恢复时必须先启动 app，等待 `/health` 返回 `ok`，再启动本轮实际暂停的影子 worker。
app 的启动生命周期会规范化并加载日 K；不能在该过程结束前让研究读取器抢读输入。
这不改变模型、校准、阈值、重复键校验或冻结账户。

服务器的统一恢复入口：

```sh
python3 /data/tick-stock-panel-v035/scripts/restore_panel_services.py \
  --maintenance-record /path/to/this-run/maintenance-started.json \
  --roles app frozen-pair-shadow decision-pair-shadow repaired-pair-shadow
```

只传入本次实际暂停的角色，不用旧维护记录启动原本禁用的服务。
记录必须包含停机前的 `images` 字典；入口会先验证镜像，再恢复服务。
主服务启动超时则保持影子 worker 暂停并返回错误，检查故障后重试恢复入口。
仅检查当前条件、不启动服务时加 `--check-only`。

停写期的全量回读和覆盖审计仍在恢复服务之前进行。
Ubuntu 的分钟事务 `finally` 应调用这个入口，恢复后再做接口和启动后引用复核。
已完成的历史导入脚本保留原字节，不能为更新模板而改写或重放。
下一轮日期脚本的恢复段参考
`output/shadow-price-keys-fix-20261029/run_import_remote_next.py`；它是版本化模板，
必须先按新日期、轮次生成脚本，不能直接执行已经完成日期的事务。

该入口只恢复容器并检查健康，不写行情、模型、协议、订单或账本。
日价真正不一致时研究读取器仍应阻断，不能任意去重或选择一条价格强行通过。
