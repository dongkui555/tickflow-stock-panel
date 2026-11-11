import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Bell, RefreshCw } from 'lucide-react'
import { api } from '@/lib/api'
import { QK } from '@/lib/queryKeys'
import { fmtPct } from '@/lib/format'

const labels: Record<string, string> = {
  waiting_data: '等待当天数据', predicted: '预测已留档', prediction_failed: '预测失败',
  sent: '已发送', failed: '发送失败，等待重试', sending: '发送中或回执待核验',
  unconfigured: '智能机器人未配置或未启用', disabled: '已关闭',
  missing_recipient: '待绑定单聊接收用户', not_connected: '等待智能机器人连接',
  unknown: '发送回执待核验，已暂停自动重发',
  blocked: '数据日期或版本校验未通过，已阻止发送',
  no_candidates: '无合格候选，已留档',
}

export function PredictionWorkflowPanel() {
  const client = useQueryClient()
  const query = useQuery({ queryKey: QK.predictionWorkflow, queryFn: api.predictionWorkflow, refetchInterval: 30000 })
  const save = useMutation({
    mutationFn: ({ enabled, notify, chat_id }: { enabled: boolean; notify: boolean; chat_id?: string }) => api.setPredictionWorkflow(enabled, notify, chat_id),
    onSuccess: data => {
      client.setQueryData(QK.predictionWorkflow, data)
      client.invalidateQueries({ queryKey: ['short-term-ai'] })
    },
  })
  const data = query.data
  const note = data?.state.notification?.status
  return <section className="border-y border-border py-4">
    <div className="flex flex-wrap items-center justify-between gap-3">
      <h2 className="text-sm font-semibold">盘后预测 / 次日执行</h2>
      <div className="flex flex-wrap items-center gap-4 text-xs">
        <label className="flex items-center gap-2"><input type="checkbox" checked={data?.config.enabled ?? false}
          disabled={!data || save.isPending} onChange={e => data && save.mutate({ notify: data.config.notify, enabled: e.target.checked })} />自动预测</label>
        <label className="flex items-center gap-2"><Bell className="h-3.5 w-3.5" /><input type="checkbox" checked={data?.config.notify ?? false}
          disabled={!data || save.isPending} onChange={e => data && save.mutate({ enabled: data.config.enabled, notify: e.target.checked })} />企业微信通知</label>
        <button type="button" title="刷新预测任务状态" aria-label="刷新预测任务状态" onClick={() => query.refetch()}
          className="flex h-8 w-8 items-center justify-center rounded border border-border"><RefreshCw className="h-4 w-4" /></button>
      </div>
    </div>
    {query.isLoading && <p className="mt-3 text-xs text-muted">读取预测记录中…</p>}
    {query.isError && <p className="mt-3 text-xs text-danger">预测任务状态读取失败</p>}
    {data && <>
      <dl className="mt-3 grid gap-3 text-xs sm:grid-cols-2 xl:grid-cols-4">
        <div><dt className="text-muted">盘后数据</dt><dd className="mt-1 break-words">{data.state.readiness?.reason || '等待盘后数据校验'}{data.state.readiness?.date && ` · ${data.state.readiness.date}`}</dd></div>
        <div><dt className="text-muted">候选快照</dt><dd className="mt-1 break-words">{data.snapshot.as_of || '暂无'} · {labels[data.state.status || ''] || '等待首次运行'}</dd></div>
        <div><dt className="text-muted">盘前复核</dt><dd className="mt-1 break-words">{data.premarket.reason || '尚未复核'}{data.premarket.date && ` · ${data.premarket.date}`}</dd></div>
        <div><dt className="text-muted">企业微信智能机器人</dt><dd className="mt-1 break-words">{!data.config.notify ? '已关闭' : !data.config.bot_configured || !data.config.bot_enabled ? labels.unconfigured : !data.config.chat_id ? labels.missing_recipient : !data.bot?.connected ? labels.not_connected : labels[note || ''] || '等待预测结果'}</dd></div>
      </dl>
      <label className="mt-3 flex max-w-md flex-col gap-2 text-xs text-secondary">单聊接收用户
        <select aria-label="单聊接收用户" className="h-9 min-w-0 rounded border border-border bg-surface px-2"
          disabled={save.isPending || !data.config.bot_configured} value={data.config.chat_id || ''}
          onChange={e => save.mutate({ enabled: data.config.enabled, notify: data.config.notify, chat_id: e.target.value })}>
          <option value="">{data.bot?.chats.some(c => c.kind === 'single') ? '未选择接收用户' : '暂无单聊会话'}</option>
          {data.config.chat_id && !data.bot?.chats.some(c => c.chatid === data.config.chat_id) && <option value={data.config.chat_id}>{data.config.chat_id}（待重新识别）</option>}
          {data.bot?.chats.filter(c => c.kind === 'single').map(c => <option value={c.chatid} key={c.chatid}>{c.chatid}</option>)}
        </select>
      </label>
      {save.isError && <p className="mt-2 text-xs text-danger">接收用户或开关保存失败</p>}
      {data.state.error && <p className="mt-2 text-xs text-danger">{data.state.error}</p>}
      {data.snapshot.no_candidates && <p className="mt-2 text-xs text-secondary">
        本日已记录 0 个候选：模型尚无通过验证的预测周期，继续观察。
        {data.snapshot.universe_total != null && ` 本次覆盖 ${data.snapshot.universe_total} 只股票。`}
      </p>}
      {data.state.data_retry && <div className="mt-3 border-t border-border pt-3 text-xs">
        <p className="text-secondary">增量补同步{data.state.data_retry.next_retry_at && ` · 下次检查 ${new Date(data.state.data_retry.next_retry_at).toLocaleString('zh-CN', { timeZone: 'Asia/Shanghai', hour12: false })}`}</p>
        {data.state.data_retry.reason && <p className="mt-1 text-muted">{data.state.data_retry.reason}</p>}
        <ul className="mt-1 space-y-1 text-muted">{data.state.data_retry.datasets.map(item => <li key={item.dataset}>
          {item.dataset} · {item.status === 'succeeded' ? '补齐' : item.status === 'needs_attention' ? '持续未齐，低频重试' : '等待重试'}
          {` · 已尝试 ${item.attempts || 0} 次 · 仍缺 ${item.missing_symbols?.length || 0} 只`}{item.error && ` · ${item.error}`}
        </li>)}</ul>
      </div>}
      {(!data.config.bot_configured || !data.config.bot_enabled) && <a href="/settings?tab=monitoring" className="mt-2 inline-block text-xs text-accent">配置企业微信智能机器人</a>}
      {!!data.snapshot.rows?.length && <details className="mt-3 text-xs">
        <summary className="cursor-pointer text-secondary">盘后留档候选 · {data.snapshot.rows.length} 只 · {data.snapshot.generated_at}</summary>
        <div className="mt-2 overflow-x-auto"><table className="w-full min-w-[420px] text-left">
          <thead><tr className="border-b border-border text-muted"><th className="py-2">股票</th><th>1日预测</th><th>3日预测</th><th>5日预测</th></tr></thead>
          <tbody>{data.snapshot.rows.map(row => <tr key={row.symbol} className="border-b border-border/50">
            <td className="py-2">{row.name || '名称待补'} {row.symbol}</td><td>{fmtPct(row.pred_ret_1d)}</td><td>{fmtPct(row.pred_ret_3d)}</td><td>{fmtPct(row.pred_ret_5d)}</td>
          </tr>)}</tbody>
        </table></div>
      </details>}
    </>}
  </section>
}
