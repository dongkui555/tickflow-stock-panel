import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { CheckCircle2, Clock, RefreshCw, ShieldAlert } from 'lucide-react'
import { api } from '@/lib/api'
import { QK } from '@/lib/queryKeys'

const STATUS = { pending: '排队中', running: '执行中', retry: '等待重试', succeeded: '已完成', paused_review: '待核验（已暂停）', paused_authorization: '待授权替换（已暂停）', reviewed: '核验已处理' }
const CATEGORY_ORDER = [
  ['pending_gap', '真实待补', 'text-accent'],
  ['source_incomplete', '分钟不足', 'text-warning'],
  ['source_empty', '源无记录', 'text-secondary'],
  ['source_unavailable', '源不可用/退避', 'text-accent'],
  ['conflict', '量价冲突', 'text-danger'],
  ['verified_candidate', '已验证待入库', 'text-accent'],
  ['not_requested', '尚未请求', 'text-secondary'],
  ['review', '待核验/隔离', 'text-warning'],
  ['complete', '已完成/终态', 'text-secondary'],
] as const

export function MaintenanceQueues({ onJobClick }: { onJobClick: (id: string) => void }) {
  const queryClient = useQueryClient()
  const query = useQuery({
    queryKey: QK.maintenanceQueue,
    queryFn: api.maintenanceQueue,
    refetchInterval: 5_000,
  })
  const authorizationQuery = useQuery({
    queryKey: ['minute-repair-authorizations'],
    queryFn: () => api.maintenanceAuthorizations(),
    refetchInterval: 5_000,
  })
  const approve = useMutation({
    mutationFn: ({ taskId, id }: { taskId: string; id: string }) =>
      api.approveMaintenanceAuthorizations(taskId, [id]),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: QK.maintenanceQueue })
      void queryClient.invalidateQueries({ queryKey: ['minute-repair-authorizations'] })
    },
  })
  return (
    <section className="border-t border-border pt-4 space-y-3">
      <div className="flex items-center justify-between gap-2">
        <h3 className="text-sm font-medium flex items-center gap-2"><Clock size={15} />增量与历史任务</h3>
        <button type="button" aria-label="刷新任务队列" title="刷新任务队列" onClick={() => query.refetch()}
          className="p-2 text-secondary hover:text-primary" disabled={query.isFetching}>
          <RefreshCw size={14} className={query.isFetching ? 'animate-spin' : ''} />
        </button>
      </div>
      {query.isError && <div className="text-sm text-danger" role="alert">任务队列读取失败</div>}
      {query.data?.minute_gap_queue && (
        <div className="border-b border-border pb-3 space-y-2" role="status">
          <div className="text-xs font-medium">股票日级分钟补差队列</div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-[11px]">
            {CATEGORY_ORDER.map(([key, label, tone]) => {
              const categories = query.data?.minute_gap_queue?.categories
              const item = categories?.[key]
              return <div key={key} className="text-secondary">
                <span className={tone}>
                  {label} {item?.count ?? 0} 个股票日
                </span>
                {item?.first_date && <span className="block text-muted">{item.first_date} 至 {item.last_date}</span>}
              </div>
            })}
          </div>
        </div>
      )}
      {!!authorizationQuery.data?.pending?.length && <div className="border border-warning/40 bg-warning/5 p-3 space-y-2" role="status">
        <div className="flex items-center gap-2 text-sm text-warning">
          <ShieldAlert size={15} />待授权替换 {authorizationQuery.data.pending.length} 项
        </div>
        <p className="text-xs text-secondary">只列出新出现的量价冲突；新增记录、重复记录、通过校验的 239/240 根分钟不会进入这里。身份未知项仍在核验队列，不会被授权替换。</p>
        <div className="space-y-1">
          {authorizationQuery.data.pending.slice(0, 30).map(item => <div key={item.id} className="flex flex-wrap items-center justify-between gap-2 text-xs">
            <span>{item.qualified_symbol || item.symbol} · {item.trade_date} · {item.source}</span>
            <button type="button" className="inline-flex items-center gap-1 text-accent hover:underline disabled:opacity-50"
              disabled={approve.isPending} onClick={() => approve.mutate({ taskId: item.task_id, id: item.id })}>
              <CheckCircle2 size={13} />批准本项替换
            </button>
          </div>)}
        </div>
        {authorizationQuery.data.pending.length > 30 && <p className="text-xs text-secondary">其余 {authorizationQuery.data.pending.length - 30} 项可通过接口按任务范围批量审批。</p>}
        {approve.isError && <p className="text-xs text-danger">授权失败，请刷新后重试</p>}
      </div>}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {(['minute', 'history'] as const).map(queue => {
          const rows = (query.data?.tasks ?? []).filter(t => t.queue === queue)
            .sort((a, b) => Number(a.status === 'succeeded') - Number(b.status === 'succeeded')
              || b.updated_at.localeCompare(a.updated_at))
          const pendingCount = rows.filter(t => t.status === 'pending').length
          const runningCount = rows.filter(t => t.status === 'running').length
          const retryCount = rows.filter(t => t.status === 'retry').length
          const reviewCount = rows.filter(t => t.status === 'paused_review').length
          return (
            <div key={queue} className="min-w-0 space-y-2">
              <h4 className="text-xs text-secondary">
                {queue === 'minute' ? '分钟增量补差' : '历史补齐与重算'} ·
                待执行 {pendingCount} · 执行中 {runningCount} · 待重试 {retryCount} · 待核验 {reviewCount}
              </h4>
              {query.isLoading ? <p className="text-xs text-secondary">加载中</p> : !rows.length &&
                <p className="text-xs text-secondary">暂无排队任务</p>}
              <div className="max-h-80 overflow-y-auto space-y-3">
                {rows.map(t => (
                  <div key={t.id} className="border-b border-border pb-3 text-xs space-y-2">
                    {(() => {
                      const noProgress = t.kind === 'minute' && /(?:added 0 rows|新增 0 行|repaired 0 stock-days)/i.test(t.message ?? '')
                      const statusLabel = noProgress ? '本轮无有效修复' : (STATUS[t.status] ?? t.status)
                      const gateChecked = t.gate_completed ?? t.completed
                      const gateTotal = t.gate_total_items ?? t.total
                      return <>
                    {t.kind === 'minute' && t.gate_date && <div className="text-accent">
                      日期闸门 {t.gate_position ?? 1}/{t.gate_total_dates ?? 1}：{t.gate_date} · 当日剩余 {t.gate_remaining ?? t.remaining ?? 0} 只
                    </div>}
                    <div className="flex flex-wrap justify-between gap-2">
                      <span>{t.label}</span><span className={noProgress ? 'text-danger' : 'text-secondary'}>{statusLabel}</span>
                    </div>
                    <div className="flex flex-wrap justify-between gap-2 text-secondary">
                      {t.kind === 'minute' && t.gate_date
                        ? <span>窗口检查 {gateChecked}/{gateTotal} 只，当前日待执行 {t.gate_remaining ?? t.remaining ?? 0} 只</span>
                        : <span>完成 {t.completed}/{t.total} {['minute', 'adjustment', 'index_daily'].includes(t.kind) ? '只' : '日'}，待执行 {t.remaining ?? Math.max(0, t.total - t.completed)}</span>}
                      {t.status !== 'paused_review' && t.status !== 'reviewed' && <span>本批 {t.batch_progress}%</span>}
                    </div>
                    {!!t.review_count && <p className="text-amber-500">待核验 {t.review_count} 日（不自动重算）</p>}
                    {!!t.skipped && <p className="text-secondary">终态跳过 {t.skipped} 项</p>}
                    {!!t.unknown && <p className="text-warning">未知覆盖 {t.unknown} 项：不伪造分钟数据</p>}
                    {!!t.authorization_count && <p className="text-warning">待授权替换 {t.authorization_count} 项：仅新冲突需要确认</p>}
                    {!!t.identity_pending_count && <p className="text-warning">身份待核验 {t.identity_pending_count} 项：不进入替换授权</p>}
                    {!!t.resolved && <p className="text-secondary">已由独立修复任务处理 {t.resolved} 日</p>}
                    {noProgress && <p className="text-danger">本批已完成来源检查但没有可验证新增，不能计入修复完成；请等待退避或转第三方核验。</p>}
                    {t.kind === 'minute' && t.repair_summary && <p className="text-secondary">
                      本批新增 {t.repair_summary.inserted_rows ?? 0} 行，更新 {t.repair_summary.updated_rows ?? 0} 行，
                      重复 {t.repair_summary.duplicate_rows ?? 0} 行，拒绝 {t.repair_summary.rejected_rows ?? 0} 行
                    </p>}
                    <div role="progressbar" aria-label={`${t.label}总体进度`} aria-valuenow={t.kind === 'minute' && t.gate_date ? gateChecked : t.completed}
                      aria-valuemin={0} aria-valuemax={Math.max(1, t.total)} className="h-1.5 w-full bg-elevated overflow-hidden">
                      <div className="h-full bg-accent" style={{ width: `${t.kind === 'minute' && t.gate_total_dates && t.gate_total_items
                        ? (t.status === 'succeeded' ? 100 : 100 * ((t.gate_position! - 1) + (t.gate_completed ?? 0) / t.gate_total_items) / t.gate_total_dates)
                        : 100 * t.completed / Math.max(1, t.total)}%` }} />
                    </div>
                    <p className="text-secondary break-words">{t.message}</p>
                    {t.error && <p className="text-danger break-words">{t.error}</p>}
                    <div className="flex flex-wrap gap-x-3 gap-y-1 text-secondary">
                      {t.status === 'retry' && t.retry_at && <span>重试时间 {new Date(t.retry_at).toLocaleString('zh-CN')}</span>}
                    {t.job_id && <button type="button" className="text-blue-500 hover:underline" onClick={() => onJobClick(t.job_id!)}>任务日志 {t.job_id}</button>}
                    </div>
                      </>
                    })()}
                  </div>
                ))}
              </div>
            </div>
          )
        })}
      </div>
    </section>
  )
}
