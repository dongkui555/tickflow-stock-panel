import { useRef, useMemo, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Loader2, CheckCircle2, XCircle, AlertTriangle } from 'lucide-react'
import { formatDuration, formatLogTime } from '@/lib/format'
import { Pill } from './StatCard'
import type { PipelineJob } from '@/lib/api'

export const STAGE_LABELS: Record<string, string> = {
  init: '初始化',
  integrity_check: '日K数据自检',
  resolve_universe: '解析标的池',
  sync_instruments: '同步个股维表',
  sync_daily: '同步日 K',
  sync_index: '同步指数 / ETF',
  index_daily: '指数日K增量补缺',
  sync_adj: '同步除权因子',
  compute_enriched: '计算技术指标',
  sync_core_history: '数据增量检查',
  sync_financials: '同步财务历史',
  sync_moneyflow: '同步主力资金流',
  sync_auction: '同步集合竞价',
  sync_news: '同步国内外新闻',
  sync_minute: '同步分钟 K',
  queue_minute: '分钟补差排队',
  minute: '分钟增量补差',
  adjustment: '复权历史重算',
  rebuild_adjustment: '复权历史重算',
  enriched_dates: '历史指标补算',
  regime: '历史市场环境重算',
  compute_regime: '计算市场环境',
  extend_history: '扩展日K历史',
  extend_minute: '扩展分钟K历史',
  rebuild_enriched: '全量计算',
  refresh_views: '刷新视图',
  done: '完成',
}

function LogViewer({ log }: { log: PipelineJob['log'] }) {
  const containerRef = useRef<HTMLDivElement>(null)

  const displayLog = useMemo(() => {
    if (!log.length) return []
    return log.filter((line, i) => {
      const next = log[i + 1]
      if (
        next &&
        line.stage === next.stage &&
        /\d+\/\d+/.test(line.msg) &&
        /\d+\/\d+/.test(next.msg)
      ) {
        return false
      }
      return true
    })
  }, [log])

  useEffect(() => {
    const el = containerRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [displayLog])

  return (
    <div ref={containerRef} className="rounded-btn bg-base/60 border border-border max-h-48 overflow-y-auto px-3 py-2 font-mono text-[11px] space-y-0.5">
      {displayLog.map((line, i) => (
        <div key={`${line.ts}-${i}`} className="flex flex-wrap items-baseline gap-x-2 text-secondary">
          <span className="text-muted shrink-0">{formatLogTime(line.ts)}</span>
          <span className="text-accent/70 shrink-0">[{STAGE_LABELS[line.stage] ?? line.stage}]</span>
          <span className="basis-full min-w-0 break-words md:basis-auto md:flex-1">{line.msg}</span>
        </div>
      ))}
      {displayLog.length === 0 && <div className="text-muted">等待启动…</div>}
    </div>
  )
}

export function ActiveJobCard({ job }: { job: PipelineJob }) {
  const statusMap = {
    running:   { icon: Loader2,     color: 'text-accent',   label: '运行中', spinning: true,  border: 'border-accent/40', bg: 'bg-accent/5' },
    pending:   { icon: Loader2,     color: 'text-muted',    label: '排队中', spinning: true,  border: 'border-border',    bg: 'bg-surface' },
    succeeded: { icon: CheckCircle2, color: 'text-bear',     label: '完成',   spinning: false, border: 'border-bear/30',   bg: 'bg-bear/5' },
    failed:    { icon: XCircle,     color: 'text-danger',   label: '失败',   spinning: false, border: 'border-danger/40', bg: 'bg-danger/5' },
  } as const
  const partialIndex = job.status === 'succeeded' && job.result?.index_sync?.complete === false
  const meta = partialIndex
    ? { icon: AlertTriangle, color: 'text-warning', label: '部分完成', spinning: false, border: 'border-warning/40', bg: 'bg-warning/5' }
    : statusMap[job.status]
  const Icon = meta.icon
  const isDone = job.status === 'succeeded' || job.status === 'failed'
  const stageLabel = isDone ? meta.label : (STAGE_LABELS[job.stage] ?? job.stage)

  return (
    <div className={`rounded-card border ${meta.border} ${meta.bg} p-5`}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2.5">
          <Icon className={`h-5 w-5 ${meta.color} ${meta.spinning ? 'animate-spin' : ''}`} />
          <div>
            <div className="text-sm font-medium text-foreground">
              {meta.label}{!isDone && ` · ${stageLabel}`}
            </div>
            <div className="text-xs text-secondary font-mono mt-0.5">
              {job.id} · {job.duration_s != null ? formatDuration(job.duration_s) : '进行中'}
            </div>
          </div>
        </div>
        {!isDone && (
          <div className="font-mono text-2xl font-bold tracking-tight">
            {job.progress}<span className="text-[16px] leading-6 text-muted">%</span>
          </div>
        )}
      </div>

      {!isDone && (
        <div className="mb-2">
          <div className="h-1.5 rounded-full bg-elevated overflow-hidden">
            <motion.div
              className="h-full bg-accent"
              animate={{ width: `${job.progress}%` }}
              transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
            />
          </div>
          {job.stage_pct > 0 && (
            <div className="flex items-center justify-between mt-1">
              <span className="text-[10px] text-muted">当前阶段</span>
              <span className="text-[10px] font-mono text-secondary">{job.stage_pct}%</span>
            </div>
          )}
        </div>
      )}

      <LogViewer log={job.log} />

      {job.result?.index_sync && (
        <div className={`mt-3 text-xs ${partialIndex ? 'text-warning' : 'text-secondary'}`} role="status">
          指数日K {job.result.index_sync.start_date} 至 {job.result.index_sync.target_date}：已完成 {job.result.index_sync.completed_count}/{job.result.index_sync.total_count} 只，
          未完成 {job.result.index_sync.remaining_count} 只（{job.result.index_sync.remaining_stock_days} 个指数日），
          本轮写入 {job.result.index_sync.written_rows} 行。
          {partialIndex && '剩余缺口由增量队列继续处理。'}
        </div>
      )}

      {job.status === 'succeeded' && job.result
        && (job.result.universe_size != null || !job.result.index_sync) && (() => {
        const skipped = new Set(job.result.skipped_stages ?? [])
        const cell = (stage: string | null, v: string) =>
          stage && skipped.has(stage) ? '跳过' : v
        // 按结果里实际存在的字段渲染: 独立除权因子同步没有日K/分钟K字段, 不显示 "0 天"
        const pills: { label: string; value: string }[] = []
        if (job.result.universe_size != null) pills.push({ label: '标的池', value: String(job.result.universe_size) })
        if (job.result.daily_days != null) pills.push({ label: '日 K', value: cell(null, `${job.result.daily_days} 天`) })
        if (job.result.adj_factor_symbols != null) pills.push({ label: '除权因子', value: cell('sync_adj', `${job.result.adj_factor_symbols} 只`) })
        if (job.result.enriched_days != null) pills.push({ label: 'enriched', value: cell(null, `${job.result.enriched_days} 行`) })
        if (job.result.remaining_gaps != null) pills.push({ label: '剩余缺口', value: `${job.result.remaining_gaps} 条` })
        if (job.result.minute_rows != null) pills.push({ label: job.result.requested_gaps != null ? '新增分钟K' : '分钟K', value: cell('sync_minute', `${job.result.minute_rows} 行`) })
        if (job.result.financial_rows) pills.push({ label: '财务历史', value: `${Object.values(job.result.financial_rows).reduce((sum, value) => sum + value, 0).toLocaleString()} 行` })
        if (job.result.moneyflow_rows != null) pills.push({ label: '历史资金', value: `${job.result.moneyflow_rows.toLocaleString()} 行` })
        if (job.result.auction_rows != null) pills.push({ label: '集合竞价', value: `${job.result.auction_rows.toLocaleString()} 行` })
        if (job.result.news_rows) pills.push({ label: '新闻', value: `${((job.result.news_rows.domestic ?? 0) + (job.result.news_rows.overseas ?? 0)).toLocaleString()} 行` })
        if (job.result.minute_task_id) pills.push({ label: '分钟任务', value: '后台队列处理' })
        if (job.result.history_rebuild_queued) pills.push({ label: '历史指标', value: '后台队列处理' })
        if (pills.length === 0) return null
        return (
          <div className="mt-3 grid grid-cols-2 md:grid-cols-5 gap-3 text-xs">
            {pills.map(p => <Pill key={p.label} {...p} />)}
          </div>
        )
      })()}
      {job.status === 'succeeded' && (job.result?.unavailable_datasets?.length ?? 0) > 0 && (
        <div className="mt-3 rounded-btn border border-warning/30 bg-warning/5 px-3 py-2 text-[11px] text-warning/90 space-y-1">
          {job.result!.unavailable_datasets!.map((item) => (
            <div key={item.dataset}>{item.dataset}：{item.reason}</div>
          ))}
        </div>
      )}
      {job.status === 'succeeded' && (job.result?.news_unavailable?.length ?? 0) > 0 && (
        <div className="mt-3 rounded-btn border border-warning/30 bg-warning/5 px-3 py-2 text-[11px] text-warning/90 space-y-1">
          {job.result!.news_unavailable!.map((item) => (
            <div key={item.dataset}>{item.dataset}：{item.reason}</div>
          ))}
        </div>
      )}
      {job.status === 'failed' && job.error && (
        <div className="mt-3 rounded-btn border border-danger/40 bg-danger/5 px-3 py-2 text-xs text-danger">
          {job.error}
          {job.error.includes('超时自动取消') && (
            <div className="mt-1 text-[10px] text-muted">
              判定依据是「无进度」而非总时长, 任务只要仍在推进就不会被中断;
              若网络环境较慢可在 设置 → 网络设置 中调大停滞阈值。
            </div>
          )}
        </div>
      )}
    </div>
  )
}
