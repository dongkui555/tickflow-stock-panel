import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import {
  Activity,
  AlertTriangle,
  BrainCircuit,
  DatabaseZap,
  Layers3,
  Gauge,
  Loader2,
  RefreshCw,
  ScanSearch,
  ShieldAlert,
  Sparkles,
  Target,
  TrendingUp,
} from 'lucide-react'
import { PageHeader } from '@/components/PageHeader'
import { PredictionWorkflowPanel } from '@/components/PredictionWorkflowPanel'
import { SectorResearchPanel } from '@/components/SectorResearchPanel'
import { GlobalRiskPanel } from '@/components/GlobalRiskPanel'
import { StockPreviewDialog } from '@/components/StockPreviewDialog'
import { StockQuickActions } from '@/components/StockQuickActions'
import { StockIdentity } from '@/lib/stockDisplay'
import { api, type MarketGatePayload, type SectorPredictionPayload, type SectorPredictionRow, type SectorPredictionSnapshot, type SectorRotationPayload, type ShortTermAIAgentReviewResponse, type ShortTermAIProfile, type ShortTermAIRecommendationRow, type StockFormerDataCoverageEntry } from '@/lib/api'
import { QK } from '@/lib/queryKeys'
import { cn } from '@/lib/cn'
import { fmtBigNum, fmtPct, fmtPrice, priceColorClass } from '@/lib/format'

const panelCls = 'rounded-card border border-border bg-surface/80 p-4 shadow-[0_1px_2px_hsl(var(--border)/0.35)]'

const profiles: Array<{ value: ShortTermAIProfile; label: string }> = [
  { value: 'balanced', label: '均衡' },
  { value: 'aggressive', label: '进攻' },
  { value: 'conservative', label: '保守' },
]

function fmtScore(v: number | null | undefined) {
  if (v == null || Number.isNaN(v)) return '—'
  return v.toFixed(1)
}

function fmtPoint(v: number | null | undefined) {
  if (v == null || Number.isNaN(v)) return '—'
  return `${v > 0 ? '+' : ''}${v.toFixed(1)}分`
}

function normalizedScore(v: number | null | undefined) {
  if (v == null || Number.isNaN(v)) return null
  return v <= 1 ? v * 100 : v
}

function pct(v: number | null | undefined, digits = 2) {
  return fmtPct(v, digits)
}

function renderScalar(value: unknown) {
  if (value == null || value === '') return '—'
  if (typeof value === 'object') return '—'
  return String(value)
}

function scoreTone(v: number | null | undefined) {
  if (v == null || Number.isNaN(v)) return 'text-muted'
  if (v >= 75) return 'text-bull'
  if (v >= 60) return 'text-warning'
  return 'text-bear'
}

function decisionCls(value: string) {
  if (value === '强关注') return 'border-danger/25 bg-danger/10 text-danger'
  if (value === '候选') return 'border-warning/25 bg-warning/10 text-warning'
  if (value === '回避') return 'border-bear/25 bg-bear/10 text-bear'
  return 'border-border bg-elevated text-secondary'
}

function signalCls(value?: string | null) {
  const text = String(value ?? '')
  if (text.includes('吸筹') || text.includes('建仓') || text.includes('流入') || text.includes('突破')) return 'border-danger/25 bg-danger/10 text-danger'
  if (text.includes('派发') || text.includes('流出')) return 'border-bear/25 bg-bear/10 text-bear'
  if (text.includes('暂无') || text.includes('不足') || text.includes('缺失') || text.includes('异常')) return 'border-warning/25 bg-warning/10 text-warning'
  return 'border-border bg-elevated text-secondary'
}

function ratingCls(value?: string | null) {
  const text = String(value ?? '')
  if (text.includes('优先')) return 'border-danger/25 bg-danger/10 text-danger'
  if (text.includes('放弃') || text.includes('回避')) return 'border-bear/25 bg-bear/10 text-bear'
  return 'border-warning/25 bg-warning/10 text-warning'
}

function marketGateCls(value?: string | null) {
  const text = String(value ?? '')
  if (text === 'open') return 'border-bull/25 bg-bull/10 text-bull'
  if (text === 'strict' || text === 'neutral') return 'border-warning/25 bg-warning/10 text-warning'
  if (text === 'observe' || text === 'closed') return 'border-bear/25 bg-bear/10 text-bear'
  return 'border-border bg-elevated text-secondary'
}

function confirmationCls(value?: string | null) {
  const text = String(value ?? '')
  if (text === 'confirmed') return 'border-bull/25 bg-bull/10 text-bull'
  if (text === 'blocked') return 'border-bear/25 bg-bear/10 text-bear'
  return 'border-warning/25 bg-warning/10 text-warning'
}

function sectorSignalCls(value?: string | null) {
  const text = String(value ?? '')
  if (text === 'pre_layout') return 'border-danger/25 bg-danger/10 text-danger'
  if (text === 'early_layout') return 'border-danger/25 bg-danger/10 text-danger'
  if (text === 'accelerating') return 'border-warning/25 bg-warning/10 text-warning'
  if (text === 'launch_confirm') return 'border-warning/25 bg-warning/10 text-warning'
  if (text === 'overheated' || text === 'fading' || text === 'layout_diverge') return 'border-bear/25 bg-bear/10 text-bear'
  if (text === 'watch_layout' || text === 'early_watch') return 'border-accent/25 bg-accent/10 text-accent'
  return 'border-border bg-elevated text-secondary'
}

function predictionLevelCls(value?: string | null) {
  const text = String(value ?? '')
  if (text === 'strong') return 'border-danger/25 bg-danger/10 text-danger'
  if (text === 'watch') return 'border-warning/25 bg-warning/10 text-warning'
  if (text === 'weak_watch') return 'border-accent/25 bg-accent/10 text-accent'
  return 'border-border bg-elevated text-secondary'
}

function validationCls(value?: string | null) {
  const text = String(value ?? '')
  if (text === 'hit') return 'border-bull/25 bg-bull/10 text-bull'
  if (text === 'miss' || text === 'missing') return 'border-bear/25 bg-bear/10 text-bear'
  return 'border-warning/25 bg-warning/10 text-warning'
}

function verdictCls(value?: string | null) {
  if (value === '强预测命中') return 'border-bull/25 bg-bull/10 text-bull'
  if (value === '观察命中') return 'border-warning/25 bg-warning/10 text-warning'
  if (value === '漏判') return 'border-bear/25 bg-bear/10 text-bear'
  return 'border-border bg-elevated text-muted'
}

function Metric({ icon, label, value, tone }: { icon: ReactNode; label: string; value: ReactNode; tone?: string }) {
  return (
    <div className="rounded-card border border-border/70 bg-base/40 px-3 py-2">
      <div className="flex items-center gap-1.5 text-[10px] text-muted">
        {icon}
        <span>{label}</span>
      </div>
      <div className={cn('mt-1 min-h-5 truncate text-sm font-semibold text-foreground', tone)}>{value}</div>
    </div>
  )
}

function SectionTitle({ icon: Icon, title, hint }: { icon: typeof Target; title: string; hint?: ReactNode }) {
  return (
    <div className="mb-3 flex items-center gap-2">
      <Icon className="h-4 w-4 text-accent" />
      <h2 className="text-sm font-semibold text-foreground">{title}</h2>
      {hint && <span className="ml-auto text-[10px] text-muted">{hint}</span>}
    </div>
  )
}

function ScoreBar({ value, parts }: { value: number; parts?: Array<{ label: string; value: number | null | undefined }> }) {
  const width = Math.max(0, Math.min(100, value))
  return (
    <div className="min-w-[8rem]">
      <div className="flex items-center justify-between gap-2">
        <span className={cn('font-semibold', scoreTone(value))}>{fmtScore(value)}</span>
        <span className="text-[10px] text-muted">/100</span>
      </div>
      <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-elevated">
        <div className="h-full rounded-full bg-accent" style={{ width: `${width}%` }} />
      </div>
      {parts && (
        <div className="mt-1 grid grid-cols-2 gap-x-2 gap-y-0.5 text-[9px] text-muted">
          {parts.map(part => <span key={part.label}>{part.label}{fmtScore(part.value)}</span>)}
        </div>
      )}
    </div>
  )
}

function CoverageWarning({ coverage }: { coverage?: Record<string, StockFormerDataCoverageEntry> }) {
  const missing = Object.values(coverage ?? {}).filter(item => !item.available)
  if (!missing.length) return null
  return (
    <div className="rounded-card border border-warning/20 bg-warning/10 px-3 py-2 text-xs text-warning">
      增强数据未完整覆盖：{missing.map(item => item.label).join('、')} 按中性值或降权处理。
    </div>
  )
}

function KlineQualityWarning({ meta }: { meta?: Record<string, any> }) {
  if (!meta?.warning && !meta?.expected_days) return null
  return (
    <div className="rounded-card border border-warning/20 bg-warning/10 px-3 py-2 text-xs text-warning">
      日K质量检查：近窗交易日 {meta?.expected_days ?? '—'} 天，覆盖股票 {meta?.symbols ?? '—'} 只；{meta?.warning || '覆盖不足或复权异常会自动降级为回避。'}
    </div>
  )
}

function SectorRotationPanel({ payload }: { payload?: SectorRotationPayload }) {
  const rows = (payload?.early_layout?.length ? payload.early_layout : payload?.rows ?? []).slice(0, 8)
  const meta = payload?.meta ?? {}
  if (!payload) return null
  return (
    <section className={panelCls}>
      <SectionTitle
        icon={Layers3}
        title="当日板块状态（描述性指标）"
        hint={`${payload.as_of ?? '—'} · ${meta.regime ?? '轮动识别'}`}
      />
      <div className="mb-3 grid gap-2 md:grid-cols-4">
        <Metric icon={<Layers3 className="h-3.5 w-3.5" />} label="轮动状态" value={meta.regime ?? '—'} />
        <Metric icon={<Activity className="h-3.5 w-3.5" />} label="Top5换手" value={pct(meta.top5_turnover_avg_20d, 1)} />
        <Metric icon={<TrendingUp className="h-3.5 w-3.5" />} label="行业分化" value={pct(meta.industry_spread_avg_20d, 2)} />
        <Metric icon={<DatabaseZap className="h-3.5 w-3.5" />} label="资金日期" value={meta.flow_latest_date ?? '—'} />
      </div>
      <div className="overflow-auto rounded-card border border-border/70">
        <table className="w-full min-w-[1320px] text-xs">
          <thead className="bg-elevated text-[10px] text-muted">
            <tr>
              <th className="px-3 py-2 text-right font-medium">排名</th>
              <th className="px-3 py-2 text-left font-medium">板块</th>
              <th className="px-3 py-2 text-left font-medium">阶段</th>
              <th className="px-3 py-2 text-right font-medium">总分</th>
              <th className="px-3 py-2 text-right font-medium">旧版观察分</th>
              <th className="px-3 py-2 text-right font-medium">今日确认</th>
              <th className="px-3 py-2 text-right font-medium">昨日前资金</th>
              <th className="px-3 py-2 text-right font-medium">5日资金</th>
              <th className="px-3 py-2 text-right font-medium">昨日前涨幅</th>
              <th className="px-3 py-2 text-right font-medium">5日涨幅</th>
              <th className="px-3 py-2 text-right font-medium">宽度</th>
              <th className="px-3 py-2 text-right font-medium">量能</th>
              <th className="px-3 py-2 text-left font-medium">状态证据</th>
            </tr>
          </thead>
          <tbody>
            {!rows.length ? (
              <tr><td colSpan={13} className="px-3 py-8 text-center text-muted">暂无板块轮动数据</td></tr>
            ) : rows.map(row => (
              <tr key={row.sector} className="border-t border-border/60">
                <td className="px-3 py-3 text-right font-mono text-muted">{row.rank}</td>
                <td className="px-3 py-3 font-medium text-foreground">{row.sector}</td>
                <td className="px-3 py-3">
                  <span className={cn('inline-flex rounded-full border px-2 py-0.5 text-[10px] font-medium', sectorSignalCls(row.signal))}>
                    {row.phase}
                  </span>
                </td>
                <td className={cn('px-3 py-3 text-right font-mono font-semibold', scoreTone(row.score))}>{fmtScore(row.score)}</td>
                <td className={cn('px-3 py-3 text-right font-mono font-semibold', scoreTone(row.setup_score ?? row.pre_layout_score))}>{fmtScore(row.setup_score ?? row.pre_layout_score)}</td>
                <td className={cn('px-3 py-3 text-right font-mono font-semibold', scoreTone(row.today_confirm_score))}>{fmtScore(row.today_confirm_score)}</td>
                <td className="px-3 py-3 text-right font-mono">{fmtBigNum(row.pre_net_5d)}</td>
                <td className="px-3 py-3 text-right font-mono">{fmtBigNum(row.net_5d)}</td>
                <td className={cn('px-3 py-3 text-right font-mono', priceColorClass(row.pre_ret_5d))}>{pct(row.pre_ret_5d)}</td>
                <td className={cn('px-3 py-3 text-right font-mono', priceColorClass(row.ret_5d))}>{pct(row.ret_5d)}</td>
                <td className="px-3 py-3 text-right font-mono">{pct(row.up_ratio, 1)}</td>
                <td className="px-3 py-3 text-right font-mono">{fmtScore(row.amount_ratio_5d)}x</td>
                <td className="px-3 py-3 text-[10px] leading-4 text-secondary">
                  {(row.reasons ?? []).slice(0, 3).join('；') || '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-3 rounded-card border border-accent/15 bg-accent/5 px-3 py-2 text-xs text-secondary">
        {meta.method ?? '优先寻找资金开始流入、量能放大、宽度扩散且短期涨幅未过热的板块。'}
      </div>
    </section>
  )
}

function hitText(hit?: number, total?: number) {
  if (!total) return '待验证'
  return `${hit ?? 0}/${total}`
}

function SectorPredictionRecordsPanel({
  payload,
  loading,
  refreshing,
  onRefresh,
  origin, onOrigin, generation, error,
}: {
  payload?: SectorPredictionPayload
  loading: boolean
  refreshing: boolean
  onRefresh: () => void
  origin: 'published' | 'replay' | 'legacy'
  onOrigin: (value: 'published' | 'replay' | 'legacy') => void
  generation?: SectorPredictionPayload['generation']
  error?: string
}) {
  const [horizon, setHorizon] = useState('1')
  const snapshots = payload?.snapshots ?? []
  const latest = snapshots[0]
  const latestRows = (latest?.rows ?? []).slice(0, 10)
  const verified = snapshots.filter(item => item.validation_summary?.verify_1d_date)
  const latestVerified = verified[0]
  const latestVerifiedSummary = latestVerified?.validation_summary ?? {}
  const actualTop10 = latestVerifiedSummary.actual_top10_1d ?? []
  const missedLatest = latestVerifiedSummary.missed_actual_top10_1d ?? []
  const aggregate = payload?.aggregate ?? {}
  const horizonStats = aggregate.horizons?.[horizon] ?? {}
  return (
    <section className={panelCls}>
      <SectionTitle
        icon={Target}
        title="板块启动提前预测记录"
        hint={`${payload?.model_version ?? 'sector_confluence_v2'} · ${origin === 'published' ? '正式留痕' : '回放成绩'}`}
      />
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <select aria-label="板块预测记录类型" value={origin} onChange={e => onOrigin(e.target.value as typeof origin)} className="h-8 rounded border border-border bg-base px-2 text-xs">
          <option value="published">正式预测</option><option value="replay">新版历史回放</option><option value="legacy">旧版回放存档</option>
        </select>
        <select aria-label="板块验证期限" value={horizon} onChange={e => setHorizon(e.target.value)} className="h-8 rounded border border-border bg-base px-2 text-xs">
          <option value="1">T+1 验证</option><option value="2">T+2 验证</option>
        </select>
        <button
          type="button"
          onClick={onRefresh}
          disabled={refreshing || origin === 'legacy'}
          className="inline-flex h-8 items-center gap-1.5 rounded-btn bg-accent px-3 text-xs font-medium text-white shadow-sm hover:bg-accent/90 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {refreshing ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />}
          生成预测 / 追加验证
        </button>
        <span className="text-[11px] text-muted">{origin === 'published' ? '不可覆盖快照 · 盘后数据门槛98%' : '历史回放 · 不计入正式成绩'}</span>
      </div>
      {error && <div role="alert" className="mb-3 text-xs text-bear">{error}</div>}
      {generation?.summary?.warning && <div role="status" className="mb-3 text-xs text-warning">{generation.as_of} · {generation.summary.warning}</div>}
      {latest && <div className="mb-3 text-xs text-muted">生成：{latest.generated_at} · 数据截止：{latest.data_cutoff ?? '旧记录未留存'}</div>}
      <div className="mb-3 grid gap-2 md:grid-cols-5">
        <Metric icon={<Target className="h-3.5 w-3.5" />} label="记录天数" value={aggregate.snapshots ?? snapshots.length} />
        <Metric icon={<Sparkles className="h-3.5 w-3.5" />} label={`T+${horizon}强预测`} value={hitText(horizonStats.strong_hits, horizonStats.strong_verified)} />
        <Metric icon={<Layers3 className="h-3.5 w-3.5" />} label={`T+${horizon} Top5`} value={hitText(horizonStats.top5_hits, horizonStats.top5_verified)} />
        <Metric icon={<Activity className="h-3.5 w-3.5" />} label="尚未启动 → 启动" value={hitText(horizonStats.start_hits, horizonStats.start_verified)} />
        <Metric icon={<TrendingUp className="h-3.5 w-3.5" />} label="已启动 → 延续" value={hitText(horizonStats.continuation_hits, horizonStats.continuation_verified)} />
      </div>
      {latestVerified && (
        <div className="mb-3 rounded-card border border-border/70 bg-base/40 p-3">
          <div className="mb-2 flex flex-wrap items-center gap-2">
            <span className="text-xs font-semibold text-foreground">昨日预测 / 今日验证</span>
            <span className="text-[10px] text-muted">
              {latestVerified.as_of} 预测 · {latestVerifiedSummary.verify_1d_date ?? '待验证'} 验证
            </span>
          </div>
          <div className="grid gap-2 md:grid-cols-5">
            <Metric icon={<Sparkles className="h-3.5 w-3.5" />} label="强预测命中" value={`${latestVerifiedSummary.actual_top10_strong_hits ?? 0}/${latestVerifiedSummary.actual_top10_verified ?? 0}`} tone="text-bull" />
            <Metric icon={<Target className="h-3.5 w-3.5" />} label="观察命中" value={`${latestVerifiedSummary.actual_top10_watch_hits ?? 0}/${latestVerifiedSummary.actual_top10_verified ?? 0}`} tone="text-warning" />
            <Metric icon={<AlertTriangle className="h-3.5 w-3.5" />} label="漏判" value={`${latestVerifiedSummary.actual_top10_misses ?? missedLatest.length}/${latestVerifiedSummary.actual_top10_verified ?? 0}`} tone={(latestVerifiedSummary.actual_top10_misses ?? missedLatest.length) ? 'text-bear' : 'text-bull'} />
            <Metric icon={<TrendingUp className="h-3.5 w-3.5" />} label="预测Top10超额" value={pct(latestVerifiedSummary.top10_excess_1d, 2)} tone={priceColorClass(latestVerifiedSummary.top10_excess_1d)} />
            <Metric icon={<Activity className="h-3.5 w-3.5" />} label="强预测池" value={hitText(latestVerifiedSummary.strong_hits, latestVerifiedSummary.strong_verified)} />
          </div>
          <div className="mt-3 overflow-auto rounded-card border border-border/70">
            <table className="w-full min-w-[980px] text-xs">
              <thead className="bg-elevated text-[10px] text-muted">
                <tr>
                  <th className="px-3 py-2 text-right font-medium">实际排名</th>
                  <th className="px-3 py-2 text-left font-medium">今日强势板块</th>
                  <th className="px-3 py-2 text-left font-medium">昨日状态</th>
                  <th className="px-3 py-2 text-right font-medium">预测排名</th>
                  <th className="px-3 py-2 text-right font-medium">预测分</th>
                  <th className="px-3 py-2 text-right font-medium">今日涨幅</th>
                  <th className="px-3 py-2 text-right font-medium">扩散</th>
                  <th className="px-3 py-2 text-right font-medium">涨停数</th>
                </tr>
              </thead>
              <tbody>
                {actualTop10.slice(0, 10).map(item => (
                  <tr key={`${latestVerified.as_of}-${item.sector}`} className="border-t border-border/60">
                    <td className="px-3 py-2 text-right font-mono text-muted">{item.actual_rank ?? '—'}</td>
                    <td className="px-3 py-2 font-medium text-foreground">{item.sector}</td>
                    <td className="px-3 py-2">
                      <span className={cn('inline-flex rounded-full border px-2 py-0.5 text-[10px] font-medium', verdictCls(item.verdict))}>
                        {item.verdict}
                      </span>
                    </td>
                    <td className="px-3 py-2 text-right font-mono">{item.prediction_rank ?? '—'}</td>
                    <td className={cn('px-3 py-2 text-right font-mono', scoreTone(item.prediction_score ?? 0))}>{item.prediction_score ?? '—'}</td>
                    <td className={cn('px-3 py-2 text-right font-mono', priceColorClass(item.actual_ret_1d))}>{pct(item.actual_ret_1d)}</td>
                    <td className="px-3 py-2 text-right font-mono">{pct(item.actual_up_ratio, 1)}</td>
                    <td className="px-3 py-2 text-right font-mono">{item.actual_limit_up_count ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      {loading && !snapshots.length ? (
        <div className="rounded-card border border-border/70 bg-base/40 px-3 py-8 text-center text-xs text-muted">
          <Loader2 className="mx-auto mb-2 h-4 w-4 animate-spin" />
          正在读取预测记录
        </div>
      ) : !latest ? (
        <div className="rounded-card border border-warning/20 bg-warning/10 px-3 py-3 text-xs text-warning">
          暂无该类型预测记录。正式预测等待当日盘后数据通过校验，允许无强推荐。
        </div>
      ) : (
        <div className="space-y-3">
          <div className="overflow-auto rounded-card border border-border/70">
            <table className="w-full min-w-[1580px] text-xs">
              <thead className="bg-elevated text-[10px] text-muted">
                <tr>
                  <th className="px-3 py-2 text-right font-medium">预测排名</th>
                  <th className="px-3 py-2 text-left font-medium">板块</th>
                  <th className="px-3 py-2 text-left font-medium">预测</th>
                  <th className="px-3 py-2 text-right font-medium">启动优先</th>
                  <th className="px-3 py-2 text-right font-medium">总分</th>
                  <th className="px-3 py-2 text-right font-medium">1日</th>
                  <th className="px-3 py-2 text-right font-medium">2日</th>
                  <th className="px-3 py-2 text-left font-medium">验证</th>
                  <th className="px-3 py-2 text-right font-medium">实际排名</th>
                  <th className="px-3 py-2 text-right font-medium">实际涨幅</th>
                  <th className="px-3 py-2 text-right font-medium">扩散</th>
                  <th className="px-3 py-2 text-left font-medium">启动前证据</th>
                  <th className="px-3 py-2 text-left font-medium">风险</th>
                </tr>
              </thead>
              <tbody>
                {latestRows.map((row: SectorPredictionRow) => {
                  const best = row.validations?.find(v => v.horizon_days === Number(horizon) && ['hit', 'miss'].includes(v.status))
                  return (
                    <tr key={`${latest.as_of}-${row.sector}`} className="border-t border-border/60">
                      <td className="px-3 py-3 text-right font-mono text-muted">{row.prediction_rank}</td>
                      <td className="px-3 py-3">
                        <div className="font-medium text-foreground">{row.sector}</div>
                        <div className="mt-1 text-[10px] text-muted">原轮动#{row.source_rank ?? '—'} · {row.source_phase ?? '—'}</div>
                      </td>
                      <td className="px-3 py-3">
                        <span className={cn('inline-flex rounded-full border px-2 py-0.5 text-[10px] font-medium', predictionLevelCls(row.prediction_level))}>
                          {row.prediction_label}
                        </span>
                        <div className="mt-1 text-[10px] text-muted">{row.forecast_kind === 'continuation' ? '启动后延续' : '尚未启动'} · 规则分，非概率</div>
                      </td>
                      <td className={cn('px-3 py-3 text-right font-mono font-semibold', scoreTone(row.priority_score))}>{fmtScore(row.priority_score)}</td>
                      <td className={cn('px-3 py-3 text-right font-mono', scoreTone(row.prediction_score))}>{fmtScore(row.prediction_score)}</td>
                      <td className={cn('px-3 py-3 text-right font-mono', scoreTone(row.horizon_1_score))}>{fmtScore(row.horizon_1_score)}</td>
                      <td className={cn('px-3 py-3 text-right font-mono', scoreTone(row.horizon_2_score))}>{fmtScore(row.horizon_2_score)}</td>
                      <td className="px-3 py-3">
                        <span className={cn('inline-flex rounded-full border px-2 py-0.5 text-[10px] font-medium', validationCls(best?.status))}>
                          {best?.status_label ?? '待验证'}
                        </span>
                        <div className="mt-1 text-[10px] text-muted">
                          {(row.validations ?? []).map(v => `${v.horizon_days}日:${v.status_label}`).join(' · ') || '未来交易日确认'}
                        </div>
                      </td>
                      <td className="px-3 py-3 text-right font-mono">{best?.actual_rank ?? '—'}</td>
                      <td className={cn('px-3 py-3 text-right font-mono', priceColorClass(best?.actual_ret_1d))}>{pct(best?.actual_ret_1d)}</td>
                      <td className="px-3 py-3 text-right font-mono">{pct(best?.actual_up_ratio, 1)}</td>
                      <td className="px-3 py-3 text-[10px] leading-4 text-secondary">
                        {(row.reasons ?? []).slice(0, 4).join('；') || '—'}
                      </td>
                      <td className="px-3 py-3 text-[10px] leading-4 text-bear">
                        {(row.risk_flags ?? []).slice(0, 3).join('；') || '—'}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>

          <div className="overflow-auto rounded-card border border-border/70">
            <table className="w-full min-w-[1320px] text-xs">
              <thead className="bg-elevated text-[10px] text-muted">
                <tr>
                  <th className="px-3 py-2 text-left font-medium">预测日</th>
                  <th className="px-3 py-2 text-left font-medium">验证日</th>
                  <th className="px-3 py-2 text-right font-medium">Top5</th>
                  <th className="px-3 py-2 text-right font-medium">Top10</th>
                  <th className="px-3 py-2 text-right font-medium">强预测池</th>
                  <th className="px-3 py-2 text-right font-medium">强命中</th>
                  <th className="px-3 py-2 text-right font-medium">观察命中</th>
                  <th className="px-3 py-2 text-right font-medium">漏判</th>
                  <th className="px-3 py-2 text-right font-medium">超额</th>
                  <th className="px-3 py-2 text-left font-medium">实际强势前5</th>
                  <th className="px-3 py-2 text-left font-medium">漏判板块</th>
                </tr>
              </thead>
              <tbody>
                {verified.slice(0, 8).map((snapshot: SectorPredictionSnapshot) => {
                  const summary = snapshot.validation_summary ?? {}
                  const actualTop = (summary.actual_top10_1d ?? []).slice(0, 5)
                  const misses = summary.missed_actual_top10_1d ?? []
                  return (
                    <tr key={snapshot.id ?? snapshot.as_of} className="border-t border-border/60">
                      <td className="px-3 py-3 font-mono text-foreground">{snapshot.as_of}</td>
                      <td className="px-3 py-3 font-mono text-muted">{summary.verify_1d_date ?? '—'}</td>
                      <td className="px-3 py-3 text-right font-mono">{hitText(summary.top5_hits, summary.top5_verified)}</td>
                      <td className="px-3 py-3 text-right font-mono">{hitText(summary.top10_hits, summary.top10_verified)}</td>
                      <td className="px-3 py-3 text-right font-mono">{hitText(summary.strong_hits, summary.strong_verified)}</td>
                      <td className="px-3 py-3 text-right font-mono text-bull">{summary.actual_top10_strong_hits ?? 0}</td>
                      <td className="px-3 py-3 text-right font-mono text-warning">{summary.actual_top10_watch_hits ?? 0}</td>
                      <td className={cn('px-3 py-3 text-right font-mono', (summary.actual_top10_misses ?? 0) > 0 ? 'text-bear' : 'text-bull')}>{summary.actual_top10_misses ?? 0}</td>
                      <td className={cn('px-3 py-3 text-right font-mono', priceColorClass(summary.top10_excess_1d))}>{pct(summary.top10_excess_1d)}</td>
                      <td className="px-3 py-3 text-[10px] leading-4 text-secondary">
                        {actualTop.map(item => `${item.sector}${item.prediction_rank ? `#${item.prediction_rank}` : ''}`).join('；') || '—'}
                      </td>
                      <td className="px-3 py-3 text-[10px] leading-4 text-bear">
                        {misses.slice(0, 4).map(item => `${item.sector} ${pct(item.actual_ret_1d)}`).join('；') || '—'}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </section>
  )
}

function MarketGatePanel({ gate }: { gate?: MarketGatePayload | null }) {
  if (!gate) return null
  const adjustment = gate.recommendation_adjustment ?? {}
  const metrics = gate.metrics ?? {}
  const reasons = gate.reasons ?? []
  return (
    <section className={panelCls}>
      <SectionTitle icon={ShieldAlert} title="A股市场闸门" hint={`${gate.as_of ?? '—'} · ${gate.generated_at}`} />
      <div className="grid gap-2 md:grid-cols-5">
        <Metric icon={<ShieldAlert className="h-3.5 w-3.5" />} label="开仓状态" value={gate.gate_label} tone={gate.gate === 'open' ? 'text-bull' : gate.gate === 'closed' || gate.gate === 'observe' ? 'text-bear' : 'text-warning'} />
        <Metric icon={<Gauge className="h-3.5 w-3.5" />} label="A股环境分" value={fmtScore(gate.local_score)} tone={scoreTone(gate.local_score)} />
        <Metric icon={<Activity className="h-3.5 w-3.5" />} label="市场状态" value={`${gate.state_label ?? '—'} / ${gate.phase_label ?? '—'}`} />
        <Metric icon={<Target className="h-3.5 w-3.5" />} label="仓位系数" value={`${fmtScore(adjustment.position_multiplier)}x`} />
        <Metric icon={<TrendingUp className="h-3.5 w-3.5" />} label="买入门槛" value={fmtPoint(adjustment.buy_threshold_add)} tone={Number(adjustment.buy_threshold_add ?? 0) > 0 ? 'text-warning' : 'text-bull'} />
      </div>
      <div className="mt-3 grid gap-2 md:grid-cols-6">
        <Metric icon={<TrendingUp className="h-3.5 w-3.5" />} label="上涨率" value={pct(Number(metrics.up_pct ?? 0) / 100, 1)} />
        <Metric icon={<AlertTriangle className="h-3.5 w-3.5" />} label="下跌率" value={pct(Number(metrics.down_pct ?? 0) / 100, 1)} tone={Number(metrics.down_pct ?? 0) >= 62 ? 'text-bear' : undefined} />
        <Metric icon={<AlertTriangle className="h-3.5 w-3.5" />} label="大跌股" value={pct(Number(metrics.strong_down_pct ?? 0) / 100, 1)} tone={Number(metrics.strong_down_pct ?? 0) >= 12 ? 'text-bear' : undefined} />
        <Metric icon={<Layers3 className="h-3.5 w-3.5" />} label="MA20上方" value={metrics.above_ma20_pct == null ? '—' : pct(Number(metrics.above_ma20_pct) / 100, 1)} />
        <Metric icon={<Sparkles className="h-3.5 w-3.5" />} label="涨停/跌停" value={`${metrics.limit_up ?? 0}/${metrics.limit_down ?? 0}`} />
        <Metric icon={<TrendingUp className="h-3.5 w-3.5" />} label="指数均值" value={pct(Number(metrics.index_pct ?? 0) / 100, 2)} tone={priceColorClass(Number(metrics.index_pct ?? 0) / 100)} />
      </div>
      <div className="mt-3 rounded-card border border-warning/20 bg-warning/10 px-3 py-2 text-xs text-warning">
        {adjustment.open_rule || '按市场闸门控制买入门槛、仓位和确认要求。'}
        {!!reasons.length && <div className="mt-1 text-[11px]">{reasons.slice(0, 4).join('；')}</div>}
      </div>
    </section>
  )
}

function Tags({ items, cls = 'border-border bg-elevated text-secondary', limit = 4 }: { items?: string[]; cls?: string; limit?: number }) {
  const visible = (items ?? []).slice(0, limit)
  if (!visible.length) return <span className="text-muted">—</span>
  return (
    <div className="flex max-w-[20rem] flex-wrap gap-1">
      {visible.map(item => (
        <span key={item} className={cn('rounded-full border px-2 py-0.5 text-[10px]', cls)}>{item}</span>
      ))}
    </div>
  )
}

function AgentReviewPanel({
  review,
  loading,
  onRun,
  disabled,
}: {
  review: ShortTermAIAgentReviewResponse | null
  loading: boolean
  onRun: () => void
  disabled: boolean
}) {
  const result = review?.review
  return (
    <section className={panelCls}>
      <SectionTitle
        icon={BrainCircuit}
        title="AI Agent复核"
        hint={review ? `${review.model || 'AI'} · ${review.generated_at}` : 'Top候选二次判断'}
      />
      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={onRun}
          disabled={disabled || loading}
          className="inline-flex h-8 items-center gap-1.5 rounded-btn bg-accent px-3 text-xs font-medium text-white shadow-sm hover:bg-accent/90 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {loading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Sparkles className="h-3.5 w-3.5" />}
          AI复核Top候选
        </button>
        <span className="text-[11px] text-muted">用大模型复核买点、卖点、止损、仓位和放弃条件。</span>
      </div>

      {review?.error && (
        <div className="mt-3 rounded-card border border-warning/20 bg-warning/10 px-3 py-2 text-xs text-warning">
          {review.error}
        </div>
      )}

      {result && (
        <div className="mt-3 space-y-3">
          <div className="grid gap-2 md:grid-cols-[1fr_11rem_1fr]">
            <div className="rounded-card border border-border/60 bg-base/40 px-3 py-2">
              <div className="text-[10px] text-muted">总体结论</div>
              <div className="mt-1 text-xs leading-5 text-foreground">{result.summary}</div>
            </div>
            <div className="rounded-card border border-border/60 bg-base/40 px-3 py-2">
              <div className="text-[10px] text-muted">市场姿态</div>
              <div className="mt-1 text-sm font-semibold text-foreground">{result.market_bias}</div>
            </div>
            <div className="rounded-card border border-border/60 bg-base/40 px-3 py-2">
              <div className="text-[10px] text-muted">StockFormer作用</div>
              <div className="mt-1 text-xs leading-5 text-secondary">{result.stockformer_role}</div>
            </div>
          </div>

          <div className="overflow-auto rounded-card border border-border/70">
            <table className="w-full min-w-[1080px] text-xs">
              <thead className="bg-elevated text-[10px] text-muted">
                <tr>
                  <th className="px-3 py-2 text-left font-medium">股票</th>
                  <th className="px-3 py-2 text-left font-medium">AI评级</th>
                  <th className="px-3 py-2 text-right font-medium">AI分</th>
                  <th className="px-3 py-2 text-left font-medium">买点</th>
                  <th className="px-3 py-2 text-left font-medium">卖点</th>
                  <th className="px-3 py-2 text-left font-medium">止损</th>
                  <th className="px-3 py-2 text-left font-medium">仓位</th>
                  <th className="px-3 py-2 text-left font-medium">理由/失效</th>
                </tr>
              </thead>
              <tbody>
                {result.recommendations.map(item => (
                  <tr key={item.symbol} className="border-t border-border/60">
                    <td className="px-3 py-3"><StockIdentity symbol={item.symbol} name={item.name} /></td>
                    <td className="px-3 py-3">
                      <span className={cn('inline-flex rounded-full border px-2 py-0.5 text-[10px] font-medium', ratingCls(item.agent_rating))}>
                        {item.agent_rating}
                      </span>
                    </td>
                    <td className={cn('px-3 py-3 text-right font-mono font-semibold', scoreTone(item.ai_score))}>{fmtScore(item.ai_score)}</td>
                    <td className="px-3 py-3 leading-5 text-secondary">{item.buy_point}</td>
                    <td className="px-3 py-3 leading-5 text-secondary">{item.sell_point}</td>
                    <td className="px-3 py-3 leading-5 text-secondary">{item.stop_loss}</td>
                    <td className="px-3 py-3 leading-5 text-secondary">{item.position}<div className="text-[10px] text-muted">{item.holding_period}</div></td>
                    <td className="px-3 py-3 leading-5 text-secondary">
                      <div>{item.reason}</div>
                      <div className="mt-1 text-[10px] text-bear">失效：{item.invalidation}</div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {!!result.risk_controls.length && (
            <div className="rounded-card border border-warning/20 bg-warning/10 px-3 py-2 text-xs text-warning">
              {result.risk_controls.join('；')}
            </div>
          )}
          <div className="text-[10px] text-muted">{result.disclaimer}</div>
        </div>
      )}
    </section>
  )
}

function RecommendationTable({ rows, onOpen }: { rows: ShortTermAIRecommendationRow[]; onOpen: (row: ShortTermAIRecommendationRow) => void }) {
  return (
    <div className="overflow-auto rounded-card border border-border/70">
      <table className="w-full min-w-[1840px] text-xs">
        <thead className="sticky top-0 z-10 bg-elevated text-[10px] text-muted">
          <tr>
            <th className="px-3 py-2 text-right font-medium">排名</th>
            <th className="px-3 py-2 text-left font-medium">股票</th>
            <th className="px-3 py-2 text-left font-medium">操作</th>
            <th className="px-3 py-2 text-left font-medium">建议</th>
            <th className="px-3 py-2 text-left font-medium">确认</th>
            <th className="px-3 py-2 text-left font-medium">入场分</th>
            <th className="px-3 py-2 text-right font-medium">StockFormer</th>
            <th className="px-3 py-2 text-right font-medium">预测</th>
            <th className="px-3 py-2 text-left font-medium">早期资金</th>
            <th className="px-3 py-2 text-left font-medium">板块预判</th>
            <th className="px-3 py-2 text-left font-medium">策略/来源</th>
            <th className="px-3 py-2 text-left font-medium">买卖计划</th>
            <th className="px-3 py-2 text-left font-medium">风险</th>
            <th className="px-3 py-2 text-left font-medium">隔夜</th>
          </tr>
        </thead>
        <tbody>
          {!rows.length ? (
            <tr>
              <td colSpan={14} className="px-3 py-10 text-center text-muted">暂无推荐结果</td>
            </tr>
          ) : rows.map(row => (
            <tr key={row.symbol} className="border-t border-border/60 hover:bg-elevated/40">
              <td className="px-3 py-3 text-right font-mono text-muted">{row.rank}</td>
              <td className="px-3 py-3">
                <button type="button" onClick={() => onOpen(row)} className="text-left">
                  <StockIdentity symbol={row.symbol} name={row.name} />
                  <div className="mt-1 flex flex-wrap gap-1 text-[10px] text-muted">
                    {row.board && <span>{row.board}</span>}
                    {row.industry && <span className="max-w-[10rem] truncate">{row.industry}</span>}
                    {row.close != null && <span>{fmtPrice(row.close)}</span>}
                  </div>
                </button>
              </td>
              <td className="px-3 py-3">
                <StockQuickActions
                  symbol={row.symbol}
                  name={row.name}
                  price={row.close}
                  watchNote="短线AI荐股中心"
                  paperReason={`短线AI荐股：${row.decision}，入场分${fmtScore(row.entry_score ?? row.final_score)}`}
                />
              </td>
              <td className="px-3 py-3">
                <span className={cn('inline-flex rounded-full border px-2 py-0.5 text-[10px] font-medium', decisionCls(row.decision))}>
                  {row.decision}
                </span>
                <div className="mt-1 max-w-[11rem] text-[10px] leading-4 text-secondary">{row.action}</div>
              </td>
              <td className="px-3 py-3">
                <div className="flex flex-wrap gap-1">
                  <span className={cn('rounded-full border px-2 py-0.5 text-[10px]', confirmationCls(row.buy_confirmation_status))}>
                    {row.buy_confirmation_label || '等待确认'}
                  </span>
                  <span className={cn('rounded-full border px-2 py-0.5 text-[10px]', marketGateCls(row.market_gate))}>
                    {row.market_gate_label || '市场闸门'}
                  </span>
                </div>
                <div className="mt-1 grid gap-0.5 font-mono text-[10px] text-muted">
                  <span>确认 {fmtScore(row.buy_confirmation_score)} · 市场 {fmtScore(row.market_local_score)}</span>
                  {row.five_minute_confirmation && <span title={row.five_minute_confirmation.bar_end ?? undefined}>
                    {row.five_minute_confirmation.label}
                    {row.five_minute_confirmation.bar_end ? ` · ${row.five_minute_confirmation.bar_end.slice(11, 16)}` : ''}
                  </span>}
                  <span className={row.minute_data_eligible ? 'text-bull' : 'text-warning'}>
                    分钟 {row.minute_coverage_status === 'complete' ? `${row.minute_coverage_minutes ?? 0}/${row.minute_expected_minutes ?? 240}` : `${row.minute_coverage_status || 'unknown'} · 日K模型`}
                  </span>
                  <span>仓位 {row.combined_position_multiplier != null ? `${fmtScore(row.combined_position_multiplier)}x` : '—'} · 门槛 {fmtPoint(row.market_buy_threshold_add)}</span>
                  <span>数据覆盖 {fmtScore(row.data_coverage_score)}% · 分钟/资金/竞价/新闻 {[
                    row.minute_data_available, row.moneyflow_data_available,
                    row.auction_data_available, row.news_data_available,
                  ].filter(Boolean).length}/4</span>
                  <span>{row.market_state_label || '—'} / {row.market_phase_label || '—'}</span>
                  {!!row.buy_confirmation_failed?.length && <span className="text-warning">缺 {row.buy_confirmation_failed.slice(0, 2).join('、')}</span>}
                </div>
              </td>
              <td className="px-3 py-3">
                <ScoreBar
                  value={row.final_score}
                  parts={[
                    { label: '入', value: row.entry_score ?? row.final_score },
                    { label: '技', value: row.technical_score },
                    { label: '模', value: row.stockformer_score },
                    { label: '资', value: row.fund_score },
                    { label: '板', value: row.sector_score },
                    { label: '险', value: row.risk_score },
                  ]}
                />
                <div className="mt-1 font-mono text-[10px] text-muted">
                  入场 {fmtScore(row.entry_score)} · 综合 {fmtScore(row.multifactor_score ?? row.base_final_score)}
                </div>
              </td>
              <td className="px-3 py-3 text-right font-mono">
                <div className={scoreTone(row.stockformer_score)}>{fmtScore(row.stockformer_score)}</div>
                <div className="mt-1 text-[10px] text-muted">模型#{row.model_rank ?? '—'} 置信{pct(row.confidence, 1)}</div>
              </td>
              <td className="px-3 py-3 text-right font-mono">
                <div className={priceColorClass(row.pred_ret_1d)}>{pct(row.pred_ret_1d)}</div>
                <div className={priceColorClass(row.pred_ret_3d)}>{pct(row.pred_ret_3d)}</div>
                <div className="text-[10px] text-muted">涨{pct(row.up_prob_5d, 1)} 趋{pct(row.trend_prob_10d, 1)}</div>
              </td>
              <td className="px-3 py-3">
                <div className="flex flex-wrap gap-1">
                  {row.daily_fund_label && <span className={cn('rounded-full border px-2 py-0.5 text-[10px]', signalCls(row.daily_fund_label))}>{row.daily_fund_label}</span>}
                  {row.intraday_fund_label && <span className={cn('rounded-full border px-2 py-0.5 text-[10px]', signalCls(row.intraday_fund_label))}>{row.intraday_fund_label}</span>}
                  {!row.daily_fund_label && !row.intraday_fund_label && (
                    <span className={cn('rounded-full border px-2 py-0.5 text-[10px]', signalCls(row.fund_status_label))}>
                      {row.fund_status_label || '暂无资金确认'}
                    </span>
                  )}
                </div>
                <div className="mt-1 grid gap-0.5 font-mono text-[10px] text-muted">
                  <span>吸筹 {fmtScore(row.main_accumulation_score ?? normalizedScore(row.moneyflow_accumulation_score))} · 早期 {fmtScore(row.early_fund_setup_score ?? normalizedScore(row.moneyflow_early_layout_score))} · 后手 {fmtScore(row.chase_risk_score)}</span>
                  <span>3日净流 {fmtBigNum(row.net_3d)} · 前3日 {fmtBigNum(row.prev_net_3d)}</span>
                  <span>效率 {fmtScore(row.absorption_efficiency == null ? normalizedScore(row.moneyflow_absorption_efficiency) : row.absorption_efficiency * 100)} · 位置 {fmtScore(row.price_position_20d == null ? null : row.price_position_20d * 100)} · 收敛 {pct(row.price_range_10d, 1)}</span>
                  <span>加速度 {pct(row.daily_moneyflow_acceleration_3d ?? row.moneyflow_acceleration_3d, 2)} · 拐点 {row.moneyflow_turn_positive ? '是' : '否'}</span>
                  <span>1日比 {pct(row.net_ratio_1d, 2)} 3日比 {pct(row.net_ratio_3d, 2)}</span>
                  <span>15m估算 {fmtBigNum(row.est_net_15m)}</span>
                </div>
              </td>
              <td className="px-3 py-3">
                {row.sector_phase ? (
                  <>
                    <span className={cn('inline-flex rounded-full border px-2 py-0.5 text-[10px] font-medium', sectorSignalCls(row.sector_signal))}>
                      {row.sector_phase}
                    </span>
                    <div className="mt-1 max-w-[12rem] truncate text-[10px] text-secondary">{row.sector || row.industry || '—'}</div>
                    <div className="mt-1 grid gap-0.5 font-mono text-[10px] text-muted">
                      <span>板块分 {fmtScore(row.sector_score)} · 排名 {row.sector_rank ?? '—'}</span>
                      <span>提前 {fmtScore(row.sector_setup_score ?? row.sector_pre_layout_score)} · 今日确认 {fmtScore(row.sector_today_confirm_score)}</span>
                      <span>今日板块 {pct(row.industry_ret_1d)} · 拥挤 {fmtScore(normalizedScore(row.sector_crowding_risk))}</span>
                      <span>昨日前 {pct(row.sector_pre_ret_5d)} · 资金 {fmtBigNum(row.sector_pre_net_5d)}</span>
                      <span>5日 {pct(row.sector_ret_5d)} · 资金 {fmtBigNum(row.sector_net_5d)}</span>
                    </div>
                  </>
                ) : (
                  <span className="text-muted">—</span>
                )}
              </td>
              <td className="px-3 py-3">
                <Tags items={row.sources} />
                <div className="mt-1 text-[10px] text-muted">策略 {row.strategy_count} 个 · 共振 {row.support_count} 项</div>
                <div className="mt-1 max-w-[16rem] text-[10px] leading-4 text-secondary">
                  {row.reasons.slice(0, 3).join('；') || '—'}
                </div>
              </td>
              <td className="px-3 py-3 text-[10px] leading-5 text-secondary">
                <div>买点：{row.trade_plan.entry}</div>
                <div>止盈：{row.trade_plan.take_profit}</div>
                <div>止损：{row.trade_plan.stop_loss}</div>
                {row.trade_plan.position && <div>仓位：{row.trade_plan.position}</div>}
              </td>
              <td className="px-3 py-3">
                <Tags items={row.risk_warnings} cls="border-bear/25 bg-bear/10 text-bear" limit={3} />
                {row.kline_quality_label && (
                  <div className={cn('mt-1 text-[10px]', row.kline_quality_label === '正常' ? 'text-muted' : 'text-warning')}>
                    日K {row.kline_quality_label} · {row.kline_rows ?? 0}日
                  </div>
                )}
                <div className="mt-1 font-mono text-[10px] text-muted">异动 {pct(row.abnormal_closeness, 1)}</div>
              </td>
              <td className="px-3 py-3 text-[10px] leading-5 text-secondary">
                <span className={cn('rounded-full border px-2 py-0.5', row.global_risk_label === '防守' || row.global_risk_label === '谨慎' ? 'border-bear/25 bg-bear/10 text-bear' : row.global_risk_label === '进攻' ? 'border-bull/25 bg-bull/10 text-bull' : 'border-border bg-elevated text-secondary')}>
                  {row.global_risk_label || '—'}
                </span>
                <div className="mt-1 font-mono text-muted">偏好 {fmtScore(row.global_appetite_score)}</div>
                <div className="font-mono text-muted">修正 {fmtScore(row.global_score_adjustment)}</div>
                <div className="font-mono text-muted">仓位 {row.global_position_multiplier != null ? `${fmtScore(row.global_position_multiplier)}x` : '—'}</div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function ShortTermAI() {
  const [profile, setProfile] = useState<ShortTermAIProfile>('balanced')
  const [limit, setLimit] = useState(20)
  const [modelId, setModelId] = useState<string>('latest')
  const [asOf, setAsOf] = useState('')
  const [preview, setPreview] = useState<ShortTermAIRecommendationRow | null>(null)
  const [agentReview, setAgentReview] = useState<ShortTermAIAgentReviewResponse | null>(null)
  const [sectorOrigin, setSectorOrigin] = useState<'published' | 'replay' | 'legacy'>('published')

  const modelsQ = useQuery({ queryKey: QK.stockformerModels, queryFn: api.stockformerModels, staleTime: 30_000 })
  const globalRiskQ = useQuery({
    queryKey: QK.globalRisk,
    queryFn: () => api.shortTermAIGlobalRisk(),
    refetchInterval: 15 * 60_000,
    staleTime: 5 * 60_000,
  })
  const sectorPredictionsQ = useQuery({
    queryKey: QK.shortTermAISectorPredictions(asOf || undefined, 8, 30, sectorOrigin),
    queryFn: () => api.shortTermAISectorPredictions({
      origin: sectorOrigin,
      asOf: asOf || null,
      days: 8,
      limit: 30,
    }),
    staleTime: 30_000,
  })
  const refreshSectorPredictionsMutation = useMutation({
    mutationFn: () => api.refreshShortTermAISectorPredictions({
      asOf: asOf || null,
      days: 8,
      lookbackDays: 70,
      limit: 30,
    }),
    onSuccess: (result) => {
      if (result.origin) setSectorOrigin(result.origin)
      sectorPredictionsQ.refetch()
    },
  })
  const recommendationsQ = useQuery({
    queryKey: QK.shortTermAi(profile, modelId, asOf || undefined, limit),
    queryFn: () => api.shortTermAIRecommendations({
      profile,
      modelId: modelId === 'latest' ? null : modelId,
      asOf: asOf || null,
      limit,
    }),
    refetchInterval: 30_000,
    staleTime: 10_000,
  })
  const agentReviewMutation = useMutation({
    mutationFn: () => api.shortTermAIAgentReview({
      profile,
      modelId: modelId === 'latest' ? null : modelId,
      asOf: asOf || null,
      limit,
      reviewCount: Math.min(5, limit),
    }),
    onSuccess: setAgentReview,
  })

  useEffect(() => {
    setAgentReview(null)
  }, [profile, modelId, asOf, limit])

  const data = recommendationsQ.data
  const rows = data?.rows ?? []
  const globalRisk = data?.global_risk ?? globalRiskQ.data
  const marketGate = data?.market_gate
  const strongCount = rows.filter(row => row.decision === '强关注').length
  const avoidCount = rows.filter(row => row.decision === '回避').length
  const confirmedCount = rows.filter(row => row.buy_confirmation_status === 'confirmed').length
  const top = rows[0]
  const modelOptions = useMemo(() => modelsQ.data?.models ?? [], [modelsQ.data])

  return (
    <div className="flex min-h-full flex-col bg-base">
      <PageHeader
        title="短线AI荐股"
        subtitle="策略共振 · StockFormer预测 · 早期资金 · 板块轮动 · 风险闸门"
        className="shrink-0 flex-col items-stretch bg-base/95 md:flex-row md:items-center [&>div:first-child]:flex-wrap [&_h1]:whitespace-nowrap"
        right={
          <div className="flex min-w-0 items-center gap-2">
            <select
              value={modelId}
              onChange={event => setModelId(event.target.value)}
              className="h-8 min-w-0 max-w-[16rem] flex-1 rounded-input border border-border bg-surface px-2 text-xs outline-none focus:border-accent md:flex-none"
            >
              <option value="latest">最新 StockFormer</option>
              {modelOptions.map(model => (
                <option key={model.model_id} value={model.model_id}>{model.model_name} · {model.trained_at}</option>
              ))}
            </select>
            <button
              type="button"
              onClick={() => recommendationsQ.refetch()}
              className="inline-flex h-8 items-center gap-1.5 rounded-btn border border-border bg-surface px-3 text-xs text-secondary hover:bg-elevated hover:text-foreground"
            >
              {recommendationsQ.isFetching ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />}
              刷新
            </button>
          </div>
        }
      />

      <main className="flex-1 space-y-4 px-3 pb-4 pt-3 lg:px-4">
        <PredictionWorkflowPanel />
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-8">
          <Metric icon={<Target className="h-3.5 w-3.5" />} label="Top推荐" value={top ? `${top.name || top.symbol} / ${fmtScore(top.final_score)}` : '—'} tone={top ? scoreTone(top.final_score) : undefined} />
          <Metric icon={<Sparkles className="h-3.5 w-3.5" />} label="强关注" value={strongCount} tone={strongCount > 0 ? 'text-bull' : undefined} />
          <Metric icon={<ShieldAlert className="h-3.5 w-3.5" />} label="买入确认" value={confirmedCount} tone={confirmedCount > 0 ? 'text-bull' : undefined} />
          <Metric icon={<ShieldAlert className="h-3.5 w-3.5" />} label="回避" value={avoidCount} tone={avoidCount > 0 ? 'text-bear' : undefined} />
          <Metric icon={<Gauge className="h-3.5 w-3.5" />} label="市场闸门" value={marketGate ? `${marketGate.gate_label} / ${fmtScore(marketGate.local_score)}` : '—'} tone={marketGate?.gate === 'closed' || marketGate?.gate === 'observe' ? 'text-bear' : marketGate?.gate === 'open' ? 'text-bull' : 'text-warning'} />
          <Metric icon={<AlertTriangle className="h-3.5 w-3.5" />} label="隔夜风险" value={globalRisk ? `${globalRisk.market_bias} / ${fmtScore(globalRisk.appetite_score)}` : '—'} tone={globalRisk && globalRisk.appetite_score < 45 ? 'text-bear' : undefined} />
          <Metric icon={<BrainCircuit className="h-3.5 w-3.5" />} label="StockFormer" value={data?.model?.model_name ?? '—'} />
          <Metric icon={<Activity className="h-3.5 w-3.5" />} label="数据日期" value={data?.as_of ?? '—'} />
        </div>

        <section className={panelCls}>
          <SectionTitle icon={TrendingUp} title="统一权重" hint={data?.generated_at} />
          <div className="flex flex-wrap items-center gap-2">
            {profiles.map(item => (
              <button
                key={item.value}
                type="button"
                onClick={() => setProfile(item.value)}
                className={cn(
                  'inline-flex h-8 items-center rounded-btn border px-3 text-xs',
                  profile === item.value ? 'border-accent bg-accent text-white' : 'border-border bg-base text-secondary hover:bg-elevated',
                )}
              >
                {item.label}
              </button>
            ))}
            <label className="ml-auto flex items-center gap-2 text-[11px] text-secondary">
              日期
              <input
                type="date"
                value={asOf}
                onChange={event => setAsOf(event.target.value)}
                className="h-8 rounded-input border border-border bg-base px-2 text-xs outline-none focus:border-accent"
              />
            </label>
            <label className="flex items-center gap-2 text-[11px] text-secondary">
              Top
              <input
                type="number"
                min={5}
                max={100}
                value={limit}
                onChange={event => setLimit(Number(event.target.value))}
                className="h-8 w-20 rounded-input border border-border bg-base px-2 text-xs outline-none focus:border-accent"
              />
            </label>
          </div>
          <div className="mt-3 grid gap-2 md:grid-cols-5">
            {Object.entries(data?.weights ?? {}).map(([key, value]) => (
              <div key={key} className="rounded border border-border/60 bg-base/40 px-3 py-2">
                <div className="text-[10px] text-muted">{key}</div>
                <div className="mt-1 font-mono text-sm font-semibold text-foreground">{(value * 100).toFixed(0)}%</div>
              </div>
            ))}
          </div>
          {data?.notes?.length ? (
            <div className="mt-3 rounded-card border border-accent/15 bg-accent/5 px-3 py-2 text-xs text-secondary">
              {data.notes.join('；')}
            </div>
          ) : null}
        </section>

        <GlobalRiskPanel
          risk={globalRisk}
          loading={globalRiskQ.isFetching || recommendationsQ.isFetching}
          onRefresh={() => {
            globalRiskQ.refetch()
            recommendationsQ.refetch()
          }}
        />

        <MarketGatePanel gate={marketGate} />

        <SectorRotationPanel payload={data?.sector_rotation} />

        <SectorPredictionRecordsPanel
          origin={sectorOrigin}
          onOrigin={setSectorOrigin}
          generation={refreshSectorPredictionsMutation.data?.generation}
          error={sectorPredictionsQ.error?.message ?? refreshSectorPredictionsMutation.error?.message}
          payload={sectorPredictionsQ.data}
          loading={sectorPredictionsQ.isFetching}
          refreshing={refreshSectorPredictionsMutation.isPending}
          onRefresh={() => refreshSectorPredictionsMutation.mutate()}
        />

        <SectorResearchPanel />
        <CoverageWarning coverage={data?.data_coverage} />
        <KlineQualityWarning meta={data?.kline_quality_meta} />
        {data?.errors?.length ? (
          <div className="rounded-card border border-warning/20 bg-warning/10 px-3 py-2 text-xs text-warning">
            {data.errors.join('；')}
          </div>
        ) : null}

        <AgentReviewPanel
          review={agentReview}
          loading={agentReviewMutation.isPending}
          onRun={() => agentReviewMutation.mutate()}
          disabled={!rows.length || recommendationsQ.isFetching}
        />

        <section className={panelCls}>
          <SectionTitle
            icon={ScanSearch}
            title="今日推荐"
            hint={`候选 ${data?.total ?? 0} · 展示 ${rows.length}`}
          />
          <RecommendationTable rows={rows} onOpen={setPreview} />
        </section>

        <section className={panelCls}>
          <SectionTitle icon={DatabaseZap} title="来源覆盖" />
          <div className="grid gap-2 md:grid-cols-3 xl:grid-cols-6">
            {Object.entries(data?.source_counts ?? {}).map(([key, value]) => (
              <Metric key={key} icon={<DatabaseZap className="h-3.5 w-3.5" />} label={key} value={renderScalar(value)} />
            ))}
          </div>
          <div className="mt-3 grid gap-2 text-[11px] text-muted md:grid-cols-2">
            <div>模型截面：{data?.cross_section?.universe_total ?? data?.cross_section?.raw_universe_total ?? '—'}</div>
            <div>策略缓存：{data?.strategy_meta?.as_of ?? '—'}</div>
            <div>资金日期：{data?.fund_meta?.daily?.flow_meta?.date_start ?? '—'} 至 {data?.fund_meta?.daily?.flow_meta?.date_end ?? '—'}</div>
            <div>分钟数据：{data?.fund_meta?.intraday?.minute_meta?.latest_minute ?? data?.fund_meta?.intraday?.minute_meta?.date_end ?? '—'}</div>
          </div>
        </section>

        <div className="rounded-card border border-warning/20 bg-warning/10 px-3 py-2 text-xs text-warning">
          <div className="flex items-start gap-2">
            <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            <span>模型预测不是确定性涨跌结论，短线买入仍需等盘口和资金确认；样本外未稳定跑赢前，只适合模拟盘或小仓验证。</span>
          </div>
        </div>
      </main>

      <StockPreviewDialog
        symbol={preview?.symbol ?? null}
        name={preview?.name ?? ''}
        onClose={() => setPreview(null)}
      />
    </div>
  )
}
