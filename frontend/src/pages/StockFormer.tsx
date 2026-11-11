import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import type { EChartsOption } from 'echarts'
import {
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  Clock,
  Database,
  FileJson,
  GitCompare,
  Info,
  Layers3,
  LineChart,
  Loader2,
  Play,
  RefreshCw,
  RotateCcw,
  ShieldCheck,
  Trophy,
  XCircle,
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import { GlobalRiskPanel } from '@/components/GlobalRiskPanel'
import { MultiPeriodExperiment } from '@/components/MultiPeriodExperiment'
import { PageHeader } from '@/components/PageHeader'
import { StockQuickActions } from '@/components/StockQuickActions'
import { StockFormerDailyShadow } from '@/components/StockFormerDailyShadow'
import { api, type StockFormerDataCoverageEntry, type StockFormerFactorResult, type StockFormerForwardValidationResult, type StockFormerModel, type StockFormerMultiFactorResult, type StockFormerPredictionResult, type StockFormerPortfolioResearchResult, type StockFormerStrictOOSResult } from '@/lib/api'
import { QK } from '@/lib/queryKeys'
import { cn } from '@/lib/cn'
import { useChartTheme } from '@/lib/theme'
import { useECharts } from './backtest/charts/useECharts'

const panelCls = 'min-w-0 rounded-card border border-border bg-surface/80 p-4 shadow-[0_1px_2px_hsl(var(--border)/0.35)]'
const inputCls = 'w-full rounded-input border border-border bg-base px-2.5 py-1.5 text-xs text-foreground outline-none transition-colors focus:border-accent'

function dateString(d: Date) {
  return d.toISOString().slice(0, 10)
}

function yearsAgo(years: number) {
  const d = new Date()
  d.setFullYear(d.getFullYear() - years)
  return dateString(d)
}

function n(v: unknown) {
  if (v == null || v === '') return null
  const x = Number(v)
  return Number.isFinite(x) ? x : null
}

function fmt(v: unknown, digits = 2) {
  const x = n(v)
  return x == null ? '--' : x.toFixed(digits)
}

function pct(v: unknown, digits = 2) {
  const x = n(v)
  return x == null ? '--' : `${(x * 100).toFixed(digits)}%`
}

function pctPoint(v: unknown, digits = 2) {
  const x = n(v)
  return x == null ? '--' : `${x.toFixed(digits)}%`
}

function crossSectionHint(result?: Pick<StockFormerPredictionResult | StockFormerMultiFactorResult, 'as_of' | 'universe_total' | 'selection_scope'>) {
  const asOf = result?.as_of ?? '--'
  const total = n(result?.universe_total)
  const scope = result?.selection_scope || '全市场有效截面'
  return `${asOf} · ${scope}${total == null ? '' : ` ${total.toFixed(0)}只`}`
}

function crossSectionNote(result?: Pick<StockFormerPredictionResult | StockFormerMultiFactorResult, 'cross_section'>) {
  const skipped = result?.cross_section?.skipped_sparse_dates
  const unready = result?.cross_section?.skipped_unready_dates
  const parts: string[] = []
  if (Array.isArray(skipped) && skipped.length > 0) {
    const text = skipped
      .map(item => `${item.date ?? '--'} ${Number(item.symbols ?? 0).toFixed(0)}只`)
      .join('、')
    parts.push(`覆盖不足：${text}`)
  }
  if (Array.isArray(unready) && unready.length > 0) {
    const text = unready
      .map(item => `${item.date ?? '--'} ${Number(item.feature_ready_symbols ?? 0).toFixed(0)}/${Number(item.symbols ?? 0).toFixed(0)}只`)
      .join('、')
    parts.push(`特征不完整：${text}`)
  }
  return parts.length ? `已跳过最新不可用截面：${parts.join('；')}。` : null
}

function toneByNumber(v: unknown) {
  const x = n(v)
  if (x == null || x === 0) return 'text-muted'
  return x > 0 ? 'text-bull' : 'text-bear'
}

function stockDisplay(symbol?: string | null, name?: string | null) {
  const cleanSymbol = String(symbol ?? '').trim()
  const cleanName = String(name ?? '').trim()
  if (!cleanSymbol) return cleanName || '--'
  if (!cleanName || cleanName === cleanSymbol) return cleanSymbol
  return `${cleanName} (${cleanSymbol})`
}

function compactSymbol(symbol?: string | null) {
  const cleanSymbol = String(symbol ?? '').trim().toUpperCase()
  if (!cleanSymbol) return ''
  const base = cleanSymbol.replace(/\.(SH|SZ|BJ)$/, '').replace(/^(SH|SZ|BJ)/, '')
  const digits = base.replace(/\D/g, '')
  return digits ? digits.slice(-6).padStart(6, '0') : base
}

function stockNameOf(names: Record<string, string>, symbol?: string | null) {
  const cleanSymbol = String(symbol ?? '').trim()
  const compact = compactSymbol(cleanSymbol)
  return names[cleanSymbol] ?? names[compact] ?? ''
}

function StockLabel({ symbol, name }: { symbol?: string | null; name?: string | null }) {
  const cleanSymbol = String(symbol ?? '').trim()
  const cleanName = String(name ?? '').trim()
  return (
    <div className="min-w-0">
      <div className="truncate font-medium text-foreground">{cleanName || cleanSymbol || '--'}</div>
      {cleanName && cleanSymbol && cleanName !== cleanSymbol && <div className="font-mono text-[10px] text-muted">{cleanSymbol}</div>}
    </div>
  )
}

function SectionTitle({ icon: Icon, title, hint }: { icon: LucideIcon; title: string; hint?: ReactNode }) {
  return (
    <div className="mb-3 flex flex-wrap items-center gap-2">
      <span className="h-3 w-0.5 shrink-0 rounded-full bg-gradient-to-b from-accent to-accent/30" />
      <Icon className="h-3.5 w-3.5 shrink-0 text-accent" />
      <h2 className="text-xs font-semibold text-foreground">{title}</h2>
      {hint != null && <span className="ml-auto text-[10px] font-mono text-muted">{hint}</span>}
    </div>
  )
}

function Metric({ label, value, tone }: { label: string; value: ReactNode; tone?: string }) {
  return (
    <div className="min-w-0 break-words rounded-card border border-border/70 bg-base/40 px-3 py-2">
      <div className="text-[10px] text-muted">{label}</div>
      <div className={cn('mt-1 text-sm font-semibold text-foreground', tone)}>{value}</div>
    </div>
  )
}

function ErrorLine({ error }: { error?: string | null }) {
  if (!error) return null
  return <div className="rounded-card border border-warning/20 bg-warning/10 px-3 py-2 text-xs text-warning">{error}</div>
}

function formatDuration(seconds: number) {
  if (seconds < 60) return `${seconds}秒`
  const minutes = Math.floor(seconds / 60)
  const remaining = seconds % 60
  return `${minutes}分${String(remaining).padStart(2, '0')}秒`
}

function TrainingState({
  pending,
  elapsedSeconds,
  error,
  model,
  stage,
  progress,
  message,
}: {
  pending: boolean
  elapsedSeconds: number
  error?: string | null
  model?: StockFormerModel | null
  stage?: string | null
  progress?: number | null
  message?: string | null
}) {
  if (pending) {
    return (
      <div role="status" className="flex items-start gap-3 rounded-card border border-accent/25 bg-accent/5 px-3 py-3 text-xs">
        <Loader2 className="mt-0.5 h-4 w-4 shrink-0 animate-spin text-accent" />
        <div className="min-w-0">
          <div className="font-medium text-foreground">正在训练模型</div>
          <div className="mt-1 text-secondary">
            已运行 {formatDuration(elapsedSeconds)} · {stage || '后台计算'}
            {progress != null ? ` · ${Math.round(progress)}%` : ''}，训练期间不会重复提交。
          </div>
          {message && <div className="mt-1 break-words leading-5 text-muted">{message}</div>}
        </div>
      </div>
    )
  }
  if (error) {
    return (
      <div role="alert" className="flex items-start gap-3 rounded-card border border-bear/25 bg-bear/5 px-3 py-3 text-xs text-bear">
        <XCircle className="mt-0.5 h-4 w-4 shrink-0" />
        <div className="min-w-0">
          <div className="font-medium text-foreground">训练未完成</div>
          <div className="mt-1 break-words leading-5">{error}</div>
        </div>
      </div>
    )
  }
  if (model) {
    return (
      <div role="status" className="flex items-start gap-3 rounded-card border border-bull/25 bg-bull/5 px-3 py-3 text-xs">
        <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-bull" />
        <div className="min-w-0">
          <div className="font-medium text-foreground">模型已就绪</div>
          <div className="mt-1 text-secondary">{model.model_name} · {model.trained_at ?? '训练时间未知'}，预测和样本外验证已刷新。</div>
        </div>
      </div>
    )
  }
  return (
    <div className="flex items-start gap-3 rounded-card border border-border/70 bg-base/40 px-3 py-3 text-xs">
      <Info className="mt-0.5 h-4 w-4 shrink-0 text-muted" />
      <div className="min-w-0 text-secondary">调整参数后点击“训练模型”。建议先确认训练日期和数据覆盖，再开始耗时任务。</div>
    </div>
  )
}

function CoverageWarning({ coverage }: { coverage?: Record<string, StockFormerDataCoverageEntry> }) {
  const items = Object.values(coverage ?? {})
  const missing = items.filter(item => !item.available)
  if (missing.length === 0) return null
  const reasons = missing.flatMap(item => (item.reasons ?? []).map(reason => `${item.label}: ${reason}`))
  const available = items.filter(item => item.available)
  return (
    <div className="rounded-card border border-warning/20 bg-warning/10 px-3 py-2 text-xs text-warning">
      增强数据未完整覆盖：{missing.map(item => item.label).join('、')} 暂无可用记录，相关特征当前按中性值参与评分。
      {available.length > 0 && (
        <div className="mt-1 text-[11px] leading-5 text-muted">
          已有数据：{available.map(item => {
            const range = item.earliest_date && item.latest_date ? `${item.earliest_date}~${item.latest_date}` : `${item.files} 个文件`
            return `${item.label} ${range}`
          }).join('；')}
        </div>
      )}
      {reasons.length > 0 && (
        <div className="mt-1 space-y-0.5 text-[11px] leading-5">
          {reasons.slice(0, 5).map(reason => <div key={reason}>{reason}</div>)}
        </div>
      )}
    </div>
  )
}

interface Column<T> {
  key: string
  label: string
  align?: 'left' | 'right'
  render?: (row: T, index: number) => ReactNode
}

function ForwardValidation({ result }: { result?: StockFormerForwardValidationResult }) {
  if (!result?.rolling) return null
  const waiting = result.status === 'waiting_for_new_oos_window'
  const stability = result.rolling.stability
  const stats = result.final_backtest?.full_oos_portfolio_stats
  const observation = result.signal_observation
  return (
    <div className="space-y-3" aria-label="冻结策略前向验证">
      <div role="status" className="text-xs leading-5 text-secondary">
        {waiting ? '等待新增样本外窗口' : result.activation_decision?.eligible ? '验证达标，等待人工激活' : '前向验证未达激活条件'}
        {observation?.note && <div className="mt-1">{observation.note}</div>}
      </div>
      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
        <Metric label="新增预测交易日" value={fmt(observation?.new_oos_sessions ?? result.sample_out?.trading_days, 0)} />
        <Metric label="完整窗口" value={fmt(stability.window_count, 0)} />
        <Metric label="连续盈利窗口" value={`${stability.trailing_positive_run} / ${stability.required_positive_windows}`} />
        <Metric label="激活状态" value={result.activation_decision?.eligible ? '等待人工激活' : '研究观察'} />
        <Metric label="组合成本后收益" value={stats ? pctPoint(stats.total_return) : '未评估'} tone={toneByNumber(stats?.total_return)} />
        <Metric label="组合最大回撤" value={stats ? pctPoint(stats.max_drawdown) : '未评估'} tone={toneByNumber(stats?.max_drawdown)} />
        <Metric label="模拟成交笔数" value={stats ? fmt(stats.trade_count, 0) : observation?.evaluated ? fmt(observation.trade_count, 0) : '未评估'} />
        <Metric label="观察区间" value={`${result.sample_out?.start ?? result.sample_out?.available_start ?? '--'} 至 ${result.sample_out?.end ?? result.sample_out?.available_end ?? '--'}`} />
      </div>
      {result.rolling.windows.length > 0 && (
        <DataTable
          rows={result.rolling.windows}
          columns={[
            { key: 'window', label: '窗口' },
            { key: 'start', label: '开始' },
            { key: 'end', label: '结束' },
            { key: 'return', label: '成本后收益', render: r => pctPoint(r.stats?.total_return) },
            { key: 'drawdown', label: '最大回撤', render: r => pctPoint(r.stats?.max_drawdown) },
            { key: 'trades', label: '成交笔数', render: r => fmt(r.stats?.trade_count, 0) },
          ]}
        />
      )}
      {result.activation_decision?.reasons.map(reason => <div key={reason} className="text-xs leading-5 text-warning">{reason}</div>)}
    </div>
  )
}

function DataTable<T extends Record<string, any>>({
  rows,
  columns,
  emptyText = '暂无数据',
  maxHeight = 'max-h-[360px]',
}: {
  rows: T[]
  columns: Column<T>[]
  emptyText?: string
  maxHeight?: string
}) {
  return (
    <div className={cn('min-w-0 max-w-full overflow-auto rounded-card border border-border/70', maxHeight)}>
      <table className="w-full min-w-max text-xs">
        <thead className="sticky top-0 z-10 bg-elevated text-[10px] text-muted">
          <tr>
            {columns.map(col => (
              <th key={col.key} className={cn('px-3 py-2 font-medium', col.align === 'right' ? 'text-right' : 'text-left')}>
                {col.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 ? (
            <tr>
              <td className="px-3 py-8 text-center text-muted" colSpan={columns.length}>{emptyText}</td>
            </tr>
          ) : rows.map((row, index) => (
            <tr key={`${row.symbol ?? row.model_id ?? row.date ?? index}-${index}`} className="border-t border-border/60 hover:bg-elevated/50">
              {columns.map(col => (
                <td key={col.key} className={cn('px-3 py-2', col.align === 'right' ? 'text-right font-mono' : '')}>
                  {col.render ? col.render(row, index) : (row[col.key] ?? '--')}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function MiniChart({ option, deps }: { option: EChartsOption | null; deps: unknown[] }) {
  const ref = useECharts(option, deps)
  return <div ref={ref} className="h-[240px]" />
}

function FactorChart({ result }: { result?: StockFormerFactorResult }) {
  const ct = useChartTheme()
  const option = useMemo<EChartsOption | null>(() => {
    const rows = result?.ic_series ?? []
    if (rows.length === 0) return null
    const dates = rows.map(r => r.date)
    return {
      animation: false,
      grid: { left: 48, right: 18, top: 18, bottom: 34 },
      tooltip: {
        trigger: 'axis',
        backgroundColor: ct.tooltipBg,
        borderColor: ct.tooltipBorder,
        textStyle: { color: ct.tooltipText, fontSize: 12 },
      },
      xAxis: {
        type: 'category',
        data: dates,
        axisTick: { show: false },
        axisLine: { lineStyle: { color: ct.border } },
        axisLabel: { color: ct.text, fontSize: 10, interval: Math.max(1, Math.floor(dates.length / 8)) },
      },
      yAxis: {
        type: 'value',
        axisLine: { show: false },
        axisLabel: { color: ct.text, fontSize: 10, formatter: (v: number) => `${(v * 100).toFixed(0)}%` },
        splitLine: { lineStyle: { color: ct.grid } },
      },
      series: [
        {
          name: 'Rank IC',
          type: 'bar',
          data: rows.map(r => r.rank_ic ?? null),
          barMaxWidth: 8,
          itemStyle: { color: '#3b82f6' },
        },
        {
          name: 'IC',
          type: 'line',
          data: rows.map(r => r.ic ?? null),
          symbol: 'none',
          lineStyle: { color: '#f59e0b', width: 1.4 },
        },
      ],
    }
  }, [result?.ic_series, ct])
  return <MiniChart option={option} deps={[result?.horizon, result?.ic_series?.length ?? 0, ct]} />
}

function OOSChart({ result }: { result?: StockFormerStrictOOSResult }) {
  const ct = useChartTheme()
  const option = useMemo<EChartsOption | null>(() => {
    const rows = result?.nav ?? []
    if (rows.length === 0) return null
    const dates = rows.map(r => r.date)
    return {
      animation: false,
      grid: { left: 52, right: 18, top: 18, bottom: 34 },
      tooltip: {
        trigger: 'axis',
        backgroundColor: ct.tooltipBg,
        borderColor: ct.tooltipBorder,
        textStyle: { color: ct.tooltipText, fontSize: 12 },
      },
      xAxis: {
        type: 'category',
        data: dates,
        axisTick: { show: false },
        axisLine: { lineStyle: { color: ct.border } },
        axisLabel: { color: ct.text, fontSize: 10, interval: Math.max(1, Math.floor(dates.length / 8)) },
      },
      yAxis: {
        type: 'value',
        scale: true,
        axisLine: { show: false },
        axisLabel: { color: ct.text, fontSize: 10 },
        splitLine: { lineStyle: { color: ct.grid } },
      },
      series: [
        {
          name: '策略',
          type: 'line',
          data: rows.map(r => r.strategy_nav),
          symbol: 'none',
          lineStyle: { color: '#2563eb', width: 2 },
        },
        {
          name: '候选池基准',
          type: 'line',
          data: rows.map(r => r.benchmark_nav),
          symbol: 'none',
          lineStyle: { color: '#64748b', type: 'dashed', width: 1.2 },
        },
      ],
    }
  }, [result?.nav, ct])
  return <MiniChart option={option} deps={[result?.nav?.length ?? 0, ct]} />
}

function latestModel(models: StockFormerModel[], selectedModelId: string, latestModelId?: string | null) {
  return (
    models.find(m => m.model_id === selectedModelId)
    ?? models.find(m => m.model_id === latestModelId)
    ?? models[0]
    ?? null
  )
}

export function StockFormer() {
  const qc = useQueryClient()
  const [selectedModelId, setSelectedModelId] = useState('')
  const defaultTrainStart = yearsAgo(2)
  const defaultTrainEnd = dateString(new Date())
  const [trainStart, setTrainStart] = useState(defaultTrainStart)
  const [trainEnd, setTrainEnd] = useState(defaultTrainEnd)
  const [oosRatio, setOosRatio] = useState(30)
  const [maxSymbols, setMaxSymbols] = useState(1000)
  const [ridgeAlpha, setRidgeAlpha] = useState(1)
  const [targetMode, setTargetMode] = useState<'absolute' | 'industry_relative'>('industry_relative')
  const [trainingProfile, setTrainingProfile] = useState<'daily_baseline' | 'minute_enhanced'>('daily_baseline')
  const [trainStartedAt, setTrainStartedAt] = useState<number | null>(null)
  const [trainElapsedSeconds, setTrainElapsedSeconds] = useState(0)
  const [trainingJobId, setTrainingJobId] = useState<string | null>(() => (
    typeof window === 'undefined' ? null : window.sessionStorage.getItem('stockformer-training-job')
  ))
  const [oosJobId, setOosJobId] = useState<string | null>(() => (
    typeof window === 'undefined' ? null : window.sessionStorage.getItem('stockformer-oos-job')
  ))
  const [portfolioJobId, setPortfolioJobId] = useState<string | null>(() => (
    typeof window === 'undefined' ? null : window.sessionStorage.getItem('stockformer-portfolio-job')
  ))
  const [predictionLimit, setPredictionLimit] = useState(100)
  const [inferenceJobId, setInferenceJobId] = useState<string | null>(null)
  const [inferenceContext, setInferenceContext] = useState('')
  const [factorHorizon, setFactorHorizon] = useState(3)
  const [oosPositions, setOosPositions] = useState(2)
  const [holdingDays, setHoldingDays] = useState(3)
  const [factorRequested, setFactorRequested] = useState(false)
  const [oosRequested, setOosRequested] = useState(false)

  const modelsQ = useQuery({
    queryKey: QK.stockformerModels,
    queryFn: () => api.stockformerModels(),
    staleTime: 30_000,
  })
  const models = modelsQ.data?.models ?? []

  useEffect(() => {
    if (!selectedModelId && modelsQ.data?.latest_model_id) setSelectedModelId(modelsQ.data.latest_model_id)
  }, [modelsQ.data?.latest_model_id, selectedModelId])

  const model = latestModel(models, selectedModelId, modelsQ.data?.latest_model_id)
  const modelId = model?.model_id ?? selectedModelId
  const frozenForward = Boolean(model?.frozen_signal_policy_file || model?.frozen_signal_policy?.version)

  const predictionQ = useQuery({
    queryKey: QK.stockformerPredictions(modelId, undefined, predictionLimit),
    queryFn: () => api.stockformerPredict({ model_id: modelId || null, limit: predictionLimit }),
    enabled: false,
    staleTime: 60_000,
  })

  const multifactorQ = useQuery({
    queryKey: QK.stockformerMultifactor(modelId, undefined, predictionLimit),
    queryFn: () => api.stockformerMultifactor({ model_id: modelId || null, limit: predictionLimit }),
    enabled: false,
    staleTime: 60_000,
  })

  const inferenceJobQ = useQuery({
    queryKey: QK.pipelineJob(inferenceJobId ?? ''),
    queryFn: () => api.pipelineJob(inferenceJobId!),
    enabled: Boolean(inferenceJobId),
    refetchInterval: q => {
      const job = q.state.data
      return job && (job.status === 'succeeded' || job.status === 'failed') ? false : 1_000
    },
  })
  const inferenceM = useMutation({
    mutationFn: (payload: { model_id: string; limit: number }) => api.stockformerInferenceJob(payload),
    onSuccess: data => setInferenceJobId(data.job_id),
  })
  const inferenceJob = inferenceJobQ.data
  const inferenceRunning = inferenceM.isPending || inferenceJob?.status === 'pending' || inferenceJob?.status === 'running'
  useEffect(() => {
    const context = `${modelId}:${predictionLimit}`
    if (!modelId || inferenceRunning || inferenceContext === context) return
    setInferenceContext(context)
    inferenceM.mutate({ model_id: modelId, limit: predictionLimit })
  }, [modelId, predictionLimit, inferenceRunning, inferenceContext])
  useEffect(() => {
    if (inferenceJob?.status !== 'succeeded') return
    const result = inferenceJob.result as unknown as {
      model_id: string; limit: number; predictions: StockFormerPredictionResult; multifactor: StockFormerMultiFactorResult
    }
    if (!result?.predictions || !result?.multifactor) return
    qc.setQueryData(QK.stockformerPredictions(result.model_id, undefined, result.limit), result.predictions)
    qc.setQueryData(QK.stockformerMultifactor(result.model_id, undefined, result.limit), result.multifactor)
  }, [inferenceJob, qc])

  const stockSymbols = useMemo(() => {
    const symbols = new Set<string>()
    for (const row of predictionQ.data?.rows ?? []) if (row.symbol) symbols.add(row.symbol)
    for (const row of multifactorQ.data?.rows ?? []) if (row.symbol) symbols.add(row.symbol)
    return [...symbols].sort()
  }, [predictionQ.data?.rows, multifactorQ.data?.rows])

  const stockNameSymbols = useMemo(() => {
    const symbols = new Set<string>()
    for (const symbol of stockSymbols) {
      symbols.add(symbol)
      const compact = compactSymbol(symbol)
      if (compact) symbols.add(compact)
    }
    return [...symbols].sort()
  }, [stockSymbols])

  const stockNamesQ = useQuery({
    queryKey: ['stockformer-stock-names', stockNameSymbols.join(',')],
    queryFn: () => api.instrumentNames(stockNameSymbols),
    enabled: stockNameSymbols.length > 0,
    staleTime: 24 * 60 * 60 * 1000,
  })
  const stockNames = stockNamesQ.data?.names ?? {}

  const factorQ = useQuery({
    queryKey: QK.stockformerFactor(modelId, factorHorizon),
    queryFn: () => api.stockformerFactorAnalysis({ model_id: modelId || null, horizon: factorHorizon, n_groups: 5 }),
    enabled: Boolean(modelId) && factorRequested,
    staleTime: 60_000,
  })

  const oosJobQ = useQuery({
    queryKey: QK.pipelineJob(oosJobId ?? ''),
    queryFn: () => api.pipelineJob(oosJobId!),
    enabled: Boolean(oosJobId),
    refetchInterval: (q: any) => {
      const job = q.state.data
      return job && (job.status === 'succeeded' || job.status === 'failed') ? false : 1_000
    },
  })
  const oosM = useMutation({
    mutationFn: () => api.stockformerStrictOos({ model_id: modelId || null, max_positions: oosPositions, holding_days: frozenForward ? 5 : holdingDays }),
    onSuccess: data => {
      setOosRequested(true)
      setOosJobId(data.job_id)
      window.sessionStorage.setItem('stockformer-oos-job', data.job_id)
    },
  })
  const globalRiskQ = useQuery({
    queryKey: QK.globalRisk,
    queryFn: () => api.shortTermAIGlobalRisk(),
    staleTime: 5 * 60_000,
    refetchInterval: 15 * 60_000,
  })
  const transformerQ = useQuery({
    queryKey: ['stockformer-transformer-experiment'],
    queryFn: api.stockformerTransformerLatest,
    staleTime: 30_000,
  })
  const comparisonQ = useQuery({
    queryKey: ['stockformer-model-comparison'],
    queryFn: api.stockformerModelComparisonLatest,
    staleTime: 30_000,
  })
  const trainingJobQ = useQuery({
    queryKey: QK.pipelineJob(trainingJobId ?? ''),
    queryFn: () => api.pipelineJob(trainingJobId!),
    enabled: Boolean(trainingJobId),
    refetchInterval: (q: any) => {
      const job = q.state.data
      return job && (job.status === 'succeeded' || job.status === 'failed') ? false : 1_000
    },
  })
  const trainingJob = trainingJobQ.data
  const oosJob = oosJobQ.data
  const completedOosResult = oosJob?.result as unknown as StockFormerStrictOOSResult | null
  const oosResult = oosJob?.status === 'succeeded'
    && (completedOosResult?.model_id ?? completedOosResult?.model?.model_id) === modelId
    ? (oosJob.result as unknown as StockFormerStrictOOSResult)
    : undefined
  const oosRunning = oosM.isPending || oosJob?.status === 'pending' || oosJob?.status === 'running'
  const portfolioJobQ = useQuery({
    queryKey: QK.pipelineJob(portfolioJobId ?? ''),
    queryFn: () => api.pipelineJob(portfolioJobId!),
    enabled: Boolean(portfolioJobId),
    refetchInterval: (q: any) => {
      const job = q.state.data
      return job && (job.status === 'succeeded' || job.status === 'failed') ? false : 1_000
    },
  })
  const portfolioJob = portfolioJobQ.data
  const completedPortfolioResult = portfolioJob?.result as unknown as StockFormerPortfolioResearchResult | null
  const portfolioResult = portfolioJob?.status === 'succeeded'
    && (completedPortfolioResult?.model_id ?? completedPortfolioResult?.model?.model_id) === modelId
    ? (portfolioJob.result as unknown as StockFormerPortfolioResearchResult)
    : undefined
  const portfolioM = useMutation({
    mutationFn: () => api.stockformerPortfolioResearch({ model_id: modelId || null, max_positions: oosPositions }),
    onSuccess: data => {
      setPortfolioJobId(data.job_id)
      window.sessionStorage.setItem('stockformer-portfolio-job', data.job_id)
    },
  })
  const portfolioRunning = portfolioM.isPending || portfolioJob?.status === 'pending' || portfolioJob?.status === 'running'
  const comparisonM = useMutation({
    mutationFn: () => api.stockformerModelComparisonRun({
      start: trainStart || null,
      end: trainEnd || null,
      max_symbols: Math.min(maxSymbols, 2000),
      sample_out_ratio: oosRatio / 100,
    }),
    onSuccess: data => {
      if (!data.error) qc.invalidateQueries({ queryKey: ['stockformer-model-comparison'] })
    },
  })
  const transformerM = useMutation({
    mutationFn: () => api.stockformerTransformerRun({
      start: trainStart || null,
      end: trainEnd || null,
      max_symbols: Math.min(maxSymbols, 1000),
      epochs: 8,
    }),
    onSuccess: data => {
      if (!data.error) qc.invalidateQueries({ queryKey: ['stockformer-transformer-experiment'] })
    },
  })

  const trainingConfigError = useMemo(() => {
    if (!trainStart || !trainEnd) return '请选择完整的训练日期。'
    if (trainStart >= trainEnd) return '开始日期必须早于结束日期。'
    if (!Number.isFinite(oosRatio) || oosRatio < 10 || oosRatio > 60) return '样本外比例需在10%到60%之间。'
    if (!Number.isFinite(maxSymbols) || maxSymbols < 20 || maxSymbols > 2000) return '股票上限需在20到2000之间。'
    if (!Number.isFinite(ridgeAlpha) || ridgeAlpha < 0) return '正则强度不能小于0。'
    return null
  }, [maxSymbols, oosRatio, ridgeAlpha, trainEnd, trainStart])

  const trainM = useMutation({
    mutationFn: () => api.stockformerTrain({
      start: trainStart || null,
      end: trainEnd || null,
      sample_out_ratio: oosRatio / 100,
      max_symbols: maxSymbols,
      ridge_alpha: ridgeAlpha,
      target_mode: targetMode,
      training_profile: trainingProfile,
    }),
    onMutate: () => {
      setTrainStartedAt(Date.now())
      setTrainElapsedSeconds(0)
    },
    onSuccess: data => {
      setTrainingJobId(data.job_id)
      window.sessionStorage.setItem('stockformer-training-job', data.job_id)
    },
    onError: () => setTrainStartedAt(null),
  })
  const trainingRunning = trainM.isPending
    || trainingJob?.status === 'pending'
    || trainingJob?.status === 'running'

  useEffect(() => {
    if (!trainingRunning || trainStartedAt == null) return
    const updateElapsed = () => setTrainElapsedSeconds(Math.max(0, Math.floor((Date.now() - trainStartedAt) / 1000)))
    updateElapsed()
    const timer = window.setInterval(updateElapsed, 1000)
    return () => window.clearInterval(timer)
  }, [trainStartedAt, trainingRunning])

  useEffect(() => {
    if (!trainingJob || (trainingJob.status !== 'succeeded' && trainingJob.status !== 'failed')) return
    setTrainStartedAt(null)
    window.sessionStorage.removeItem('stockformer-training-job')
    if (trainingJob.status !== 'succeeded') return
    setFactorRequested(false)
    setOosRequested(false)
    setOosJobId(null)
    window.sessionStorage.removeItem('stockformer-oos-job')
    setPortfolioJobId(null)
    window.sessionStorage.removeItem('stockformer-portfolio-job')
    const trainedModel = trainingJob.result?.model
    if (trainedModel?.model_id) setSelectedModelId(trainedModel.model_id)
    qc.invalidateQueries({ queryKey: QK.stockformerModels })
    qc.invalidateQueries({ queryKey: ['stockformer-predictions'] })
    qc.invalidateQueries({ queryKey: ['stockformer-multifactor'] })
    qc.invalidateQueries({ queryKey: ['stockformer-factor'] })
    qc.invalidateQueries({ queryKey: ['pipeline-job'] })
  }, [qc, trainingJob])

  const resetTrainingConfig = () => {
    setTrainStart(defaultTrainStart)
    setTrainEnd(defaultTrainEnd)
    setOosRatio(30)
    setMaxSymbols(1000)
    setRidgeAlpha(1)
    setTargetMode('absolute')
    setTrainingProfile('daily_baseline')
  }

  const selectedModelOptions = (
    <div className="flex w-full min-w-0 items-center gap-2 sm:w-auto">
      <select
        value={modelId}
        onChange={e => setSelectedModelId(e.target.value)}
        className="h-8 min-w-0 flex-1 sm:flex-none sm:max-w-[360px] rounded-input border border-border bg-surface px-2 text-xs outline-none focus:border-accent"
      >
        {models.length === 0 && <option value="">暂无模型</option>}
        {models.map(m => (
          <option key={m.model_id} value={m.model_id}>{m.model_name} · {m.training_profile === 'minute_enhanced' ? '分钟增强' : '日K基线'} · {m.trained_at}</option>
        ))}
      </select>
      <button
        type="button"
        onClick={() => {
          modelsQ.refetch()
          inferenceM.mutate({ model_id: modelId, limit: predictionLimit })
          globalRiskQ.refetch()
        }}
        disabled={inferenceRunning || !modelId}
        className="inline-flex h-8 shrink-0 items-center gap-1.5 rounded-btn border border-border bg-surface px-3 text-xs text-secondary hover:bg-elevated hover:text-foreground disabled:opacity-60"
      >
        {inferenceRunning ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />}
        刷新
      </button>
    </div>
  )

  const topPrediction = predictionQ.data?.rows?.[0]
  const topMulti = multifactorQ.data?.rows?.[0]
  const oosStats = oosResult?.stats
  const sectorAblation = oosResult?.sector_ablation
  const pass = oosResult?.diagnosis?.passed
  const oosChecks = oosResult?.diagnosis?.checks ?? []
  const transformer = transformerM.data?.experiment ?? transformerQ.data?.experiment
  const trainingError = trainM.error?.message
    ?? trainingJobQ.error?.message
    ?? (trainingJob?.status === 'failed' ? trainingJob.error : null)
  const trainingModel = trainingJob?.status === 'succeeded' ? trainingJob.result?.model : null
  const trainingLog = trainingJob?.log?.[trainingJob.log.length - 1]

  return (
    <div className="min-h-full bg-base flex flex-col">
      <PageHeader
        title="StockFormer"
        subtitle="模型训练、样本外验证与预测因子管理"
        right={selectedModelOptions}
        className="shrink-0 flex-wrap bg-base/95 [&>div:first-child]:pl-8 sm:[&>div:first-child]:pl-0"
      />

      <main className="flex-1 space-y-4 px-3 pb-4 pt-3 lg:px-4">
        <ErrorLine error={inferenceM.error?.message || inferenceJobQ.error?.message || (inferenceJob?.status === 'failed' ? inferenceJob.error : null)} />
        {inferenceRunning && <div role="status" className="flex items-center gap-2 text-xs text-secondary"><Loader2 className="h-4 w-4 animate-spin" />{inferenceJob?.log?.[inferenceJob.log.length - 1]?.msg ?? '准备预测任务'} · {inferenceJob?.progress ?? 0}%</div>}
        <section className="grid gap-2 md:grid-cols-2 xl:grid-cols-5" aria-label="模型状态概览">
          <Metric label="当前模型" value={model?.model_name ?? '尚未训练'} />
          <Metric label="最后训练" value={model?.trained_at ?? '--'} />
          <Metric label="特征版本" value={model?.feature_version ?? '--'} />
          <Metric label="样本外超额" value={oosStats ? pctPoint(oosStats.excess_return) : '--'} tone={toneByNumber(oosStats?.excess_return)} />
          <Metric label="模型数量" value={`${models.length} 个`} />
        </section>

        <div className="flex flex-wrap items-center gap-x-4 gap-y-2 rounded-card border border-border/70 bg-surface/60 px-3 py-2 text-[11px] text-secondary">
          <span className="font-medium text-foreground">当前信号</span>
          <span className={cn('font-semibold', predictionQ.data?.signal_status === 'no_signal' ? 'text-bear' : 'text-bull')}>{predictionQ.data?.signal_label ?? '等待预测'}</span>
          <span className="text-border">|</span>
          <span>预测截面：{crossSectionHint(predictionQ.data)}</span>
          <span className="text-border">|</span>
          <span>验证期限：{oosResult?.target_horizon ? `${oosResult.target_horizon}日` : '--'}</span>
          <span className="text-border">|</span>
          <span>Top预测：{topPrediction ? stockDisplay(topPrediction.symbol, stockNameOf(stockNames, topPrediction.symbol)) : '--'}</span>
          <span className="text-border">|</span>
          <span>综合排名：{topMulti ? `${stockDisplay(topMulti.symbol, stockNameOf(stockNames, topMulti.symbol))} / ${fmt(topMulti.final_score, 1)}` : '--'}</span>
        </div>

        <StockFormerDailyShadow />

        <GlobalRiskPanel
          risk={globalRiskQ.data}
          loading={globalRiskQ.isFetching}
          onRefresh={() => globalRiskQ.refetch()}
        />

        <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,0.95fr)_minmax(0,1.05fr)]">
          <section className={cn(panelCls, 'order-2')}>
            <SectionTitle icon={GitCompare} title="XGBoost / LightGBM 对照实验" hint="研究区 · 不替换生产模型" />
            <div className="mb-3 text-xs leading-relaxed text-secondary">
              同一数据、同一切分、同一特征和同一3日交易规则，对比单模型与融合模型；结果不自动替换当前模型。
            </div>
            <div className="mb-3 flex items-center gap-2">
              <button
                type="button"
                disabled={comparisonM.isPending}
                onClick={() => comparisonM.mutate()}
                className="inline-flex h-9 items-center gap-2 rounded-btn border border-accent/40 bg-accent/10 px-4 text-xs font-semibold text-accent hover:bg-accent/15 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {comparisonM.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <GitCompare className="h-4 w-4" />}
                运行三模型对照
              </button>
              <ErrorLine error={comparisonM.data?.error || comparisonM.error?.message || comparisonQ.data?.error} />
            </div>
            {comparisonQ.data?.report && (
              <DataTable
                rows={Object.entries(comparisonQ.data.report.models).map(([name, value]) => ({ name, ...value }))}
                columns={[
                  { key: 'name', label: '模型' },
                  { key: 'rank1', label: '1日Rank IC', render: r => fmt(r.prediction_metrics?.['1d']?.rank_ic, 4) },
                  { key: 'rank3', label: '3日Rank IC', render: r => fmt(r.prediction_metrics?.['3d']?.rank_ic, 4) },
                  { key: 'trade3', label: '3日交易模型IC', render: r => fmt(r.prediction_metrics?.['3d']?.trade_model?.rank_ic, 4) },
                  { key: 'rank5', label: '5日Rank IC', render: r => fmt(r.prediction_metrics?.['5d']?.rank_ic, 4) },
                  { key: 'recent1', label: '近3月1日IC', render: r => fmt(r.recent_3m_metrics?.['1d']?.rank_ic, 4) },
                  { key: 'recent3', label: '近3月3日IC', render: r => fmt(r.recent_3m_metrics?.['3d']?.rank_ic, 4) },
                  { key: 'recent5', label: '近3月5日IC', render: r => fmt(r.recent_3m_metrics?.['5d']?.rank_ic, 4) },
                  { key: 'top2', label: 'Top2可成交净收益', render: r => pctPoint(r.top2_after_cost_return) },
                  { key: 'return', label: '策略收益', render: r => pctPoint(r.strategy?.strategy_return) },
                  { key: 'excess', label: '超额收益', render: r => pctPoint(r.strategy?.excess_return) },
                  { key: 'dd', label: '最大回撤', render: r => pctPoint(r.strategy?.max_drawdown) },
                  { key: 'signals', label: '信号数/胜率', render: r => `${fmt(r.strategy?.signal_count, 0)} / ${pctPoint(r.strategy?.win_rate)}` },
                  { key: 'days', label: '有效日/空仓日', render: r => `${fmt(r.strategy?.eligible_days, 0)} / ${fmt(r.strategy?.no_signal_days, 0)}` },
                  { key: 'activation', label: '准入', render: r => r.promotion_check?.allow_simulation ? '可模拟' : '无信号' },
                ]}
              />
            )}
          </section>

          <section className={cn(panelCls, 'order-1 xl:col-span-2')}>
            <SectionTitle icon={BrainCircuit} title="训练工作台" hint={modelsQ.isFetching ? '同步中' : `${models.length} 个模型`} />
            <div className="mb-4 flex flex-wrap items-center justify-between gap-3 rounded-card border border-border/70 bg-base/40 px-3 py-3">
              <div>
                <div className="text-sm font-semibold text-foreground">训练新的 StockFormer 模型</div>
                <div className="mt-1 text-[11px] text-secondary">训练完成后会自动刷新预测、因子分析和样本外验证；历史模型不会被覆盖。</div>
              </div>
              <div className="flex items-center gap-2 text-[10px] text-muted">
                <span>当前模型：{model?.model_name ?? '暂无'}</span>
                <span className="text-border">·</span>
                <span>{trainingProfile === 'minute_enhanced' ? '分钟增强' : '日K基线'}</span>
              </div>
            </div>
            <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1.35fr)_minmax(0,0.65fr)]">
              <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
              <label className="text-[11px] text-secondary">
                开始日期
                <input value={trainStart} onChange={e => setTrainStart(e.target.value)} type="date" className={cn(inputCls, 'mt-1')} />
              </label>
              <label className="text-[11px] text-secondary">
                结束日期
                <input value={trainEnd} onChange={e => setTrainEnd(e.target.value)} type="date" className={cn(inputCls, 'mt-1')} />
              </label>
              <label className="text-[11px] text-secondary">
                样本外%
                <input value={oosRatio} onChange={e => setOosRatio(Number(e.target.value))} type="number" min={10} max={60} className={cn(inputCls, 'mt-1')} />
              </label>
              <label className="text-[11px] text-secondary">
                股票上限
                <input value={maxSymbols} onChange={e => setMaxSymbols(Number(e.target.value))} type="number" min={20} max={2000} className={cn(inputCls, 'mt-1')} />
              </label>
              <label className="text-[11px] text-secondary">
                正则强度
                <input value={ridgeAlpha} onChange={e => setRidgeAlpha(Number(e.target.value))} type="number" min={0} step={0.1} className={cn(inputCls, 'mt-1')} />
              </label>
              <label className="text-[11px] text-secondary">
                收益目标
                <select value={targetMode} onChange={e => setTargetMode(e.target.value as 'absolute' | 'industry_relative')} className={cn(inputCls, 'mt-1')}>
                  <option value="absolute">绝对收益（主模型）</option>
                  <option value="industry_relative">行业横截面超额收益</option>
                </select>
              </label>
              <label className="text-[11px] text-secondary">
                训练数据层
                <select value={trainingProfile} onChange={e => setTrainingProfile(e.target.value as 'daily_baseline' | 'minute_enhanced')} className={cn(inputCls, 'mt-1')}>
                  <option value="daily_baseline">日K基线（不依赖分钟）</option>
                  <option value="minute_enhanced">分钟增强（至少228根，QMT稀疏日可用）</option>
                </select>
              </label>
              </div>
              <div className="space-y-2 rounded-card border border-border/70 bg-surface/45 p-3">
                <div className="text-[11px] font-medium text-foreground">本次训练摘要</div>
                <div className="grid gap-1.5 text-[11px] text-secondary">
                  <div className="flex justify-between gap-3"><span>训练区间</span><span className="font-mono text-foreground">{trainStart || '--'} 至 {trainEnd || '--'}</span></div>
                  <div className="flex justify-between gap-3"><span>样本外比例</span><span className="font-mono text-foreground">{Number.isFinite(oosRatio) ? `${oosRatio}%` : '--'}</span></div>
                  <div className="flex justify-between gap-3"><span>股票上限</span><span className="font-mono text-foreground">{Number.isFinite(maxSymbols) ? maxSymbols : '--'} 只</span></div>
                  <div className="flex justify-between gap-3"><span>收益目标</span><span className="text-foreground">{targetMode === 'absolute' ? '绝对收益' : '行业中性'}</span></div>
                  <div className="flex justify-between gap-3"><span>数据层</span><span className="text-foreground">{trainingProfile === 'minute_enhanced' ? '分钟增强' : '日K基线'}</span></div>
                </div>
                {trainingConfigError && <div className="mt-2 rounded border border-warning/20 bg-warning/10 px-2 py-1.5 text-[10px] leading-4 text-warning">{trainingConfigError}</div>}
                {!trainingConfigError && trainingProfile === 'minute_enhanced' && <div className="mt-2 rounded border border-accent/15 bg-accent/5 px-2 py-1.5 text-[10px] leading-4 text-secondary">分钟增强使用至少228根且覆盖率不低于95%的样本；QMT的238/239根稀疏日可用，明显缺失的股票日会被排除。</div>}
              </div>
            </div>
            <div className="mt-4 grid grid-cols-1 gap-3 xl:grid-cols-[auto_minmax(0,1fr)] xl:items-start">
              <div className="flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  disabled={trainingRunning || Boolean(trainingConfigError)}
                  onClick={() => trainM.mutate()}
                  className="inline-flex h-9 items-center gap-2 rounded-btn bg-accent px-4 text-xs font-semibold text-white shadow-sm hover:bg-accent/90 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {trainingRunning ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                  {trainingRunning ? '训练中…' : '开始训练'}
                </button>
                <button
                  type="button"
                  disabled={trainingRunning}
                  onClick={resetTrainingConfig}
                  className="inline-flex h-9 items-center gap-1.5 rounded-btn border border-border bg-surface px-3 text-xs text-secondary hover:bg-elevated hover:text-foreground disabled:cursor-not-allowed disabled:opacity-60"
                >
                  <RotateCcw className="h-3.5 w-3.5" />
                  恢复默认
                </button>
              </div>
              <TrainingState
                pending={trainingRunning}
                elapsedSeconds={trainElapsedSeconds}
                error={trainingError}
                model={trainingModel}
                stage={trainingJob?.stage}
                progress={trainingJob?.progress}
                message={trainingLog?.msg}
              />
            </div>
            <div className="mt-4 grid gap-2 md:grid-cols-3">
              <Metric label="样本内" value={model ? `${model.train_start} 至 ${model.train_end}` : '--'} />
              <Metric label="样本外" value={model ? `${model.oos_start} 至 ${model.oos_end}` : '--'} />
              <Metric label="训练耗时" value={model ? `${fmt(model.elapsed_ms, 0)} ms` : '--'} />
              <Metric label="训练行数" value={model ? fmt(model.n_rows_train, 0) : '--'} />
              <Metric label="样本外行数" value={model ? fmt(model.n_rows_oos, 0) : '--'} />
              <Metric label="股票数" value={model ? fmt(model.n_symbols, 0) : '--'} />
              <Metric label="训练层" value={model?.training_profile === 'minute_enhanced' ? '分钟增强' : model?.training_profile === 'daily_baseline' ? '日K基线' : '--'} />
              <Metric label="分钟有效行" value={model?.training_coverage ? fmt(model.training_coverage.ready_rows, 0) : '--'} />
            </div>
            {model?.validation && (
              <div className="mt-4 space-y-2">
                <SectionTitle icon={GitCompare} title="模型优化对照" hint="研究候选" />
                <div className="text-xs text-secondary">
                  验证集 {model.validation.start} 至 {model.validation.end_exclusive} 前，标签跨界样本已隔离
                </div>
                <DataTable
                  rows={Object.entries(model.holdout_comparison ?? {}).map(([horizon, value]) => ({
                    horizon, ...value, choice: model.validation?.horizons[horizon],
                  }))}
                  columns={[
                    { key: 'horizon', label: '周期' },
                    { key: 'choice', label: '验证集选择', render: r => `${r.choice?.family === 'compact' ? '精简量价' : '完整多因子'} / ${r.choice?.feature_count ?? '--'}项` },
                    { key: 'alpha', label: '正则', render: r => fmt(r.choice?.alpha, 0) },
                    { key: 'before', label: '基线留出RankIC', render: r => fmt(r.baseline?.rank_ic, 4) },
                    { key: 'after', label: '候选留出RankIC', render: r => fmt(r.selected?.rank_ic, 4) },
                    { key: 'top', label: 'Top2标签均值(扣费假设)', render: r => pct(r.selected?.top2_after_cost_proxy) },
                  ]}
                />
                <details className="text-xs text-muted">
                  <summary className="cursor-pointer">训练口径与限制</summary>
                  <ul className="mt-2 space-y-1 list-disc pl-4">
                {model.training_notes?.map(note => <li key={note}>{note}</li>)}
              </ul>
              {model.rolling_ablation?.windows?.length ? (
                <div className="mt-3 rounded border border-border/60 bg-base/40 p-3">
                  <div className="text-xs font-medium text-foreground">滚动消融实验（仅样本外前历史）</div>
                  <div className="mt-1 text-[10px] text-muted">目标：{model.target_mode === 'industry_relative' ? '行业横截面超额收益' : '绝对收益'}；仅用于滚动对照，不参与自动激活。</div>
                  <div className="mt-2 grid gap-1 text-[10px] text-secondary">
                    {model.rolling_ablation.windows.map((window, index) => {
                      const families = Object.entries(window.families ?? {})
                      const best = families
                        .map(([name, metrics]) => ({ name, score: Number(metrics?.['3d']?.rank_ic ?? -Infinity) }))
                        .sort((a, b) => b.score - a.score)[0]
                      return <div key={`${window.train_months}-${index}`}>{window.train_months}个月训练 / {window.test_months}个月测试：最佳 {best?.name ?? '--'}，3日Rank IC {Number.isFinite(best?.score) ? best!.score.toFixed(4) : '--'}</div>
                    })}
                  </div>
                </div>
              ) : null}
                </details>
                {model.recent_oos_validation && (
                  <div className="mt-3 rounded border border-border/60 bg-base/40 p-3">
                    <div className="text-xs font-medium text-foreground">最近3个月独立验证</div>
                    <div className="mt-1 text-[10px] text-muted">
                      {model.recent_oos_validation.start ?? '--'} 至 {model.recent_oos_validation.end ?? '--'}，
                      {model.recent_oos_validation.test_trading_days ?? '--'} 个交易日；不参与调参和自动激活。
                    </div>
                    <div className="mt-2 grid gap-1 text-[10px] text-secondary">
                      {Object.entries(model.recent_oos_validation.metrics ?? {}).map(([horizon, metrics]) => (
                        <div key={horizon}>{horizon} Rank IC {fmt(metrics.rank_ic, 4)}，扣费后Top2 {pct(metrics.top2_after_cost_proxy)}</div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
            {model && (
              <div className="mt-3 grid gap-2 text-[10px] text-muted">
                <div className="flex items-center gap-2 rounded bg-base/50 px-2 py-1"><FileJson className="h-3.5 w-3.5" /><span className="truncate">{model.model_file}</span></div>
                <div className="flex items-center gap-2 rounded bg-base/50 px-2 py-1"><Database className="h-3.5 w-3.5" /><span className="truncate">{model.factor_file}</span></div>
                <div className="flex items-center gap-2 rounded bg-base/50 px-2 py-1"><Clock className="h-3.5 w-3.5" /><span className="truncate">{model.prediction_file}</span></div>
              </div>
            )}
          </section>

          <section className={cn(panelCls, 'order-3')}>
            <SectionTitle icon={GitCompare} title="Transformer 独立实验" hint="研究区 · 不替换生产模型" />
            <div className="mb-3 text-xs leading-relaxed text-secondary">
              使用20日序列预测未来5日收益，模型文件和指标独立保存，不覆盖XGBoost生产模型。
            </div>
            <div className="mb-3 flex items-center gap-2">
              <button
                type="button"
                disabled={transformerM.isPending}
                onClick={() => transformerM.mutate()}
                className="inline-flex h-9 items-center gap-2 rounded-btn border border-accent/40 bg-accent/10 px-4 text-xs font-semibold text-accent hover:bg-accent/15 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {transformerM.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                训练 Transformer
              </button>
              <ErrorLine error={transformerM.data?.error || transformerM.error?.message || transformerQ.data?.error} />
            </div>
            <div className="grid gap-2 md:grid-cols-3">
              <Metric label="样本外区间" value={transformer ? `${transformer.oos_start} 至 ${transformer.oos_end}` : '--'} />
              <Metric label="方向准确率" value={transformer ? pct(transformer.oos_direction_accuracy) : '--'} />
              <Metric label="样本外 IC" value={fmt(transformer?.oos_ic, 4)} tone={toneByNumber(transformer?.oos_ic)} />
              <Metric label="简单基线准确率" value={transformer ? pct(transformer.baseline_direction_accuracy) : '--'} />
              <Metric label="训练/样本外" value={transformer ? `${fmt(transformer.train_samples, 0)} / ${fmt(transformer.oos_samples, 0)}` : '--'} />
              <Metric label="训练耗时" value={transformer ? `${fmt(transformer.elapsed_ms, 0)} ms` : '--'} />
            </div>
            {transformer && (
              <div className="mt-3 grid gap-2 text-[10px] text-muted">
                <div className="flex items-center gap-2 rounded bg-base/50 px-2 py-1"><FileJson className="h-3.5 w-3.5" /><span className="truncate">{transformer.model_file}</span></div>
                <div className="flex items-center gap-2 rounded bg-base/50 px-2 py-1"><Database className="h-3.5 w-3.5" /><span className="truncate">{transformer.metrics_file}</span></div>
              </div>
            )}
          </section>

          <section className={cn(panelCls, 'order-4')}>
            <SectionTitle icon={Trophy} title={oosResult?.rolling ? '冻结策略前向验证' : '样本外标签诊断'} hint={oosResult?.rolling ? (oosResult.activation_decision?.eligible ? '待人工激活' : '研究观察') : pass == null ? '待验证' : pass ? '通过门槛' : '未通过'} />
            <ErrorLine error={oosResult?.error || (oosJob?.status === 'failed' ? oosJob.error : null) || oosM.error?.message} />
            {oosResult?.evaluation_warning && <div role="status" className="mb-3 text-xs leading-relaxed text-warning">{oosResult.evaluation_warning}</div>}
            <CoverageWarning coverage={oosResult?.data_coverage} />
            <ForwardValidation result={oosResult} />
            {!oosResult?.rolling && (
            <div className="grid gap-2 md:grid-cols-4">
              <Metric label="标签收益估算" value={oosStats ? pctPoint(oosStats.strategy_return) : '--'} tone={toneByNumber(oosStats?.strategy_return)} />
              <Metric label="候选池基准" value={oosStats ? pctPoint(oosStats.benchmark_return) : '--'} tone={toneByNumber(oosStats?.benchmark_return)} />
              <Metric label="超额收益" value={oosStats ? pctPoint(oosStats.excess_return) : '--'} tone={toneByNumber(oosStats?.excess_return)} />
              <Metric label="估算回撤" value={oosStats ? pctPoint(oosStats.max_drawdown) : '--'} tone={toneByNumber(oosStats?.max_drawdown)} />
              <Metric label="信号日上涨率" value={oosStats ? pctPoint(oosStats.win_rate) : '--'} />
              <Metric label="盈亏比" value={fmt(oosStats?.profit_loss_ratio, 2)} />
              <Metric label="候选信号条数" value={fmt(oosStats?.trade_count, 0)} />
              <Metric label="周期换手假设" value={oosStats ? pctPoint(oosStats.turnover_rate) : '--'} />
              <Metric label="买入确认分" value={fmt(oosStats?.avg_buy_confirmation_score, 1)} />
              <Metric label="平均仓位系数" value={fmt(oosStats?.avg_position_multiplier, 2)} />
              <Metric label="弱市过滤日" value={fmt(oosStats?.market_closed_days, 0)} />
            </div>
            )}
            {oosResult?.layer_diagnostics && (
              <div className="mt-3 rounded-card border border-border/70 bg-base/40 p-3">
                <div className="mb-2 text-[11px] font-medium text-foreground">过滤漏斗</div>
                <div className="grid gap-2 md:grid-cols-4">
                  <Metric label="样本外总行" value={fmt(oosResult.layer_diagnostics.total_rows, 0)} />
                  <Metric label="预测通过" value={`${fmt(oosResult.layer_diagnostics.prediction_pass.rows, 0)} / ${fmt(oosResult.layer_diagnostics.prediction_pass.dates, 0)}日`} />
                  <Metric label="入场规则通过" value={`${fmt(oosResult.layer_diagnostics.entry_rules_pass.rows, 0)} / ${fmt(oosResult.layer_diagnostics.entry_rules_pass.dates, 0)}日`} />
                  <Metric label="市场闸门通过" value={`${fmt(oosResult.layer_diagnostics.market_gate_pass.rows, 0)} / ${fmt(oosResult.layer_diagnostics.market_gate_pass.dates, 0)}日`} />
                  <Metric label="预测+入场" value={fmt(oosResult.layer_diagnostics.cumulative.prediction_and_entry, 0)} />
                  <Metric label="预测+闸门" value={fmt(oosResult.layer_diagnostics.cumulative.prediction_and_market, 0)} />
                  <Metric label="最终可交易" value={fmt(oosResult.layer_diagnostics.cumulative.prediction_entry_market, 0)} />
                </div>
              </div>
            )}
            {oosResult?.threshold_ablation && (
              <div className="mt-3 rounded-card border border-accent/15 bg-accent/5 p-3">
                <div className="mb-2 text-[11px] font-medium text-foreground">同模型阈值消融</div>
                <DataTable
                  rows={Object.entries(oosResult.threshold_ablation).map(([key, value]) => ({ key, ...value }))}
                  columns={[
                    { key: 'label', label: '方案' },
                    { key: 'forecast_floor', label: '预测门槛', render: r => fmt(r.forecast_floor, 4) },
                    { key: 'entry_score_min', label: '入场分', render: r => fmt(r.entry_score_min, 0) },
                    { key: 'market_gate_mode', label: '市场闸门' },
                    { key: 'return', label: '收益', render: r => pctPoint(r.stats?.strategy_return) },
                    { key: 'excess', label: '超额', render: r => pctPoint(r.stats?.excess_return) },
                    { key: 'dd', label: '回撤', render: r => pctPoint(r.stats?.max_drawdown) },
                    { key: 'trades', label: '信号数', render: r => fmt(r.stats?.trade_count, 0) },
                  ]}
                />
              </div>
            )}
            <div className="mt-3 rounded-card border border-border/70 bg-base/40 p-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <div className="text-[11px] font-medium text-foreground">真实组合回测</div>
                  <div className="mt-1 text-[10px] text-muted">使用真实成交撮合、佣金、印花税、滑点、持仓上限和到期平仓；研究结果不自动激活模型。</div>
                </div>
                <button
                  type="button"
                  disabled={portfolioRunning || !modelId}
                  onClick={() => portfolioM.mutate()}
                  className="inline-flex h-8 items-center gap-1.5 rounded-btn border border-accent/40 bg-accent/10 px-3 text-xs text-accent hover:bg-accent/15 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {portfolioRunning ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
                  {portfolioRunning ? '回测中…' : '运行真实组合回测'}
                </button>
              </div>
              {portfolioJob && (portfolioRunning || portfolioJob.status === 'failed') && (
                <div className="mt-2 text-[10px] text-secondary" role="status">
                  {portfolioJob.log?.[portfolioJob.log.length - 1]?.msg || portfolioJob.stage || '准备中'} · {portfolioJob.progress ?? 0}%
                </div>
              )}
              {portfolioResult?.execution_warning && <div className="mt-2 text-[10px] text-warning">{portfolioResult.execution_warning}</div>}
              {portfolioResult?.error && <ErrorLine error={portfolioResult.error} />}
              <ForwardValidation result={portfolioResult} />
              {portfolioResult?.horizons && (
                <DataTable
                  rows={Object.values(portfolioResult.horizons).map(value => ({
                    horizon: value.horizon,
                    selected: value.selected,
                    trials: value.trials,
                  }))}
                  columns={[
                    { key: 'horizon', label: '周期', render: r => `${r.horizon}日` },
                    { key: 'floor', label: '选中预测门槛', render: r => fmt(r.selected?.forecast_floor, 4) },
                    { key: 'return', label: '真实收益', render: r => pctPoint(r.selected?.stats?.total_return) },
                    { key: 'dd', label: '真实回撤', render: r => pctPoint(r.selected?.stats?.max_drawdown) },
                    { key: 'trades', label: '成交笔数', render: r => fmt(r.selected?.stats?.trade_count, 0) },
                    { key: 'win', label: '胜率', render: r => pctPoint(r.selected?.stats?.win_rate) },
                  ]}
                />
              )}
              {portfolioResult?.model_tuning && (
                <div className="mt-3 border-t border-border/60 pt-3">
                  <div className="mb-2 text-[11px] font-medium text-foreground">模型候选与确认期对比</div>
                  <div className="mb-2 text-[10px] text-muted">
                    调优期：{portfolioResult.tuning_period?.start ?? '--'} 至 {portfolioResult.tuning_period?.end ?? '--'}；
                    确认期：{portfolioResult.confirmation_period?.start ?? '--'} 至 {portfolioResult.confirmation_period?.end ?? '--'}。
                    选参只读取调优期。
                  </div>
                  <DataTable
                    rows={Object.values(portfolioResult.model_tuning).flatMap(value => (value.candidates ?? []).map(candidate => ({
                      horizon: value.horizon,
                      ...candidate,
                    })))}
                    columns={[
                      { key: 'horizon', label: '周期', render: r => `${r.horizon}日` },
                      { key: 'family', label: '特征族' },
                      { key: 'alpha', label: '正则', render: r => fmt(r.alpha, 2) },
                      { key: 'tuning_return', label: '调优收益', render: r => pctPoint(r.tuning_stats?.total_return) },
                      { key: 'tuning_dd', label: '调优回撤', render: r => pctPoint(r.tuning_stats?.max_drawdown) },
                      { key: 'confirm_return', label: '确认收益', render: r => pctPoint(r.confirmation_stats?.total_return) },
                      { key: 'confirm_dd', label: '确认回撤', render: r => pctPoint(r.confirmation_stats?.max_drawdown) },
                      { key: 'trades', label: '确认成交', render: r => fmt(r.confirmation_stats?.trade_count, 0) },
                      { key: 'selected', label: '入选', render: r => r.selected ? '是' : (r.status === 'ok' ? '否' : '失败') },
                    ]}
                  />
                  {Object.values(portfolioResult.model_tuning).some(value => value.selected) && (
                    <div className="mt-2 text-[10px] text-secondary">
                      {Object.values(portfolioResult.model_tuning).filter(value => value.selected).map(value => (
                        <span key={value.horizon} className="mr-3">
                          {value.horizon}日入选：{value.selected?.family} / alpha {fmt(value.selected?.alpha, 2)}，确认收益 {pctPoint(value.selected?.stats?.total_return)}，确认回撤 {pctPoint(value.selected?.stats?.max_drawdown)}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
            {sectorAblation?.delta && (
              <div className="mt-3 rounded-card border border-accent/15 bg-accent/5 px-3 py-2">
                <div className="mb-2 text-[11px] font-medium text-foreground">板块轮动消融对比</div>
                <div className="grid gap-2 md:grid-cols-5">
                  <Metric label="收益差" value={pctPoint(sectorAblation.delta.strategy_return)} tone={toneByNumber(sectorAblation.delta.strategy_return)} />
                  <Metric label="超额差" value={pctPoint(sectorAblation.delta.excess_return)} tone={toneByNumber(sectorAblation.delta.excess_return)} />
                  <Metric label="回撤差" value={pctPoint(sectorAblation.delta.max_drawdown)} tone={toneByNumber(sectorAblation.delta.max_drawdown)} />
                  <Metric label="胜率差" value={pctPoint(sectorAblation.delta.win_rate)} tone={toneByNumber(sectorAblation.delta.win_rate)} />
                  <Metric label="交易差" value={fmt(sectorAblation.delta.trade_count, 0)} tone={toneByNumber(sectorAblation.delta.trade_count)} />
                </div>
                {sectorAblation.note && <div className="mt-2 text-[10px] text-muted">{sectorAblation.note}</div>}
              </div>
            )}
            {oosChecks.length > 0 && (
              <div className="mt-3 grid gap-2 md:grid-cols-2 xl:grid-cols-5">
                {oosChecks.map(item => (
                  <div
                    key={item.label}
                    className={cn(
                      'rounded-card border px-3 py-2 text-[11px]',
                      item.passed === true
                        ? 'border-bull/20 bg-bull/5 text-bull'
                        : item.passed === null
                          ? 'border-warning/20 bg-warning/5 text-warning'
                          : 'border-bear/20 bg-bear/5 text-bear',
                    )}
                  >
                    <div className="font-medium text-foreground">{item.label}</div>
                    <div className="mt-1 font-mono">{item.passed === null ? '未验证' : `${item.value ?? '--'} / ${item.criteria ?? '--'}`}</div>
                  </div>
                ))}
              </div>
            )}
            <div className="mt-3 grid grid-cols-1 gap-3 md:grid-cols-[220px_minmax(0,1fr)]">
              <div className="space-y-2">
                <label className="text-[11px] text-secondary">
                  最大持仓数
                  <input value={oosPositions} onChange={e => setOosPositions(Number(e.target.value))} type="number" min={1} max={20} className={cn(inputCls, 'mt-1')} />
                </label>
                <label className="text-[11px] text-secondary">
                  {frozenForward ? '持有天数（冻结）' : '持有天数'}
                  <input value={frozenForward ? 5 : holdingDays} disabled={frozenForward} onChange={e => setHoldingDays(Number(e.target.value))} type="number" min={1} max={30} className={cn(inputCls, 'mt-1 disabled:opacity-60')} />
                </label>
                <button
                  type="button"
                  disabled={oosRunning || !modelId}
                  onClick={() => {
                    setOosRequested(true)
                    oosM.mutate()
                  }}
                  className="inline-flex h-8 w-full items-center justify-center gap-1.5 rounded-btn border border-border bg-surface px-3 text-xs text-secondary hover:bg-elevated hover:text-foreground disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {oosRunning ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />}
                  {oosRunning ? '样本外计算中…' : '重算样本外'}
                </button>
                {oosJob && (oosRunning || oosJob.status === 'failed') && (
                  <div className="rounded border border-border/70 bg-base/50 px-2 py-1.5 text-[10px] text-secondary" role="status">
                    <div className="flex items-center justify-between gap-2">
                      <span>{oosJob.log?.[oosJob.log.length - 1]?.msg || oosJob.stage || '准备中'}</span>
                      <span className="font-mono">{oosJob.progress ?? 0}%</span>
                    </div>
                    <div className="mt-1 h-1 overflow-hidden rounded bg-border/50"><div className="h-full bg-accent transition-all" style={{ width: `${Math.max(0, Math.min(100, oosJob.progress ?? 0))}%` }} /></div>
                  </div>
                )}
              </div>
              {!oosResult?.rolling && <OOSChart result={oosResult} />}
            </div>
          </section>
        </div>

        <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
          <section className={panelCls}>
            <SectionTitle icon={LineChart} title="模型预测结果" hint={crossSectionHint(predictionQ.data)} />
            <ErrorLine error={predictionQ.data?.error} />
            <div className="mb-3 flex items-center justify-between gap-3">
              <label className="flex items-center gap-2 text-[11px] text-secondary">
                展示数量
                <input value={predictionLimit} onChange={e => setPredictionLimit(Number(e.target.value))} type="number" min={20} max={5000} className="h-8 w-24 rounded-input border border-border bg-base px-2 text-xs outline-none focus:border-accent" />
              </label>
              {inferenceRunning && <Loader2 className="h-4 w-4 animate-spin text-muted" />}
            </div>
            {crossSectionNote(predictionQ.data) && (
              <div className="mb-3 rounded-card border border-accent/15 bg-accent/5 px-3 py-2 text-[11px] text-secondary">
                {crossSectionNote(predictionQ.data)}
              </div>
            )}
            <DataTable
              rows={predictionQ.data?.rows ?? []}
              columns={[
                { key: 'model_rank', label: '排名', align: 'right' },
                { key: 'symbol', label: '股票', render: r => <StockLabel symbol={r.symbol} name={stockNameOf(stockNames, r.symbol)} /> },
                {
                  key: 'actions',
                  label: '操作',
                  render: r => {
                    const name = stockNameOf(stockNames, r.symbol)
                    return (
                      <StockQuickActions
                        symbol={r.symbol}
                        name={name}
                        price={r.close}
                        watchNote="StockFormer模型预测"
                        paperReason={`StockFormer模型预测：模型分${fmt(r.stockformer_score, 1)}，5日预测${pct(r.pred_ret_5d)}`}
                      />
                    )
                  },
                },
                { key: 'close', label: '收盘', align: 'right', render: r => fmt(r.close, 2) },
                { key: 'pred_ret_1d', label: '1日', align: 'right', render: r => <span className={toneByNumber(r.pred_ret_1d)}>{pct(r.pred_ret_1d)}</span> },
                { key: 'pred_ret_3d', label: '3日', align: 'right', render: r => <span className={toneByNumber(r.pred_ret_3d)}>{pct(r.pred_ret_3d)}</span> },
                { key: 'pred_ret_5d', label: '5日', align: 'right', render: r => <span className={toneByNumber(r.pred_ret_5d)}>{pct(r.pred_ret_5d)}</span> },
                { key: 'pred_ret_10d', label: '10日', align: 'right', render: r => <span className={toneByNumber(r.pred_ret_10d)}>{pct(r.pred_ret_10d)}</span> },
                { key: 'up_prob_5d', label: '上涨概率', align: 'right', render: r => pct(r.up_prob_5d) },
                { key: 'trend_prob_10d', label: '趋势概率', align: 'right', render: r => pct(r.trend_prob_10d) },
                { key: 'confidence', label: '置信度', align: 'right', render: r => pct(r.confidence) },
                { key: 'stockformer_score', label: '模型分', align: 'right', render: r => fmt(r.stockformer_score, 1) },
                { key: 'minute_coverage_status', label: '分钟覆盖', render: r => `${r.minute_coverage_status || 'unknown'}${r.minute_coverage_minutes != null ? ` ${r.minute_coverage_minutes}/240` : ''}` },
              ]}
            />
          </section>

          <section className={panelCls}>
            <SectionTitle icon={Layers3} title="AI推荐股票" hint={crossSectionHint(multifactorQ.data)} />
            <ErrorLine error={multifactorQ.data?.error} />
            <CoverageWarning coverage={multifactorQ.data?.data_coverage} />
            {crossSectionNote(multifactorQ.data) && (
              <div className="mb-3 rounded-card border border-accent/15 bg-accent/5 px-3 py-2 text-[11px] text-secondary">
                {crossSectionNote(multifactorQ.data)}
              </div>
            )}
            <DataTable
              rows={multifactorQ.data?.rows ?? []}
              columns={[
                { key: 'rank', label: '排名', align: 'right' },
                { key: 'symbol', label: '股票', render: r => <StockLabel symbol={r.symbol} name={stockNameOf(stockNames, r.symbol)} /> },
                {
                  key: 'actions',
                  label: '操作',
                  render: r => {
                    const name = stockNameOf(stockNames, r.symbol)
                    return (
                      <StockQuickActions
                        symbol={r.symbol}
                        name={name}
                        price={r.close}
                        watchNote="StockFormer多因子推荐"
                        paperReason={`StockFormer多因子推荐：入场分${fmt(r.entry_score, 1)}，综合分${fmt(r.final_score, 1)}`}
                      />
                    )
                  },
                },
                { key: 'entry_score', label: '入场分', align: 'right', render: r => fmt(r.entry_score, 1) },
                { key: 'final_score', label: '综合分', align: 'right', render: r => fmt(r.final_score, 1) },
                { key: 'technical_score', label: '技术', align: 'right', render: r => fmt(r.technical_score, 1) },
                { key: 'stockformer_score', label: '模型', align: 'right', render: r => fmt(r.stockformer_score, 1) },
                { key: 'fundamental_score', label: '基本面', align: 'right', render: r => fmt(r.fundamental_score, 1) },
                { key: 'fund_score', label: '资金', align: 'right', render: r => fmt(r.fund_score, 1) },
                { key: 'sector_score', label: '板块', align: 'right', render: r => fmt(r.sector_score, 1) },
                { key: 'sentiment_score', label: '情绪', align: 'right', render: r => fmt(r.sentiment_score, 1) },
                { key: 'risk_score', label: '风险', align: 'right', render: r => fmt(r.risk_score, 1) },
                { key: 'main_net_inflow_rate', label: '主力净流', align: 'right', render: r => <span className={toneByNumber(r.main_net_inflow_rate)}>{pct(r.main_net_inflow_rate)}</span> },
                { key: 'turnover_rate', label: '换手', align: 'right', render: r => pctPoint(r.turnover_rate) },
                { key: 'open_strength', label: '开盘强度', align: 'right', render: r => <span className={toneByNumber(r.open_strength)}>{pct(r.open_strength)}</span> },
                { key: 'limit_up_streak', label: '连板', align: 'right', render: r => fmt(r.limit_up_streak, 0) },
                { key: 'industry_heat_rank', label: '行业热度', align: 'right', render: r => pct(r.industry_heat_rank) },
                { key: 'sector_fund_layout_score', label: '布局分', align: 'right', render: r => pct(r.sector_fund_layout_score) },
                { key: 'news_sentiment_score', label: '新闻情绪', align: 'right', render: r => <span className={toneByNumber(r.news_sentiment_score)}>{fmt(r.news_sentiment_score, 2)}</span> },
                { key: 'pred_ret_5d', label: '5日预测', align: 'right', render: r => <span className={toneByNumber(r.pred_ret_5d)}>{pct(r.pred_ret_5d)}</span> },
              ]}
            />
          </section>
        </div>

        <MultiPeriodExperiment />

        <section className={panelCls}>
          <SectionTitle icon={BarChart3} title="因子分析" hint={`H${factorHorizon}`} />
          <ErrorLine error={factorQ.data?.error} />
          <div className="mb-3 flex items-center gap-2">
            {[1, 3, 5, 10].map(h => (
              <button
                key={h}
                type="button"
                onClick={() => setFactorHorizon(h)}
                className={cn(
                  'inline-flex h-7 items-center rounded-btn border px-3 text-xs',
                  factorHorizon === h ? 'border-accent bg-accent text-white' : 'border-border bg-surface text-secondary hover:bg-elevated',
                )}
              >
                {h}日
              </button>
            ))}
            {factorQ.isFetching && <Loader2 className="h-4 w-4 animate-spin text-muted" />}
            <button
              type="button"
              onClick={() => {
                setFactorRequested(true)
                void factorQ.refetch()
              }}
              className="inline-flex h-7 items-center gap-1.5 rounded-btn border border-border bg-surface px-3 text-xs text-secondary hover:bg-elevated hover:text-foreground"
            >
              {factorQ.isFetching ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <BarChart3 className="h-3.5 w-3.5" />}
              运行因子分析
            </button>
          </div>
          {!factorRequested && !oosRequested && (
            <div className="mb-3 rounded-card border border-warning/20 bg-warning/10 px-3 py-2 text-xs text-warning">
              因子分析和严格样本外回测属于高内存研究任务，默认不在页面打开时自动运行；点击按钮后按需执行。
            </div>
          )}
          <div className="grid gap-3 md:grid-cols-4">
            <Metric label="IC" value={fmt(factorQ.data?.ic, 4)} tone={toneByNumber(factorQ.data?.ic)} />
            <Metric label="Rank IC" value={fmt(factorQ.data?.rank_ic, 4)} tone={toneByNumber(factorQ.data?.rank_ic)} />
            <Metric label="ICIR" value={fmt(factorQ.data?.icir, 4)} tone={toneByNumber(factorQ.data?.icir)} />
            <Metric label="多空期数" value={fmt(factorQ.data?.long_short_nav?.length, 0)} />
          </div>
          <div className="mt-3 grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1.2fr)_minmax(0,0.8fr)]">
            <FactorChart result={factorQ.data} />
            <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-1">
              <DataTable
                rows={factorQ.data?.quantile_returns ?? []}
                maxHeight="max-h-[190px]"
                columns={[
                  { key: 'group', label: '分位组', align: 'right' },
                  { key: 'avg_return', label: '平均收益', align: 'right', render: r => pct(r.avg_return) },
                  { key: 'cum_return', label: '累计收益', align: 'right', render: r => pct(r.cum_return) },
                  { key: 'periods', label: '期数', align: 'right' },
                ]}
              />
              <DataTable
                rows={factorQ.data?.decay_curve ?? []}
                maxHeight="max-h-[190px]"
                columns={[
                  { key: 'horizon', label: '周期', align: 'right', render: r => `${r.horizon}日` },
                  { key: 'rank_ic', label: 'Rank IC', align: 'right', render: r => fmt(r.rank_ic, 4) },
                ]}
              />
            </div>
          </div>
        </section>

        <section className={panelCls}>
          <SectionTitle icon={GitCompare} title="模型列表与样本外明细" hint="StockFormer artifacts" />
          <DataTable
            rows={models}
            maxHeight="max-h-[280px]"
            columns={[
              {
                key: 'selected',
                label: '',
                render: r => (
                  <button
                    type="button"
                    onClick={() => setSelectedModelId(r.model_id)}
                    className={cn(
                      'inline-flex h-6 items-center gap-1 rounded-btn border px-2 text-[10px]',
                      r.model_id === modelId ? 'border-accent bg-accent text-white' : 'border-border bg-surface text-secondary hover:bg-elevated',
                    )}
                  >
                    <ShieldCheck className="h-3 w-3" />
                    选中
                  </button>
                ),
              },
              { key: 'model_name', label: '模型' },
              { key: 'trained_at', label: '训练时间' },
              { key: 'feature_version', label: '特征版本' },
              { key: 'train', label: '样本内', render: r => `${r.train_start} 至 ${r.train_end}` },
              { key: 'oos', label: '样本外', render: r => `${r.oos_start} 至 ${r.oos_end}` },
              { key: 'n_symbols', label: '股票数', align: 'right' },
              { key: 'oos5', label: '5日RankIC', align: 'right', render: r => fmt(r.oos_metrics?.['5d']?.rank_ic, 4) },
              { key: 'oos10', label: '10日RankIC', align: 'right', render: r => fmt(r.oos_metrics?.['10d']?.rank_ic, 4) },
              { key: 'elapsed_ms', label: '耗时ms', align: 'right', render: r => fmt(r.elapsed_ms, 0) },
            ]}
          />
          <div className="mt-3 rounded-card border border-warning/20 bg-warning/10 px-3 py-2 text-xs text-warning">
            StockFormer 分数只作为独立因子接入；样本外收益、超额收益、交易数和回撤未过门槛前，不进入模拟盘和实盘决策。
          </div>
        </section>
      </main>
    </div>
  )
}
