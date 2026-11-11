import { useState, type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import {
  Activity,
  Clock3,
  DatabaseZap,
  Gauge,
  RefreshCw,
  ShieldAlert,
  TrendingDown,
  TrendingUp,
} from 'lucide-react'
import { PageHeader } from '@/components/PageHeader'
import { StockPreviewDialog } from '@/components/StockPreviewDialog'
import {
  api,
  type FundMonitorDailyRow,
  type FundMonitorDailySignal,
  type FundMonitorIntradayRow,
  type FundMonitorIntradaySignal,
} from '@/lib/api'
import { QK } from '@/lib/queryKeys'
import { fmtBigNum, fmtPct, fmtPrice, priceColorClass } from '@/lib/format'

type FundTab = 'daily' | 'intraday'

const DAILY_SIGNAL_OPTIONS: Array<{ value: 'all' | FundMonitorDailySignal; label: string }> = [
  { value: 'all', label: '全部信号' },
  { value: 'main_accumulation', label: '主力吸筹' },
  { value: 'early_layout', label: '资金初动' },
  { value: 'low_absorption', label: '低位吸筹' },
  { value: 'strong_inflow', label: '持续强流入' },
  { value: 'inflow_confirm', label: '流入确认' },
  { value: 'distribution_risk', label: '拉高派发风险' },
  { value: 'outflow', label: '资金流出' },
  { value: 'watch', label: '观察' },
]

const INTRADAY_SIGNAL_OPTIONS: Array<{ value: 'all' | FundMonitorIntradaySignal; label: string }> = [
  { value: 'all', label: '全部信号' },
  { value: 'active_inflow', label: '盘中主动流入' },
  { value: 'breakout_flow', label: '放量突破' },
  { value: 'distribution_risk', label: '冲高派发风险' },
  { value: 'outflow', label: '盘中流出' },
  { value: 'watch', label: '观察' },
]

const SIGNAL_CLASS: Record<string, string> = {
  main_accumulation: 'border-danger/25 bg-danger/10 text-danger',
  early_layout: 'border-danger/25 bg-danger/10 text-danger',
  low_absorption: 'border-accent/25 bg-accent/10 text-accent',
  strong_inflow: 'border-danger/25 bg-danger/10 text-danger',
  active_inflow: 'border-danger/25 bg-danger/10 text-danger',
  breakout_flow: 'border-orange-400/30 bg-orange-400/10 text-orange-500 dark:text-orange-300',
  inflow_confirm: 'border-amber-400/30 bg-amber-400/10 text-amber-600 dark:text-amber-300',
  watch: 'border-border bg-elevated text-secondary',
  distribution_risk: 'border-purple-500/25 bg-purple-500/10 text-purple-600 dark:text-purple-300',
  outflow: 'border-bear/25 bg-bear/10 text-bear',
}

function signalBadge(signal: string, label: string) {
  return (
    <span className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] font-medium ${SIGNAL_CLASS[signal] ?? SIGNAL_CLASS.watch}`}>
      {label}
    </span>
  )
}

function fmtScore(v: number | null | undefined) {
  if (v == null || Number.isNaN(v)) return '—'
  return v.toFixed(1)
}

function fmtTurnover(v: number | null | undefined) {
  if (v == null || Number.isNaN(v)) return '—'
  return `${v.toFixed(2)}%`
}

function fmtRatio(v: number | null | undefined) {
  if (v == null || Number.isNaN(v)) return '—'
  return `${v.toFixed(2)}x`
}

function fmtMoney(v: number | null | undefined) {
  if (v == null || Number.isNaN(v)) return '—'
  const sign = v < 0 ? '-' : ''
  return `${sign}${fmtBigNum(Math.abs(v))}`
}

function fmtTime(value: string | null | undefined) {
  if (!value) return '—'
  const text = String(value)
  return text.length > 16 ? text.slice(5, 16) : text
}

function formatAge(ms: number | null | undefined) {
  if (ms == null || Number.isNaN(Number(ms))) return '—'
  if (ms < 1000) return `${Math.round(ms)}ms`
  const seconds = Math.round(ms / 1000)
  if (seconds < 60) return `${seconds}s`
  return `${Math.floor(seconds / 60)}m ${seconds % 60}s`
}

function MetricCard({ icon, label, value, hint, tone = 'neutral' }: {
  icon: ReactNode
  label: string
  value: ReactNode
  hint?: ReactNode
  tone?: 'neutral' | 'bull' | 'bear' | 'warn'
}) {
  const toneCls = tone === 'bull'
    ? 'text-danger'
    : tone === 'bear'
      ? 'text-bear'
      : tone === 'warn'
        ? 'text-warning'
        : 'text-foreground'
  return (
    <div className="rounded border border-border bg-surface px-3 py-2">
      <div className="flex items-center gap-1.5 text-[10px] text-muted">
        {icon}
        <span>{label}</span>
      </div>
      <div className={`mt-1 text-lg font-semibold tabular-nums ${toneCls}`}>{value}</div>
      {hint && <div className="mt-0.5 truncate text-[10px] text-muted">{hint}</div>}
    </div>
  )
}

function ScoreCell({ score, parts }: { score: number; parts: Array<{ label: string; value: number }> }) {
  return (
    <div className="min-w-[7.5rem]">
      <div className="flex items-center justify-between text-[11px]">
        <span className="font-semibold text-foreground">{fmtScore(score)}</span>
        <span className="text-muted">/100</span>
      </div>
      <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-elevated">
        <div className="h-full rounded-full bg-accent" style={{ width: `${Math.max(0, Math.min(100, score))}%` }} />
      </div>
      <div className="mt-1 flex flex-wrap gap-x-2 gap-y-0.5 text-[9px] text-muted">
        {parts.map(p => <span key={p.label}>{p.label}{fmtScore(p.value)}</span>)}
      </div>
    </div>
  )
}

function StockButton({ row, onOpenStock }: {
  row: { symbol: string; name?: string | null; board?: string | null; industry?: string | null }
  onOpenStock: (symbol: string, name: string) => void
}) {
  const name = row.name || row.symbol
  return (
    <button
      type="button"
      onClick={() => onOpenStock(row.symbol, name)}
      className="group text-left"
      title="打开个股分析"
    >
      <div className="font-medium text-foreground group-hover:text-accent">{name}</div>
      <div className="mt-0.5 flex flex-wrap items-center gap-1 text-[10px] text-muted">
        <span>{row.symbol}</span>
        {row.board && <span className="rounded bg-elevated px-1 py-px">{row.board}</span>}
        {row.industry && <span className="max-w-[12rem] truncate rounded bg-elevated px-1 py-px">{row.industry}</span>}
      </div>
    </button>
  )
}

function Reasons({ reasons }: { reasons?: string[] }) {
  if (!reasons?.length) return <span className="text-muted">—</span>
  return (
    <div className="flex max-w-[20rem] flex-wrap gap-1">
      {reasons.map((reason, idx) => (
        <span key={`${idx}-${reason}`} className="rounded bg-elevated px-1.5 py-0.5 text-[10px] text-secondary">
          {reason}
        </span>
      ))}
    </div>
  )
}

function NumberInput({ label, value, onChange, min, max, step = 1 }: {
  label: string
  value: number
  onChange: (value: number) => void
  min: number
  max: number
  step?: number
}) {
  return (
    <label className="flex flex-col gap-1 text-[11px] text-muted">
      {label}
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={event => onChange(Math.max(min, Math.min(max, Number(event.target.value) || min)))}
        className="h-8 w-28 rounded border border-border bg-base px-2 text-xs text-foreground outline-none focus:border-accent"
      />
    </label>
  )
}

function DailyBoard({ onOpenStock }: { onOpenStock: (symbol: string, name: string) => void }) {
  const [windowDays, setWindowDays] = useState(20)
  const [limit, setLimit] = useState(100)
  const [minScore, setMinScore] = useState(0)
  const [signal, setSignal] = useState<(typeof DAILY_SIGNAL_OPTIONS)[number]['value']>('all')
  const query = useQuery({
    queryKey: QK.fundMonitorDaily(windowDays, limit, minScore, signal),
    queryFn: () => api.fundMonitorDaily({ window: windowDays, limit, minScore, signal }),
    refetchInterval: 60_000,
    staleTime: 10_000,
  })
  const rows = query.data?.rows ?? []
  const flowMeta = query.data?.flow_meta ?? {}
  const counts = query.data?.counts ?? {}

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3">
      <div className="rounded border border-border bg-surface px-3 py-2 text-xs text-secondary">
        {query.data?.data_note ?? '日级资金监控使用本地 moneyflow 数据。'}
      </div>

      <div className="grid grid-cols-2 gap-2 lg:grid-cols-5">
        <MetricCard icon={<DatabaseZap className="h-3.5 w-3.5" />} label="资金最新日" value={String(flowMeta.date_end ?? '—')} hint={query.data?.source_label} />
        <MetricCard icon={<Gauge className="h-3.5 w-3.5" />} label="覆盖标的" value={String(flowMeta.symbols ?? '—')} hint={`${flowMeta.trading_days ?? 0} 个资金交易日`} />
        <MetricCard icon={<TrendingUp className="h-3.5 w-3.5" />} label="早期信号" value={String((counts.main_accumulation ?? 0) + (counts.early_layout ?? 0) + (counts.low_absorption ?? 0))} tone="bull" hint="主力吸筹/资金初动/低位吸筹" />
        <MetricCard icon={<ShieldAlert className="h-3.5 w-3.5" />} label="派发风险" value={String(counts.distribution_risk ?? 0)} tone="warn" hint="上涨但资金转弱" />
        <MetricCard icon={<Clock3 className="h-3.5 w-3.5" />} label="跳过分区" value={String(flowMeta.skipped_partitions ?? 0)} hint="坏 parquet 自动隔离" />
      </div>

      <div className="flex flex-wrap items-end gap-3 rounded border border-border bg-surface px-3 py-2">
        <label className="flex flex-col gap-1 text-[11px] text-muted">
          资金窗口
          <select
            value={windowDays}
            onChange={event => setWindowDays(Number(event.target.value))}
            className="h-8 w-32 rounded border border-border bg-base px-2 text-xs text-foreground outline-none focus:border-accent"
          >
            {[5, 10, 20, 30, 60].map(v => <option key={v} value={v}>近{v}个交易日</option>)}
          </select>
        </label>
        <label className="flex flex-col gap-1 text-[11px] text-muted">
          信号
          <select
            value={signal}
            onChange={event => setSignal(event.target.value as typeof signal)}
            className="h-8 w-36 rounded border border-border bg-base px-2 text-xs text-foreground outline-none focus:border-accent"
          >
            {DAILY_SIGNAL_OPTIONS.map(opt => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
          </select>
        </label>
        <NumberInput label="最低分" value={minScore} min={0} max={100} onChange={setMinScore} />
        <NumberInput label="显示数量" value={limit} min={20} max={1000} onChange={setLimit} />
        <button
          type="button"
          onClick={() => query.refetch()}
          className="inline-flex h-8 items-center gap-1.5 rounded border border-border bg-base px-3 text-xs text-secondary transition-colors hover:text-foreground"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${query.isFetching ? 'animate-spin' : ''}`} />
          刷新
        </button>
        <div className="ml-auto text-[10px] text-muted">
          权重: 吸筹30% · 早期资金25% · 趋势25% · 风险20%
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-auto rounded border border-border bg-surface">
        <table className="min-w-[1780px] w-full text-left text-xs">
          <thead className="sticky top-0 z-10 bg-elevated text-[11px] text-muted">
            <tr>
              <th className="w-12 px-3 py-2">#</th>
              <th className="px-3 py-2">股票</th>
              <th className="px-3 py-2">信号</th>
              <th className="px-3 py-2">综合分</th>
              <th className="px-3 py-2 text-right">1日净额</th>
              <th className="px-3 py-2 text-right">3日净额</th>
              <th className="px-3 py-2 text-right">5日净额</th>
              <th className="px-3 py-2 text-right">3日净额率</th>
              <th className="px-3 py-2 text-right">资金加速度</th>
              <th className="px-3 py-2 text-right">吸筹分</th>
              <th className="px-3 py-2 text-right">吸筹效率</th>
              <th className="px-3 py-2 text-right">早期分</th>
              <th className="px-3 py-2 text-right">后手风险</th>
              <th className="px-3 py-2 text-right">连续流入</th>
              <th className="px-3 py-2 text-right">现价</th>
              <th className="px-3 py-2 text-right">涨跌</th>
              <th className="px-3 py-2 text-right">换手</th>
              <th className="px-3 py-2">证据</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row: FundMonitorDailyRow, idx) => (
              <tr key={row.symbol} className="border-t border-border/70 hover:bg-elevated/50">
                <td className="px-3 py-2 text-muted">{idx + 1}</td>
                <td className="px-3 py-2"><StockButton row={row} onOpenStock={onOpenStock} /></td>
                <td className="px-3 py-2">{signalBadge(row.signal, row.signal_label)}</td>
                <td className="px-3 py-2">
                  <ScoreCell
                    score={row.score}
                    parts={[
                      { label: '资', value: row.fund_score },
                      { label: '趋', value: row.trend_score },
                      { label: '险', value: row.risk_score },
                    ]}
                  />
                </td>
                <td className={`px-3 py-2 text-right tabular-nums ${priceColorClass(row.net_1d)}`}>{fmtMoney(row.net_1d)}</td>
                <td className={`px-3 py-2 text-right tabular-nums ${priceColorClass(row.net_3d)}`}>{fmtMoney(row.net_3d)}</td>
                <td className={`px-3 py-2 text-right tabular-nums ${priceColorClass(row.net_5d)}`}>{fmtMoney(row.net_5d)}</td>
                <td className={`px-3 py-2 text-right tabular-nums ${priceColorClass(row.net_ratio_3d)}`}>{fmtPct(row.net_ratio_3d)}</td>
                <td className={`px-3 py-2 text-right tabular-nums ${priceColorClass(row.moneyflow_acceleration_3d)}`}>{fmtPct(row.moneyflow_acceleration_3d)}</td>
                <td className="px-3 py-2 text-right tabular-nums">{fmtScore(row.accumulation_score)}</td>
                <td className="px-3 py-2 text-right tabular-nums">{fmtScore(row.absorption_efficiency == null ? null : row.absorption_efficiency * 100)}</td>
                <td className="px-3 py-2 text-right tabular-nums">{fmtScore(row.early_layout_score)}</td>
                <td className={`px-3 py-2 text-right tabular-nums ${row.inflow_age_risk && row.inflow_age_risk >= 0.45 ? 'text-danger' : 'text-muted'}`}>
                  {fmtScore(row.inflow_age_risk == null ? null : row.inflow_age_risk * 100)}
                </td>
                <td className="px-3 py-2 text-right tabular-nums">{row.positive_streak ?? 0}日</td>
                <td className="px-3 py-2 text-right tabular-nums">{fmtPrice(row.close)}</td>
                <td className={`px-3 py-2 text-right tabular-nums ${priceColorClass(row.change_pct)}`}>{fmtPct(row.change_pct)}</td>
                <td className="px-3 py-2 text-right tabular-nums">{fmtTurnover(row.turnover_rate)}</td>
                <td className="px-3 py-2"><Reasons reasons={row.reasons} /></td>
              </tr>
            ))}
            {!query.isLoading && rows.length === 0 && (
              <tr>
                <td colSpan={18} className="px-3 py-8 text-center text-muted">暂无符合条件的资金信号</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function IntradayBoard({ onOpenStock }: { onOpenStock: (symbol: string, name: string) => void }) {
  const [limit, setLimit] = useState(100)
  const [minScore, setMinScore] = useState(0)
  const [signal, setSignal] = useState<(typeof INTRADAY_SIGNAL_OPTIONS)[number]['value']>('all')
  const query = useQuery({
    queryKey: QK.fundMonitorIntraday(limit, minScore, signal),
    queryFn: () => api.fundMonitorIntraday({ limit, minScore, signal }),
    refetchInterval: 10_000,
    staleTime: 3_000,
  })
  const rows = query.data?.rows ?? []
  const counts = query.data?.counts ?? {}
  const quote = query.data?.quote_status ?? {}
  const minuteMeta = query.data?.minute_meta ?? {}
  const minuteRefresh = query.data?.minute_refresh ?? {}
  const fresh = query.data?.quote_is_fresh

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3">
      <div className="rounded border border-border bg-surface px-3 py-2 text-xs text-secondary">
        {query.data?.data_note ?? '盘中资金雷达使用实时行情和分钟K做近似估算。'}
      </div>

      <div className="grid grid-cols-2 gap-2 lg:grid-cols-6">
        <MetricCard icon={<Activity className="h-3.5 w-3.5" />} label="行情模式" value={String(query.data?.quote_mode ?? '—')} hint={fresh ? '实时缓存可用' : '行情可能未刷新'} tone={fresh ? 'bull' : 'warn'} />
        <MetricCard icon={<Clock3 className="h-3.5 w-3.5" />} label="行情年龄" value={formatAge(quote.quote_age_ms)} hint={`标的 ${quote.symbol_count ?? '—'} 只`} />
        <MetricCard icon={<DatabaseZap className="h-3.5 w-3.5" />} label="分钟K日期" value={String(minuteMeta.date ?? '—')} hint={fmtTime(minuteMeta.latest_minute)} />
        <MetricCard icon={<Gauge className="h-3.5 w-3.5" />} label="分钟增量" value={minuteRefresh.enabled ? '开启' : '关闭'} hint={minuteRefresh.running ? '后台运行中' : String(minuteRefresh.gate_reason ?? '未运行')} tone={minuteRefresh.enabled ? 'bull' : 'warn'} />
        <MetricCard icon={<TrendingUp className="h-3.5 w-3.5" />} label="主动流入" value={String((counts.active_inflow ?? 0) + (counts.breakout_flow ?? 0))} tone="bull" hint="含放量突破" />
        <MetricCard icon={<TrendingDown className="h-3.5 w-3.5" />} label="派发/流出" value={String((counts.distribution_risk ?? 0) + (counts.outflow ?? 0))} tone="bear" hint="风险信号" />
      </div>

      <div className="flex flex-wrap items-end gap-3 rounded border border-border bg-surface px-3 py-2">
        <label className="flex flex-col gap-1 text-[11px] text-muted">
          信号
          <select
            value={signal}
            onChange={event => setSignal(event.target.value as typeof signal)}
            className="h-8 w-40 rounded border border-border bg-base px-2 text-xs text-foreground outline-none focus:border-accent"
          >
            {INTRADAY_SIGNAL_OPTIONS.map(opt => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
          </select>
        </label>
        <NumberInput label="最低分" value={minScore} min={0} max={100} onChange={setMinScore} />
        <NumberInput label="显示数量" value={limit} min={20} max={1000} onChange={setLimit} />
        <button
          type="button"
          onClick={() => query.refetch()}
          className="inline-flex h-8 items-center gap-1.5 rounded border border-border bg-base px-3 text-xs text-secondary transition-colors hover:text-foreground"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${query.isFetching ? 'animate-spin' : ''}`} />
          刷新
        </button>
        <div className="ml-auto flex items-center gap-3 text-[10px] text-muted">
          <span>自动刷新 10s</span>
          <Link to="/settings?tab=monitoring&highlight=minute-refresh" className="text-accent hover:underline">分钟增量设置</Link>
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-auto rounded border border-border bg-surface">
        <table className="min-w-[1340px] w-full text-left text-xs">
          <thead className="sticky top-0 z-10 bg-elevated text-[11px] text-muted">
            <tr>
              <th className="w-12 px-3 py-2">#</th>
              <th className="px-3 py-2">股票</th>
              <th className="px-3 py-2">盘中信号</th>
              <th className="px-3 py-2">强度分</th>
              <th className="px-3 py-2 text-right">现价</th>
              <th className="px-3 py-2 text-right">涨跌</th>
              <th className="px-3 py-2 text-right">预计量比</th>
              <th className="px-3 py-2 text-right">15m估算净额</th>
              <th className="px-3 py-2 text-right">5m涨跌</th>
              <th className="px-3 py-2 text-right">15m涨跌</th>
              <th className="px-3 py-2 text-right">日级确认</th>
              <th className="px-3 py-2 text-right">换手</th>
              <th className="px-3 py-2">证据</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row: FundMonitorIntradayRow, idx) => (
              <tr key={row.symbol} className="border-t border-border/70 hover:bg-elevated/50">
                <td className="px-3 py-2 text-muted">{idx + 1}</td>
                <td className="px-3 py-2"><StockButton row={row} onOpenStock={onOpenStock} /></td>
                <td className="px-3 py-2">{signalBadge(row.signal, row.signal_label)}</td>
                <td className="px-3 py-2">
                  <ScoreCell
                    score={row.score}
                    parts={[
                      { label: '资', value: row.fund_score },
                      { label: '价', value: row.price_score },
                      { label: '险', value: row.risk_score },
                    ]}
                  />
                </td>
                <td className="px-3 py-2 text-right tabular-nums">{fmtPrice(row.close)}</td>
                <td className={`px-3 py-2 text-right tabular-nums ${priceColorClass(row.change_pct)}`}>{fmtPct(row.change_pct)}</td>
                <td className="px-3 py-2 text-right tabular-nums">{fmtRatio(row.amount_ratio_20d)}</td>
                <td className={`px-3 py-2 text-right tabular-nums ${priceColorClass(row.est_net_15m)}`}>
                  <div>{fmtMoney(row.est_net_15m)}</div>
                  {row.estimate_source === 'quote_fallback' && <div className="text-[9px] text-muted">行情估算</div>}
                </td>
                <td className={`px-3 py-2 text-right tabular-nums ${priceColorClass(row.ret_5m)}`}>{fmtPct(row.ret_5m)}</td>
                <td className={`px-3 py-2 text-right tabular-nums ${priceColorClass(row.ret_15m)}`}>{fmtPct(row.ret_15m)}</td>
                <td className={`px-3 py-2 text-right tabular-nums ${priceColorClass(row.daily_net_5d)}`}>{fmtMoney(row.daily_net_5d)}</td>
                <td className="px-3 py-2 text-right tabular-nums">{fmtTurnover(row.turnover_rate)}</td>
                <td className="px-3 py-2"><Reasons reasons={row.reasons} /></td>
              </tr>
            ))}
            {!query.isLoading && rows.length === 0 && (
              <tr>
                <td colSpan={13} className="px-3 py-8 text-center text-muted">暂无符合条件的盘中资金信号</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export function FundMonitor() {
  const [tab, setTab] = useState<FundTab>('intraday')
  const [preview, setPreview] = useState<{ symbol: string; name: string } | null>(null)
  const onOpenStock = (symbol: string, name: string) => setPreview({ symbol, name })

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="shrink-0">
        <PageHeader
          title="资金监控"
          subtitle="日级主力资金榜 · 盘中资金雷达"
          right={
            <div className="flex items-center gap-2">
              <Link
                to="/monitor"
                className="inline-flex h-7 items-center gap-1 rounded border border-border bg-base px-2 text-[11px] text-secondary transition-colors hover:text-foreground"
              >
                <ShieldAlert className="h-3 w-3" />
                告警规则
              </Link>
            </div>
          }
        />
      </div>

      <div className="flex shrink-0 flex-wrap items-center gap-3 px-5 pt-3">
        <div className="inline-flex rounded-full border border-border/50 bg-base/70 p-0.5">
          <button
            type="button"
            onClick={() => setTab('daily')}
            className={`inline-flex items-center gap-1.5 rounded-full px-3.5 py-1.5 text-xs transition-all ${
              tab === 'daily' ? 'bg-accent/15 font-medium text-accent shadow-sm' : 'text-secondary hover:text-foreground'
            }`}
          >
            <DatabaseZap className="h-3.5 w-3.5" />
            日级主力资金榜
          </button>
          <button
            type="button"
            onClick={() => setTab('intraday')}
            className={`inline-flex items-center gap-1.5 rounded-full px-3.5 py-1.5 text-xs transition-all ${
              tab === 'intraday' ? 'bg-accent/15 font-medium text-accent shadow-sm' : 'text-secondary hover:text-foreground'
            }`}
          >
            <Activity className="h-3.5 w-3.5" />
            盘中资金雷达
          </button>
        </div>
        <span className="text-[10px] text-muted">
          {tab === 'daily' ? 'Tushare moneyflow · 盘后确认' : '实时行情 + 分钟K · 盘中估算'}
        </span>
      </div>

      <div className="flex min-h-0 flex-1 flex-col px-5 pb-4 pt-3">
        {tab === 'daily'
          ? <DailyBoard onOpenStock={onOpenStock} />
          : <IntradayBoard onOpenStock={onOpenStock} />
        }
      </div>

      {preview && (
        <StockPreviewDialog
          symbol={preview.symbol}
          name={preview.name}
          onClose={() => setPreview(null)}
        />
      )}
    </div>
  )
}
