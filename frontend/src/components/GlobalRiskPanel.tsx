import { Globe2, Loader2, RefreshCw, ShieldAlert, TrendingUp } from 'lucide-react'
import type { GlobalRiskPayload } from '@/lib/api'
import { cn } from '@/lib/cn'

const groupLabels: Record<string, string> = {
  us_broad: '美股宽基',
  tech_ai: '科技/AI链',
  china_proxy: '中国资产',
  macro: '汇率/波动率',
  overseas_news: '海外新闻',
}

function fmt(v: number | null | undefined, digits = 1) {
  const value = Number(v)
  return Number.isFinite(value) ? value.toFixed(digits) : '--'
}

function pctPoint(v: number | null | undefined) {
  const value = Number(v)
  return Number.isFinite(value) ? `${value > 0 ? '+' : ''}${value.toFixed(2)}%` : '--'
}

function scorePoint(v: number | null | undefined) {
  const value = Number(v)
  return Number.isFinite(value) ? `${value > 0 ? '+' : ''}${value.toFixed(1)}分` : '--'
}

function scoreTone(score?: number | null) {
  const value = Number(score)
  if (!Number.isFinite(value)) return 'text-muted'
  if (value >= 60) return 'text-bull'
  if (value <= 42) return 'text-bear'
  return 'text-warning'
}

function chipTone(tone?: string | null) {
  if (tone === 'positive') return 'border-bull/25 bg-bull/10 text-bull'
  if (tone === 'negative') return 'border-bear/25 bg-bear/10 text-bear'
  return 'border-border bg-elevated text-secondary'
}

export function GlobalRiskPanel({
  risk,
  loading,
  onRefresh,
  className,
}: {
  risk?: GlobalRiskPayload | null
  loading?: boolean
  onRefresh?: () => void
  className?: string
}) {
  if (!risk) return null
  const adjustment = risk.recommendation_adjustment
  const groups = Object.entries(risk.group_scores ?? {})

  return (
    <section className={cn('rounded-card border border-border bg-surface/80 p-4 shadow-[0_1px_2px_hsl(var(--border)/0.35)]', className)}>
      <div className="mb-3 flex items-center gap-2">
        <span className="h-3 w-0.5 rounded-full bg-gradient-to-b from-accent to-accent/30" />
        <Globe2 className="h-4 w-4 text-accent" />
        <h2 className="text-sm font-semibold text-foreground">隔夜全球风险</h2>
        <span className="ml-auto text-[10px] text-muted">{risk.updated_at || risk.generated_at}</span>
        {onRefresh && (
          <button
            type="button"
            onClick={onRefresh}
            disabled={loading}
            className="inline-flex h-7 items-center gap-1 rounded-btn border border-border bg-base px-2 text-[10px] text-secondary hover:bg-elevated disabled:cursor-not-allowed disabled:opacity-60"
          >
            {loading ? <Loader2 className="h-3 w-3 animate-spin" /> : <RefreshCw className="h-3 w-3" />}
            刷新
          </button>
        )}
      </div>

      <div className="grid gap-2 md:grid-cols-5">
        <div className="rounded-card border border-border/70 bg-base/40 px-3 py-2">
          <div className="flex items-center gap-1.5 text-[10px] text-muted"><ShieldAlert className="h-3 w-3" />风险偏好</div>
          <div className={cn('mt-1 text-lg font-semibold', scoreTone(risk.appetite_score))}>{fmt(risk.appetite_score, 1)}</div>
          <div className="mt-0.5 text-[10px] text-muted">{risk.market_bias}</div>
        </div>
        <div className="rounded-card border border-border/70 bg-base/40 px-3 py-2">
          <div className="text-[10px] text-muted">仓位系数</div>
          <div className="mt-1 font-mono text-sm font-semibold text-foreground">{fmt(adjustment?.position_multiplier, 2)}x</div>
          <div className="mt-0.5 text-[10px] text-muted">盘前建议</div>
        </div>
        <div className="rounded-card border border-border/70 bg-base/40 px-3 py-2">
          <div className="text-[10px] text-muted">买入门槛</div>
          <div className={cn('mt-1 font-mono text-sm font-semibold', scoreTone(50 - Number(adjustment?.buy_threshold_add ?? 0)))}>{scorePoint(adjustment?.buy_threshold_add)}</div>
          <div className="mt-0.5 text-[10px] text-muted">分数修正</div>
        </div>
        <div className="rounded-card border border-border/70 bg-base/40 px-3 py-2">
          <div className="text-[10px] text-muted">行情覆盖</div>
          <div className="mt-1 font-mono text-sm font-semibold text-foreground">{risk.quote_count}</div>
          <div className="mt-0.5 text-[10px] text-muted">{risk.cache_hit ? '缓存' : '实时'}{risk.cache_fallback ? ' · 兜底' : ''}</div>
        </div>
        <div className="rounded-card border border-border/70 bg-base/40 px-3 py-2">
          <div className="text-[10px] text-muted">新闻信号</div>
          <div className={cn('mt-1 font-mono text-sm font-semibold', scoreTone(risk.news_signal?.score))}>{fmt(risk.news_signal?.score, 1)}</div>
          <div className="mt-0.5 text-[10px] text-muted">{risk.news_signal?.count ?? 0} 条</div>
        </div>
      </div>

      <div className="mt-3 grid gap-2 md:grid-cols-5">
        {groups.map(([key, item]) => (
          <div key={key} className="rounded border border-border/60 bg-base/30 px-3 py-2">
            <div className="flex items-center justify-between gap-2">
              <span className="text-[10px] text-muted">{groupLabels[key] ?? key}</span>
              <span className={cn('font-mono text-xs font-semibold', scoreTone(item.score))}>{fmt(item.score, 1)}</span>
            </div>
            <div className="mt-1 text-[10px] text-muted">
              加权变动 {pctPoint(item.weighted_change)} · {item.available ?? 0} 项
            </div>
          </div>
        ))}
      </div>

      <div className="mt-3 flex flex-wrap gap-1.5">
        {risk.quote_highlights?.slice(0, 8).map(item => (
          <span key={item.symbol} className={cn('inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px]', scoreTone(50 + Number(item.change_pct ?? 0) * 12))}>
            <TrendingUp className="h-3 w-3" />
            {item.name || item.symbol} {pctPoint(item.change_pct)}
          </span>
        ))}
      </div>

      <div className="mt-3 grid gap-2 lg:grid-cols-[1fr_1fr]">
        <div className="rounded-card border border-accent/15 bg-accent/5 px-3 py-2 text-xs text-secondary">
          {risk.guidance}
          <div className="mt-1 text-[11px] text-muted">{adjustment?.open_rule}</div>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {risk.sector_impacts?.length ? risk.sector_impacts.map(item => (
            <span key={`${item.key}-${item.label}`} className={cn('rounded-full border px-2 py-0.5 text-[10px]', chipTone(item.tone))}>
              {item.label}：{item.summary}
            </span>
          )) : risk.risk_tags?.map(tag => (
            <span key={tag} className="rounded-full border border-border bg-elevated px-2 py-0.5 text-[10px] text-secondary">{tag}</span>
          ))}
        </div>
      </div>

      {risk.errors?.length ? (
        <div className="mt-3 rounded-card border border-warning/20 bg-warning/10 px-3 py-2 text-[11px] text-warning">
          {risk.errors.slice(0, 2).join('；')}
        </div>
      ) : null}
    </section>
  )
}
