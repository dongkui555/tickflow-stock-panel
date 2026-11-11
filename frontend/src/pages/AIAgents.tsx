import { useMemo, useState, type ReactNode } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Activity,
  Database,
  Loader2,
  Newspaper,
  RadioTower,
  RefreshCw,
  Send,
  ShieldAlert,
  Sparkles,
  TrendingUp,
  Wallet,
} from 'lucide-react'
import {
  api,
  type AIAgentsAnalysis,
  type AIAgentsContext,
  type AIAgentsDecision,
} from '@/lib/api'
import { QK } from '@/lib/queryKeys'
import { cn } from '@/lib/cn'
import { StockIdentity } from '@/lib/stockDisplay'

const panelCls = 'rounded-card border border-border bg-surface/80 p-4 shadow-[0_1px_2px_hsl(var(--border)/0.35)]'

function n(v: unknown) {
  const x = Number(v)
  return Number.isFinite(x) ? x : null
}

function fmt(v: unknown, digits = 2) {
  const x = n(v)
  return x == null ? '--' : x.toFixed(digits)
}

function fmtMoney(v: unknown) {
  const x = n(v)
  if (x == null) return '--'
  if (Math.abs(x) >= 100000000) return `${(x / 100000000).toFixed(2)}亿`
  if (Math.abs(x) >= 10000) return `${(x / 10000).toFixed(2)}万`
  return x.toFixed(2)
}

function pctClass(v: unknown) {
  const x = n(v)
  if (x == null || x === 0) return 'text-muted'
  return x > 0 ? 'text-bull' : 'text-bear'
}

function SectionTitle({ icon: Icon, title, hint }: { icon: typeof Activity; title: string; hint?: ReactNode }) {
  return (
    <div className="mb-3 flex items-center gap-2">
      <span className="h-3 w-0.5 rounded-full bg-gradient-to-b from-accent to-accent/30" />
      <Icon className="h-3.5 w-3.5 text-accent" />
      <h2 className="text-xs font-semibold text-foreground">{title}</h2>
      {hint != null && <span className="ml-auto text-[10px] font-mono text-muted">{hint}</span>}
    </div>
  )
}

function Metric({ label, value, tone }: { label: string; value: ReactNode; tone?: string }) {
  return (
    <div className="rounded-card border border-border/70 bg-base/40 px-3 py-2">
      <div className="text-[10px] text-muted">{label}</div>
      <div className={cn('mt-1 text-sm font-semibold text-foreground', tone)}>{value}</div>
    </div>
  )
}

function DecisionCard({ decision }: { decision?: AIAgentsDecision }) {
  if (!decision) return null
  return (
    <div className={panelCls}>
      <SectionTitle icon={ShieldAlert} title="结构化决策" hint={decision.holding_period} />
      <div className="grid grid-cols-2 gap-2 md:grid-cols-5">
        <Metric label="综合分" value={fmt(decision.score, 1)} tone={decision.score >= 70 ? 'text-bull' : decision.score < 50 ? 'text-bear' : 'text-warning'} />
        <Metric label="评级" value={decision.rating} />
        <Metric label="支撑" value={fmt(decision.support)} />
        <Metric label="压力" value={fmt(decision.resistance)} />
        <Metric label="保护线" value={fmt(decision.reference_stop)} tone="text-warning" />
      </div>
      <div className="mt-3 grid gap-3 md:grid-cols-2">
        <div>
          <div className="mb-1 text-[11px] text-secondary">正向证据</div>
          <div className="space-y-1">
            {(decision.evidence ?? []).slice(0, 6).map((item, i) => (
              <div key={i} className="rounded bg-bull/10 px-2 py-1 text-[11px] text-bull">{item}</div>
            ))}
            {(decision.evidence ?? []).length === 0 && <div className="text-[11px] text-muted">暂无</div>}
          </div>
        </div>
        <div>
          <div className="mb-1 text-[11px] text-secondary">风险项</div>
          <div className="space-y-1">
            {(decision.risks ?? []).slice(0, 6).map((item, i) => (
              <div key={i} className="rounded bg-bear/10 px-2 py-1 text-[11px] text-bear">{item}</div>
            ))}
            {(decision.risks ?? []).length === 0 && <div className="text-[11px] text-muted">暂无</div>}
          </div>
        </div>
      </div>
    </div>
  )
}

function ContextPanel({ context }: { context?: AIAgentsContext }) {
  if (!context) return null
  const quote = context.quote ?? {}
  const tech = context.technical ?? {}
  const fund = context.fund_flow ?? {}
  const news = context.news?.news_data ?? {}
  const overseasQuotes = context.overseas?.market?.quotes ?? []
  return (
    <div className="grid gap-4 xl:grid-cols-[1.1fr_0.9fr]">
      <div className={panelCls}>
        <SectionTitle icon={TrendingUp} title="行情与技术位置" hint={context.generated_at} />
        <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
          <Metric label="现价" value={fmt(quote.price ?? tech.latest_close)} />
          <Metric label="涨跌幅" value={`${fmt(quote.change_percent ?? tech.change_pct)}%`} tone={pctClass(quote.change_percent ?? tech.change_pct)} />
          <Metric label="MA20" value={fmt(tech.ma20)} />
          <Metric label="量比" value={fmt(tech.volume_ratio)} />
          <Metric label="趋势" value={tech.trend ?? '--'} />
          <Metric label="RSI14" value={fmt(tech.rsi14)} />
          <Metric label="支撑" value={fmt(tech.support)} />
          <Metric label="压力" value={fmt(tech.resistance)} />
        </div>
      </div>

      <div className={panelCls}>
        <SectionTitle icon={RadioTower} title="资金确认" hint={fund.source || 'multi-source'} />
        <div className="grid grid-cols-2 gap-2">
          <Metric label="近5日主力" value={fmtMoney(fund.main_net_5d)} tone={pctClass(fund.main_net_5d)} />
          <Metric label="近20日主力" value={fmtMoney(fund.main_net_20d)} tone={pctClass(fund.main_net_20d)} />
          <Metric label="可用天数" value={fund.days ?? '--'} />
          <Metric label="20日流入天数" value={fund.positive_days_20 ?? '--'} />
        </div>
        {!fund.data_success && <div className="mt-3 rounded bg-warning/10 px-2 py-1 text-[11px] text-warning">{fund.error || '资金数据暂不可用'}</div>}
      </div>

      <div className={panelCls}>
        <SectionTitle icon={Newspaper} title="新闻与海外环境" hint={`${news.count ?? 0} 条`} />
        <div className="grid gap-2 md:grid-cols-2">
          {(news.items ?? []).slice(0, 6).map((item: any, index: number) => (
            <a
              key={`${item.title}-${index}`}
              href={item.url || undefined}
              target="_blank"
              rel="noreferrer"
              className="rounded-card border border-border/70 bg-base/40 px-3 py-2 text-[11px] hover:bg-elevated"
            >
              <div className="line-clamp-2 text-foreground">{item.title || item.标题 || '--'}</div>
              <div className="mt-1 truncate text-muted">{item.source || item.来源 || '未知来源'} · {item.date || item.time || ''}</div>
            </a>
          ))}
          {(news.items ?? []).length === 0 && <div className="text-[11px] text-muted">暂无可用新闻</div>}
        </div>
      </div>

      <div className={panelCls}>
        <SectionTitle icon={Database} title="海外行情" hint={context.overseas?.market?.source || ''} />
        <div className="grid grid-cols-2 gap-2">
          {overseasQuotes.slice(0, 8).map((item: any) => (
            <Metric
              key={item.symbol}
              label={item.name || item.symbol}
              value={`${fmt(item.change_pct)}%`}
              tone={pctClass(item.change_pct)}
            />
          ))}
          {overseasQuotes.length === 0 && <div className="text-[11px] text-muted">暂无海外行情</div>}
        </div>
      </div>
    </div>
  )
}

export function AIAgents() {
  const qc = useQueryClient()
  const [symbol, setSymbol] = useState('000001')
  const [fundDays, setFundDays] = useState(60)
  const [context, setContext] = useState<AIAgentsContext | null>(null)
  const [analysis, setAnalysis] = useState<AIAgentsAnalysis | null>(null)
  const [order, setOrder] = useState({ symbol: '000001', side: 'BUY' as 'BUY' | 'SELL', quantity: 100, price: '', reason: '' })

  const health = useQuery({
    queryKey: QK.aiagentsHealth,
    queryFn: api.aiagentsHealth,
  })
  const paper = useQuery({
    queryKey: QK.aiagentsPaper,
    queryFn: api.aiagentsPaper,
  })
  const holdings = useQuery({
    queryKey: QK.aiagentsHoldings,
    queryFn: api.aiagentsHoldings,
    enabled: false,
  })

  const contextMutation = useMutation({
    mutationFn: () => api.aiagentsContext({
      symbol,
      include_fund_flow: true,
      include_news: true,
      include_overseas: true,
      fund_flow_days: fundDays,
    }),
    onSuccess: setContext,
  })

  const analyzeMutation = useMutation({
    mutationFn: () => api.aiagentsAnalyze({ symbol, include_fund_flow: true, include_news: true }),
    onSuccess: data => {
      setAnalysis(data)
      setContext(data.context)
    },
  })

  const orderMutation = useMutation({
    mutationFn: () => api.aiagentsPaperOrder({
      symbol: order.symbol,
      side: order.side,
      quantity: order.quantity,
      price: order.price ? Number(order.price) : null,
      reason: order.reason,
    }),
    onSuccess: data => {
      qc.setQueryData(QK.aiagentsPaper, data)
      qc.invalidateQueries({ queryKey: QK.aiagentsPaper })
    },
  })

  const statusTone = health.data?.provider_ready ? 'text-bull' : 'text-warning'
  const agentSummaries = useMemo(() => Object.entries(analysis?.agents ?? {}), [analysis])

  return (
    <div className="mx-auto flex max-w-[1500px] flex-col gap-4 p-5">
      <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-lg font-semibold text-foreground">AI 增强</h1>
        </div>
        <button
          onClick={() => health.refetch()}
          className="inline-flex h-8 items-center justify-center gap-1.5 rounded-btn border border-border px-3 text-xs hover:bg-elevated"
        >
          <RefreshCw className="h-3.5 w-3.5" />
          刷新状态
        </button>
      </div>

      <div className={panelCls}>
        <SectionTitle icon={Activity} title="适配器状态" hint={health.data?.checked_at} />
        <div className="grid gap-2 md:grid-cols-4">
          <Metric label="本地数据目录" value={health.data?.data_root_exists ? '已就绪' : '未找到'} tone={statusTone} />
          <Metric label="数据源" value={health.data?.provider_ready ? '已就绪' : '不可用'} tone={statusTone} />
          <Metric label="Tushare" value={health.data?.tushare_available ? '可用' : '未配置'} />
          <Metric label="缓存目录" value={<span className="truncate text-[11px]">{health.data?.cache_dir || '--'}</span>} />
        </div>
        {health.data?.provider_error && <div className="mt-3 rounded bg-warning/10 px-2 py-1 text-[11px] text-warning">{health.data.provider_error}</div>}
      </div>

      <div className={panelCls}>
        <SectionTitle icon={Sparkles} title="个股增强研究" />
        <div className="grid gap-3 md:grid-cols-[18rem_10rem_1fr]">
          <label className="text-[11px] text-secondary">
            股票代码
            <input
              value={symbol}
              onChange={e => setSymbol(e.target.value)}
              className="mt-1 h-9 w-full rounded-btn border border-border bg-base px-3 text-sm text-foreground outline-none focus:border-accent"
            />
          </label>
          <label className="text-[11px] text-secondary">
            资金天数
            <input
              type="number"
              min={5}
              max={120}
              value={fundDays}
              onChange={e => setFundDays(Number(e.target.value))}
              className="mt-1 h-9 w-full rounded-btn border border-border bg-base px-3 text-sm text-foreground outline-none focus:border-accent"
            />
          </label>
          <div className="flex items-end gap-2">
            <button
              onClick={() => contextMutation.mutate()}
              disabled={contextMutation.isPending}
              className="inline-flex h-9 items-center justify-center gap-1.5 rounded-btn bg-elevated px-3 text-xs hover:bg-elevated/80 disabled:opacity-60"
            >
              {contextMutation.isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />}
              拉取上下文
            </button>
            <button
              onClick={() => analyzeMutation.mutate()}
              disabled={analyzeMutation.isPending}
              className="inline-flex h-9 items-center justify-center gap-1.5 rounded-btn bg-accent px-3 text-xs font-medium text-white hover:bg-accent/90 disabled:opacity-60"
            >
              {analyzeMutation.isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Sparkles className="h-3.5 w-3.5" />}
              结构化分析
            </button>
          </div>
        </div>
      </div>

      <DecisionCard decision={analysis?.decision} />
      <ContextPanel context={context ?? undefined} />

      {agentSummaries.length > 0 && (
        <div className={panelCls}>
          <SectionTitle icon={Sparkles} title="分析师摘要" hint={analysis?.generated_at} />
          <div className="grid gap-2 md:grid-cols-2">
            {agentSummaries.map(([key, item]) => (
              <div key={key} className="rounded-card border border-border/70 bg-base/40 px-3 py-2">
                <div className="mb-1 flex items-center justify-between">
                  <span className="text-xs font-medium text-foreground">{key}</span>
                  <span className={cn('text-[10px]', item.status === 'ok' ? 'text-bull' : 'text-warning')}>{item.status}</span>
                </div>
                <div className="text-[11px] leading-relaxed text-secondary">{item.summary}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="grid gap-4 xl:grid-cols-2">
        <div className={panelCls}>
          <SectionTitle icon={Wallet} title="模拟盘" hint={paper.data?.db_path} />
          <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
            <Metric label="总资产" value={fmtMoney(paper.data?.account?.total_asset)} />
            <Metric label="现金" value={fmtMoney(paper.data?.account?.cash)} />
            <Metric label="浮动盈亏" value={fmtMoney(paper.data?.account?.unrealized_pnl)} tone={pctClass(paper.data?.account?.unrealized_pnl)} />
            <Metric label="收益率" value={`${fmt(paper.data?.account?.return_pct)}%`} tone={pctClass(paper.data?.account?.return_pct)} />
          </div>
          <div className="mt-3 grid gap-2 md:grid-cols-[1fr_6rem_6rem_7rem_1fr_auto]">
            <input value={order.symbol} onChange={e => setOrder({ ...order, symbol: e.target.value })} className="h-8 rounded-btn border border-border bg-base px-2 text-xs outline-none focus:border-accent" />
            <select value={order.side} onChange={e => setOrder({ ...order, side: e.target.value as 'BUY' | 'SELL' })} className="h-8 rounded-btn border border-border bg-base px-2 text-xs outline-none focus:border-accent">
              <option value="BUY">BUY</option>
              <option value="SELL">SELL</option>
            </select>
            <input type="number" min={100} step={100} value={order.quantity} onChange={e => setOrder({ ...order, quantity: Number(e.target.value) })} className="h-8 rounded-btn border border-border bg-base px-2 text-xs outline-none focus:border-accent" />
            <input value={order.price} onChange={e => setOrder({ ...order, price: e.target.value })} placeholder="市价" className="h-8 rounded-btn border border-border bg-base px-2 text-xs outline-none focus:border-accent" />
            <input value={order.reason} onChange={e => setOrder({ ...order, reason: e.target.value })} placeholder="原因" className="h-8 rounded-btn border border-border bg-base px-2 text-xs outline-none focus:border-accent" />
            <button onClick={() => orderMutation.mutate()} disabled={orderMutation.isPending} className="inline-flex h-8 items-center justify-center rounded-btn bg-accent px-3 text-xs text-white disabled:opacity-60">
              {orderMutation.isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Send className="h-3.5 w-3.5" />}
            </button>
          </div>
          <div className="mt-3 max-h-56 overflow-auto rounded-card border border-border/70">
            <table className="w-full text-xs">
              <thead className="bg-elevated/60 text-muted">
                <tr><th className="px-2 py-2 text-left">股票</th><th className="px-2 py-2 text-right">数量</th><th className="px-2 py-2 text-right">成本</th><th className="px-2 py-2 text-right">现价</th><th className="px-2 py-2 text-right">盈亏%</th></tr>
              </thead>
              <tbody>
                {(paper.data?.positions ?? []).map((p: any) => (
                  <tr key={p.symbol} className="border-t border-border/60">
                    <td className="px-2 py-2"><StockIdentity symbol={p.symbol} name={p.name} compact /></td>
                    <td className="px-2 py-2 text-right">{p.quantity}</td>
                    <td className="px-2 py-2 text-right">{fmt(p.cost_price)}</td>
                    <td className="px-2 py-2 text-right">{fmt(p.current_price)}</td>
                    <td className={cn('px-2 py-2 text-right', pctClass(p.unrealized_pnl_pct))}>{fmt(p.unrealized_pnl_pct)}%</td>
                  </tr>
                ))}
                {(paper.data?.positions ?? []).length === 0 && <tr><td colSpan={5} className="px-2 py-6 text-center text-muted">暂无持仓</td></tr>}
              </tbody>
            </table>
          </div>
        </div>

        <div className={panelCls}>
          <SectionTitle icon={ShieldAlert} title="持仓优化" hint={holdings.data?.source} />
          <button
            onClick={() => holdings.refetch()}
            disabled={holdings.isFetching}
            className="mb-3 inline-flex h-8 items-center justify-center gap-1.5 rounded-btn bg-elevated px-3 text-xs hover:bg-elevated/80 disabled:opacity-60"
          >
            {holdings.isFetching ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />}
            刷新持仓优化
          </button>
          <div className="space-y-2">
            {(holdings.data?.rows ?? []).map((row: any) => (
              <div key={row.code} className="rounded-card border border-border/70 bg-base/40 px-3 py-2">
                <div className="flex items-center justify-between gap-2">
                  <div className="min-w-0">
                    <div className="truncate text-xs font-medium text-foreground">{row.name ? `${row.name} (${row.code})` : row.code}</div>
                    <div className="mt-1 text-[11px] text-muted">成本 {fmt(row.cost_price)} · 现价 {fmt(row.quote?.price ?? row.technical?.latest_close)}</div>
                  </div>
                  <div className="text-right">
                    <div className={cn('text-sm font-semibold', row.optimization?.score >= 70 ? 'text-bull' : row.optimization?.score < 50 ? 'text-bear' : 'text-warning')}>{fmt(row.optimization?.score, 1)}</div>
                    <div className="text-[10px] text-muted">{row.optimization?.rating}</div>
                  </div>
                </div>
                <div className="mt-2 grid grid-cols-3 gap-2 text-[11px] text-secondary">
                  <div>保护 {fmt(row.optimization?.reference_stop)}</div>
                  <div>止盈 {fmt(row.optimization?.reference_take_profit)}</div>
                  <div>趋势 {row.technical?.trend ?? '--'}</div>
                </div>
              </div>
            ))}
            {holdings.isFetched && (holdings.data?.rows ?? []).length === 0 && <div className="py-8 text-center text-xs text-muted">暂无持仓</div>}
            {!holdings.isFetched && <div className="py-8 text-center text-xs text-muted">尚未加载</div>}
          </div>
        </div>
      </div>
    </div>
  )
}
