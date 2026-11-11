import { useState, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  Sparkles, LineChart, History as HistoryIcon, Loader2, ExternalLink, Bell, AlertTriangle,
  RadioTower, Newspaper, Globe2, Database, Activity, Users, Zap,
} from 'lucide-react'
import { PageHeader } from '@/components/PageHeader'
import { EmptyState } from '@/components/EmptyState'
import { StockFinancialSearch } from '@/components/financials/StockFinancialSearch'
import { StockPreviewDialog } from '@/components/StockPreviewDialog'
import { LastStockChip } from '@/components/LastStockChip'
import { AnalysisKChart, type PriceLevel, type LevelType } from '@/components/stock-analysis/AnalysisKChart'
import { PriceAlertDialog } from '@/components/stock-analysis/PriceAlertDialog'
import { api, type AIAgentsContext, type StockAnalysisMode } from '@/lib/api'
import { useLastStock } from '@/lib/useLastStock'
import { QK } from '@/lib/queryKeys'
import { toast } from '@/components/Toast'
import {
  startAnalysis, findTodayReport, useHistoryReports,
  deleteReport, openHistoryReport, loadHistory,
} from '@/lib/stockAnalysisStore'

/**
 * 个股分析页 —— 日 K + 关键价位 + 多 Agent 数据覆盖 + AI 综合分析。
 *
 * 与财务分析页的区别:
 *  - 以【行情 + 关键价位】为视觉主体(专用日 K 图表,不复用个股对话框图表)
 *  - AI 分析输出客观技术状态与风险提示(非买卖建议、非财务质量评级)
 *  - 报告胶囊用蓝色系,与财务分析(紫色)并存
 */
export function StockAnalysis() {
  const [symbol, setSymbol] = useState<string>('')
  const [name, setName] = useState<string>('')
  const [checking, setChecking] = useState(false)
  const [confirmReport, setConfirmReport] = useState<{ id: string; created_at: string; focus: string } | null>(null)
  const [previewSymbol, setPreviewSymbol] = useState<string | null>(null)
  const [showPriceAlerts, setShowPriceAlerts] = useState(false)
  const [analysisMode, setAnalysisMode] = useState<StockAnalysisMode>('multi_agent')
  const { last: lastStock, remember: rememberStock } = useLastStock('stock-analysis')

  // 进入页面立即加载历史报告(供右侧常驻列表)。store 内部有 historyLoaded 去重, 重复调用安全。
  useEffect(() => { loadHistory() }, [])

  // 自动恢复上次选中的股票(切走再回来不丢)。useLastStock 的 last 来自 localStorage, 同步可用。
  useEffect(() => {
    if (!symbol && lastStock) {
      setSymbol(lastStock.symbol)
      setName(lastStock.name)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const onSelect = (sym: string, nm: string) => {
    setSymbol(sym)
    setName(nm)
    setConfirmReport(null)
    setShowPriceAlerts(false)
    rememberStock(sym, nm)
  }

  const handleAnalyze = async () => {
    if (!symbol || checking) return
    setChecking(true)
    try {
      // 当日已分析过 → 二次确认(查看今日报告 / 重新分析)
      const today = await findTodayReport(symbol)
      if (today) {
        setConfirmReport({ id: today.id, created_at: today.created_at, focus: today.focus })
      } else {
        await doAnalysis()
      }
    } catch {
      await doAnalysis()
    } finally {
      setChecking(false)
    }
  }

  const doAnalysis = async () => {
    const r = await startAnalysis(symbol, name, '', analysisMode)
    if (r.error) toast(r.error, 'error')
  }

  return (
    <>
      <PageHeader
        title="个股分析"
        subtitle="日 K · 关键价位 · 多 Agent 数据覆盖 · AI 综合分析"
        right={
          <div className="flex items-center gap-2">
            <LastStockChip stock={lastStock} onSelect={onSelect} />
          </div>
        }
      />

      <div className="w-full px-8 py-6 space-y-6">
        {/* 搜索栏 */}
        <div className="flex items-center gap-3">
          <div className="w-72">
            <StockFinancialSearch onSelect={onSelect} assetTypes="stock,index" />
          </div>
          {symbol && (
            <>
              <button
                onClick={() => setPreviewSymbol(symbol)}
                title="查看个股日 K 详情"
                className="group flex items-center gap-2 text-sm rounded-md px-1.5 py-0.5 -mx-1.5 hover:bg-elevated transition-colors"
              >
                <span className="text-foreground font-medium group-hover:text-sky-300 transition-colors">{name || symbol}</span>
                <span className="text-[10px] font-mono text-muted">{symbol}</span>
                <ExternalLink className="h-3 w-3 text-muted opacity-0 group-hover:opacity-100 transition-opacity" />
              </button>
              <AnalysisModeToggle value={analysisMode} onChange={setAnalysisMode} />
              <button
                onClick={handleAnalyze}
                disabled={checking}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-btn bg-gradient-to-r from-sky-500/25 to-blue-500/15 border border-sky-400/30 text-sky-300 text-xs font-medium hover:from-sky-500/35 hover:to-blue-500/25 transition-all disabled:opacity-40 disabled:cursor-not-allowed"
              >
                {checking ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Sparkles className="h-3.5 w-3.5" />}
                {analysisMode === 'multi_agent' ? '6 Agent 分析' : '快速分析'}
              </button>
              <button
                onClick={() => setShowPriceAlerts(true)}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-btn border border-sky-400/25 bg-sky-400/[0.08] text-sky-300 text-xs font-medium hover:border-sky-400/40 hover:bg-sky-400/[0.12] transition-all"
                title="设置价格点位提醒"
              >
                <Bell className="h-3.5 w-3.5" />
                点位提醒
              </button>
            </>
          )}
        </div>

        {/* 主体:左侧当前个股看板 + 右侧常驻历史报告 */}
        <div className="grid grid-cols-[1fr_288px] gap-6 items-start">
          <div className="min-w-0">
            {!symbol ? (
              <EmptyState
                icon={LineChart}
                title="选择一只股票开始分析"
                hint="搜索代码或名称,查看日 K、关键价位与多源数据覆盖,并可让多 Agent 进行技术 / 基本面 / 资金 / 新闻 / 海外综合分析。"
              />
            ) : (
          <StockAnalysisBoard symbol={symbol} />
            )}
          </div>
          <HistorySidebar />
        </div>
      </div>

      {/* 二次确认:已有历史报告 */}
      {confirmReport && (
        <ConfirmModal
          report={confirmReport}
          onView={() => { openHistoryReport(confirmReport.id); setConfirmReport(null) }}
          onRedo={async () => { setConfirmReport(null); await doAnalysis() }}
          onClose={() => setConfirmReport(null)}
        />
      )}

      {/* 个股日 K 详情对话框(点击名称/代码打开) */}
      <StockPreviewDialog
        symbol={previewSymbol}
        name={previewSymbol === symbol ? name : undefined}
        triggerInfo={null}
        onClose={() => setPreviewSymbol(null)}
      />

      {showPriceAlerts && symbol && (
        <PriceAlertDialog
          key={symbol}
          symbol={symbol}
          name={name}
          onClose={() => setShowPriceAlerts(false)}
        />
      )}
    </>
  )
}

function AnalysisModeToggle({
  value,
  onChange,
}: {
  value: StockAnalysisMode
  onChange: (value: StockAnalysisMode) => void
}) {
  const items: Array<{ value: StockAnalysisMode; label: string; icon: any; title: string }> = [
    { value: 'multi_agent', label: '6Agent', icon: Users, title: '六位分析师交叉验证' },
    { value: 'quick', label: '快速', icon: Zap, title: '单模型快速分析' },
  ]
  return (
    <div className="inline-flex h-8 items-center rounded-lg border border-border/60 bg-elevated/30 p-0.5">
      {items.map(item => {
        const active = value === item.value
        const Icon = item.icon
        return (
          <button
            key={item.value}
            type="button"
            title={item.title}
            onClick={() => onChange(item.value)}
            className={`inline-flex h-7 items-center gap-1.5 rounded-md px-2 text-[11px] transition-colors ${
              active
                ? 'bg-sky-500/18 text-sky-300'
                : 'text-muted hover:bg-elevated hover:text-secondary'
            }`}
          >
            <Icon className="h-3.5 w-3.5" />
            {item.label}
          </button>
        )
      })}
    </div>
  )
}

// ===== 分析看板:日 K + 关键价位 =====
function StockAnalysisBoard({ symbol }: { symbol: string }) {
  const kline = useQuery({
    // QK 集中键 (start/end 空串 = 滚动最近窗口, 无日期边界)。收益: 键进集中
    // 管理, 且 useQuoteStream 的焦点股增量合并对无边界键生效 — 此前私拼 3 元
    // 键永远修补不到; 注意合并依赖 _focusSymbol (预览弹窗设置焦点), 看板单独
    // 打开时仍靠 60s staleTime 重拉, 与其他日K入口的键 (带真实日期) 不共享。
    queryKey: QK.kline(symbol, '', ''),
    queryFn: () => api.klineDaily(symbol, 250),
    enabled: !!symbol,
    staleTime: 60_000,
    placeholderData: (prev: any, prevQuery: any) =>
      (prevQuery?.queryKey as readonly unknown[] | undefined)?.[1] === symbol ? prev : undefined,
  })

  const agents = useQuery({
    queryKey: ['aiagents-context', symbol],
    queryFn: () => api.aiagentsContext({
      symbol,
      include_fund_flow: true,
      include_news: true,
      include_overseas: true,
      fund_flow_days: 60,
    }),
    enabled: !!symbol,
    staleTime: 60_000,
  })

  const levelsQ = useQuery({
    queryKey: QK.stockLevels(symbol),
    queryFn: () => api.stockAnalysisLevels(symbol, 250),
    enabled: !!symbol,
    staleTime: 60_000,
  })

  if (kline.isLoading) {
    return <div className="flex items-center justify-center py-20"><Loader2 className="h-5 w-5 animate-spin text-muted" /></div>
  }

  if (kline.isError) {
    return (
      <EmptyState
        icon={AlertTriangle}
        title="日 K 数据加载失败"
        hint="请检查网络或数据源配置后重试。"
      />
    )
  }

  const rows = kline.data?.rows ?? []
  if (rows.length === 0) {
    return <EmptyState icon={LineChart} title="暂无日 K 数据" hint="该标的尚未同步日 K,请先在数据页或自选页同步。" />
  }

  const levels = (levelsQ.data?.levels ?? {}) as Record<LevelType, PriceLevel[]>
  const agentContext = agents.data as AIAgentsContext | undefined

  // 涨跌色:最后一根 K 线收 vs 前一根收(无前日则按开收判断)
  const last = rows[rows.length - 1]
  const prev = rows[rows.length - 2]
  const curClose = levelsQ.data?.close
  const isUp = prev ? (last.close >= prev.close) : (last.close >= last.open)

  return (
    <div className="space-y-4">
      <AgentCoverageBar context={agentContext} loading={agents.isLoading} />
      <div className="rounded-card border border-border/60 bg-surface/40 overflow-hidden">
        <div className="px-4 py-3 border-b border-border/40">
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2 min-w-0">
              <LineChart className="h-4 w-4 text-sky-400 shrink-0" />
              <span className="text-sm font-medium text-foreground">关键价位分析</span>
            </div>
            <div className="flex items-baseline gap-2 shrink-0">
              <span className="text-[10px] text-muted">{rows.length} 个交易日</span>
              <span className="text-[10px] text-muted/60">·</span>
              <span className="text-[10px] text-muted">当前价</span>
              <span className={`text-base font-mono font-bold ${isUp ? 'text-bull' : 'text-bear'}`}>
                {curClose?.toFixed(2) ?? '—'}
              </span>
            </div>
          </div>
        </div>
        <div className="p-3">
          <AnalysisKChart
            rows={rows}
            levels={levels}
            series={levelsQ.data?.series}
            seriesDates={levelsQ.data?.dates}
            defaultLevelTypes={['sr', 'pivot', 'keltner_s']}
            scopeKey={symbol}
            height={480}
          />
        </div>
      </div>
    </div>
  )
}

function AgentCoverageBar({ context, loading }: { context?: AIAgentsContext; loading?: boolean }) {
  const quote = context?.quote ?? {}
  const tech = context?.technical ?? {}
  const fund = context?.fund_flow ?? {}
  const news = context?.news?.news_data ?? {}
  const overseas = context?.overseas?.market ?? {}
  const basic = context?.basic ?? {}
  return (
    <div className="rounded-card border border-border/60 bg-surface/40 p-3">
      <div className="flex items-center gap-2 mb-2">
        <Activity className="h-4 w-4 text-sky-400" />
        <span className="text-sm font-medium text-foreground">多 Agent 数据覆盖</span>
        <span className="ml-auto text-[10px] text-muted">{loading ? '加载中' : context?.generated_at || '--'}</span>
      </div>
      <div className="grid gap-2 md:grid-cols-3 xl:grid-cols-6">
        <CoverageMetric icon={LineChart} label="行情" value={quote.price ?? tech.latest_close} status={tech.data_success ? 'ok' : 'degraded'} />
        <CoverageMetric icon={RadioTower} label="资金" value={fund.data_success ? `${fund.days ?? 0} 天` : '降级'} status={fund.data_success ? 'ok' : 'degraded'} />
        <CoverageMetric icon={Newspaper} label="新闻" value={news.count ?? 0} status={context?.news?.data_success ? 'ok' : 'degraded'} />
        <CoverageMetric icon={Globe2} label="海外" value={overseas.data_success ? '可用' : '降级'} status={overseas.data_success ? 'ok' : 'degraded'} />
        <CoverageMetric icon={Database} label="基本面" value={basic.name || basic.industry || '--'} status={basic && Object.keys(basic).length > 0 ? 'ok' : 'degraded'} />
        <CoverageMetric icon={Sparkles} label="多Agent" value={context?.disclaimer ? '已接入' : '--'} status={context ? 'ok' : 'degraded'} />
      </div>
    </div>
  )
}

function CoverageMetric({
  icon: Icon,
  label,
  value,
  status,
}: {
  icon: any
  label: string
  value: any
  status: 'ok' | 'degraded'
}) {
  return (
    <div className="rounded-card border border-border/60 bg-base/40 px-3 py-2">
      <div className="flex items-center gap-2 text-[10px] text-muted">
        <Icon className="h-3.5 w-3.5" />
        <span>{label}</span>
        <span className={`ml-auto ${status === 'ok' ? 'text-bull' : 'text-warning'}`}>{status === 'ok' ? '正常' : '降级'}</span>
      </div>
      <div className="mt-1 text-sm font-semibold text-foreground truncate">
        {typeof value === 'number' ? value.toFixed(2) : (value ?? '--')}
      </div>
    </div>
  )
}

// ===== 左侧常驻:历史报告侧栏(所有股票,按时间倒序平铺) =====
function HistorySidebar() {
  const { reports, loaded } = useHistoryReports()

  return (
    <aside className="self-start sticky top-0">
      <div className="rounded-card border border-border/60 bg-surface/40 overflow-hidden">
        <div className="px-3 py-2.5 border-b border-border/40 flex items-center gap-2">
          <HistoryIcon className="h-3.5 w-3.5 text-sky-400 shrink-0" />
          <span className="text-xs font-medium text-foreground">历史报告</span>
          {loaded && reports.length > 0 && (
            <span className="ml-auto text-[10px] text-muted">{reports.length}</span>
          )}
        </div>

        {!loaded ? (
          <div className="flex items-center justify-center py-16">
            <Loader2 className="h-4 w-4 animate-spin text-muted" />
          </div>
        ) : reports.length === 0 ? (
          <div className="px-3 py-10 text-center">
            <p className="text-xs text-muted">还没有任何个股分析报告</p>
            <p className="text-[10px] text-muted/60 mt-1">选一只股票,点「AI 个股分析」生成</p>
          </div>
        ) : (
          <div className="max-h-[calc(100vh-220px)] overflow-y-auto p-2 space-y-1.5">
            {reports.map(r => (
              <div
                key={r.id}
                className="group rounded-lg border border-border/40 bg-elevated/20 p-2.5 hover:border-border hover:bg-elevated/40 transition-colors"
              >
                <div className="flex items-center justify-between gap-2">
                  <button
                    onClick={() => openHistoryReport(r.id)}
                    className="flex-1 text-left min-w-0"
                  >
                    <div className="flex items-center gap-1.5 min-w-0">
                      <span className="text-xs font-medium text-foreground truncate">{r.name || r.symbol}</span>
                      <span className="text-[10px] font-mono text-muted shrink-0">{r.symbol}</span>
                    </div>
                    <div className="mt-0.5 flex items-center gap-2 text-[10px] text-muted">
                      <span>{fmtRelative(r.created_at)}</span>
                      {r.close != null && <span className="font-mono">价 {r.close.toFixed(2)}</span>}
                      {r.focus && <span className="text-sky-300/70 truncate">关注: {r.focus}</span>}
                    </div>
                    {r.summary && (
                      <div className="mt-1 text-[11px] text-muted truncate">{r.summary}</div>
                    )}
                  </button>
                  <button
                    onClick={() => { deleteReport(r.id); toast('已删除', 'success') }}
                    className="shrink-0 text-[10px] text-muted/60 hover:text-danger transition-colors px-1 py-0.5 opacity-0 group-hover:opacity-100"
                    title="删除"
                  >
                    删除
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </aside>
  )
}

// ===== 二次确认弹窗 =====
function ConfirmModal({ report, onView, onRedo, onClose }: {
  report: { id: string; created_at: string; focus: string }
  onView: () => void
  onRedo: () => void
  onClose: () => void
}) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4" onClick={onClose}>
      <div
        className="w-full max-w-sm bg-surface border border-border rounded-2xl p-5 shadow-2xl"
        onClick={e => e.stopPropagation()}
      >
        <div className="flex items-center gap-2 mb-2">
          <HistoryIcon className="h-4 w-4 text-sky-400" />
          <span className="text-sm font-medium text-foreground">该个股已有分析报告</span>
        </div>
        <p className="text-xs text-secondary leading-relaxed mb-1">
          最近一次报告生成于 <span className="text-foreground">{fmtRelative(report.created_at)}</span>。
        </p>
        {report.focus && <p className="text-xs text-muted mb-1">关注点: {report.focus}</p>}
        <p className="text-xs text-muted mb-4">可直接查看历史,或重新生成一份新报告。</p>
        <div className="flex gap-2">
          <button onClick={onView}
            className="flex-1 h-8 rounded-lg bg-elevated border border-border text-xs text-secondary hover:text-foreground transition-colors">
            查看历史
          </button>
          <button onClick={onRedo}
            className="flex-1 h-8 rounded-lg bg-gradient-to-r from-sky-500/20 to-blue-500/15 border border-sky-400/30 text-xs text-sky-300 hover:from-sky-500/30 transition-all">
            重新分析
          </button>
        </div>
      </div>
    </div>
  )
}

function fmtRelative(iso: string): string {
  try {
    const t = new Date(iso).getTime()
    const diff = Date.now() - t
    if (diff < 60_000) return '刚刚'
    if (diff < 3600_000) return `${Math.floor(diff / 60_000)} 分钟前`
    if (diff < 86400_000) return `${Math.floor(diff / 3600_000)} 小时前`
    if (diff < 7 * 86400_000) return `${Math.floor(diff / 86400_000)} 天前`
    return new Date(iso).toLocaleDateString('zh-CN')
  } catch { return iso }
}
