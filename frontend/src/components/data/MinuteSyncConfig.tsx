import { useState, useEffect, useId, useMemo, useRef } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Loader2, Trash2, Download, Calendar, SlidersHorizontal, Activity, RefreshCw } from 'lucide-react'
import { api } from '@/lib/api'
import { QK } from '@/lib/queryKeys'
import { MissingCapChip } from '@/lib/capability-labels'
import { MinuteGapPanel } from '@/components/data/MinuteGapPanel'

type MinuteSyncScope = 'combined' | 'watchlist' | 'realtime_watchlist' | 'holdings' | 'stockformer_top' | 'manual' | 'all'

const SCOPE_OPTIONS: { value: MinuteSyncScope; label: string; hint: string }[] = [
  { value: 'combined', label: '持仓+自选+推荐', hint: '默认：覆盖短线盯盘重点池' },
  { value: 'watchlist', label: '自选股', hint: '只同步自选列表' },
  { value: 'holdings', label: '持仓股', hint: '只同步模拟/持仓股票' },
  { value: 'stockformer_top', label: '推荐TopN', hint: '按 StockFormer 多因子推荐池' },
  { value: 'manual', label: '手动输入', hint: '只同步输入的股票代码' },
  { value: 'realtime_watchlist', label: '实时盯盘列表', hint: '当前实时行情列表' },
  { value: 'all', label: '全市场A股', hint: '极慢，仅少量天数调试' },
]

function normalizeSymbols(raw: string): string[] {
  const out: string[] = []
  const seen = new Set<string>()
  raw.replace(/，/g, ',').split(/[\s,;；]+/).forEach(item => {
    let symbol = item.trim().toUpperCase()
    if (!symbol) return
    if (symbol.includes('.')) symbol = symbol.split('.')[0]
    if (/^\d+$/.test(symbol)) symbol = symbol.padStart(6, '0')
    if (!seen.has(symbol)) {
      seen.add(symbol)
      out.push(symbol)
    }
  })
  return out
}

// hasCap: 分钟K能力当前是否可用 (路由矩阵判定, 生效源含插件/自定义源)
export function MinuteSyncConfig({ hasCap, onJobStart, onClose }: { hasCap: boolean; onJobStart?: (jobId: string) => void; onClose?: () => void }) {
  const [tab, setTab] = useState<'sync' | 'health'>('sync')
  const tabId = useId()
  const syncTabRef = useRef<HTMLButtonElement>(null)
  const healthTabRef = useRef<HTMLButtonElement>(null)
  const qc = useQueryClient()
  const prefs = useQuery({
    queryKey: QK.preferences,
    queryFn: api.preferences,
  })
  const refreshStatus = useQuery({
    queryKey: ['minute-refresh-status'],
    queryFn: api.minuteRefreshStatus,
    refetchInterval: 5000,
  })
  const update = useMutation({
    mutationFn: ({ enabled, days, segmentDays, scope, topN, symbols }: {
      enabled: boolean
      days: number
      segmentDays?: number
      scope?: MinuteSyncScope
      topN?: number
      symbols?: string[]
    }) => api.updateMinuteSync(enabled, days, { segmentDays, scope, topN, symbols }),
    onSuccess: () => qc.invalidateQueries({ queryKey: QK.preferences }),
  })

  const hasMinuteCap = hasCap
  const enabled = prefs.data?.minute_sync_enabled ?? false
  const days = prefs.data?.minute_sync_days ?? 5
  const segmentDays = prefs.data?.minute_sync_segment_days ?? 20
  const scope = (prefs.data?.minute_sync_scope ?? 'combined') as MinuteSyncScope
  const topN = prefs.data?.minute_sync_top_n ?? 100
  const manualSymbols = prefs.data?.minute_sync_symbols ?? []
  const [localDays, setLocalDays] = useState(days)
  const [localSegment, setLocalSegment] = useState(segmentDays)
  const [localScope, setLocalScope] = useState<MinuteSyncScope>(scope)
  const [localTopN, setLocalTopN] = useState(topN)
  const [manualText, setManualText] = useState(manualSymbols.join('\n'))
  const parsedSymbols = useMemo(() => normalizeSymbols(manualText), [manualText])

  useEffect(() => { setLocalDays(days) }, [days])
  useEffect(() => { setLocalSegment(segmentDays) }, [segmentDays])
  useEffect(() => { setLocalScope(scope) }, [scope])
  useEffect(() => { setLocalTopN(topN) }, [topN])
  useEffect(() => { setManualText(manualSymbols.join('\n')) }, [manualSymbols.join('|')])

  const savePrefs = (overrides: {
    enabled?: boolean
    days?: number
    segmentDays?: number
    scope?: MinuteSyncScope
    topN?: number
    symbols?: string[]
  } = {}) => {
    update.mutate({
      enabled: overrides.enabled ?? enabled,
      days: overrides.days ?? localDays,
      segmentDays: overrides.segmentDays,
      scope: overrides.scope ?? localScope,
      topN: overrides.topN ?? localTopN,
      symbols: overrides.symbols ?? parsedSymbols,
    })
  }

  const handleToggle = () => {
    if (!hasMinuteCap) return
    savePrefs({ enabled: !enabled })
  }

  const setDays = (v: number) => {
    const clamped = Math.max(1, Math.min(30, v))
    setLocalDays(clamped)
    savePrefs({ days: clamped })
  }

  // 段大小: 交易日/段, 步进 ±5, 范围 [5, 30]。越小越省内存但越慢。
  const setSegment = (v: number) => {
    const clamped = Math.max(5, Math.min(30, v))
    setLocalSegment(clamped)
    savePrefs({ segmentDays: clamped })
  }

  const setScope = (v: MinuteSyncScope) => {
    setLocalScope(v)
    savePrefs({ scope: v })
  }

  const setTopN = (v: number) => {
    const clamped = Math.max(1, Math.min(1000, v))
    setLocalTopN(clamped)
    savePrefs({ topN: clamped })
  }

  const saveManual = () => {
    savePrefs({ symbols: parsedSymbols })
  }

  // 清空分钟K数据 (二次确认)
  const [confirmClear, setConfirmClear] = useState(false)
  const clearMutation = useMutation({
    mutationFn: () => api.clearMinute(),
    onSuccess: () => {
      setConfirmClear(false)
      qc.invalidateQueries({ queryKey: QK.dataStatus })
    },
  })

  const intradayMutation = useMutation({
    mutationFn: () => api.minuteRefreshRun(),
    onSuccess: () => {
      refreshStatus.refetch()
      qc.invalidateQueries({ queryKey: QK.dataStatus })
    },
  })

  // 历史分钟补齐 (补最新尾巴 / 向前补历史互相独立, 不影响盘中增量刷新)
  const [fetchingMode, setFetchingMode] = useState<'' | 'latest' | 'backfill' | '1y' | '2y'>('')
  const handleFetch = (mode: 'latest' | 'backfill' | '1y' | '2y') => {
    if (!hasMinuteCap) return
    const fetchDays = mode === '2y' ? 730 : mode === '1y' ? 365 : mode === 'backfill' ? localSegment : localDays
    const extend = mode !== 'latest'
    setFetchingMode(mode)
    api.syncMinute({
      days: fetchDays,
      extend,
      mode: extend ? 'history_backfill' : 'history_increment',
      scope: localScope,
      topN: localTopN,
      symbols: parsedSymbols,
    }).then((res) => {
      qc.invalidateQueries({ queryKey: QK.pipelineJobs })
      qc.invalidateQueries({ queryKey: QK.dataStatus })
      // 通知主页面跟踪 job 进度 (ActiveJobCard 会显示实时进度+日志)
      if (res.job_id && onJobStart) onJobStart(res.job_id)
    }).finally(() => setFetchingMode(''))
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div
        role="tablist"
        aria-label="分钟数据管理"
        className="flex shrink-0 gap-5 border-b border-border px-4 sm:px-5"
        onKeyDown={e => {
          if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key)) return
          e.preventDefault()
          const next = e.key === 'Home' ? 'sync' : e.key === 'End' ? 'health' : tab === 'sync' ? 'health' : 'sync'
          setTab(next)
          ;(next === 'sync' ? syncTabRef : healthTabRef).current?.focus()
        }}
      >
        {([
          { key: 'sync', label: '同步与获取', icon: SlidersHorizontal, ref: syncTabRef },
          { key: 'health', label: '数据体检', icon: Activity, ref: healthTabRef },
        ] as const).map(({ key, label, icon: Icon, ref }) => (
          <button
            key={key}
            ref={ref}
            role="tab"
            id={`${tabId}-${key}-tab`}
            aria-controls={`${tabId}-${key}-panel`}
            aria-selected={tab === key}
            tabIndex={tab === key ? 0 : -1}
            onClick={() => setTab(key)}
            className={`inline-flex items-center gap-1.5 border-b-2 py-3 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-accent/60 ${tab === key ? 'border-accent text-accent' : 'border-transparent text-muted hover:text-foreground'}`}
          >
            <Icon className="h-3.5 w-3.5" />{label}
          </button>
        ))}
      </div>
      <div
        role="tabpanel"
        id={`${tabId}-sync-panel`}
        aria-labelledby={`${tabId}-sync-tab`}
        hidden={tab !== 'sync'}
        tabIndex={0}
        className="min-h-0 flex-1 space-y-5 overflow-y-auto overscroll-contain p-4 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-accent/60 sm:p-5"
      >
      {/* 区块 A: 自动同步 (盘后定时拉取的偏好设置) */}
      <div className="space-y-4 rounded-card border border-border bg-base/30 p-3 sm:p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2.5">
          <button
            role="switch"
            aria-label="自动同步"
            aria-checked={enabled}
            onClick={handleToggle}
            disabled={!hasMinuteCap}
            className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors duration-200 shrink-0 ${
              enabled ? 'bg-accent shadow-[0_0_6px_rgba(61,214,140,0.3)]' : 'bg-elevated'
            } ${!hasMinuteCap ? 'opacity-40 cursor-not-allowed' : 'cursor-pointer'}`}
          >
            <span
              className={`inline-block h-3.5 w-3.5 rounded-full bg-white shadow-sm transition-transform duration-200 ${
                enabled ? 'translate-x-[18px]' : 'translate-x-0.5'
              }`}
            />
          </button>
          <span className="text-xs text-foreground font-medium">
            自动同步{enabled ? '已开启' : '已关闭'}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex items-center">
            <button
              onClick={() => setDays(localDays - 1)}
              aria-label="减少自动同步天数"
              disabled={!hasMinuteCap || !enabled || localDays <= 1}
              className="h-6 w-6 flex items-center justify-center rounded-l-btn bg-elevated border border-border text-secondary hover:bg-border/50 disabled:opacity-30 transition-colors text-xs"
            >−</button>
            <div className={`h-6 w-8 flex items-center justify-center border-y border-border text-[11px] font-mono tabular-nums ${enabled ? 'text-foreground bg-base' : 'text-muted bg-elevated/50'}`}>
              {localDays}
            </div>
            <button
              onClick={() => setDays(localDays + 1)}
              aria-label="增加自动同步天数"
              disabled={!hasMinuteCap || !enabled || localDays >= 30}
              className="h-6 w-6 flex items-center justify-center rounded-r-btn bg-elevated border border-border text-secondary hover:bg-border/50 disabled:opacity-30 transition-colors text-xs"
            >+</button>
          </div>
          <span className="text-[10px] text-muted">天</span>
          {!hasMinuteCap && (
            <MissingCapChip capKey="kline.minute.batch" />
          )}
        </div>
      </div>

      {/* 分段大小: 控制单次 SDK 请求覆盖的交易日数。每段拉完即落盘,避免全量攒内存 OOM。
          「往前获取」与「获取 1 年」共用此设置 (两者都经过 sync_and_persist_minute)。 */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-t border-border/60 pt-4">
        <div className="flex items-center gap-1.5">
          <span className="text-xs text-foreground font-medium">分段大小</span>
          <span className="text-[10px] text-muted px-1 py-px rounded bg-warning/8 text-warning/80">内存优化</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex items-center">
            <button
              onClick={() => setSegment(localSegment - 5)}
              aria-label="减小分段大小"
              disabled={!hasMinuteCap || localSegment <= 5}
              className="h-6 w-6 flex items-center justify-center rounded-l-btn bg-elevated border border-border text-secondary hover:bg-border/50 disabled:opacity-30 transition-colors text-xs"
            >−</button>
            <div className="h-6 w-8 flex items-center justify-center border-y border-border text-[11px] font-mono tabular-nums bg-base text-foreground">
              {localSegment}
            </div>
            <button
              onClick={() => setSegment(localSegment + 5)}
              aria-label="增大分段大小"
              disabled={!hasMinuteCap || localSegment >= 30}
              className="h-6 w-6 flex items-center justify-center rounded-r-btn bg-elevated border border-border text-secondary hover:bg-border/50 disabled:opacity-30 transition-colors text-xs"
            >+</button>
          </div>
          <span className="text-[10px] text-muted">交易日/段</span>
        </div>
      </div>
      <div className="text-[11px] text-muted leading-relaxed">
        每段拉完即写盘，避免内存堆积。分段越小越省内存，获取耗时越长；默认 20 个交易日。设置自动保存。
      </div>

      <div className="pt-2 border-t border-border space-y-2">
        <div className="flex items-center justify-between gap-2">
          <div>
            <div className="text-xs text-foreground font-medium">同步范围</div>
            <div className="text-[10px] text-muted">
              {SCOPE_OPTIONS.find(item => item.value === localScope)?.hint}
            </div>
          </div>
          <select
            value={localScope}
            onChange={event => setScope(event.target.value as MinuteSyncScope)}
            disabled={!hasMinuteCap}
            className="h-7 w-36 rounded-input border border-border bg-base px-2 text-[11px] text-foreground outline-none focus:border-accent disabled:opacity-40"
          >
            {SCOPE_OPTIONS.map(item => (
              <option key={item.value} value={item.value}>{item.label}</option>
            ))}
          </select>
        </div>

        {(localScope === 'combined' || localScope === 'stockformer_top') && (
          <div className="flex items-center justify-between gap-2">
            <span className="text-[11px] text-secondary">推荐池 TopN</span>
            <div className="flex items-center">
              <button
                onClick={() => setTopN(localTopN - 50)}
                disabled={!hasMinuteCap || localTopN <= 1}
                className="h-6 w-6 flex items-center justify-center rounded-l-btn bg-elevated border border-border text-secondary hover:bg-border/50 disabled:opacity-30 transition-colors text-xs"
              >−</button>
              <div className="h-6 w-12 flex items-center justify-center border-y border-border text-[11px] font-mono tabular-nums bg-base text-foreground">
                {localTopN}
              </div>
              <button
                onClick={() => setTopN(localTopN + 50)}
                disabled={!hasMinuteCap || localTopN >= 1000}
                className="h-6 w-6 flex items-center justify-center rounded-r-btn bg-elevated border border-border text-secondary hover:bg-border/50 disabled:opacity-30 transition-colors text-xs"
              >+</button>
            </div>
          </div>
        )}

        {(localScope === 'manual' || localScope === 'combined') && (
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-[11px] text-secondary">手动代码</span>
              <button
                onClick={saveManual}
                disabled={!hasMinuteCap || update.isPending}
                className="px-2 py-0.5 rounded-btn bg-elevated text-[10px] text-secondary hover:text-foreground hover:bg-border/50 disabled:opacity-40 transition-colors"
              >
                保存
              </button>
            </div>
            <textarea
              value={manualText}
              onChange={event => setManualText(event.target.value)}
              disabled={!hasMinuteCap}
              placeholder="000001, 600519"
              rows={2}
              className="w-full resize-none rounded-input border border-border bg-base px-2 py-1.5 text-[11px] text-foreground outline-none focus:border-accent disabled:opacity-40"
            />
            <div className="text-[10px] text-muted">
              已识别 {parsedSymbols.length} 只；可用逗号、空格或换行分隔
            </div>
          </div>
        )}

        {localScope === 'all' && (
          <div className="rounded-md border border-warning/25 bg-warning/8 px-2 py-1.5 text-[10px] text-warning/90 leading-relaxed">
            全市场分钟K数据量极大，建议只拉 1-3 天做排障；短线实盘优先使用重点股票池。
          </div>
        )}
      </div>
      </div>

      {/* 区块 B: 手动获取 (一次性操作, 独立于上方自动同步开关) */}
      <div className="space-y-3">
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
          <Download className="h-3 w-3 text-secondary" />
          <span className="text-[11px] text-secondary font-medium">历史分钟补齐</span>
          <span className="text-[10px] text-muted">补本地历史库</span>
        </div>
        <div className="grid grid-cols-2 gap-2">
        <button
          onClick={() => handleFetch('latest')}
          disabled={!hasMinuteCap || fetchingMode !== ''}
          className="inline-flex min-h-20 flex-col items-center justify-center gap-1 px-2 py-3 rounded-btn bg-accent/90 text-white text-xs font-medium hover:bg-accent disabled:opacity-40 transition-colors duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60 focus-visible:ring-offset-2 focus-visible:ring-offset-surface"
        >
          {fetchingMode === 'latest' ? (
            <><Loader2 className="h-3.5 w-3.5 animate-spin" /><span>补齐中…</span></>
          ) : (
            <><Download className="h-3.5 w-3.5" /><span>增量补最新</span><span className="text-[9px] opacity-70">{localDays}天窗口</span></>
          )}
        </button>
        <button
          onClick={() => handleFetch('backfill')}
          disabled={!hasMinuteCap || fetchingMode !== ''}
          className="inline-flex flex-col items-center justify-center gap-0.5 px-2 py-2 rounded-btn bg-elevated text-secondary text-xs font-medium hover:bg-border/50 disabled:opacity-40 transition-colors duration-150"
        >
          {fetchingMode === 'backfill' ? (
            <><Loader2 className="h-3.5 w-3.5 animate-spin" /><span>回补中…</span></>
          ) : (
            <><Download className="h-3.5 w-3.5" /><span>向前补 {localSegment} 天</span></>
          )}
        </button>
        <button
          onClick={() => handleFetch('1y')}
          disabled={!hasMinuteCap || fetchingMode !== ''}
          className="inline-flex min-h-20 flex-col items-center justify-center gap-1 px-2 py-3 rounded-btn border border-accent/30 bg-accent/5 text-accent text-xs font-medium hover:bg-accent/10 disabled:opacity-40 transition-colors duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60"
        >
          {fetchingMode === '1y' ? (
            <><Loader2 className="h-3.5 w-3.5 animate-spin" /><span>分段获取中…</span></>
          ) : (
            <><Calendar className="h-3.5 w-3.5" /><span>获取最近 1 年</span><span className="text-[9px] opacity-70">分段拉取</span></>
          )}
        </button>
        <button
          onClick={() => handleFetch('2y')}
          disabled={!hasMinuteCap || fetchingMode !== ''}
          className="inline-flex flex-col items-center justify-center gap-0.5 px-2 py-2 rounded-btn border border-danger/35 bg-danger/10 text-danger/90 text-xs font-medium hover:bg-danger/15 disabled:opacity-40 transition-colors duration-150"
        >
          {fetchingMode === '2y' ? (
            <><Loader2 className="h-3.5 w-3.5 animate-spin" /><span>分段获取中…</span></>
          ) : (
            <><Calendar className="h-3.5 w-3.5" /><span>获取最近 2 年</span><span className="text-[9px] opacity-70">长任务</span></>
          )}
        </button>
        </div>
        <div className="text-[10px] text-muted leading-relaxed">
          A股标的 · 前复权价格 · 增量补最新按单股最后分钟线补尾巴；向前补历史从本地最早数据继续回补。
        </div>
      </div>

      {/* 区块 C: 盘中增量刷新 */}
      <div className="pt-3 border-t border-border space-y-2">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-1.5">
            <RefreshCw className="h-3 w-3 text-secondary" />
            <span className="text-[11px] text-secondary font-medium">盘中增量刷新</span>
            <span className={`text-[10px] px-1.5 py-px rounded ${
              refreshStatus.data?.gate_reason ? 'bg-warning/10 text-warning' : 'bg-accent/10 text-accent'
            }`}>
              {refreshStatus.data?.running ? '服务运行中' : '服务未运行'}
            </span>
          </div>
          <button
            onClick={() => intradayMutation.mutate()}
            disabled={!hasMinuteCap || intradayMutation.isPending}
            className="inline-flex items-center gap-1.5 px-2 py-1 rounded-btn bg-elevated text-[10px] text-secondary hover:text-foreground hover:bg-border/50 disabled:opacity-40 transition-colors"
          >
            {intradayMutation.isPending ? <Loader2 className="h-3 w-3 animate-spin" /> : <RefreshCw className="h-3 w-3" />}
            立即跑一轮
          </button>
        </div>
        <div className="grid grid-cols-3 gap-2 text-[10px]">
          <div className="rounded-md border border-border bg-base px-2 py-1.5">
            <div className="text-muted">间隔</div>
            <div className="text-foreground font-mono">{refreshStatus.data?.interval_seconds ?? '-'}s</div>
          </div>
          <div className="rounded-md border border-border bg-base px-2 py-1.5">
            <div className="text-muted">上轮写入</div>
            <div className="text-foreground font-mono">{refreshStatus.data?.last_rows ?? 0}</div>
          </div>
          <div className="rounded-md border border-border bg-base px-2 py-1.5">
            <div className="text-muted">覆盖标的</div>
            <div className="text-foreground font-mono">{refreshStatus.data?.last_symbols ?? 0}</div>
          </div>
        </div>
        {(refreshStatus.data?.gate_reason || intradayMutation.data?.reason) && (
          <div className="rounded-md border border-warning/25 bg-warning/8 px-2 py-1.5 text-[10px] text-warning/90 leading-relaxed">
            {intradayMutation.data?.reason || refreshStatus.data?.gate_reason}
          </div>
        )}
      </div>
      </div>
      <div
        role="tabpanel"
        id={`${tabId}-health-panel`}
        aria-labelledby={`${tabId}-health-tab`}
        hidden={tab !== 'health'}
        tabIndex={0}
        className="min-h-0 flex-1 overflow-y-auto overscroll-contain p-4 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-accent/60 sm:p-5"
      >
        <MinuteGapPanel hasCap={hasMinuteCap} onJobStart={onJobStart} />
      </div>
      <div className="flex shrink-0 items-center justify-between gap-3 border-t border-border bg-base/20 px-4 py-3 sm:px-5">
      {/* 区块 C: 清空 (危险操作, 独立分隔) */}
      <button
        onClick={() => setConfirmClear(true)}
        disabled={clearMutation.isPending}
        title="清空分钟K数据"
        className="inline-flex items-center justify-center gap-1.5 px-2.5 py-2 rounded-btn text-danger/80 text-[11px] font-medium hover:bg-danger/10 disabled:opacity-40 transition-colors duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-danger/60"
      >
        <Trash2 className="h-3 w-3" />
        清空分钟K数据
      </button>
      {onClose && <button onClick={onClose} className="rounded-btn border border-border bg-elevated px-4 py-1.5 text-xs text-secondary hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60">关闭</button>}
      </div>

      {/* 清空确认弹窗 */}
      {confirmClear && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={() => !clearMutation.isPending && setConfirmClear(false)} />
          <div className="relative rounded-card border border-border bg-surface shadow-2xl mx-4 px-6 py-5 max-w-sm w-full space-y-4">
            <div className="text-sm text-foreground text-center font-medium">确认清空分钟K数据？</div>
            <div className="text-[11px] text-muted text-center leading-relaxed">
              此操作仅删除分钟K (kline_minute) 数据, <span className="text-foreground/80">不影响</span>日K、复权因子、指标等其他数据。清空后可重新获取。
            </div>
            <div className="flex items-center justify-center gap-3">
              <button onClick={() => setConfirmClear(false)} disabled={clearMutation.isPending}
                className="px-4 py-1.5 rounded-btn bg-elevated text-secondary text-xs hover:bg-elevated/80 transition-colors duration-150">取消</button>
              <button onClick={() => clearMutation.mutate()} disabled={clearMutation.isPending}
                className="px-4 py-1.5 rounded-btn bg-danger/90 text-foreground text-xs font-medium hover:bg-danger disabled:opacity-40 transition-colors duration-150">
                {clearMutation.isPending ? <span className="inline-flex items-center gap-1.5"><Loader2 className="h-3 w-3 animate-spin" />清空中…</span> : '确认清空'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
