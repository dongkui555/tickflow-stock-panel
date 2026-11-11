import { useMemo, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { Check, Database, Loader2, Play, Search, X } from 'lucide-react'

import { api } from '@/lib/api'

const DEFAULT_PARAMS = JSON.stringify({
  ts_code: '000001.SZ',
  start_date: '20260901',
  end_date: '20260919',
}, null, 2)

export function TushareCatalogPanel() {
  const [search, setSearch] = useState('')
  const [apiName, setApiName] = useState('daily')
  const [source, setSource] = useState<'auto' | 'promax' | 'datahub'>('auto')
  const [paramsText, setParamsText] = useState(DEFAULT_PARAMS)
  const [inputError, setInputError] = useState('')

  const catalog = useQuery({
    queryKey: ['tushare-api-catalog'],
    queryFn: api.tushareCatalog,
    staleTime: 60_000,
  })
  const query = useMutation({
    mutationFn: api.tushareQuery,
  })

  const items = useMemo(() => {
    const needle = search.trim().toLowerCase()
    const all = catalog.data?.items ?? []
    if (!needle) return all
    return all.filter(item =>
      item.api.toLowerCase().includes(needle)
      || item.label.includes(search.trim())
      || item.category.includes(search.trim()),
    )
  }, [catalog.data?.items, search])

  const run = () => {
    try {
      const params = JSON.parse(paramsText) as Record<string, unknown>
      if (!params || Array.isArray(params) || typeof params !== 'object') {
        throw new Error('参数必须是 JSON 对象')
      }
      setInputError('')
      query.mutate({ api: apiName, source, params, max_rows: 100 })
    } catch (error) {
      setInputError(error instanceof Error ? error.message : '参数格式错误')
    }
  }

  const sources = catalog.data?.sources
  return (
    <div className="rounded-card border border-border bg-surface overflow-hidden">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border px-4 py-3">
        <div className="flex items-center gap-2">
          <Database className="h-4 w-4 text-accent" />
          <div>
            <div className="text-sm font-medium text-foreground">Tushare 接口中心</div>
            <div className="text-[11px] text-muted">
              {catalog.data?.count ?? 0} 个白名单接口 · 主备链路 {catalog.data?.source_chain || '—'}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-3 text-[11px]">
          <SourceState label="Promax" ready={sources?.promax} />
          <SourceState label="datahubco" ready={sources?.datahub} />
          <span className="text-muted">竞价: {catalog.data?.auction_source ?? '—'}</span>
        </div>
      </div>

      <div className="grid gap-4 p-4 lg:grid-cols-[minmax(260px,0.8fr)_minmax(380px,1.2fr)]">
        <div className="min-w-0">
          <div className="relative mb-2">
            <Search className="pointer-events-none absolute left-2.5 top-2.5 h-3.5 w-3.5 text-muted" />
            <input
              value={search}
              onChange={event => setSearch(event.target.value)}
              className="h-9 w-full rounded-btn border border-border bg-base pl-8 pr-3 text-xs text-foreground outline-none focus:border-accent"
              placeholder="搜索接口、名称或分类"
            />
          </div>
          <div className="max-h-64 overflow-auto rounded-btn border border-border divide-y divide-border/60">
            {items.map(item => (
              <button
                type="button"
                key={item.api}
                onClick={() => setApiName(item.api)}
                className={`flex w-full items-center justify-between gap-3 px-3 py-2 text-left text-xs ${apiName === item.api ? 'bg-accent/10 text-accent' : 'text-secondary hover:bg-hover'}`}
              >
                <span className="min-w-0">
                  <span className="font-mono text-foreground">{item.api}</span>
                  <span className="ml-2">{item.label}</span>
                </span>
                <span className="flex shrink-0 items-center gap-1.5">
                  {item.auto_sync && <span className="text-[10px] text-success">自动</span>}
                  {!item.http_supported && <span className="text-[10px] text-danger">SDK限定</span>}
                  <span className="text-[10px] text-muted">{item.category}</span>
                </span>
              </button>
            ))}
          </div>
        </div>

        <div className="min-w-0 space-y-3">
          <div className="flex flex-wrap items-center gap-2">
            <select
              value={source}
              onChange={event => setSource(event.target.value as typeof source)}
              className="h-9 rounded-btn border border-border bg-base px-3 text-xs text-foreground outline-none focus:border-accent"
            >
              <option value="auto">自动主备</option>
              <option value="promax">仅 Promax</option>
              <option value="datahub">仅 datahubco</option>
            </select>
            <button
              type="button"
              onClick={run}
              disabled={query.isPending}
              className="inline-flex h-9 items-center gap-1.5 rounded-btn bg-accent px-3 text-xs font-medium text-white disabled:opacity-50"
            >
              {query.isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
              试调 {apiName}
            </button>
          </div>
          <textarea
            value={paramsText}
            onChange={event => setParamsText(event.target.value)}
            spellCheck={false}
            className="h-32 w-full resize-y rounded-btn border border-border bg-base p-3 font-mono text-xs text-foreground outline-none focus:border-accent"
            aria-label="Tushare 接口参数"
          />
          {(inputError || query.isError) && (
            <div className="text-xs text-danger">{inputError || String(query.error instanceof Error ? query.error.message : query.error)}</div>
          )}
          {query.data && (
            <div className="space-y-2">
              <div className="flex flex-wrap items-center gap-3 text-[11px] text-muted">
                <span>来源 <span className="font-mono text-foreground">{query.data.source}</span></span>
                <span>总行数 <span className="font-mono text-foreground">{query.data.row_count}</span></span>
                <span>返回 <span className="font-mono text-foreground">{query.data.returned_rows}</span></span>
                <span>字段 {query.data.fields.length}</span>
              </div>
              <pre className="max-h-52 overflow-auto rounded-btn border border-border bg-base p-3 text-[11px] text-secondary">
                {JSON.stringify(query.data.items.slice(0, 10), null, 2)}
              </pre>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function SourceState({ label, ready }: { label: string; ready?: boolean }) {
  return (
    <span className={`inline-flex items-center gap-1 ${ready ? 'text-success' : 'text-muted'}`}>
      {ready ? <Check className="h-3 w-3" /> : <X className="h-3 w-3" />}
      {label}
    </span>
  )
}
