import { useEffect, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { FlaskConical, Loader2, Play, RefreshCw, Square } from 'lucide-react'
import { api } from '@/lib/api'
import { QK } from '@/lib/queryKeys'

const arms = { daily: '日K基线', daily_60m: '日K + 60分钟', daily_15m: '日K + 15分钟', daily_60m_15m: '日K + 60分钟 + 15分钟' }
const inputClass = 'w-full rounded border border-border bg-base px-2 py-1.5 text-xs'
const value = (n: number | null | undefined, percent = false) =>
  n == null || !Number.isFinite(n) ? '--' : `${(n * (percent ? 100 : 1)).toFixed(percent ? 2 : 4)}${percent ? '%' : ''}`

export function MultiPeriodExperiment() {
  const qc = useQueryClient()
  const today = new Date().toLocaleDateString('en-CA', { timeZone: 'Asia/Shanghai' })
  const [start, setStart] = useState(`${Number(today.slice(0, 4)) - 1}${today.slice(4)}`)
  const [end, setEnd] = useState(today)
  const [symbols, setSymbols] = useState(50)
  const [horizon, setHorizon] = useState('1')
  const [split, setSplit] = useState('test')
  const [jobId, setJobId] = useState(() => localStorage.getItem('multiperiod-job') ?? '')
  const reportQ = useQuery({ queryKey: QK.stockformerMultiperiod, queryFn: api.stockformerMultiperiod })
  const jobQ = useQuery({
    queryKey: QK.pipelineJob(jobId), queryFn: () => api.pipelineJob(jobId), enabled: !!jobId,
    retry: false,
    refetchInterval: query => ['pending', 'running'].includes(query.state.data?.status ?? 'pending') ? 3000 : false,
  })
  const run = useMutation({
    mutationFn: () => api.stockformerRunMultiperiod({ start, end, max_symbols: symbols }),
    onSuccess: result => { setJobId(result.job_id); localStorage.setItem('multiperiod-job', result.job_id) },
  })
  const cancel = useMutation({
    mutationFn: () => api.pipelineCancel(jobId),
    onSuccess: () => void qc.invalidateQueries({ queryKey: QK.pipelineJob(jobId) }),
  })
  useEffect(() => {
    if (jobQ.data?.status === 'succeeded') void qc.invalidateQueries({ queryKey: QK.stockformerMultiperiod })
  }, [jobQ.data?.status, qc])
  const busy = run.isPending || !!jobId && !jobQ.isError && ['pending', 'running'].includes(jobQ.data?.status ?? 'pending')
  const report = reportQ.data
  const metrics = report?.horizons?.[horizon]?.metrics?.[split]
  const error = run.error?.message || cancel.error?.message || reportQ.error?.message || jobQ.error?.message || jobQ.data?.error

  return <section id="multiperiod-experiment" className="border-y border-border py-4" aria-label="多周期特征实验">
    <header className="mb-3 flex flex-wrap items-center gap-2">
      <FlaskConical className="h-4 w-4 text-accent" />
      <h2 className="text-sm font-semibold">多周期特征实验</h2>
      <span className="text-xs text-muted">研究结果 · 未替换生产模型</span>
      <button type="button" className="ml-auto p-2" title="刷新实验结果" aria-label="刷新实验结果" onClick={() => void reportQ.refetch()}><RefreshCw className="h-4 w-4" /></button>
    </header>
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
      <label className="text-xs text-muted">开始日期<input type="date" value={start} max={end} disabled={busy} className={inputClass} onChange={e => setStart(e.target.value)} /></label>
      <label className="text-xs text-muted">结束日期<input type="date" value={end} min={start} max={today} disabled={busy} className={inputClass} onChange={e => setEnd(e.target.value)} /></label>
      <label className="text-xs text-muted">训练前固定股票数<input type="number" value={symbols} min={20} max={200} disabled={busy} className={inputClass} onChange={e => setSymbols(Number(e.target.value))} /></label>
      <button type="button" disabled={busy || !start || !end || start >= end || symbols < 20 || symbols > 200} onClick={() => run.mutate()} className="inline-flex items-center justify-center gap-2 self-end rounded bg-accent px-3 py-2 text-xs text-white disabled:opacity-50">
        {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}运行四组对照
      </button>
    </div>
    {busy && <div role="status" className="mt-3 text-xs text-muted">
      <progress className="w-full" value={jobQ.data?.progress ?? 0} max={100} />
      {jobQ.data?.log.at(-1)?.msg || '准备实验'}
      {!!jobId && <button type="button" title="停止实验" aria-label="停止实验" disabled={cancel.isPending} onClick={() => cancel.mutate()} className="ml-2 inline-flex rounded border border-border p-1"><Square className="h-3 w-3" /></button>}
    </div>}
    {error && <p role="alert" className="mt-3 break-words text-xs text-bear">{error}</p>}
    {!report?.run_id && !reportQ.isPending && <p className="mt-4 text-xs text-muted">暂无已完成的多周期对照实验</p>}
    {report?.run_id && <>
      <div className="my-3 flex flex-wrap items-center gap-2 text-xs">
        <label>验证区间 <select aria-label="实验验证区间" className="rounded border border-border bg-base p-1" value={split} onChange={e => setSplit(e.target.value)}>
          <option value="test">独立测试集</option><option value="validation">验证集</option><option value="train">训练集</option>
        </select></label>
        <div role="group" aria-label="预测期限" className="flex gap-1">
          {['1', '3', '5', '10'].map(h => <button type="button" key={h} aria-pressed={horizon === h} onClick={() => setHorizon(h)} className={`rounded px-2 py-1 ${horizon === h ? 'bg-accent text-white' : 'bg-base text-muted'}`}>{h}日</button>)}
        </div>
        <span className="text-muted">{report.symbols?.length}只 · 共同样本 {report.coverage?.common_rows.toLocaleString()} / {report.coverage?.baseline_rows.toLocaleString()} ({value(report.coverage?.common_ratio, true)})</span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-right text-xs tabular-nums">
          <thead className="border-b border-border text-muted"><tr>
            {['模型特征', 'IC', 'Rank IC', '方向准确率', 'RMSE', 'Top5远期均值', '池内超额均值', '样本 / 交易日'].map(h => <th key={h} className="whitespace-nowrap px-3 py-2 font-normal first:text-left">{h}</th>)}
          </tr></thead>
          <tbody>{Object.entries(arms).map(([key, name]) => {
            const m = metrics?.[key]
            return <tr key={key} className="border-b border-border/60">
              <td className="whitespace-nowrap px-3 py-2 text-left">{name}</td>
              <td className="px-3 py-2">{value(m?.ic)}</td><td className="px-3 py-2">{value(m?.rank_ic)}</td>
              <td className="px-3 py-2">{value(m?.direction_accuracy, true)}</td><td className="px-3 py-2">{value(m?.rmse, true)}</td>
              <td className="px-3 py-2">{value(m?.top_k_forward_return, true)}</td><td className="px-3 py-2">{value(m?.top_k_excess, true)}</td>
              <td className="whitespace-nowrap px-3 py-2">{m ? `${m.rows} / ${m.dates}` : '--'}</td>
            </tr>
          })}</tbody>
        </table>
      </div>
      <div className="mt-3 space-y-1 text-xs text-muted">
        <p>训练起点 {report.config?.train_start} · 验证起点 {report.config?.validation_start} · 测试 {report.config?.test_start} 至 {report.config?.end}</p>
        <p>未通过分钟完整性及日K量价核验：{report.excluded_stock_days?.toLocaleString()} 个股票日</p>
        <p className="break-all">实验 {report.run_id} · 特征版本 {report.version}</p>
      </div>
      <details className="mt-3 text-xs text-muted"><summary className="cursor-pointer">实验口径与限制</summary>
        <ul className="mt-2 list-inside list-disc space-y-1">{report.limitations?.map(item => <li key={item}>{item}</li>)}</ul>
      </details>
    </>}
  </section>
}
