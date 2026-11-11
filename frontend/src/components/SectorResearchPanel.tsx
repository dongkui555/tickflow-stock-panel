import { useEffect, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { FlaskConical, Loader2, Play } from 'lucide-react'
import { api, type SectorResearchExperiment } from '@/lib/api'
import { QK } from '@/lib/queryKeys'
import { fmtPct } from '@/lib/format'
import { SectorEntryResults } from './SectorEntryResults'

const labels: Record<string, string> = {
  v2: '新版共振规则', no_flow_change: '移除资金转折', no_parent: '移除一级行业联动',
  no_resilience: '移除相对抗跌', no_volume: '移除温和放量', no_leader: '移除龙头先行', no_position: '移除位置过滤',
  no_market_gate: '移除市场过滤', legacy: '旧版排序（统一数据）',
  relative_strength: '简单相对强度', random_matched: '同数量随机基准',
}

export function SectorResearchPanel() {
  const [jobId, setJobId] = useState('')
  const [phase, setPhase] = useState('holdout')
  const [horizon, setHorizon] = useState(5)
  const [lookback, setLookback] = useState(252)
  const [train, setTrain] = useState(80)
  const [holdout, setHoldout] = useState(40)
  const [capital, setCapital] = useState(100000)
  const [experiment, setExperiment] = useState<SectorResearchExperiment>('stock_entry')
  const [recordDate, setRecordDate] = useState('all')
  const early = experiment !== 'baseline'
  const stockEntry = experiment === 'stock_entry'
  const results = useQuery({queryKey: QK.sectorResearch(experiment), queryFn: () => api.sectorResearch(experiment), staleTime: 30_000})
  const job = useQuery({queryKey: QK.pipelineJob(jobId), queryFn: () => api.pipelineJob(jobId), enabled: !!jobId,
    refetchInterval: query => ['failed', 'succeeded'].includes(query.state.data?.status ?? '') ? false : 3000})
  const start = useMutation({mutationFn: () => api.startSectorResearch({experiment, lookback_days: lookback, train_days: train,
    test_days: 20, holdout_days: holdout, initial_capital: capital}), onSuccess: data => setJobId(data.job_id)})
  const active = start.isPending || (!!jobId && !['failed', 'succeeded'].includes(job.data?.status ?? ''))
  useEffect(() => { if (job.data?.status === 'succeeded') void results.refetch() }, [job.data?.status])
  const rows = (results.data?.comparisons ?? []).filter(row => row.phase === phase && row.horizon === horizon)
  const comparison = (results.data?.comparisons ?? []).filter(row => row.phase === phase && row.variant === 'v2')
  const records = (results.data?.forecast_records ?? []).filter(r => r.phase === phase)
  const recordDates = [...new Set(records.map(r => r.signal_date))].sort().reverse()
  return <section className="border-t border-border py-4" aria-label="板块滚动样本外对照">
    <h3 className="mb-3 flex items-center gap-2 text-sm font-semibold"><FlaskConical className="h-4 w-4 text-accent" />板块滚动样本外对照</h3>
    <div role="tablist" aria-label="板块实验类型" className="mb-4 flex flex-wrap border-b border-border">
      {([{id: 'baseline', label: '原规则对照'}, {id: 'early_layout', label: '3–5日提前布局'}, {id: 'stock_entry', label: '板块内选股与入场'}] as const).map(tab =>
        <button key={tab.id} type="button" role="tab" aria-selected={experiment === tab.id} disabled={active}
          onClick={() => {setExperiment(tab.id); setHorizon(tab.id === 'stock_entry' ? 5 : 1); setRecordDate('all')}}
          className={`px-3 py-2 text-xs disabled:opacity-50 ${experiment === tab.id ? 'border-b-2 border-accent text-accent' : 'text-muted'}`}>{tab.label}</button>)}
    </div>
    <div className="mb-3 flex flex-wrap items-end gap-3">
      {([{label: '历史交易日', value: lookback, set: setLookback, min: 160, max: 504},
        {label: '训练交易日', value: train, set: setTrain, min: 40, max: 200},
        {label: early ? '留出诊断日' : '独立留出日', value: holdout, set: setHoldout, min: 20, max: 80},
        {label: '模拟本金', value: capital, set: setCapital, min: 10000, max: 10000000}]).map(field =>
        <label key={field.label} className="grid w-32 gap-1 text-xs text-muted">{field.label}
          <input aria-label={field.label} type="number" min={field.min} max={field.max} value={field.value}
            disabled={active} onChange={e => field.set(Number(e.target.value))}
            className="h-8 min-w-0 rounded border border-border bg-base px-2 text-foreground" />
        </label>)}
      <button type="button" onClick={() => start.mutate()} disabled={active || lookback < train + holdout + (stockEntry ? 38 : early ? 36 : 30)}
        className="inline-flex h-8 items-center gap-1.5 rounded bg-accent px-3 text-xs text-white disabled:opacity-50">
        {active ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}运行对照
      </button>
      {early && <span className="self-center text-xs text-muted">固定止损 4.5% · 最高总仓位 60% · 历史回放</span>}
    </div>
    {active && <div className="mb-3 text-xs text-muted" role="status">{job.data?.stage ?? '排队中'} · {job.data?.progress ?? 0}%</div>}
    {(start.error || job.data?.error || results.error) && <div role="alert" className="mb-3 text-xs text-bear">{String(start.error?.message ?? job.data?.error ?? results.error?.message)}</div>}
    <div className="mb-3 flex flex-wrap gap-3">
      <select aria-label="验证阶段" value={phase} onChange={e => {setPhase(e.target.value); setRecordDate('all')}} className="h-8 rounded border border-border bg-base px-2 text-xs">
        <option value="holdout">{early ? '留出段诊断（已观察）' : '独立留出段'}</option><option value="walk_forward">滚动样本外</option>
      </select>
      <select aria-label={stockEntry ? '入场后持有期限' : '预测期限'} value={horizon} onChange={e => setHorizon(Number(e.target.value))} className="h-8 rounded border border-border bg-base px-2 text-xs">
        {!stockEntry && <><option value={1}>T+1 当日表现</option><option value={2}>T+2 当日表现</option></>}
        {early && <><option value={3}>{stockEntry ? '入场后持有3日' : '未来3日内启动'}</option><option value={5}>{stockEntry ? '入场后持有5日' : '未来5日内启动'}</option></>}
      </select>
      <span className="self-center text-xs text-muted">{results.data?.generated_at ?? '尚无实验结果'}</span>
    </div>
    {!!results.data?.limitations?.length && <details className="mb-3 text-xs text-warning"><summary>研究限制 · 未批准实盘</summary>
      <ul className="mt-2 space-y-1">{results.data.limitations.map(text => <li key={text}>{text}</li>)}</ul></details>}
    {stockEntry && <SectorEntryResults data={results.data} phase={phase} horizon={horizon} loading={results.isLoading} />}
    {early && !stockEntry && comparison.length > 0 && <div className="mb-5 overflow-x-auto">
      <h4 className="mb-2 text-xs font-semibold">同批候选 · 四期限对照</h4>
      <table className="w-full min-w-[1050px] text-xs" aria-label="四期限对照">
        <thead className="bg-elevated text-muted"><tr>{['期限', '已验证/发出', '命中率', '第3日起启动数', '平均启动等待', '板块单日/累计收益', '板块单日/累计超额', '板块收盘回撤', '模拟收益', '组合回撤'].map(x => <th key={x} className="px-3 py-2 text-left font-medium">{x}</th>)}</tr></thead>
        <tbody>{comparison.map(r => <tr key={r.horizon} className="border-b border-border/60">
          <td className="px-3 py-2">{r.horizon <= 2 ? `T+${r.horizon} 当日确认` : `未来${r.horizon}日内启动`}</td>
          <td className="px-3 py-2">{r.selected}/{r.issued ?? r.selected}</td><td className="px-3 py-2">{fmtPct(r.precision)}</td>
          <td className="px-3 py-2">{r.horizon > 2 ? r.late_hits ?? 0 : '--'}</td>
          <td className="px-3 py-2">{r.mean_start_day == null ? '--' : `${r.mean_start_day.toFixed(1)}日`}</td>
          <td className="px-3 py-2">{fmtPct(r.mean_sector_return)}</td><td className="px-3 py-2">{fmtPct(r.mean_sector_excess)}</td>
          <td className="px-3 py-2">{fmtPct(r.mean_window_drawdown)}</td><td className="px-3 py-2">{fmtPct(r.trading.total_return)}</td>
          <td className="px-3 py-2">{fmtPct(r.trading.max_drawdown)}</td>
        </tr>)}</tbody>
      </table>
    </div>}
    {!stockEntry && (results.isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : !rows.length ? <div className="py-5 text-xs text-muted">暂无该区间实验数据</div> :
      <div className="overflow-x-auto"><table className="w-full min-w-[1380px] text-xs">
        <thead className="bg-elevated text-muted"><tr>{['方案', '有效信号', '命中率', '启动前命中/验证', '延续命中/验证', '新启动召回', '漏判', '板块超额', '模拟收益', '最大回撤', '交易数', '缺失标签', '无信号日'].map(x => <th key={x} className="px-3 py-2 text-left font-medium">{x}</th>)}</tr></thead>
        <tbody>{rows.map(r => <tr key={r.variant} className="border-b border-border/60">
          <td className="px-3 py-2">{labels[r.variant] ?? r.variant}</td><td className="px-3 py-2">{r.selected}</td>
          <td className="px-3 py-2">{fmtPct(r.precision)}</td>
          {['start', 'continuation'].map(kind => <td key={kind} className="px-3 py-2">{r.forecast_kinds?.[kind] ? `${r.forecast_kinds[kind].hits}/${r.forecast_kinds[kind].selected}` : '--'}</td>)}
          <td className="px-3 py-2">{fmtPct(r.new_start_recall)}</td>
          <td className="px-3 py-2">{r.missed_starts}</td><td className="px-3 py-2">{fmtPct(r.mean_sector_excess)}</td>
          <td className="px-3 py-2">{fmtPct(r.trading.total_return)}</td><td className="px-3 py-2">{fmtPct(r.trading.max_drawdown)}</td>
          <td className="px-3 py-2">{r.trading.n_trades ?? 0}</td><td className="px-3 py-2">{r.missing_labels ?? '--'}</td><td className="px-3 py-2">{r.abstention_days}/{r.test_days}</td>
        </tr>)}</tbody></table></div>)}
    {early && !stockEntry && <div className="mt-5 border-t border-border pt-3">
      <div className="mb-3 flex flex-wrap items-center gap-3">
        <h4 className="text-xs font-semibold">启动前预测 · 历史回放明细</h4>
        <select aria-label="预测明细日期" value={recordDate} onChange={e => setRecordDate(e.target.value)} className="h-8 rounded border border-border bg-base px-2 text-xs">
          <option value="all">全部信号日期</option>{recordDates.map(day => <option key={day}>{day}</option>)}
        </select>
        <span className="text-xs text-muted">{records.length} 条强信号 · 不计入正式预测准确率</span>
      </div>
      {!records.length ? <div className="py-3 text-xs text-muted">该区间无启动前强信号记录</div> :
        <div className="max-h-96 overflow-auto"><table className="w-full min-w-[950px] text-xs" aria-label="提前布局预测明细">
          <thead className="sticky top-0 bg-elevated text-muted"><tr>{['信号日', '板块', '分数', 'T+1', 'T+2', '3日窗口', '5日窗口', '首次启动', '5日累计收益', '5日收盘回撤'].map(x => <th key={x} className="px-3 py-2 text-left font-medium">{x}</th>)}</tr></thead>
          <tbody>{records.filter(r => recordDate === 'all' || r.signal_date === recordDate).map(r => <tr key={r.signal_id} className="border-b border-border/60">
            <td className="px-3 py-2">{r.signal_date}</td><td className="px-3 py-2">{r.sector}</td><td className="px-3 py-2">{r.prediction_score.toFixed(1)}</td>
            {[1, 2, 3, 5].map(h => <td key={h} className="px-3 py-2">{r.validations[String(h)]?.status !== 'verified' ? '缺数据' : r.validations[String(h)].hit ? '命中' : '未命中'}</td>)}
            <td className="px-3 py-2">{r.validations['5']?.first_start_date ?? '--'}</td>
            <td className="px-3 py-2">{fmtPct(r.validations['5']?.period_return)}</td><td className="px-3 py-2">{fmtPct(r.validations['5']?.max_drawdown)}</td>
          </tr>)}</tbody>
        </table></div>}
    </div>}
    {!!results.data?.folds?.length && <details className="mt-3 text-xs text-muted"><summary>训练与验证区间</summary>
      <div className="mt-2 overflow-auto"><table className="min-w-[620px] text-left"><thead><tr><th>折</th><th>训练</th><th>验证</th><th>隔离交易日</th></tr></thead>
        <tbody>{results.data.folds.map(f => <tr key={f.fold}><td className="pr-4 py-1">{f.kind === 'holdout' ? (early ? '留出诊断' : '独立留出') : f.fold}</td><td className="pr-4">{f.train_start} ~ {f.train_end}</td><td className="pr-4">{f.test_start} ~ {f.test_end}</td><td>{f.gap_sessions}</td></tr>)}</tbody></table></div></details>}
  </section>
}
