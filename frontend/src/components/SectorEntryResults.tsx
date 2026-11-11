import { useState } from 'react'
import type { SectorEntryCondition, SectorResearchResult } from '@/lib/api'
import { fmtBigNum, fmtPct } from '@/lib/format'

const variants: Record<string, string> = {
  leader_open: '原龙头 · 次日开盘', ranked_open: '原量价排序 · 次日开盘',
  leader_confirm: '原龙头 · 严格确认', ranked_confirm: '原量价排序 · 严格确认',
  ranked_no_pattern: '原排序 · 移除形态', ranked_no_candle: '原排序 · 移除收阳',
  ranked_no_location: '原排序 · 移除强收盘', ranked_no_volume: '原排序 · 移除放量',
  strength_open: '强弱分层 · 次日开盘', strength_confirm: '强弱分层 · 严格确认',
  strength_adaptive: '强弱分层 · 分形态确认',
}
const conditionLabels: Record<string, string> = {
  data: '数据质量', risk_name: '风险警示 / 涨停', liquidity: '成交额 / 换手', heat: '追涨位置',
  trend: '中期趋势', risk_distance: '波动 / 支撑距离', relative_strength: '板块内相对强度', recovery: '短线收复',
  support: '冻结支撑', advance: '信号后涨幅', market: '市场环境', limits: '涨停约束',
  pattern: '突破 / 回踩形态', candle: '收盘方向', location: '收盘位置', volume: '量能确认',
}
const fieldLabels: Record<string, string> = {
  close: '收盘', open: '开盘', ma5: 'MA5', ma20: 'MA20', ma20_prior: '5日前MA20', prior_close: '昨收',
  relative5: '5日板块超额', ret_3d: '3日涨幅', ret_5d: '5日涨幅', position20: '20日位置', atr_pct: 'ATR比例',
  support_distance: '支撑距离', ma20_distance: 'MA20偏离', close_location: '收盘位置', amount_yuan: '成交额',
  turnover_pct: '换手率', amount_ratio_5d: '量额比', price_coverage: '价格覆盖', flow_coverage: '资金覆盖',
  support: '冻结支撑', return_since_signal: '信号后涨幅', limit_up: '涨停标记', limit_known: '涨停口径已知',
  weak: '弱市', very_weak: '极弱市', event_risk: '事件风险', breakout: '突破', reclaim: '回踩收复',
  stock_available: '个股数据', history_ok: '历史连续', name: '名称', low10: '前10日低点', amount: '成交额', turnover_rate: '换手率',
}
const fractional = new Set(['relative5', 'ret_3d', 'ret_5d', 'atr_pct', 'support_distance', 'ma20_distance', 'return_since_signal'])
const prices = new Set(['close', 'open', 'ma5', 'ma20', 'ma20_prior', 'prior_close', 'support', 'low10'])
function observation(key: string, value: string | number | boolean | null) {
  if (value == null) return '--'
  if (typeof value === 'boolean') return value ? '是' : '否'
  if (typeof value !== 'number') return value
  if (fractional.has(key)) return fmtPct(value)
  if (key === 'amount_yuan' || key === 'amount') return fmtBigNum(value)
  if (key === 'turnover_pct' || key === 'turnover_rate') return `${value.toFixed(2)}%`
  if (key.endsWith('_coverage')) return `${(value * 100).toFixed(1)}%`
  return value.toFixed(prices.has(key) ? 4 : 3)
}
function ConditionDetails({ rows }: { rows?: SectorEntryCondition[] }) {
  if (!rows?.length) return null
  return <div className="my-2 space-y-2 border-l border-border pl-2">{rows.map(c => <div key={c.id}>
    <div className={c.status === 'passed' ? 'text-muted' : 'text-warning'}>
      {conditionLabels[c.id] ?? c.label} · {!c.active ? '实验移除' : c.status === 'passed' ? '通过' : c.status === 'missing' ? '缺数据' : '未通过'}{c.hard ? ' · 硬约束' : ''}
    </div>
    <div className="mt-1 text-muted">门槛: {c.required}</div>
    <div className="mt-1 break-words">{Object.entries(c.observed).map(([key, value]) => `${fieldLabels[key] ?? key}: ${observation(key, value)}`).join(' · ')}</div>
  </div>)}</div>
}
const statuses: Record<string, string> = {
  no_candidate: '无合格标的', expired: '确认超时', invalidated: '计划失效',
  missing_data: '数据不足', merged: '重复计划合并', scheduled: '入场计划',
  closed: '已平仓', opening_blocked: '开盘放弃', not_filled: '未成交或未平仓',
}

export function SectorEntryResults({ data, phase, horizon, loading }: {
  data?: SectorResearchResult; phase: string; horizon: number; loading: boolean
}) {
  const [variant, setVariant] = useState('ranked_confirm')
  const [stage, setStage] = useState<'selection' | 'confirmation'>('confirmation')
  const comparisons = (data?.entry_comparisons ?? []).filter(r => r.phase === phase && r.horizon === horizon)
  const records = (data?.entry_records ?? []).filter(r => r.phase === phase && r.variant === variant)
  const selected = comparisons.find(r => r.variant === variant)
  const diagnostics = selected?.condition_diagnostics?.[stage] ?? []
  if (loading) return <div role="status" className="py-5 text-xs text-muted">正在加载实验结果</div>
  return <div>
    <h4 className="mb-2 text-xs font-semibold">板块内选股与入场 · {comparisons.length || '--'}组对照</h4>
    {!comparisons.length ? <div className="py-5 text-xs text-muted">暂无选股与入场实验结果</div> :
      <div className="overflow-x-auto"><table className="w-full min-w-[1080px] text-xs" aria-label="选股与入场对照">
        <thead className="bg-elevated text-muted"><tr>{['方案', '原始板块信号', '入场计划', '模拟收益', '最大回撤', '平仓胜率', '平仓数', '止损数', '无标的', '确认超时/失效', '缺数据', '开盘放弃', '买入换手'].map(x => <th key={x} className="px-3 py-2 text-left font-medium">{x}</th>)}</tr></thead>
        <tbody>{comparisons.map(r => <tr key={r.variant} className="border-b border-border/60">
          <td className="px-3 py-2">{variants[r.variant] ?? r.variant}</td><td className="px-3 py-2">{r.issued}</td>
          <td className="px-3 py-2">{r.scheduled}</td><td className="px-3 py-2">{fmtPct(r.trading.total_return)}</td>
          <td className="px-3 py-2">{fmtPct(r.trading.max_drawdown)}</td><td className="px-3 py-2">{fmtPct(r.win_rate)}</td>
          <td className="px-3 py-2">{r.trading.n_trades ?? 0}</td><td className="px-3 py-2">{r.stop_count}</td>
          <td className="px-3 py-2">{r.status_counts.no_candidate ?? 0}</td>
          <td className="px-3 py-2">{r.status_counts.expired ?? 0}/{r.status_counts.invalidated ?? 0}</td>
          <td className="px-3 py-2">{r.status_counts.missing_data ?? 0}</td><td className="px-3 py-2">{r.opening_rejections}</td>
          <td className="px-3 py-2">{fmtPct(r.trading.turnover)}</td>
        </tr>)}</tbody>
      </table></div>}
    <div className="mb-2 mt-5 flex flex-wrap items-center gap-3">
      <h4 className="text-xs font-semibold">条件逐项检验</h4>
      <select aria-label="入场明细方案" value={variant} onChange={e => setVariant(e.target.value)} className="h-8 rounded border border-border bg-base px-2 text-xs">
        {Object.entries(variants).map(([key, label]) => <option key={key} value={key}>{label}</option>)}
      </select>
      <span className="text-xs text-muted">{records.length} 条记录 · 固定4.5%止损 · 仅研究</span>
    </div>
    {selected && !selected.trading.n_trades && <p role="status" className="my-3 text-xs text-warning">本方案暂无平仓交易，尚不能评估盈利能力。</p>}
    <div role="tablist" aria-label="条件检验阶段" className="mb-2 flex border-b border-border">
      {(['selection', 'confirmation'] as const).map(s => <button key={s} role="tab" aria-selected={stage === s} onClick={() => setStage(s)}
        className={`border-b-2 px-3 py-2 text-xs ${stage === s ? 'border-accent text-accent' : 'border-transparent text-muted'}`}>
        {s === 'selection' ? '选股条件' : '入场确认'}
      </button>)}
    </div>
    {!diagnostics.length ? <div className="py-3 text-xs text-muted">该方案暂无逐项条件记录</div> : <div className="overflow-x-auto">
      <table className="w-full min-w-[1000px] text-xs" aria-label="条件检验统计">
        <thead className="bg-elevated text-muted"><tr>{['条件', '约束', '检查次数', '通过', '未通过', '缺数据', '唯一阻断', '重叠阻断', '顺序剩余', '实验移除', '门槛'].map(x => <th key={x} className="px-3 py-2 text-left font-medium">{x}</th>)}</tr></thead>
        <tbody>{diagnostics.map(c => <tr key={c.id} className="border-b border-border/60 align-top">
          <td className="px-3 py-2">{conditionLabels[c.id] ?? c.label}</td><td className="px-3 py-2">{c.hard ? '硬约束' : '选股 / 确认'}</td>
          {[c.tested, c.passed, c.failed, c.missing, c.sole_blocker, c.disabled === c.tested ? 0 : c.failed + c.missing - c.sole_blocker, c.survivors, c.disabled].map((v, i) => <td key={i} className="px-3 py-2 tabular-nums">{v}</td>)}
          <td className="max-w-80 px-3 py-2">{c.required}</td>
        </tr>)}</tbody>
      </table>
    </div>}
    <h4 className="mb-2 mt-5 text-xs font-semibold">标的与确认记录</h4>
    {!records.length ? <div className="py-3 text-xs text-muted">该区间无标的记录</div> :
      <div className="max-h-[480px] overflow-auto"><table className="w-full min-w-[1050px] text-xs" aria-label="选股入场审计">
        <thead className="sticky top-0 bg-elevated text-muted"><tr>{['预测日', '板块', '股票', '评分', '确认日', '计划买入日', '结果', '单笔收益', '依据'].map(x => <th key={x} className="px-3 py-2 text-left font-medium">{x}</th>)}</tr></thead>
        <tbody>{records.map((r, i) => {
          const execution = r.executions?.[String(horizon)]
          return <tr key={`${r.signal_date}:${r.sector}:${i}`} className="border-b border-border/60 align-top">
            <td className="px-3 py-2">{r.signal_date}</td><td className="px-3 py-2">{r.sector}</td>
            <td className="px-3 py-2">{r.symbol ? <>{r.name || '名称缺失'}<div className="text-muted">{r.symbol}</div></> : '--'}</td>
            <td className="px-3 py-2">{r.score?.toFixed(1) ?? '--'}</td><td className="px-3 py-2">{r.confirmation_date ?? '--'}</td>
            <td className="px-3 py-2">{r.scheduled_entry_date ?? '--'}</td>
            <td className="px-3 py-2">{statuses[execution?.status ?? r.status] ?? r.status}{execution?.exit_reason === 'stop_loss' && <div className="text-warning">触发止损</div>}</td>
            <td className="px-3 py-2">{fmtPct(execution?.pnl_pct)}</td>
            <td className="max-w-80 px-3 py-2">
              <div className="mb-1 text-muted">{execution?.reason ?? r.reasons?.join('、')}</div>
              <details><summary className="cursor-pointer text-accent">评分 / 确认</summary>
                {r.original_symbol && <div className="mt-1 text-muted">原龙头 {r.original_symbol}</div>}
                {r.shortlist.map(s => <div key={s.symbol} className="mt-2">{s.name || s.symbol} {s.symbol} · {s.score.toFixed(1)}
                  <div className="mt-1 text-muted">{Object.entries(s.components).map(([k,v]) => `${k} ${v.toFixed(0)}`).join(' · ')}</div></div>)}
                {r.shortlist.map(s => <details key={`conditions:${s.symbol}`} className="mt-2"><summary className="cursor-pointer">{s.name || s.symbol} 选股条件</summary><ConditionDetails rows={s.conditions} /></details>)}
                {r.checks.map(c => <details key={c.date} className="mt-2"><summary className="cursor-pointer">{c.date} · {c.status === 'confirmed' ? `确认通过 (${c.pattern})` : c.reasons.join('、')}</summary><ConditionDetails rows={c.conditions} /></details>)}
                {!!r.rejected?.length && <details className="mt-2"><summary className="cursor-pointer">排除样本 ({r.rejected.length})</summary>
                  {r.rejected.map(s => <div key={s.symbol} className="mt-2">{s.name || '名称缺失'} {s.symbol} · {s.reasons.join('、')}<ConditionDetails rows={s.conditions} /></div>)}
                </details>}
                <div className="mt-2 text-muted">{Object.entries(r.filter_reasons).map(([k,v]) => `${k} ${v}`).join(' · ')}</div>
              </details>
            </td>
          </tr>
        })}</tbody>
      </table></div>}
  </div>
}
