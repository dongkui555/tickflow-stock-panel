import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api'
import { QK } from '@/lib/queryKeys'

const labels: Record<string, string> = {
  waiting_close: '等待盘后', waiting_data: '等待数据', recorded: '已记录',
  late_record: '迟到记录（排除前向）', inputs_changed: '数据变化，等待重试',
  blocked_protocol_changed: '冻结版本变化，已暂停', error: '记录失败，等待核查',
  missed_session: '错过当日快照', calendar_unknown: '交易日待确认', non_trading_day: '非交易日',
  model_no_signal: '模型无有效周期权重', prediction_gate_empty: '预测门槛清空',
  net_margin_empty: '净优势不足', entry_rules_empty: '入场规则清空',
  market_gate_empty: '市场闸门清空', candidate_selected: '已选候选', late_snapshot: '预测生成过晚',
  computing_features: '正在加载特征与计算冻结预测', before_start: '尚未到起始日期',
  decision_pool_empty: '基础股票池清空',
}

export function StockFormerDailyShadow() {
  return <div className="space-y-4"><ShadowPanel kind="repaired" /><ShadowPanel kind="decision" /><ShadowPanel kind="frozen" /><ShadowPanel kind="legacy" /></div>
}

function ShadowPanel({ kind }: { kind: 'repaired' | 'decision' | 'frozen' | 'legacy' }) {
  const frozenPair = kind !== 'legacy'
  const title = kind === 'repaired' ? '修正版研究模型 · 独立每日影子账户' : kind === 'decision' ? '717 研究模型 · 原始排序与统一股票池' : frozenPair ? '原冻结组合 · 每日影子验证' : '旧研究模型 · 每日影子记录'
  const query = useQuery({
    queryKey: kind === 'repaired' ? QK.stockformerRepairedPairShadow : kind === 'decision' ? QK.stockformerDecisionPairShadow : frozenPair ? QK.stockformerFrozenPairShadow : QK.stockformerDailyShadow,
    queryFn: kind === 'repaired' ? api.stockformerRepairedPairShadow : kind === 'decision' ? api.stockformerDecisionPairShadow : frozenPair ? api.stockformerFrozenPairShadow : api.stockformerDailyShadow,
    refetchInterval: 60_000,
    staleTime: 30_000,
  })
  const data = query.data
  if (!query.isLoading && !data?.enabled && !query.error) return null
  return (
    <section className="min-w-0 rounded-card border border-border bg-surface/80 p-4" aria-label={title}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-xs font-semibold">{title}</h2>
        <button className="text-xs text-accent" onClick={() => query.refetch()} disabled={query.isFetching}>刷新记录</button>
      </div>
      {query.error && <div role="alert" className="mt-2 text-xs text-warning">{query.error.message}</div>}
      {query.isLoading && <div className="mt-2 text-xs text-muted">正在加载影子记录…</div>}
      {data && <>
        <div className="mt-2 break-all text-[11px] text-secondary">
          {frozenPair ? `树模型＋配套校准 ${data.pair_id?.slice(0, 16) ?? ''}` : data.model_id} · 起始 {data.start_date} · {labels[data.state.status ?? ''] ?? data.state.status ?? '等待调度'}
          {data.state.reason && <span> · {data.state.reason}</span>}
        </div>
        <div className="mt-2 text-[11px] text-secondary">
          已记录 {data.recorded_days} 个交易日 · 模拟成交 {data.fills} 笔 · 账户 {data.account_id}
          {!frozenPair && data.fills === 0 && <span> · 收益待有效成交后评估</span>}
          {frozenPair && <span> · 完成交易 {data.completed_trades ?? 0} 笔
            {data.cost_after_return_pct != null
              ? <span> · 成本后组合收益 {data.cost_after_return_pct.toFixed(2)}%</span>
              : <span> · 成本后收益待有效成交与结算后评估</span>}
          </span>}
        </div>
        {frozenPair && <div className={`mt-2 text-[11px] ${data.worker?.alive ? 'text-secondary' : 'text-warning'}`}>
          后台任务{data.worker?.alive ? '运行中' : '心跳未确认，请核查'} · 树、校准和信号规则整体冻结 · 独立模拟账户
        </div>}
        {data.research_warnings?.length ? <div className="mt-2 space-y-1 text-[11px] text-warning" role="note">
          {data.research_warnings.map(warning => <p key={warning}>{warning}</p>)}
        </div> : null}
        <p className="mt-2 text-[11px] text-muted">每日{kind === 'repaired' ? '19:10' : '19:00'}后检查收盘数据，缺数据每15分钟重试。零候选仍留档；未在当日生成的快照不补记过去开盘成交。研究模型保持冻结。</p>
        <div className="mt-3 overflow-x-auto">
          <table className="w-full whitespace-nowrap text-left text-[11px]">
            <thead className="text-muted"><tr>{['日期 / 生成时间', '状态 / 原因', '进入闸门', ...(frozenPair ? ['原始正分', '校准正分'] : []), '预测通过', '净优势通过', '入场通过', '市场通过', '预测＋净优势', '再经入场', '再经市场', '正式候选'].map(label => <th className="p-2 font-medium" key={label}>{label}</th>)}</tr></thead>
            <tbody>{data.days.slice(-20).reverse().map(row => <tr key={row.as_of} className="border-t border-border/60">
              <td className="p-2">{row.as_of}<div className="text-[10px] text-muted">{row.generated_at?.replace('T', ' ') ?? '尚未生成'}</div>{row.next_actual_session_open && <div className="text-[10px] text-muted">最早执行 {row.next_actual_session_open.replace('T', ' ')}</div>}</td>
              <td className="p-2">{labels[row.status] ?? row.status}<div className="text-[10px] text-muted">{labels[row.observation?.no_trade_reason ?? ''] ?? row.reason ?? row.observation?.no_trade_reason}</div></td>
              {[row.observation?.total_rows, ...(frozenPair ? [row.raw_positive_rows, row.calibrated_positive_rows] : []), row.observation?.prediction_pass, row.observation?.net_margin_pass, row.observation?.entry_rules_pass, row.observation?.market_gate_pass, row.observation?.prediction_net_margin_pass, row.observation?.prediction_net_entry_pass, row.observation?.final_candidate_pass, row.observation?.selected_count].map((value, i) => <td className="p-2 font-mono" key={i}>{value ?? '--'}</td>)}
            </tr>)}</tbody>
          </table>
          {data.days.length === 0 && <div className="py-3 text-xs text-muted">已接入调度，等待第一个交易日盘后记录。</div>}
        </div>
      </>}
    </section>
  )
}
