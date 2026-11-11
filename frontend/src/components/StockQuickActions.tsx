import { useMutation, useQueryClient } from '@tanstack/react-query'
import { Loader2, Star, WalletCards } from 'lucide-react'
import { toast } from '@/components/Toast'
import { api } from '@/lib/api'
import { cn } from '@/lib/cn'
import { QK } from '@/lib/queryKeys'

type StockQuickActionsProps = {
  symbol?: string | null
  name?: string | null
  price?: number | null
  quantity?: number
  watchNote?: string
  paperReason?: string
  className?: string
}

function labelOf(symbol?: string | null, name?: string | null) {
  const cleanSymbol = String(symbol ?? '').trim()
  const cleanName = String(name ?? '').trim()
  if (!cleanSymbol) return cleanName || '股票'
  if (!cleanName || cleanName === cleanSymbol) return cleanSymbol
  return `${cleanName} ${cleanSymbol}`
}

function orderPrice(price?: number | null) {
  const value = Number(price)
  return Number.isFinite(value) && value > 0 ? value : null
}

export function StockQuickActions({
  symbol,
  name,
  price,
  quantity = 100,
  watchNote = '推荐列表一键加入',
  paperReason = '推荐列表一键买入模拟仓',
  className,
}: StockQuickActionsProps) {
  const queryClient = useQueryClient()
  const cleanSymbol = String(symbol ?? '').trim()
  const label = labelOf(cleanSymbol, name)
  const disabled = !cleanSymbol

  const addWatchlist = useMutation({
    mutationFn: () => api.watchlistAdd(cleanSymbol, watchNote),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: QK.watchlist })
      queryClient.invalidateQueries({ queryKey: QK.watchlistQuotes })
      queryClient.invalidateQueries({ queryKey: ['watchlist-enriched'] })
      queryClient.invalidateQueries({ queryKey: QK.quoteStatus })
      toast(`已加入自选：${label}`, 'success')
    },
  })

  const addPaper = useMutation({
    mutationFn: () => api.aiagentsPaperOrder({
      symbol: cleanSymbol,
      side: 'BUY',
      quantity,
      price: orderPrice(price),
      reason: paperReason,
    }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: QK.aiagentsPaper })
      queryClient.invalidateQueries({ queryKey: QK.aiagentsHoldings })
      queryClient.invalidateQueries({ queryKey: QK.quoteStatus })
      toast(`已买入模拟仓：${label} ${quantity}股`, 'success')
    },
  })

  const btnCls = 'inline-flex h-7 w-7 items-center justify-center rounded-btn border border-border bg-surface text-muted transition-colors hover:border-accent/50 hover:bg-accent/10 hover:text-accent disabled:cursor-not-allowed disabled:opacity-50'

  return (
    <div className={cn('flex items-center gap-1.5', className)} onClick={event => event.stopPropagation()}>
      <button
        type="button"
        aria-label={`加入自选 ${label}`}
        title={`加入自选：${label}`}
        disabled={disabled || addWatchlist.isPending}
        onClick={() => addWatchlist.mutate()}
        className={btnCls}
      >
        {addWatchlist.isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Star className="h-3.5 w-3.5" />}
      </button>
      <button
        type="button"
        aria-label={`买入模拟仓 ${label} ${quantity}股`}
        title={`买入模拟仓：${label} ${quantity}股`}
        disabled={disabled || addPaper.isPending}
        onClick={() => addPaper.mutate()}
        className={btnCls}
      >
        {addPaper.isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <WalletCards className="h-3.5 w-3.5" />}
      </button>
    </div>
  )
}
