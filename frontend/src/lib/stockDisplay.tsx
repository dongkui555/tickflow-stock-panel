import { cn } from '@/lib/cn'

function clean(value?: string | null): string {
  return String(value ?? '').trim()
}

export function stockDisplay(symbol?: string | null, name?: string | null) {
  const cleanSymbol = clean(symbol)
  const cleanName = clean(name)
  if (!cleanSymbol) return cleanName || '--'
  if (!cleanName || cleanName === cleanSymbol) return cleanSymbol
  return `${cleanName} (${cleanSymbol})`
}

interface StockIdentityProps {
  symbol?: string | null
  name?: string | null
  compact?: boolean
  className?: string
}

export function StockIdentity({ symbol, name, compact = false, className }: StockIdentityProps) {
  const cleanSymbol = clean(symbol)
  const cleanName = clean(name)

  if (compact) {
    return (
      <span className={cn('inline-flex min-w-0 items-center gap-1', className)}>
        <span className="truncate font-medium text-foreground">{cleanName || cleanSymbol || '--'}</span>
        {cleanName && cleanSymbol && cleanName !== cleanSymbol && (
          <span className="shrink-0 font-mono text-[10px] text-muted">({cleanSymbol})</span>
        )}
      </span>
    )
  }

  return (
    <div className={cn('min-w-0', className)}>
      <div className="truncate font-medium text-foreground">{cleanName || cleanSymbol || '--'}</div>
      {cleanName && cleanSymbol && cleanName !== cleanSymbol && <div className="font-mono text-[10px] text-muted">{cleanSymbol}</div>}
    </div>
  )
}
