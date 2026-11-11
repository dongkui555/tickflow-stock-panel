/** Normalize only an already market-qualified A-share symbol.
 *
 * A bare code is ambiguous across stock/index/ETF namespaces and must be
 * resolved by the backend dimension service before it reaches this helper.
 */
export function normalizeSymbol(value: string | null | undefined): string {
  const raw = String(value ?? '').trim().toUpperCase()
  if (!raw) return ''
  const match = raw.match(/^(\d{6})\.(SH|SZ|BJ)$/)
  return match ? `${match[1]}.${match[2]}` : ''
}

export function isCanonicalSymbol(value: string | null | undefined): boolean {
  return normalizeSymbol(value) !== ''
}
