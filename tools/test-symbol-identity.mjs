import assert from 'node:assert/strict'
import { normalizeSymbol } from '../frontend/src/lib/symbol.ts'

assert.equal(normalizeSymbol('000001.SZ'), '000001.SZ')
assert.equal(normalizeSymbol('920305.BJ'), '920305.BJ')
assert.equal(normalizeSymbol('000001'), '')
assert.equal(normalizeSymbol('AAPL'), '')
console.log('symbol identity tests passed')
