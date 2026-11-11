"""Apply reviewed resolutions for shared frontend contracts."""
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
conflicts = Path(r'F:\tick-panel-upgrade-20261009\integration\conflicts')
plans = {
    'frontend/src/pages/settings/DataSources.tsx': ['local'],
    'frontend/src/pages/settings/DataSourceEditor.tsx': ['datasets', 'both', 'both'],
    'frontend/src/pages/settings/Monitoring.tsx': ['up', 'monitorRoute', 'local'],
    'frontend/src/pages/Lots.tsx': ['up'] * 3,
    'frontend/src/pages/Signals.tsx': ['up'],
}
for name, plan in plans.items():
    index = [0]
    def resolve(match):
        a, b = match.groups()
        choice = plan[index[0]]
        index[0] += 1
        if choice == 'local': return a
        if choice == 'up': return b
        if choice == 'both': return a + b
        if choice == 'datasets': return b.replace("'full_minute'", "'full_minute', 'financial'")
        if choice == 'monitorRoute': return a + b[b.index('              : rs?.repair_only'):]
        if choice == 'dailyQuery': return a.replace('    queryKey:', '    ...(!minutePeriod ? klineDailyQueryOptions(symbol, dateRange, extColumns) : {}),\n    queryKey:')
        if choice == 'markerPills': return b.replace('{showMarkerToggle && showLimitMarkers &&', '{!minutePeriod && showMarkerToggle && showLimitMarkers &&').replace('{showMarkerToggle && addedDate &&', '{!minutePeriod && showMarkerToggle && addedDate &&')
        if choice == 'volumeMarker': return a + '          addedDate={minutePeriod ? null : (showAddedMark ? addedDate : null)}\n'
        if choice == 'scheduleDoc': return a.replace('15:30', '15:35')
        if choice == 'trackedCapacity': return a.replace('        result = fn(on_progress=progress)', '        result = run_with_capacity(job_id, lambda: fn(on_progress=progress))')
        if choice == 'dataSteps':
            return b.replace("    { label: '指标' },", "    { label: '指标' },\n    { label: '财务' },\n    { label: '资金' },\n    { label: '竞价' },\n    { label: '新闻' },")
        if choice == 'dashboardImports': return a.replace('useDataStatus, useCapabilityMatrix,', 'useDataStatus, useCapabilities, useCapabilityMatrix,')
        if choice == 'menuModels': return ''.join(line for line in a.splitlines(keepends=True) if "'/stock-analysis'" not in line) + b
        if choice == 'menuFunds': return ''.join(line for line in a.splitlines(keepends=True) if "'/fund-monitor'" in line) + b
        if choice == 'quotePoll': return a.replace('memory_only=self._paused)', 'memory_only=self._paused, final_boundary_ms=self._final_boundary_ms(phase))')
        if choice == 'quoteFailure': return b.replace('                    else:\n                        logger.debug', '                else:\n                    logger.debug')
        if choice == 'quoteSignature': return b.replace('final_boundary_ms: int | None = None) -> bool:', 'final_boundary_ms: int | None = None, memory_only: bool = False) -> bool:')
        if choice == 'quoteFetch':
            head = b[:b.index('    def _fetch_full_market_quotes(')].replace('self._fetch_full_market_quotes(final_boundary_ms=final_boundary_ms)', 'self._fetch_full_market_quotes(final_boundary_ms=final_boundary_ms, memory_only=memory_only)').replace('        return updated', '        if updated:\n            self._record_fetch_success()\n        else:\n            self._record_fetch_failure("empty_or_rate_limited")\n        return updated')
            tail = a[a.index('    def _record_fetch_success'):].replace('def _fetch_full_market_quotes(self, *, memory_only: bool = False) -> None:', 'def _fetch_full_market_quotes(self, *, memory_only: bool = False, final_boundary_ms: int | None = None):').replace('            self._process_full_market_records(records,', '            return self._process_full_market_records(records,').replace('require_stock=not partial)', 'require_stock=not partial, final_boundary_ms=final_boundary_ms)').replace('        self._process_full_market_records(rows,', '        return self._process_full_market_records(rows,').replace('merge_existing=True)', 'merge_existing=True, final_boundary_ms=final_boundary_ms)')
            return head + tail
        if choice == 'quoteIndices': return b.replace('records = records + fetched_indices', 'records.extend(self._coerce_realtime_records(fetched_indices))')
        if choice == 'quoteProcess': return a.replace('    ) -> None:', '        replace_index_cache: bool = True,\n        final_boundary_ms: int | None = None,\n    ) -> tuple[pl.DataFrame, pl.DataFrame | None] | None:')
        if choice == 'quoteEtf': return a[:a.index('        # ----')] + b[b.index('        # ----'):]
        if choice == 'syncImports': return b.replace('import shutil', 'import os\nimport shutil').replace('date, datetime, timedelta', 'date, datetime, time as dt_time, timedelta')
        if choice == 'syncServices': return 'from app.parquet import read_parquet_safe\n' + b
        if choice == 'dailySources':
            chunk = b[b.index('            iter_daily ='):b.index('            df = provider.get_daily(')]
            chunk = chunk.replace('            if callable(iter_daily):', '            if callable(iter_daily) and not replace_existing:')
            chunk = ''.join('    ' + line if line.strip() else line for line in chunk.splitlines(keepends=True))
            return a.replace('                candidate_df = provider.get_daily(', chunk + '                candidate_df = provider.get_daily(').replace('            df = custom_df\n', '            df = custom_df\n            seen.update(df["symbol"].cast(pl.Utf8).unique().to_list())\n')
        if choice == 'adjPersist':
            return b[:b.index('                factor_dir =')] + '                return _persist_adj_factors(\n                    repo, new_data, asset_type=asset_type,\n                    cumulative=getattr(provider, "adj_factor_kind", "event") == "cumulative",\n                )\n'
        if choice == 'minuteMerge': return a[:a.index('            day_df = pl.concat')] + b.replace('pl.concat([same, new_rows])', 'pl.concat([same, new_rows], how="diagonal_relaxed")')
        if choice == 'minutePersist': return a + '    now = cn_now().replace(tzinfo=None)\n'
        if choice == 'minuteGroups': return a.replace('                on_source_issues=on_source_issues,', '                on_source_issues=on_source_issues,\n                raw_basis=minute_adjust.minute_basis_is_raw(repo.store.data_dir),')
        if choice == 'watchImports': return b.replace('from app.market_time import CN_TZ', 'from app.market_time import CN_TZ, cn_today') + 'from app.services.security_identity import exchange_for_code\n'
        if choice == 'watchEntries': return b.replace('    symbols = [', '    raw_symbols = [').replace('    if not symbols:', '    if not raw_symbols:')
        if choice == 'navigation':
            extras = [line for line in a.splitlines(keepends=True) if any(path in line for path in ['/short-term-ai', '/stockformer', '/aiagents', '/fund-monitor'])]
            return b + ''.join(extras)
        if choice == 'depthStale': return a.replace('            return\n', '            return None\n')
        if choice == 'depthProvider':
            return a[:a.index('    def _call_primary_depth_batch')] + '    def _call_primary_depth_batch(self, symbols: list[str]) -> dict:\n' + b
        if choice == 'storageBatch': return a.replace('enriched_sym = compute_enriched(', 'enriched_sym = _compute_storage_batches(')
        if choice == 'aliasScan': return b.replace('lf_batch = lf_batch.filter(pl.col("symbol").is_in(batch_syms))', 'lf_batch = _filter_lazy_by_symbol_keys(lf_batch, batch_syms)')
        if choice == 'indentLocal': return ''.join('    ' + line if line.strip() else line for line in a.splitlines(keepends=True))
        if choice == 'storagePublish':
            return b.replace('                if symbols:\n', '                if symbols:\n                    if on_before_symbol_publish:\n                        on_before_symbol_publish(enriched)\n').replace('existing = existing.filter(~pl.col("symbol").is_in(batch_syms))', 'existing = _exclude_by_symbol_keys(existing, batch_syms)').replace('publication.write_parquet(date_df_storage, out)', '_publish_storage_if_changed(publication, date_df_storage, out)').replace('publication.write_parquet(date_df.sort(["symbol"]), out)', '_publish_storage_if_changed(publication, date_df.sort(["symbol"]), out)')
        if choice == 'analyzerImport': return a.replace('from typing import AsyncIterator', 'from collections.abc import AsyncIterator\nfrom datetime import timedelta')
        if choice == 'analyzerMode':
            return a.replace('asset_type=asset_type)', 'asset_type=asset_type, paper_positions=paper_positions,\n                                             raw_close=raw_close)').replace('        agent_outputs: dict[str, dict] = {}', '        payload["paper_positions"] = paper_positions\n        payload["raw_close"] = raw_close\n        agent_outputs: dict[str, dict] = {}')
        if choice == 'klineImport': return a.replace('Request, UploadFile', 'Request, Response, UploadFile')
        if choice == 'klineServices': return a.replace('import kline_sync\n', 'import kline_sync, trading_day, minute_adjust\n')
        if choice == 'liveGzip':
            return a.replace('            return {\n', '            return _gzip_payload(request, {\n').replace('                "source": "live",\n            }', '                "source": "live",\n            }, pref_key="minute_batch_compress")').replace('        return {**base_response, "sessions": [], "source": "none"}\n', b)
        if choice == 'import': return a.replace('api, type', 'api, friendlyStreamError, type')
        if choice == 'repoImport': return b.replace('import scan_enriched_parquet', 'import read_parquet_safe, scan_enriched_parquet')
        if choice == 'heartbeatMode': return b.replace('req.symbol, req.focus)', 'req.symbol, req.focus, req.mode)')
        if choice == 'collectLocal': return a.replace('df = lf.collect()', 'df = guarded_collect(lf)')
        if choice == 'repoMinute':
            return a.replace('return (', 'df = guarded_collect(').replace('                  .collect(streaming=True)\n', '                  , streaming=True\n')
        if choice == 'financial': return b.replace('atomic_write_parquet(df,', 'atomic_write_parquet(out_df,')
        if choice == 'routerImports': return '\n'.join(line for line in a.splitlines() if not line.startswith('const Lots')) + '\n'
        if choice == 'routerRoutes': return b + "      { path: 'fund-monitor', element: <FundMonitor /> },\n"
        if choice == 'scheduleGet': return a.replace('15:30', '15:35').replace('"minute": 30', '"minute": 35').replace('_schedule_time(d, 15, 30)', '_schedule_time(d, 15, 35)')
        if choice == 'scheduleSet': return a.replace('15 * 60:', '15 * 60 + 35:').replace('h, m = 15, 0', 'h, m = 15, 35').replace('15:00', '15:35')
        if choice == 'interfaces': return a + '}\n\n' + b
        if choice == 'sync': return b[:b.index('  updateMinuteSync:')] + a
        if choice == 'stream': return a.replace("'error' | 'done'", "'error' | 'done' | 'ping'")
        if choice == 'result':
            return re.sub(r'(universe_size|daily_days|adj_factor_symbols|enriched_days|minute_rows):', r'\1?:', a) + '    adj_written?: number\n'
        raise ValueError(choice)
    source = (conflicts / name).read_text(encoding='utf-8')
    result = re.sub(r'<<<<<<<[^\n]*\n(.*?)=======\n(.*?)>>>>>>>[^\n]*\n', resolve, source, flags=re.S)
    assert index[0] == len(plan), (name, index)
    (root / name).write_text(result, encoding='utf-8')
    print(name, index[0])
