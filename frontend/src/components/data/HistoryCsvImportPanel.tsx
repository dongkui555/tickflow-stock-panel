import { useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { FileUp, Loader2 } from 'lucide-react'
import { api } from '@/lib/api'
import { QK } from '@/lib/queryKeys'

type AssetType = 'auto' | 'stock' | 'index' | 'etf'
type DataKind = 'daily' | 'minute'

export function HistoryCsvImportPanel({
  isRunning,
  onStart,
}: {
  isRunning: boolean
  onStart: (jobId: string) => void
}) {
  const qc = useQueryClient()
  const [files, setFiles] = useState<File[]>([])
  const [sourcePath, setSourcePath] = useState('')
  const [assetType, setAssetType] = useState<AssetType>('auto')
  const [dataKind, setDataKind] = useState<DataKind>('daily')

  const upload = useMutation({
    mutationFn: () => {
      const path = sourcePath.trim()
      const fileTypeText = dataKind === 'minute' ? 'CSV / Parquet / ZIP' : 'CSV'
      if (!files.length && !path) throw new Error(`请至少选择一个 ${fileTypeText} 文件，或填写本地路径`)
      return api.importHistoryCsvBatch(files, assetType, path, dataKind)
    },
    onSuccess: ({ job_id }) => {
      onStart(job_id)
      qc.invalidateQueries({ queryKey: QK.pipelineJobs })
    },
  })

  const disabled = isRunning || upload.isPending || (!files.length && !sourcePath.trim())

  return (
    <div className="space-y-4">
      <div className="rounded-card bg-accent/8 border border-accent/20 p-3 space-y-1.5">
        <div className="text-xs text-foreground">
          导入历史行情到主行情库，直接参与看板、选股和回测。
        </div>
        <div className="text-[10px] text-muted leading-relaxed">
          日线支持 CSV；分钟K支持 CSV、Parquet、ZIP 离线包。常见列名可自动识别：
          `ts_code/symbol/code`、`trade_date/date/datetime/trade_time`、`open/high/low/close`、`vol/volume`、`amt/amount`。
        </div>
      </div>

      <div className="space-y-2">
        <label className="text-xs text-secondary">数据类型</label>
        <select
          value={dataKind}
          onChange={e => setDataKind(e.target.value as DataKind)}
          disabled={isRunning || upload.isPending}
          className="w-full rounded-btn border border-border bg-base px-3 py-2 text-xs text-foreground outline-none focus:border-accent disabled:opacity-40"
        >
          <option value="daily">日线</option>
          <option value="minute">分钟K</option>
        </select>
      </div>

      <div className="space-y-2">
        <label className="text-xs text-secondary">
          {dataKind === 'minute' ? '分钟K离线包（可多选）' : 'CSV 文件（可多选）'}
        </label>
        <input
          type="file"
          accept={dataKind === 'minute' ? '.csv,.parquet,.zip,text/csv,application/zip' : '.csv,text/csv'}
          multiple
          onChange={e => setFiles(Array.from(e.target.files ?? []))}
          disabled={isRunning || upload.isPending}
          className="block w-full rounded-btn border border-border bg-base px-3 py-2 text-xs text-foreground file:mr-3 file:rounded-btn file:border-0 file:bg-elevated file:px-3 file:py-1.5 file:text-xs file:text-secondary hover:file:text-foreground disabled:opacity-40"
        />
        {!!files.length && (
          <div className="space-y-1 text-[11px] text-muted">
            <div className="font-medium text-secondary">已选择 {files.length} 个文件</div>
            <div className="max-h-24 overflow-auto font-mono leading-relaxed">
              {files.slice(0, 6).map(file => (
                <div key={`${file.name}-${file.size}`}>{file.name} · {(file.size / 1024).toFixed(1)} KB</div>
              ))}
              {files.length > 6 && <div>... 还有 {files.length - 6} 个文件</div>}
            </div>
          </div>
        )}
      </div>

      <div className="space-y-2">
        <label className="text-xs text-secondary">服务器本地路径（可选）</label>
        <input
          type="text"
          value={sourcePath}
          onChange={e => setSourcePath(e.target.value)}
          placeholder={
            dataKind === 'minute'
              ? '例如 /app/offline-data/minute 或 D:\\chatgpt\\aiagents-stock\\offline-data\\minute'
              : '例如 D:\\data\\history.csv 或 D:\\data\\history\\'
          }
          disabled={isRunning || upload.isPending}
          className="block w-full rounded-btn border border-border bg-base px-3 py-2 text-xs text-foreground outline-none focus:border-accent disabled:opacity-40"
        />
        <div className="text-[10px] text-muted leading-relaxed">
          Docker 中推荐把离线包放到主机 offline-data 目录，页面可填 /app/offline-data 下的文件或目录。
        </div>
      </div>

      <div className="flex items-center justify-between gap-3">
        <span className="text-xs text-secondary">资产类型</span>
        <select
          value={assetType}
          onChange={e => setAssetType(e.target.value as AssetType)}
          disabled={isRunning || upload.isPending || dataKind === 'minute'}
          className="min-w-40 rounded-btn border border-border bg-base px-3 py-1.5 text-xs text-foreground outline-none focus:border-accent disabled:opacity-40"
        >
          <option value="auto">自动识别</option>
          <option value="stock">股票</option>
          <option value="index">指数</option>
          <option value="etf">ETF</option>
        </select>
      </div>

      <div className="text-[10px] text-muted leading-relaxed">
        分钟K离线包建议按股票或日期拆分，字段至少包含代码、分钟时间、开高低收；成交量和成交额可为空但会影响量价分析。
      </div>

      <button
        onClick={() => upload.mutate()}
        disabled={disabled}
        className="w-full inline-flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-btn bg-accent/90 text-base text-xs font-medium hover:bg-accent disabled:opacity-40 disabled:pointer-events-none transition-colors duration-150"
      >
        {upload.isPending ? (
          <>
            <Loader2 className="h-3 w-3 animate-spin" />
            上传中…
          </>
        ) : (
          <>
            <FileUp className="h-3 w-3" />
            开始导入
          </>
        )}
      </button>

      {upload.isError && (
        <div className="rounded-btn border border-danger/40 bg-danger/5 px-3 py-2 text-xs text-danger">
          导入失败:{String((upload.error as any).message)}
        </div>
      )}
    </div>
  )
}
