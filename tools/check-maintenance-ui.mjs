import assert from 'node:assert/strict'
import { mkdir } from 'node:fs/promises'
import path from 'node:path'
import { createServer } from '../frontend/node_modules/vite/dist/node/index.js'

const { chromium } = await import(process.argv[2])
const root = path.resolve('frontend')
const out = path.resolve('data/maintenance-ui-qa')
process.chdir(root)
await mkdir(out, { recursive: true })
const entry = `
import React from 'react';
import {createRoot} from 'react-dom/client';
import {QueryClient, QueryClientProvider} from '@tanstack/react-query';
import {MaintenanceQueues} from '/src/components/data/MaintenanceQueues.tsx';
import '/src/index.css';
const client = new QueryClient({defaultOptions:{queries:{retry:false}}});
createRoot(document.getElementById('root')).render(React.createElement(QueryClientProvider,
 {client}, React.createElement(MaintenanceQueues,{onJobClick:id=>document.getElementById('selected').textContent=id})));
`
const server = await createServer({
  root, server: { host: '127.0.0.1', port: 19188, strictPort: true },
  plugins: [{
    name: 'maintenance-qa',
    resolveId(id) { if (id === 'virtual:maintenance-qa') return '\0maintenance-qa' },
    load(id) { if (id === '\0maintenance-qa') return entry },
    configureServer(s) {
      s.middlewares.use('/__maintenance_qa', async (req, res) => {
        res.setHeader('Content-Type', 'text/html; charset=utf-8')
        res.end(await s.transformIndexHtml('/__maintenance_qa', '<html><head><meta name="viewport" content="width=device-width, initial-scale=1"></head><body class="bg-base text-foreground"><main style="padding:24px"><div id="root"></div><output id="selected"></output></main><script type="module" src="/@id/virtual:maintenance-qa"></script></body></html>'))
      })
    },
  }],
})
await server.listen()
const browser = await chromium.launch({ channel: 'msedge', headless: true })
try {
  const page = await browser.newPage()
  const errors = []
  page.on('pageerror', error => errors.push(error.message))
  let mode = 'populated'
  const base = { total: 5558, completed: 150, batch_progress: 42, attempts: 0, job_id: 'qa-minute',
    message: '2026-09-07 ~ 2026-09-08', error: null, updated_at: '2026-09-08T21:00:00+08:00' }
  await page.route('**/api/pipeline/maintenance-queue', route => {
    if (mode === 'error') return route.fulfill({ status: 503, json: { detail: 'test unavailable' } })
    return route.fulfill({ json: { tasks: mode === 'empty' ? [] : [
      { ...base, id: 'm1', kind: 'minute', queue: 'minute', label: '分钟增量补差', status: 'running' },
      { ...base, id: 'h1', kind: 'adjustment', queue: 'history', label: '复权历史重算', status: 'pending', total: 33, completed: 20, batch_progress: 0, job_id: 'qa-history' },
      { ...base, id: 'h2', kind: 'regime', queue: 'history', label: '历史市场环境重算', status: 'retry', total: 620, completed: 10, batch_progress: 0, error: '输入数据校验未通过', retry_at: '2026-09-08T21:15:00+08:00' },
      { ...base, id: 'h3', kind: 'regime', queue: 'history', label: '遗留市场环境核验', status: 'paused_review', total: 621, completed: 110, remaining: 0, review_count: 511, batch_progress: 0, job_id: null, message: '仅来源文件版本变化，缺少实际输入变化证据，不自动重算' },
    ] } })
  })
  for (const [name, width, height] of [['desktop', 1440, 900], ['mobile', 390, 844]]) {
    await page.setViewportSize({ width, height })
    await page.goto('http://127.0.0.1:19188/__maintenance_qa')
    await page.getByText('完成 150/5558 只').waitFor()
    assert.equal(await page.getByRole('progressbar').count(), 4)
    await page.getByText('待核验 511 日（不自动重算）').waitFor()
    await page.getByText('完成 110/621 日，待执行 0').waitFor()
    await page.getByRole('button', { name: '任务日志 qa-history' }).click()
    assert.equal(await page.locator('#selected').textContent(), 'qa-history')
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth))
    await page.screenshot({ path: path.join(out, `${name}.png`), fullPage: true })
  }
  mode = 'empty'
  await page.getByRole('button', { name: '刷新任务队列' }).click()
  await page.getByText('暂无排队任务').first().waitFor()
  mode = 'error'
  await page.getByRole('button', { name: '刷新任务队列' }).click()
  await page.getByText('任务队列读取失败').waitFor()
  assert.deepEqual(errors, [])
  console.log('PASS: desktop/mobile, progress, logs, refresh, empty/error states, no horizontal overflow')
} finally {
  await browser.close()
  await server.close()
}
