import { expect, it } from 'vitest'
import { createDiagnosticReport } from './db'

it('exports diagnostic facts without images, model bytes, or location', () => {
  const report = createDiagnosticReport({ modelSha256: 'abc', modelExperimental: true }, {
    browserUserAgent: 'Safari/605.1.15', serviceWorkerControlled: true, networkSignal: 'offline',
    records: [], errors: [{ time: '2026-11-18T00:00:00Z', message: 'camera denied' }],
    timings: [{ time: '2026-11-18T00:00:00Z', fps: 25, prepMs: 9, inferMs: 70, totalMs: 85 }],
    testResults: [{ name: 'camera-start', passed: false, time: '2026-11-18T00:00:00Z', detail: 'denied', source: 'automatic' }]
  })
  expect(report.browserUserAgent).toContain('Safari')
  expect(report.timings[0].inferMs).toBe(70)
  expect(report.testResults[0].passed).toBe(false)
  expect(Object.keys(report)).not.toEqual(expect.arrayContaining(['photo', 'image', 'modelBytes', 'latitude', 'longitude']))
})
