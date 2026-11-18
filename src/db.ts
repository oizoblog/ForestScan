import type { ModelManifest, RecordItem } from './types'

const NAME = 'forestscan-records'
export type LogItem = { time: string; message: string }
export type SavedModel = { key: 'active'; manifest: ModelManifest; bytes: ArrayBuffer }
export type TimingItem = { id?: number; time: string; fps: number; prepMs: number; inferMs: number; totalMs: number }
export type TestResult = { name: string; passed: boolean; time: string; detail: string; source: 'automatic' }

function open(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(NAME, 3)
    req.onupgradeneeded = () => {
      const db = req.result
      if (!db.objectStoreNames.contains('scans')) db.createObjectStore('scans', { keyPath: 'id', autoIncrement: true })
      if (!db.objectStoreNames.contains('models')) db.createObjectStore('models', { keyPath: 'key' })
      if (!db.objectStoreNames.contains('logs')) db.createObjectStore('logs', { keyPath: 'id', autoIncrement: true })
      if (!db.objectStoreNames.contains('timings')) db.createObjectStore('timings', { keyPath: 'id', autoIncrement: true })
      if (!db.objectStoreNames.contains('testResults')) db.createObjectStore('testResults', { keyPath: 'name' })
    }
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error)
  })
}
async function transact<T>(store: string, mode: IDBTransactionMode, action: (object: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  const db = await open()
  try {
    return await new Promise<T>((resolve, reject) => {
      const tx = db.transaction(store, mode)
      const req = action(tx.objectStore(store))
      let value: T
      req.onsuccess = () => { value = req.result }
      tx.oncomplete = () => resolve(value)
      tx.onerror = () => reject(tx.error)
      tx.onabort = () => reject(tx.error)
    })
  } finally { db.close() }
}
export async function saveRecord(record: RecordItem): Promise<void> { await transact('scans', 'readwrite', s => s.add(record)) }
export async function getRecords(): Promise<RecordItem[]> { return transact('scans', 'readonly', s => s.getAll()) }
export async function saveModel(manifest: ModelManifest, bytes: ArrayBuffer): Promise<void> { await transact('models', 'readwrite', s => s.put({ key: 'active', manifest, bytes } satisfies SavedModel)) }
export async function getModel(): Promise<SavedModel | undefined> { return transact('models', 'readonly', s => s.get('active')) }
export async function removeModel(): Promise<void> { await transact('models', 'readwrite', s => s.delete('active')) }
export async function saveLog(message: string): Promise<void> { await transact('logs', 'readwrite', s => s.add({ time: new Date().toISOString(), message })) }
export async function getLogs(): Promise<LogItem[]> { return transact('logs', 'readonly', s => s.getAll()) }
export async function saveTiming(timing: Omit<TimingItem, 'id'>): Promise<void> { await transact('timings', 'readwrite', s => s.add(timing)) }
export async function getRecentTimings(limit = 120): Promise<TimingItem[]> {
  const db = await open()
  try {
    return await new Promise((resolve, reject) => {
      const rows: TimingItem[] = []
      const tx = db.transaction('timings', 'readonly')
      const req = tx.objectStore('timings').openCursor(null, 'prev')
      req.onsuccess = () => {
        const cursor = req.result
        if (cursor && rows.length < limit) { rows.push(cursor.value as TimingItem); cursor.continue() }
      }
      tx.oncomplete = () => resolve(rows.reverse())
      tx.onerror = () => reject(tx.error)
    })
  } finally { db.close() }
}
export async function saveTestResult(name: string, passed: boolean, detail: string): Promise<void> {
  await transact('testResults', 'readwrite', s => s.put({ name, passed, detail, time: new Date().toISOString(), source: 'automatic' } satisfies TestResult))
}
export async function getTestResults(): Promise<TestResult[]> { return transact('testResults', 'readonly', s => s.getAll()) }
export type DiagnosticContext = { modelSha256: string | null; modelExperimental: boolean | null }
export function createDiagnosticReport(context: DiagnosticContext, data: {
  records: RecordItem[]; errors: LogItem[]; timings: TimingItem[]; testResults: TestResult[];
  browserUserAgent: string; serviceWorkerControlled: boolean; networkSignal: 'online' | 'offline'
}) {
  return {
    format: 'forestscan-diagnostics-v1', exportedAt: new Date().toISOString(),
    browserUserAgent: data.browserUserAgent,
    serviceWorkerControlled: data.serviceWorkerControlled,
    networkSignal: data.networkSignal,
    ...context, testResults: data.testResults, timings: data.timings,
    errors: data.errors, records: data.records,
    privacy: 'No photos, video frames, precise location, or model bytes are included.'
  }
}
export async function exportRecords(context: DiagnosticContext): Promise<void> {
  const [records, errors, timings, testResults] = await Promise.all([getRecords(), getLogs(), getRecentTimings(), getTestResults()])
  const report = createDiagnosticReport(context, { records, errors, timings, testResults,
    browserUserAgent: navigator.userAgent,
    serviceWorkerControlled: Boolean(navigator.serviceWorker?.controller),
    networkSignal: navigator.onLine ? 'online' : 'offline',
  })
  const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url; a.download = `forestscan-diagnostics-${new Date().toISOString().slice(0, 10)}.json`; a.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
