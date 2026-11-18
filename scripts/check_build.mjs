import fs from 'node:fs'
import path from 'node:path'

const base = process.argv[2] || '/'
const dist = 'dist'
const sw = fs.readFileSync(path.join(dist, 'sw.js'), 'utf8')
const html = fs.readFileSync(path.join(dist, 'index.html'), 'utf8')
const manifest = JSON.parse(fs.readFileSync(path.join(dist, 'manifest.webmanifest'), 'utf8'))
const entries = [...sw.matchAll(/url:"([^"]+)"/g)].map(m => m[1])
const origin = 'https://example.github.io'
if (new URL(manifest.scope, `${origin}${base}manifest.webmanifest`).pathname !== base ||
    new URL(manifest.start_url, `${origin}${base}manifest.webmanifest`).pathname !== base ||
    new URL('sw.js', `${origin}${base}`).pathname !== `${base}sw.js`)
  throw new Error('Manifest or Service Worker scope escapes the project subpath')
for (const entry of entries) {
  if (!fs.existsSync(path.join(dist, entry))) throw new Error(`Missing precache asset: ${entry}`)
}
for (const suffix of ['.wasm', 'worker-', 'manifest.webmanifest', 'index.html', 'icon-192.png']) {
  if (!entries.some(entry => entry.includes(suffix))) throw new Error(`Offline precache lacks ${suffix}`)
}
if (entries.some(entry => entry.endsWith('.onnx')) || fs.existsSync(path.join(dist, 'models')))
  throw new Error('Experimental ONNX model must not be deployed by default')
for (const asset of ['manifest.webmanifest', 'icon-192.png', 'assets/index-']) {
  if (!html.includes(`${base}${asset}`)) throw new Error(`Wrong base path for ${asset}`)
}
const workerFile = entries.find(entry => entry.includes('worker-'))
const worker = fs.readFileSync(path.join(dist, workerFile), 'utf8')
if (!worker.includes(`${base}assets/ort-wasm-simd-threaded-`)) throw new Error('WASM URL does not use deployment base')
const appFile = entries.find(entry => entry.startsWith('assets/index-') && entry.endsWith('.js'))
const app = fs.readFileSync(path.join(dist, appFile), 'utf8')
if (!app.includes(`${base}sw.js`) || !app.includes(`scope:"${base}"`))
  throw new Error('Service Worker registration or scope is wrong')
if (!app.includes(`${base}${workerFile}`)) throw new Error('Worker URL does not use deployment base')
console.log(JSON.stringify({ base, precacheEntries: entries.length, manifestScope: base,
  serviceWorkerScope: base, wasmCached: true, workerCached: true, modelNotBundled: true }))
