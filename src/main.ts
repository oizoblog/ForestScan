import './style.css'
import { exportRecords, getLogs, getModel, getRecords, removeModel, saveLog, saveModel, saveRecord, saveTestResult, saveTiming } from './db'
import type { Box, ModelManifest, ScanMode } from './types'
import { drawBoxes, demoBoxes } from './vision'
import { selectModelFiles } from './modelImport'
import { registerSW } from 'virtual:pwa-register'

registerSW({ immediate: true })
document.querySelector<HTMLDivElement>('#app')!.innerHTML = `<main class="app"><header class="top"><div class="brand"><span class="dot"></span>ForestScan</div><span class="badge" id="network">检查网络</span></header><section class="viewer" id="viewer"><video id="video" autoplay muted playsinline></video><canvas id="overlay"></canvas><div class="placeholder" id="placeholder"><div class="glyph">🍄</div><strong>对准林地中的目标</strong><p>点击下方按钮开启后置摄像头</p></div><div class="hud"><span id="scan">待启动</span><span id="fps">0 FPS</span></div><div class="notice">疑似蘑菇，不可作为食用鉴定依据</div></section><section class="panel"><div class="mode"><button id="realMode" class="active">真实模型</button><button id="demoMode">模拟演示</button></div><p class="status" id="modelStatus">未加载有效蘑菇专用模型；真实检测不可用。</p><div class="modelrow"><label for="modelFiles">加载模型和 manifest</label><input type="file" id="modelFiles" multiple><span class="small">选择配套的 .onnx 与 manifest.json</span><button id="forgetModel" class="secondary">清除缓存模型</button></div><div class="small" id="infer">推理耗时：—</div></section><div class="controls"><button id="start" class="primary">启动摄像头</button><button id="stop" class="secondary" disabled>停止</button><button id="save" class="secondary">保存当前记录</button><button id="export" class="secondary">导出诊断报告</button></div><div class="status" id="status" role="status">准备就绪</div><div class="records small" id="records">本机记录：0 条</div><details class="small"><summary>错误日志</summary><div id="logList">暂无错误</div></details><footer class="footer">本应用不识别可食用性。离线功能需在 HTTPS 页面首次载入后使用。</footer></main>`

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T
const video = $<HTMLVideoElement>('video'), canvas = $<HTMLCanvasElement>('overlay')
const viewer = $<HTMLElement>('viewer'), status = $<HTMLElement>('status')
let mode: ScanMode = 'real', stream: MediaStream | null = null, frameHandle = 0, boxes: Box[] = []
let worker: Worker | null = null, ready = false, busy = false, nextId = 0, pendingId = 0
let lastInfer = 0, lastFps = performance.now(), frames = 0, timeoutHandle = 0
let currentFps = 0, lastTimingSaved = 0
let userSelectedModel = false
let activeManifest: ModelManifest | null = null

function setStatus(message: string, error = false) {
  status.textContent = message
  status.classList.toggle('error', error)
  if (error) {
    const line = document.createElement('div')
    line.textContent = `${new Date().toLocaleString()} ${message}`
    const list = $('logList')
    if (list.textContent === '暂无错误') list.textContent = ''
    list.prepend(line)
    void saveLog(message).catch(() => undefined)
  }
}
const scanText = (message: string) => { $('scan').textContent = message }
const updateNetwork = () => { $('network').textContent = navigator.onLine ? '在线' : '离线' }
const testResult = (name: string, passed: boolean, detail: string) => { void saveTestResult(name, passed, detail).catch(() => undefined) }
window.addEventListener('online', updateNetwork)
window.addEventListener('offline', () => { updateNetwork(); testResult('offline-signal-observed', true, '浏览器报告离线；不等同于飞行模式验收') })
updateNetwork()
if (!navigator.onLine) testResult('offline-signal-observed', true, '页面启动时浏览器报告离线；不等同于飞行模式验收')
async function refreshRecords() { $('records').textContent = `本机记录：${(await getRecords()).length} 条` }
void refreshRecords().catch(() => setStatus('IndexedDB 无法使用，记录和模型缓存不可用。', true))
void getLogs().then(logs => {
  if (!logs.length) return
  const list = $('logList')
  if (list.textContent === '暂无错误') list.textContent = ''
  for (const item of logs.slice(-20).reverse()) {
    const line = document.createElement('div')
    line.textContent = `${new Date(item.time).toLocaleString()} ${item.message}`
    list.append(line)
  }
}).catch(() => undefined)

function setMode(next: ScanMode) {
  mode = next; boxes = []
  $('realMode').classList.toggle('active', next === 'real')
  $('demoMode').classList.toggle('active', next === 'demo')
  scanText(next === 'demo' ? '模拟演示 · 非真实检测' : ready ? '模型待扫描' : '未加载模型')
  setStatus(next === 'demo' ? '模拟演示：框的位置由程序生成，不代表识别结果。' : ready ? '用户模型已加载；正在等待画面。' : '未加载有效蘑菇专用模型；真实检测不可用。')
}
$('realMode').onclick = () => setMode('real'); $('demoMode').onclick = () => setMode('demo')

function resize() {
  const dpr = devicePixelRatio || 1
  canvas.width = Math.round(viewer.clientWidth * dpr)
  canvas.height = Math.round(viewer.clientHeight * dpr)
  canvas.getContext('2d')!.setTransform(dpr, 0, 0, dpr, 0, 0)
}
new ResizeObserver(resize).observe(viewer); resize()

async function start() {
  if (stream) return
  $('start').setAttribute('disabled', '')
  let acquired: MediaStream | null = null
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('摄像头不可用。请使用 HTTPS 或 localhost，并在 Safari 中允许摄像头。')
    try {
      acquired = await navigator.mediaDevices.getUserMedia({ audio: false, video: {
        facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } } })
    } catch (error) {
      if (error instanceof DOMException && ['OverconstrainedError', 'NotFoundError'].includes(error.name))
        acquired = await navigator.mediaDevices.getUserMedia({ audio: false, video: true })
      else throw error
    }
    stream = acquired
    video.srcObject = acquired
    await video.play()
    testResult('camera-start', true, '视频轨道已播放；后置摄像头方向仍需人工确认')
    $('placeholder').hidden = true
    $('stop').removeAttribute('disabled')
    setStatus(mode === 'demo' ? '模拟演示运行中。' : ready ? '用户模型扫描运行中。' : '摄像头已启动；加载有效模型后才能真实检测。')
    lastFps = performance.now(); frames = 0
    frameHandle = requestAnimationFrame(tick)
  } catch (error) {
    acquired?.getTracks().forEach(track => track.stop())
    stream = null; video.srcObject = null
    $('start').removeAttribute('disabled')
    setStatus(`无法启动摄像头：${String(error)}`, true)
    testResult('camera-start', false, String(error))
  }
}
function stop() {
  cancelAnimationFrame(frameHandle)
  stream?.getTracks().forEach(track => track.stop())
  stream = null; video.srcObject = null
  $('placeholder').hidden = false
  $('start').removeAttribute('disabled'); $('stop').setAttribute('disabled', '')
  boxes = []; scanText('已停止'); $('fps').textContent = '0 FPS'
  canvas.getContext('2d')!.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight)
  setStatus('摄像头已停止。')
}
$('start').onclick = start; $('stop').onclick = stop
document.addEventListener('visibilitychange', () => { if (document.hidden && stream) stop() })

async function tick(now: number) {
  if (!stream) return
  frames++
  if (now - lastFps >= 1000) {
    currentFps = frames * 1000 / (now - lastFps)
    $('fps').textContent = `${currentFps.toFixed(1)} FPS`
    frames = 0; lastFps = now
  }
  if (mode === 'demo') {
    boxes = demoBoxes(now); scanText('模拟演示 · 非真实检测')
  } else if (ready && !busy && now - lastInfer > 180 && video.readyState >= 2) {
    busy = true; lastInfer = now
    try {
      const bitmap = await createImageBitmap(video)
      if (!stream || !worker) { bitmap.close(); busy = false; return }
      const id = ++nextId; pendingId = id
      worker.postMessage({ type: 'infer', id, bitmap }, [bitmap])
      timeoutHandle = window.setTimeout(() => failModel('推理超时，请重新加载模型。'), 60000)
    } catch (error) { busy = false; setStatus(`画面处理失败：${String(error)}`, true) }
  } else if (!ready) { boxes = []; scanText('未加载模型') }
  drawBoxes(canvas, video, boxes, mode)
  frameHandle = requestAnimationFrame(tick)
}

function failModel(message: string) {
  clearTimeout(timeoutHandle)
  worker?.terminate(); worker = null
  ready = false; busy = false; boxes = []
  activeManifest = null
  $('modelStatus').textContent = '未加载有效蘑菇专用模型；真实检测不可用。'
  $('infer').textContent = '推理耗时：—'
  scanText('模型错误')
  setStatus(message, true)
  testResult('model-run', false, message)
  void removeModel().catch(() => undefined)
}

function validateManifest(value: unknown): ModelManifest {
  const m = value as ModelManifest
  if (!m || m.format !== 'forestscan-yolo-v1' || m.task !== 'mushroom-detection' ||
      !Number.isInteger(m.inputSize) || m.inputSize < 128 || m.inputSize > 640 ||
      !Array.isArray(m.classNames) || m.classNames.length !== 1 || m.classNames[0] !== 'mushroom' ||
      !m.modelFile?.endsWith('.onnx') || !/^[a-f0-9]{64}$/i.test(m.sha256) ||
      !m.dataset || !m.license || !Number.isFinite(m.validation?.mAP50) ||
      m.validation.mAP50 <= 0 || m.validation.mAP50 > 1)
    throw new Error('manifest 不符合蘑菇专用模型协议或缺少有效验证结果')
  return m
}
async function loadModel(value: unknown, bytes: ArrayBuffer, cache: boolean) {
  const manifest = validateManifest(value)
  const digest = await crypto.subtle.digest('SHA-256', bytes)
  const sha = [...new Uint8Array(digest)].map(x => x.toString(16).padStart(2, '0')).join('')
  if (sha.toLowerCase() !== manifest.sha256.toLowerCase()) throw new Error('模型 SHA-256 校验失败')
  if (!cache && userSelectedModel) return
  worker?.terminate(); clearTimeout(timeoutHandle)
  ready = false; busy = false; boxes = []
  const candidate = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' })
  worker = candidate
  candidate.onerror = e => { if (worker === candidate) failModel(`Worker 启动失败：${e.message}`) }
  candidate.onmessage = event => {
    if (worker !== candidate) return
    const data = event.data
    if (data.type === 'ready') {
      ready = true
      activeManifest = manifest
      $('modelStatus').textContent = `${manifest.experimental ? '实验' : '用户'}模型：${manifest.modelFile} · mAP50 ${(manifest.validation.mAP50 * 100).toFixed(1)}% · 未经独立核验`
      setStatus('用户模型通过哈希、运行与输出形状检查；数据来源和验证指标未经独立核验。')
      testResult(cache ? 'model-import' : 'model-idb-restore', true, `SHA-256 ${manifest.sha256}`)
    } else if (data.type === 'result' && data.id === pendingId) {
      clearTimeout(timeoutHandle); busy = false
      $('infer').textContent = `预处理 ${data.prepMs.toFixed(0)} ms · 推理 ${data.inferMs.toFixed(0)} ms · 总计 ${data.totalMs.toFixed(0)} ms`
      testResult('wasm-inference', true, '收到真实 ONNX Runtime Web 推理结果')
      if (performance.now() - lastTimingSaved > 5000) {
        lastTimingSaved = performance.now()
        void saveTiming({ time: new Date().toISOString(), fps: currentFps,
          prepMs: data.prepMs, inferMs: data.inferMs, totalMs: data.totalMs }).catch(() => undefined)
      }
      if (mode === 'real') { boxes = data.boxes; scanText(boxes.length ? `疑似目标 ${boxes.length}` : '扫描中 · 未发现目标') }
    } else if (data.type === 'error') {
      if (data.id === undefined || data.id === pendingId) failModel(`模型错误：${data.message}`)
    }
  }
  if (cache) {
    try { await saveModel(manifest, bytes); testResult('model-idb-save', true, 'ONNX 文件已保存到 IndexedDB') }
    catch (error) { setStatus(`模型可运行，但离线保存失败：${String(error)}`, true); testResult('model-idb-save', false, String(error)) }
  }
  if (worker !== candidate) return
  candidate.postMessage({ type: 'load', bytes, manifest }, [bytes])
  setStatus('正在加载并验证 ONNX 模型……')
}

$<HTMLInputElement>('modelFiles').onchange = async event => {
  const files = Array.from((event.target as HTMLInputElement).files || [])
  let onnx: File, manifest: ModelManifest
  try {
    const selected = selectModelFiles(files)
    onnx = selected.onnx
    manifest = validateManifest(JSON.parse(await selected.manifestFile.text()))
    if (manifest.modelFile !== onnx.name) throw new Error('模型文件名与 manifest 不一致')
  } catch (error) {
    setStatus(`模型文件选择失败：${String(error)}`, true)
    testResult('model-import', false, String(error))
    return
  }
  userSelectedModel = true
  try { await loadModel(manifest, await onnx.arrayBuffer(), true) }
  catch (error) { failModel(`模型加载失败：${String(error)}`) }
}
void getModel().then(saved => {
  if (!saved || userSelectedModel) return
  setStatus('正在恢复本机缓存模型……')
  return loadModel(saved.manifest, saved.bytes, false)
}).catch(error => { failModel(`缓存模型恢复失败：${String(error)}`) })
$('forgetModel').onclick = async () => {
  clearTimeout(timeoutHandle)
  worker?.terminate(); worker = null; ready = false; busy = false; boxes = []; activeManifest = null
  try { await removeModel(); $('modelStatus').textContent = '未加载有效蘑菇专用模型；真实检测不可用。'; setStatus('缓存模型已清除。') }
  catch (error) { setStatus(`清除模型失败：${String(error)}`, true) }
}
$('save').onclick = async () => {
  try {
    const current = mode === 'real' && !ready ? [] : boxes
    await saveRecord({ time: new Date().toISOString(), mode, count: current.length, boxes: current,
      modelSha256: mode === 'real' && ready ? activeManifest?.sha256 : undefined })
    await refreshRecords()
    testResult('record-save', true, `模式 ${mode}`)
    setStatus(mode === 'demo' ? '模拟演示记录已保存（非真实检测）。' : ready ? '用户模型扫描记录已保存。' : '已保存空记录；未加载真实模型。')
  } catch (error) { setStatus(`保存失败：${String(error)}`, true); testResult('record-save', false, String(error)) }
}
$('export').onclick = () => exportRecords({ modelSha256: activeManifest?.sha256 || null,
  modelExperimental: activeManifest?.experimental ?? null }).catch(error => setStatus(`导出失败：${String(error)}`, true))
window.addEventListener('pagehide', () => { stream?.getTracks().forEach(track => track.stop()); worker?.terminate() })
