/// <reference lib="webworker" />
import * as ort from 'onnxruntime-web/wasm'
import { decodeYolo } from './vision'
import type { ModelManifest } from './types'

let session: ort.InferenceSession | null = null
let manifest: ModelManifest | null = null
let running = false
ort.env.wasm.numThreads = 1

self.onmessage = async (event: MessageEvent) => {
  const msg = event.data
  if (msg.type === 'load') {
    try {
      if (running) throw new Error('推理进行中，暂不能更换模型')
      await session?.release()
      session = null
      manifest = null
      const candidate = await ort.InferenceSession.create(msg.bytes as ArrayBuffer, {
        executionProviders: ['wasm'], graphOptimizationLevel: 'all'
      })
      try {
        if (candidate.inputNames.length !== 1 || candidate.outputNames.length !== 1)
          throw new Error('模型必须各有一个输入和输出')
        const size = (msg.manifest as ModelManifest).inputSize
        const probe = await candidate.run({ [candidate.inputNames[0]]:
          new ort.Tensor('float32', new Float32Array(3 * size * size), [1, 3, size, size]) })
        const output = probe[candidate.outputNames[0]]
        if (!(output.data instanceof Float32Array)) throw new Error('模型输出必须是 float32')
        decodeYolo(output.data, output.dims, size, size, size)
      } catch (error) {
        await candidate.release()
        throw error
      }
      session = candidate
      manifest = msg.manifest
      self.postMessage({ type: 'ready' })
    } catch (error) {
      self.postMessage({ type: 'error', message: String(error) })
    }
    return
  }
  if (msg.type !== 'infer') return
  const bitmap = msg.bitmap as ImageBitmap
  const id = msg.id as number
  if (running || !session || !manifest) {
    bitmap.close()
    self.postMessage({ type: 'error', id, message: '推理器尚未就绪' })
    return
  }
  running = true
  try {
    const totalStart = performance.now()
    const size = manifest.inputSize
    const sourceW = bitmap.width, sourceH = bitmap.height
    if (!sourceW || !sourceH) throw new Error('无效摄像头画面')
    const scale = Math.min(size / sourceW, size / sourceH)
    const dw = sourceW * scale, dh = sourceH * scale
    const canvas = new OffscreenCanvas(size, size)
    const ctx = canvas.getContext('2d', { willReadFrequently: true })
    if (!ctx) throw new Error('Safari 未提供 Worker 2D Canvas')
    ctx.fillStyle = '#727272'
    ctx.fillRect(0, 0, size, size)
    ctx.drawImage(bitmap, (size - dw) / 2, (size - dh) / 2, dw, dh)
    const rgba = ctx.getImageData(0, 0, size, size).data
    const area = size * size, input = new Float32Array(area * 3)
    for (let i = 0; i < area; i++) {
      input[i] = rgba[i * 4] / 255
      input[area + i] = rgba[i * 4 + 1] / 255
      input[area * 2 + i] = rgba[i * 4 + 2] / 255
    }
    const prepMs = performance.now() - totalStart
    const inferStart = performance.now()
    const results = await session.run({ [session.inputNames[0]]:
      new ort.Tensor('float32', input, [1, 3, size, size]) })
    const inferMs = performance.now() - inferStart
    const output = results[session.outputNames[0]]
    if (!(output.data instanceof Float32Array)) throw new Error('模型输出必须是 float32')
    const boxes = decodeYolo(output.data, output.dims, size, sourceW, sourceH)
    self.postMessage({ type: 'result', id, boxes, prepMs, inferMs,
      totalMs: performance.now() - totalStart })
  } catch (error) {
    self.postMessage({ type: 'error', id, message: String(error) })
  } finally {
    bitmap.close()
    running = false
  }
}
