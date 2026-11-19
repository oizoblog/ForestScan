import fs from 'node:fs'
import path from 'node:path'
import * as ort from 'onnxruntime-web/wasm'

const target = process.argv[2] || 'models/oyster_experiment'
const directModel = target.endsWith('.onnx')
const dir = directModel ? path.dirname(target) : target
const manifest = directModel
  ? { modelFile: path.basename(target), inputSize: Number(process.argv[3]) }
  : JSON.parse(fs.readFileSync(`${dir}/manifest.json`, 'utf8'))
if (!Number.isInteger(manifest.inputSize)) throw new Error('input size required for direct ONNX verification')
const inputBytes = fs.readFileSync(process.argv[4] || `${dir}/verify_input.float32`)
const expectedBytes = fs.readFileSync(process.argv[5] || `${dir}/verify_output.float32`)
const input = new Float32Array(inputBytes.buffer, inputBytes.byteOffset, inputBytes.byteLength / 4)
const expected = new Float32Array(expectedBytes.buffer, expectedBytes.byteOffset, expectedBytes.byteLength / 4)
ort.env.wasm.numThreads = 1
const session = await ort.InferenceSession.create(fs.readFileSync(`${dir}/${manifest.modelFile}`), { executionProviders: ['wasm'] })
const start = performance.now()
const result = await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, manifest.inputSize, manifest.inputSize]) })
const elapsed = performance.now() - start
const output = result[session.outputNames[0]]
if (!(output.data instanceof Float32Array) || output.data.length !== expected.length) throw new Error('ONNX Web output shape/type mismatch')
let maxError = 0
for (let i = 0; i < expected.length; i++) maxError = Math.max(maxError, Math.abs(output.data[i] - expected[i]))
console.log(JSON.stringify({ dims: output.dims, wasmMs: Math.round(elapsed), maxDifferenceFromPython: maxError }))
if (maxError > 0.002) process.exitCode = 1
await session.release()
