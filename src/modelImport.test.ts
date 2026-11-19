import { describe, expect, it } from 'vitest'
import { selectModelFiles } from './modelImport'

const onnx = { name: 'mushroom.onnx', size: 10566324 }
const json = { name: 'manifest.json', size: 1227 }

describe('local model file import', () => {
  it('accepts exactly one ONNX and manifest.json in either order', () => {
    expect(selectModelFiles([onnx, json])).toEqual({ onnx, manifestFile: json })
    expect(selectModelFiles([json, onnx])).toEqual({ onnx, manifestFile: json })
  })

  it('rejects missing, extra, duplicate, and wrongly named files', () => {
    for (const files of [
      [onnx], [json], [onnx, json, { name: 'extra.txt', size: 1 }],
      [onnx, onnx], [json, json], [onnx, { name: 'other.json', size: 100 }],
      [json, { name: 'weights.pt', size: 100 }]
    ]) expect(() => selectModelFiles(files)).toThrow()
  })

})
