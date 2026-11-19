export function selectModelFiles<T extends { name: string }>(files: readonly T[]): { onnx: T; manifestFile: T } {
  if (files.length !== 2) throw new Error('请一次且仅选择 mushroom.onnx 和 manifest.json 两个文件。')
  const onnx = files.find(file => file.name.toLowerCase().endsWith('.onnx'))
  const manifestFile = files.find(file => file.name === 'manifest.json')
  if (!onnx || !manifestFile) throw new Error('必须提供一个 .onnx 文件和一个名为 manifest.json 的文件。')
  return { onnx, manifestFile }
}
