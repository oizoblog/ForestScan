export type Box = { x: number; y: number; w: number; h: number; score: number; label: string }
export type ScanMode = 'real' | 'demo'
export type RecordItem = { id?: number; time: string; mode: ScanMode; count: number; boxes: Box[]; modelSha256?: string }
export type ModelManifest = { format: 'forestscan-yolo-v1'; task: 'mushroom-detection'; inputSize: number; classNames: string[]; modelFile: string; sha256: string; dataset: string; license: string; experimental?: boolean; validation: { mAP50: number; mAP50_95: number } }
