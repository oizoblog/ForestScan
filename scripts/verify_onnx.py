"""Run a real ONNX image through the same letterbox and NMS contract as the PWA."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    return intersection / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection or 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=Path)
    parser.add_argument('image', type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.model_dir / 'manifest.json').read_text(encoding='utf-8'))
    model_path = args.model_dir / manifest['modelFile']
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == manifest['sha256']
    image = Image.open(args.image).convert('RGB')
    width, height = image.size
    size = manifest['inputSize']
    scale = min(size / width, size / height)
    dw, dh = round(width * scale), round(height * scale)
    padded = Image.new('RGB', (size, size), (114, 114, 114))
    padded.paste(image.resize((dw, dh)), ((size - dw) // 2, (size - dh) // 2))
    tensor = np.asarray(padded, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    start = time.perf_counter()
    output = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    infer_ms = (time.perf_counter() - start) * 1000
    tensor.astype(np.float32).tofile(args.model_dir / 'verify_input.float32')
    output.astype(np.float32).tofile(args.model_dir / 'verify_output.float32')
    assert output.shape[0] == 1 and 5 in output.shape[1:]
    rows = output[0].T if output.shape[1] == 5 else output[0]
    boxes = []
    pad_x, pad_y = (size - width * scale) / 2, (size - height * scale) / 2
    for cx, cy, bw, bh, score in rows:
        if not np.isfinite(score) or score < 0.4 or score > 1 or bw <= 0 or bh <= 0:
            continue
        box = [max(0, min(width, float((cx - bw / 2 - pad_x) / scale))),
               max(0, min(height, float((cy - bh / 2 - pad_y) / scale))),
               max(0, min(width, float((cx + bw / 2 - pad_x) / scale))),
               max(0, min(height, float((cy + bh / 2 - pad_y) / scale))), float(score)]
        if box[2] > box[0] and box[3] > box[1]:
            boxes.append(box)
    kept = []
    for box in sorted(boxes, key=lambda b: -b[4])[:300]:
        if all(iou(box, other) < .45 for other in kept):
            kept.append(box)
        if len(kept) >= 50:
            break
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for box in kept:
        draw.rectangle(box[:4], outline='lime', width=3)
    preview_path = args.model_dir / f'verify_{args.image.stem}.jpg'
    overlay.save(preview_path)
    report = {'image': str(args.image), 'model_sha256': manifest['sha256'],
              'output_shape': list(output.shape), 'onnx_cpu_ms': round(infer_ms, 1),
              'candidate_count': len(boxes), 'after_nms': len(kept),
              'max_score': float(rows[:, 4].max()), 'preview': str(preview_path)}
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
