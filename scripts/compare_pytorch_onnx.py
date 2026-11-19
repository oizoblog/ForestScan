"""Compare Colab PyTorch raw detections with the local ONNX model."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


def iou(a: dict, b: dict) -> float:
    x1, y1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    x2, y2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    return intersection / (area_a + area_b - intersection or 1.0)


def decode(output: np.ndarray, width: int, height: int, size: int,
           confidence: float = 0.25, nms_iou: float = 0.45) -> tuple[int, list[dict]]:
    if output.shape != (1, 5, 3549) or output.dtype != np.float32:
        raise ValueError(f"Expected float32 [1,5,3549], got {output.dtype} {output.shape}")
    scale = min(size / width, size / height)
    pad_x, pad_y = (size - width * scale) / 2, (size - height * scale) / 2
    candidates = []
    for cx, cy, bw, bh, score in output[0].T:
        if not np.isfinite([cx, cy, bw, bh, score]).all() or score < confidence or score > 1 or bw <= 0 or bh <= 0:
            continue
        x1 = max(0.0, min(float(width), float((cx - bw / 2 - pad_x) / scale)))
        y1 = max(0.0, min(float(height), float((cy - bh / 2 - pad_y) / scale)))
        x2 = max(0.0, min(float(width), float((cx + bw / 2 - pad_x) / scale)))
        y2 = max(0.0, min(float(height), float((cy + bh / 2 - pad_y) / scale)))
        if x2 > x1 and y2 > y1:
            candidates.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": float(score)})
    kept = []
    for box in sorted(candidates, key=lambda b: -b["score"])[:300]:
        if all(iou(box, other) < nms_iou for other in kept):
            kept.append(box)
        if len(kept) >= 50:
            break
    return len(candidates), kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, default=Path("models/comparison"))
    parser.add_argument("--model", type=Path, default=Path("models/mushroom_v1/mushroom.onnx"))
    args = parser.parse_args()
    info = json.loads((args.comparison / "comparison_info.json").read_text(encoding="utf-8"))
    image = Image.open(args.comparison / info["image"])
    width, height = image.size
    if [width, height] != info["original_size"]:
        raise ValueError("Image dimensions differ from comparison_info.json")
    source = np.load(args.comparison / "input_tensor.npy", allow_pickle=False)
    pytorch = np.load(args.comparison / "pytorch_raw_output.npy", allow_pickle=False)
    if source.shape != (1, 3, 416, 416) or source.dtype != np.float32 or not np.isfinite(source).all():
        raise ValueError(f"Expected finite float32 [1,3,416,416], got {source.dtype} {source.shape}")
    if pytorch.shape != (1, 5, 3549) or pytorch.dtype != np.float32 or not np.isfinite(pytorch).all():
        raise ValueError(f"Unexpected PyTorch output {pytorch.dtype} {pytorch.shape}")
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    if [(i.name, i.shape, i.type) for i in session.get_inputs()] != [("images", [1, 3, 416, 416], "tensor(float)")]:
        raise ValueError("Unexpected ONNX input contract")
    started = time.perf_counter()
    onnx = session.run(["output0"], {"images": source})[0]
    inference_ms = (time.perf_counter() - started) * 1000
    if onnx.shape != (1, 5, 3549) or onnx.dtype != np.float32 or not np.isfinite(onnx).all():
        raise ValueError(f"Unexpected ONNX output {onnx.dtype} {onnx.shape}")
    delta = np.abs(onnx.astype(np.float64) - pytorch.astype(np.float64))
    pytorch_candidates, pytorch_boxes = decode(pytorch, width, height, 416)
    onnx_candidates, onnx_boxes = decode(onnx, width, height, 416)
    pairs = []
    for a, b in zip(pytorch_boxes, onnx_boxes):
        pairs.append({"iou": iou(a, b),
                      "max_coordinate_error_px": max(abs(a[k] - b[k]) for k in ("x1", "y1", "x2", "y2")),
                      "score_error": abs(a["score"] - b["score"])})
    report = {
        "image": str(args.comparison / info["image"]), "image_size": [width, height],
        "input_shape": list(source.shape), "input_dtype": str(source.dtype),
        "output_shape": list(onnx.shape), "output_dtype": str(onnx.dtype),
        "onnx_cpu_ms": round(inference_ms, 2),
        "raw_error": {
            "max_abs": float(delta.max()), "mean_abs": float(delta.mean()),
            "median_abs": float(np.median(delta)),
            "p90_abs": float(np.quantile(delta, 0.9)), "p99_abs": float(np.quantile(delta, 0.99)),
            "p999_abs": float(np.quantile(delta, 0.999)),
            "count_gt_1e-4": int((delta > 1e-4).sum()),
            "count_gt_1e-3": int((delta > 1e-3).sum()),
        },
        "conf": 0.25, "nms_iou": 0.45,
        "pytorch_candidates": pytorch_candidates, "onnx_candidates": onnx_candidates,
        "pytorch_boxes": pytorch_boxes, "onnx_boxes": onnx_boxes,
        "matched_box_differences": pairs,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
