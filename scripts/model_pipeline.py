"""Train, validate and export a one-class mushroom YOLO detector.

The user must provide licensed, annotated images in Ultralytics YOLO layout.
No public dataset or pretrained weights are silently downloaded.
"""
import argparse
import hashlib
import json
from pathlib import Path

import yaml
from ultralytics import YOLO


def check_dataset(path: Path) -> tuple[dict, dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data.get("names") not in ({0: "mushroom"}, ["mushroom"]):
        raise ValueError("dataset.yaml must define exactly one class: mushroom")
    root = (path.parent / data.get("path", ".")).resolve()
    for split in ("train", "val"):
        if not data.get(split):
            raise ValueError(f"Missing {split} split")
        if not (root / data[split]).exists():
            raise FileNotFoundError(root / data[split])
        label_dir = root / str(data[split]).replace("images", "labels")
        if not label_dir.exists() or not list(label_dir.glob("*.txt")):
            raise FileNotFoundError(f"Missing YOLO labels in {label_dir}")
    license_file = path.parent / "DATA_LICENSE.json"
    if not license_file.exists():
        raise FileNotFoundError("DATA_LICENSE.json required beside dataset.yaml")
    license_info = json.loads(license_file.read_text(encoding="utf-8"))
    for key in ("dataset", "source", "license", "attribution"):
        if not license_info.get(key) or "REPLACE_" in str(license_info[key]):
            raise ValueError(f"DATA_LICENSE.json missing {key}")
    data["path"] = str(root)
    return license_info, data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, help="Path to dataset.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("models/mushroom"))
    parser.add_argument("--model", default="yolo11n.yaml", help="Architecture YAML or explicitly chosen pretrained .pt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.imgsz < 128 or args.imgsz > 640 or args.imgsz % 32:
        parser.error("--imgsz must be a multiple of 32 between 128 and 640 for the PWA")
    if args.epochs < 1 or args.batch < 1:
        parser.error("--epochs and --batch must be positive")
    license_info, data = check_dataset(args.dataset)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    resolved_yaml = args.output / "resolved-dataset.yaml"
    resolved_yaml.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    model = YOLO(args.model)
    model.train(data=str(resolved_yaml), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device, workers=0,
                project=str(args.output), name="training", exist_ok=True)
    best = args.output / "training/weights/best.pt"
    detector = YOLO(str(best))
    metrics = detector.val(data=str(resolved_yaml), imgsz=args.imgsz, device=args.device, workers=0,
                           project=str(args.output), name="validation", exist_ok=True)
    onnx_path = Path(detector.export(format="onnx", imgsz=args.imgsz,
                                     simplify=False, dynamic=False, nms=False, opset=12))
    target = args.output / "mushroom.onnx"
    target.write_bytes(onnx_path.read_bytes())
    manifest = {
        "format": "forestscan-yolo-v1", "task": "mushroom-detection",
        "experimental": True,
        "inputSize": args.imgsz, "classNames": ["mushroom"],
        "modelFile": target.name, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "dataset": license_info["dataset"], "license": license_info["license"],
        "source": license_info["source"], "attribution": license_info["attribution"],
        "validation": {"mAP50": float(metrics.box.map50), "mAP50_95": float(metrics.box.map)},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
