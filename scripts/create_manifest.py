"""Create a ForestScan manifest for an existing, independently exported ONNX model.

The dataset provenance is supplied by the user; ONNX metadata alone cannot
establish the training images' license or the reported validation metrics.
"""
import argparse
import hashlib
import json
from pathlib import Path

import onnx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Existing mushroom.onnx")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--license-url", required=True)
    parser.add_argument("--attribution", required=True)
    parser.add_argument("--modification", action="append", default=[])
    parser.add_argument("--license-verification-status", required=True)
    parser.add_argument("--precision", type=float, required=True)
    parser.add_argument("--recall", type=float, required=True)
    parser.add_argument("--map50", type=float, required=True)
    parser.add_argument("--map50-95", type=float, required=True)
    parser.add_argument("--output", type=Path, help="Defaults to manifest.json beside the model")
    args = parser.parse_args()

    if not args.model.is_file() or args.model.name != "mushroom.onnx":
        parser.error("model must be an existing file named mushroom.onnx")
    for key in ("dataset", "source", "license", "license_url", "attribution", "license_verification_status"):
        value = getattr(args, key).strip()
        if not value or value.lower() in {"unknown", "unverified", "tbd", "none"}:
            parser.error(f"--{key} must contain verified provenance, not a placeholder")
    for key in ("precision", "recall", "map50", "map50_95"):
        value = getattr(args, key)
        if not 0 <= value <= 1:
            parser.error(f"--{key.replace('_', '-')} must be between 0 and 1")

    graph = onnx.load(str(args.model), load_external_data=False)
    onnx.checker.check_model(graph)
    if len(graph.graph.input) != 1 or len(graph.graph.output) != 1:
        parser.error("model must have exactly one input and one output")
    model_input, model_output = graph.graph.input[0], graph.graph.output[0]
    def shape(value):
        tensor = value.type.tensor_type
        if tensor.elem_type != onnx.TensorProto.FLOAT:
            parser.error(f"{value.name} must be float32")
        return [dim.dim_value for dim in tensor.shape.dim]
    input_shape, output_shape = shape(model_input), shape(model_output)
    if model_input.name != "images" or input_shape[:2] != [1, 3] or len(input_shape) != 4:
        parser.error(f"unexpected input: {model_input.name} {input_shape}")
    size = input_shape[2]
    if size != input_shape[3] or size < 128 or size > 640 or size % 32:
        parser.error(f"unsupported input size: {input_shape}")
    if model_output.name != "output0" or output_shape != [1, 5, 3549]:
        parser.error(f"unexpected output: {model_output.name} {output_shape}")
    metadata = {item.key: item.value for item in graph.metadata_props}
    if metadata.get("task") != "detect" or metadata.get("names") != "{0: 'mushroom'}":
        parser.error("ONNX metadata must identify one mushroom detection class")
    if "'nms': False" not in metadata.get("args", ""):
        parser.error("ONNX export must leave NMS to ForestScan")

    digest = hashlib.sha256()
    with args.model.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    manifest = {
        "format": "forestscan-yolo-v1", "task": "mushroom-detection",
        "inputSize": size, "classNames": ["mushroom"],
        "modelFile": args.model.name, "sha256": digest.hexdigest(),
        "fileSizeBytes": args.model.stat().st_size,
        "dataset": args.dataset.strip(), "source": args.source.strip(),
        "license": args.license.strip(), "attribution": args.attribution.strip(),
        "licenseUrl": args.license_url.strip(),
        "modifications": args.modification,
        "licenseVerificationStatus": args.license_verification_status.strip(),
        "modelLicense": metadata.get("license", ""),
        "validation": {
            "precision": args.precision, "recall": args.recall,
            "mAP50": args.map50, "mAP50_95": args.map50_95,
        },
        "experimental": True,
    }
    output = args.output or args.model.with_name("manifest.json")
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(output), "sha256": digest.hexdigest(),
                      "fileSizeBytes": manifest["fileSizeBytes"],
                      "input": [model_input.name, input_shape],
                      "output": [model_output.name, output_shape]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
