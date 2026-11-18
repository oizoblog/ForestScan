"""Download a small, attributed CC BY 4.0 oyster mushroom detection subset.

Source: Duman et al., Annotated Oyster Mushroom Images, DOI 10.17632/hf55tkx489.1.
Annotations are converted from normalized top-left xywh to YOLO center xywh.
Downloaded images stay under ignored data/ and are never bundled in the PWA.
"""
import hashlib
import io
import json
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image

DATASET = "Project-AgML/oyster_mushroom_maturity_detection"
REVISION = "c299436e0dfb5b6db2e55ea5eccf37c994917b2a"
TRAIN = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380]
VAL = [440, 470, 500, 530]
ROOT = Path("data/oyster_experiment")


def get_row(index: int) -> dict:
    query = urllib.parse.urlencode({"dataset": DATASET, "config": "default", "split": "train", "offset": index, "length": 1})
    with urllib.request.urlopen(f"https://datasets-server.huggingface.co/rows?{query}", timeout=30) as response:
        return json.load(response)["rows"][0]["row"]


def main() -> None:
    evidence = []
    for split, indices in (("train", TRAIN), ("val", VAL)):
        image_dir = ROOT / "images" / split
        label_dir = ROOT / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for index in indices:
            row = get_row(index)
            with urllib.request.urlopen(row["image"]["src"], timeout=45) as response:
                raw = response.read()
            image = Image.open(io.BytesIO(raw)).convert("RGB")
            if image.size != (row["image"]["width"], row["image"]["height"]):
                raise ValueError(f"Image dimensions mismatch at row {index}")
            # Resize without crop; normalized annotations remain valid.
            image.thumbnail((640, 640))
            name = f"oyster_{index:03d}"
            image.save(image_dir / f"{name}.jpg", quality=85)
            lines = []
            for x, y, w, h in row["objects"]["bbox"]:
                if min(x, y, w, h) < 0 or x + w > 1.001 or y + h > 1.001:
                    raise ValueError(f"Invalid box at row {index}: {(x, y, w, h)}")
                lines.append(f"0 {x + w / 2:.6f} {y + h / 2:.6f} {w:.6f} {h:.6f}")
            (label_dir / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            evidence.append({"row": index, "split": split, "boxes": len(lines), "source_sha256": hashlib.sha256(raw).hexdigest()})
            print(f"{split} {index}: {len(lines)} boxes", flush=True)
    (ROOT / "dataset.yaml").write_text("path: .\ntrain: images/train\nval: images/val\nnames:\n  0: mushroom\n", encoding="utf-8")
    license_info = {
        "dataset": "Annotated Oyster Mushroom Images (24-image experimental subset)",
        "source": "https://data.mendeley.com/datasets/hf55tkx489/1",
        "license": "CC BY 4.0",
        "attribution": "Sonay Duman, Abdullah Elewi, Abdulsalam Hajhamed, Rasheed Khankan, Amina Souag, Asma Ahmed (2024), Annotated Oyster Mushroom Images, Mendeley Data V1, doi:10.17632/hf55tkx489.1",
        "modifications": "24 source images selected, resized to max 640 px, two maturity labels merged into mushroom",
        "mirror": f"https://huggingface.co/datasets/{DATASET}/tree/{REVISION}",
        "rows": evidence,
    }
    (ROOT / "DATA_LICENSE.json").write_text(json.dumps(license_info, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
