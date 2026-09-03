#!/usr/bin/env python3
"""
D2 - subject-region removal sample code.

This is a reference implementation for Diagnosis 2. It compares a
model's prediction on the original visual input with the same input after the
subject bounding box is replaced by a neutral gray rectangle.

Expected manifest columns:
  item_id,image_path,gold_labels,bbox_x1,bbox_y1,bbox_x2,bbox_y2

`gold_labels` can contain one label or a semicolon-separated label set. Image
paths may be absolute or relative to the manifest directory.

Examples:
  python run_D2.py --dry-run --limit 3 --out /tmp/d2_dryrun.csv
  python run_D2.py --manifest data/emotic_d2_manifest.csv --dataset EMOTIC \
      --model-family qwen --model-path /path/to/Qwen3.5-9B \
      --model-name Qwen3.5-9B --out results/d2_qwen.csv
"""
import argparse
import csv
import os
from pathlib import Path

from PIL import Image, ImageDraw

from run_D1 import ABSTENTION_RE, LABEL_RE, build_model, make_blank


EMOTIC_LABELS = [
    "Affection", "Anger", "Annoyance", "Anticipation", "Aversion",
    "Confidence", "Disapproval", "Disconnection", "Disquietment",
    "Doubt/Confusion", "Embarrassment", "Engagement", "Esteem",
    "Excitement", "Fatigue", "Fear", "Happiness", "Pain", "Peace",
    "Pleasure", "Sadness", "Sensitivity", "Suffering", "Surprise",
    "Sympathy", "Yearning",
]
CAER_LABELS = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


CSV_FIELDS = [
    "experiment", "model", "model_version", "dataset", "item_id", "condition",
    "visual_input", "gold_labels", "pred_raw", "pred_canonical", "response_type",
    "correct", "prediction_changed_from_original", "seed", "image_path",
    "subject_bbox",
]


class DryRunModel:
    version = "dry-run-stub"

    def __init__(self, label_options):
        self.label_options = label_options
        self._i = 0

    def generate(self, image, seed, prompt):
        label = self.label_options[self._i % len(self.label_options)]
        self._i += 1
        return f"[[{label}]]"


def default_label_options(dataset):
    name = dataset.lower()
    if name == "emotic":
        return EMOTIC_LABELS
    if name == "caer":
        return CAER_LABELS
    raise ValueError("--labels is required for datasets other than EMOTIC or CAER")


def parse_label_options(spec, dataset):
    if spec:
        return [x.strip() for x in spec.split(",") if x.strip()]
    return default_label_options(dataset)


def load_label_map(path, label_options):
    label_map = {label.lower(): label for label in label_options}
    if path and Path(path).exists():
        import json
        with open(path) as f:
            for key, value in json.load(f).get("map", {}).items():
                if value.lower() in label_map:
                    label_map[key.lower()] = label_map[value.lower()]
    return label_map


def parse_prediction(raw, label_map):
    if raw is None:
        return "parse_fail"
    match = LABEL_RE.search(raw)
    if match:
        token = match.group(1).strip().lower()
        return label_map.get(token, "other") if token else "parse_fail"
    low = raw.strip().lower()
    if low in label_map:
        return label_map[low]
    for label_key in sorted(label_map, key=len, reverse=True):
        if label_key in low:
            return label_map[label_key]
    return "parse_fail"


def classify_response(raw, label_map):
    if raw is not None and ABSTENTION_RE.search(raw):
        return "abstention", "abstention"
    pred = parse_prediction(raw, label_map)
    if pred == "parse_fail":
        return "parse_fail", "parse_fail"
    return "valid_emotion", pred


def build_prompt(label_options):
    labels = ", ".join(label_options)
    return (
        "What emotion is the subject in this image expressing?\n"
        f"Choose exactly one from: {labels}.\n"
        "Respond with only the label as: [[label]]."
    )


def parse_label_set(value, label_map):
    labels = []
    for token in str(value or "").replace("|", ";").replace(",", ";").split(";"):
        token = token.strip()
        if token and token.lower() in label_map:
            labels.append(label_map[token.lower()])
    return labels


def parse_bbox(row):
    return tuple(float(row[k]) for k in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"))


def bbox_to_str(bbox):
    return ",".join(f"{x:.1f}" for x in bbox)


def clamp_bbox(bbox, size):
    w, h = size
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(w, int(round(x1))))
    y1 = max(0, min(h, int(round(y1))))
    x2 = max(0, min(w, int(round(x2))))
    y2 = max(0, min(h, int(round(y2))))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid subject bbox after clamping: {bbox} for image size {size}")
    return x1, y1, x2, y2


def load_manifest(manifest_path, label_map, limit=None):
    base = Path(manifest_path).parent
    required = {"item_id", "image_path", "gold_labels", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"}
    items = []
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest is missing columns: {sorted(missing)}")
        for row in reader:
            path = row["image_path"]
            if path and not os.path.isabs(path):
                path = str(base / path)
            items.append({
                "item_id": row["item_id"],
                "image_path": path,
                "gold_labels": parse_label_set(row["gold_labels"], label_map),
                "bbox": parse_bbox(row),
            })
    return items[:limit] if limit else items


def dummy_items(n, label_options):
    return [{
        "item_id": f"dummy_{idx:04d}",
        "image_path": None,
        "gold_labels": [label_options[idx % len(label_options)]],
        "bbox": (56, 40, 168, 200),
    } for idx in range(n)]


def load_image(path):
    if path:
        return Image.open(path).convert("RGB")
    return make_blank()


def remove_subject_region(image, bbox, fill=(128, 128, 128), draw_box=False):
    out = image.copy()
    box = clamp_bbox(bbox, out.size)
    draw = ImageDraw.Draw(out)
    draw.rectangle(box, fill=fill)
    if draw_box:
        draw.rectangle(box, outline=(220, 20, 60), width=3)
    return out


def build_row(args, model, item, condition, visual_input, raw, rtype, pred, seed, changed):
    gold = item["gold_labels"]
    return {
        "experiment": "D2_subject_region_removal",
        "model": args.model_name,
        "model_version": model.version,
        "dataset": args.dataset,
        "item_id": item["item_id"],
        "condition": condition,
        "visual_input": visual_input,
        "gold_labels": ";".join(gold),
        "pred_raw": (raw or "").replace("\n", " ").strip(),
        "pred_canonical": pred,
        "response_type": rtype,
        "correct": int(pred in gold),
        "prediction_changed_from_original": "" if changed is None else int(changed),
        "seed": seed,
        "image_path": item["image_path"] or "",
        "subject_bbox": bbox_to_str(item["bbox"]),
    }


def run(args):
    label_options = parse_label_options(args.labels, args.dataset)
    label_map = load_label_map(args.label_map, label_options)
    prompt = build_prompt(label_options)
    items = dummy_items(args.limit or 4, label_options) if args.dry_run else load_manifest(
        args.manifest, label_map, args.limit)
    if args.num_shards > 1:
        items = items[args.shard::args.num_shards]
        print(f"[run_D2] shard {args.shard}/{args.num_shards}: {len(items)} items")

    model = DryRunModel(label_options) if args.dry_run else build_model(
        args.model_family, args.model_path, args.temperature, args.max_new_tokens)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append and out_path.exists() else "w"
    write_header = mode == "w"
    n_rows = 0

    with open(out_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for item in items:
            image = load_image(item["image_path"])
            removed = remove_subject_region(image, item["bbox"], draw_box=args.draw_box)

            original_pred = None
            for condition, visual_input, current_image in [
                ("original", "V_i", image),
                ("subject_region_removed", "V_i_subject_removed", removed),
            ]:
                raw = model.generate(current_image, args.seed, prompt)
                rtype, pred = classify_response(raw, label_map)
                if condition == "original":
                    original_pred = pred
                    changed = None
                else:
                    changed = pred != original_pred
                writer.writerow(build_row(
                    args, model, item, condition, visual_input, raw, rtype, pred,
                    args.seed, changed))
                n_rows += 1
                f.flush()

    print(f"[run_D2] wrote {n_rows} rows ({mode}) -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="D2: subject-region removal")
    ap.add_argument("--manifest", help="CSV with item_id,image_path,gold_labels,bbox_x1,bbox_y1,bbox_x2,bbox_y2")
    ap.add_argument("--dataset", default="EMOTIC")
    ap.add_argument("--model-path", default=os.environ.get("D2_MODEL_PATH"))
    ap.add_argument("--model-name", default="Qwen3.5-9B")
    ap.add_argument("--model-family", default="qwen", choices=["qwen", "mistral3", "internvl", "hf"])
    ap.add_argument("--labels", help="comma-separated label names; defaults to the EMOTIC or CAER label space")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--label-map", default=str(Path(__file__).parent / "label_map.json"))
    ap.add_argument("--out", default="results/D2_subject_region_removal.csv")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--draw-box", action="store_true", help="draw a red outline around the removed subject region")
    ap.add_argument("--dry-run", action="store_true", help="run without loading a VLM")
    args = ap.parse_args()
    if not args.dry_run and not args.manifest:
        ap.error("--manifest is required unless --dry-run")
    if not args.dry_run and not args.model_path:
        ap.error("--model-path is required unless --dry-run")
    run(args)


if __name__ == "__main__":
    main()
