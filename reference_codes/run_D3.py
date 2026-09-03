#!/usr/bin/env python3
"""
D3 - subject-scene recomposition sample code.

This is a reference implementation for Diagnosis 3. For a paired sample
`i, j`, it evaluates the original visual input of sample i and two composite
inputs: one that keeps the subject from i with the scene context from j, and one
that keeps the scene context from i with the subject from j.

Expected manifest columns:
  item_id_i,image_path_i,gold_labels_i,bbox_i_x1,bbox_i_y1,bbox_i_x2,bbox_i_y2,
  item_id_j,image_path_j,gold_labels_j,bbox_j_x1,bbox_j_y1,bbox_j_x2,bbox_j_y2

`gold_labels_i` and `gold_labels_j` can contain one label or semicolon-separated
label sets. Image paths may be absolute or relative to the manifest directory.

Examples:
  python run_D3.py --dry-run --limit 2 --out /tmp/d3_dryrun.csv
  python run_D3.py --manifest data/emotic_d3_pairs.csv --dataset EMOTIC \
      --model-family internvl --model-path /path/to/InternVL3.5-8B \
      --model-name InternVL3.5-8B --out results/d3_internvl.csv
"""
import argparse
import csv
import os
from pathlib import Path

from PIL import Image, ImageDraw

from run_D1 import ABSTENTION_RE, LABEL_RE, build_model


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
    "experiment", "model", "model_version", "dataset", "pair_id", "item_id_i",
    "item_id_j", "condition", "visual_input", "subject_label_set",
    "scene_context_label_set", "pred_raw", "pred_canonical", "response_type",
    "recognition_against_sample_i", "prediction_changed_from_original_i",
    "source_alignment", "seed", "image_path_i", "image_path_j",
    "subject_bbox_i", "subject_bbox_j",
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


def parse_bbox(row, prefix):
    return tuple(float(row[f"{prefix}_{k}"]) for k in ("x1", "y1", "x2", "y2"))


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
        raise ValueError(f"invalid bbox after clamping: {bbox} for image size {size}")
    return x1, y1, x2, y2


def load_manifest(manifest_path, label_map, limit=None):
    base = Path(manifest_path).parent
    required = {
        "item_id_i", "image_path_i", "gold_labels_i",
        "bbox_i_x1", "bbox_i_y1", "bbox_i_x2", "bbox_i_y2",
        "item_id_j", "image_path_j", "gold_labels_j",
        "bbox_j_x1", "bbox_j_y1", "bbox_j_x2", "bbox_j_y2",
    }
    pairs = []
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest is missing columns: {sorted(missing)}")
        for row in reader:
            path_i = row["image_path_i"]
            path_j = row["image_path_j"]
            if path_i and not os.path.isabs(path_i):
                path_i = str(base / path_i)
            if path_j and not os.path.isabs(path_j):
                path_j = str(base / path_j)
            pairs.append({
                "item_id_i": row["item_id_i"],
                "item_id_j": row["item_id_j"],
                "image_path_i": path_i,
                "image_path_j": path_j,
                "labels_i": parse_label_set(row["gold_labels_i"], label_map),
                "labels_j": parse_label_set(row["gold_labels_j"], label_map),
                "bbox_i": parse_bbox(row, "bbox_i"),
                "bbox_j": parse_bbox(row, "bbox_j"),
            })
    return pairs[:limit] if limit else pairs


def dummy_pairs(n, label_options):
    return [{
        "item_id_i": f"dummy_i_{idx:04d}",
        "item_id_j": f"dummy_j_{idx:04d}",
        "image_path_i": None,
        "image_path_j": None,
        "labels_i": [label_options[idx % len(label_options)]],
        "labels_j": [label_options[(idx + 3) % len(label_options)]],
        "bbox_i": (56, 40, 168, 200),
        "bbox_j": (56, 40, 168, 200),
    } for idx in range(n)]


def load_image(path, tint):
    if path:
        return Image.open(path).convert("RGB")
    return Image.new("RGB", (224, 224), tint)


def paste_subject(subject_image, subject_bbox, scene_image, scene_bbox, draw_box=False):
    out = scene_image.copy()
    src_box = clamp_bbox(subject_bbox, subject_image.size)
    dst_box = clamp_bbox(scene_bbox, out.size)
    patch = subject_image.crop(src_box).resize(
        (dst_box[2] - dst_box[0], dst_box[3] - dst_box[1]), Image.BICUBIC)
    out.paste(patch, dst_box)
    if draw_box:
        ImageDraw.Draw(out).rectangle(dst_box, outline=(220, 20, 60), width=3)
    return out


def source_alignment(pred, subject_labels, scene_context_labels):
    in_subject = pred in subject_labels
    in_scene = pred in scene_context_labels
    if in_subject and not in_scene:
        return "subject_only"
    if in_scene and not in_subject:
        return "scene_context_only"
    if in_subject and in_scene:
        return "shared"
    return "neither"


def build_row(args, model, pair, condition, visual_input, subject_labels,
              scene_context_labels, raw, rtype, pred, original_i_pred, seed):
    changed = "" if condition == "original_i" else int(pred != original_i_pred)
    return {
        "experiment": "D3_subject_scene_recomposition",
        "model": args.model_name,
        "model_version": model.version,
        "dataset": args.dataset,
        "pair_id": f"{pair['item_id_i']}__{pair['item_id_j']}",
        "item_id_i": pair["item_id_i"],
        "item_id_j": pair["item_id_j"],
        "condition": condition,
        "visual_input": visual_input,
        "subject_label_set": ";".join(subject_labels),
        "scene_context_label_set": ";".join(scene_context_labels),
        "pred_raw": (raw or "").replace("\n", " ").strip(),
        "pred_canonical": pred,
        "response_type": rtype,
        "recognition_against_sample_i": int(pred in pair["labels_i"]),
        "prediction_changed_from_original_i": changed,
        "source_alignment": source_alignment(pred, subject_labels, scene_context_labels),
        "seed": seed,
        "image_path_i": pair["image_path_i"] or "",
        "image_path_j": pair["image_path_j"] or "",
        "subject_bbox_i": bbox_to_str(pair["bbox_i"]),
        "subject_bbox_j": bbox_to_str(pair["bbox_j"]),
    }


def maybe_save_visual(path_dir, pair, condition, image):
    if not path_dir:
        return
    out_dir = Path(path_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image.save(out_dir / f"{pair['item_id_i']}__{pair['item_id_j']}__{condition}.png")


def run(args):
    label_options = parse_label_options(args.labels, args.dataset)
    label_map = load_label_map(args.label_map, label_options)
    prompt = build_prompt(label_options)
    pairs = dummy_pairs(args.limit or 2, label_options) if args.dry_run else load_manifest(
        args.manifest, label_map, args.limit)
    if args.num_shards > 1:
        pairs = pairs[args.shard::args.num_shards]
        print(f"[run_D3] shard {args.shard}/{args.num_shards}: {len(pairs)} pairs")

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
        for pair in pairs:
            image_i = load_image(pair["image_path_i"], (160, 170, 190))
            image_j = load_image(pair["image_path_j"], (190, 170, 160))
            subject_i_scene_j = paste_subject(
                image_i, pair["bbox_i"], image_j, pair["bbox_j"], draw_box=args.draw_box)
            subject_j_scene_i = paste_subject(
                image_j, pair["bbox_j"], image_i, pair["bbox_i"], draw_box=args.draw_box)

            conditions = [
                ("original_i", "V_i=(S_i,C_i)", image_i, pair["labels_i"], pair["labels_i"]),
                ("subject_i_scene_j", "(S_i,C_j)", subject_i_scene_j, pair["labels_i"], pair["labels_j"]),
                ("subject_j_scene_i", "(S_j,C_i)", subject_j_scene_i, pair["labels_j"], pair["labels_i"]),
            ]

            original_i_pred = None
            cached_rows = []
            for condition, visual_input, image, subject_labels, scene_context_labels in conditions:
                maybe_save_visual(args.save_visuals, pair, condition, image)
                raw = model.generate(image, args.seed, prompt)
                rtype, pred = classify_response(raw, label_map)
                if condition == "original_i":
                    original_i_pred = pred
                cached_rows.append((condition, visual_input, subject_labels,
                                    scene_context_labels, raw, rtype, pred))

            for row_data in cached_rows:
                writer.writerow(build_row(
                    args, model, pair, *row_data, original_i_pred=original_i_pred,
                    seed=args.seed))
                n_rows += 1
                f.flush()

    print(f"[run_D3] wrote {n_rows} rows ({mode}) -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="D3: subject-scene recomposition")
    ap.add_argument("--manifest", help="CSV with paired samples i and j")
    ap.add_argument("--dataset", default="EMOTIC")
    ap.add_argument("--model-path", default=os.environ.get("D3_MODEL_PATH"))
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
    ap.add_argument("--out", default="results/D3_subject_scene_recomposition.csv")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--draw-box", action="store_true", help="draw a red outline around the pasted subject region")
    ap.add_argument("--save-visuals", help="optional directory for saving constructed visual inputs")
    ap.add_argument("--dry-run", action="store_true", help="run without loading a VLM")
    args = ap.parse_args()
    if not args.dry_run and not args.manifest:
        ap.error("--manifest is required unless --dry-run")
    if not args.dry_run and not args.model_path:
        ap.error("--model-path is required unless --dry-run")
    run(args)


if __name__ == "__main__":
    main()
