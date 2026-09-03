#!/usr/bin/env python3
"""
Exp1 - image-present vs image-absent emotion-distribution baseline.

7-class taxonomy, frozen prompt + [[label]] parse, single CSV schema,
temperature=1, thinking OFF, blank/noise image specs.

Runs each item under conditions {image_present, blank, noise} for an open-source
vision-language model (open models are always fed a visual token -> blank/noise,
never a true visual-absent input). Writes one row per (item, condition, seed) to
the results CSV.

Model deps (torch/transformers) are imported lazily ONLY when actually generating,
so `--dry-run` validates parsing / CSV / image logic on a CPU login node with no GPU.
"""
import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

CANONICAL7 = ["neutral", "happy", "sad", "anger", "surprise", "disgust", "fear"]

# Frozen prompt (image_present / blank / noise / true_absent all share it).
PROMPT = (
    "What emotion is the person in this image expressing?\n"
    "Choose exactly one: happy, sad, angry, fear, surprise, disgust, neutral.\n"
    "Respond with only the label as: [[label]]."
)
# Variant B: reworded to drop the "look at the image" handle.
PROMPT_REWORDED = (
    "A person is described in this input. What emotion is this person most likely "
    "expressing? Choose exactly one: happy, sad, angry, fear, surprise, disgust, neutral. "
    "Respond with only the label as: [[label]]."
)
# Variant C: fake metadata hinting an image is attached (mirage induction).
PROMPT_MIRAGE_HINT = (
    "number of image attachments: 1\n"
    "What emotion is the person in this image expressing? Choose exactly one: happy, sad, "
    "angry, fear, surprise, disgust, neutral. Respond with only the label as: [[label]]."
)

# condition -> prompt text
CONDITION_PROMPT = {
    "image_present": PROMPT, "blank": PROMPT, "noise": PROMPT, "true_absent": PROMPT,
    "blank_reworded": PROMPT_REWORDED, "blank_mirage_hint": PROMPT_MIRAGE_HINT,
}
# conditions whose visual input is a 224x224 mid-gray blank
BLANK_CONDITIONS = {"blank", "blank_reworded", "blank_mirage_hint"}

# ---- Section C variants (abstention-allowed / unrelated image / label-order permutation).
# Loaded from configs/c_variants.json so that different model runs use byte-identical prompts,
# orderings, and unrelated image. Additive: absent file -> only the frozen conditions above. ----
NOISE_CONDITIONS = {"noise"}
UNRELATED_CONDITIONS = set()
_CVAR = {}
_CVAR_PATH = Path(__file__).resolve().parent / "configs" / "c_variants.json"
if _CVAR_PATH.exists():
    _CVAR = json.load(open(_CVAR_PATH))
    _abstain = _CVAR["abstain_prompt"]
    CONDITION_PROMPT["blank_abstain"] = _abstain          # abstention-allowed over blank
    CONDITION_PROMPT["noise_abstain"] = _abstain          # abstention-allowed over noise
    BLANK_CONDITIONS.add("blank_abstain")
    NOISE_CONDITIONS.add("noise_abstain")
    _open = _CVAR["open_prompt"]                           # C1 (main): open-ended, no menu, no escape
    CONDITION_PROMPT["blank_open"] = _open
    CONDITION_PROMPT["noise_open"] = _open
    BLANK_CONDITIONS.add("blank_open")
    NOISE_CONDITIONS.add("noise_open")
    CONDITION_PROMPT["unrelated"] = PROMPT                 # C2: unrelated image, forced-choice
    UNRELATED_CONDITIONS.add("unrelated")
    _CVAR["_unrelated_abspath"] = str(Path(__file__).resolve().parent / _CVAR["unrelated_image"])
    for _k, _order in enumerate(_CVAR["orderings"]):       # C3: label-order permutation over blank
        _cond = "blank_order%d" % _k
        CONDITION_PROMPT[_cond] = PROMPT.replace(_CVAR["forced_option_line"], ", ".join(_order))
        BLANK_CONDITIONS.add(_cond)

CSV_FIELDS = [
    "experiment", "model", "model_version", "dataset", "item_id", "condition",
    "gold_face", "gold_context", "pred_raw", "pred_canonical", "seed",
    "response_type",  # additive 12th col: valid_emotion / abstention / parse_fail
]

# FER2013 integer label -> canonical (0..6)
FER2013_IDX2CANON = {
    0: "anger", 1: "disgust", 2: "fear", 3: "happy", 4: "sad", 5: "surprise", 6: "neutral",
}

LABEL_RE = re.compile(r"\[\[\s*(.*?)\s*\]\]", re.DOTALL)


# ----------------------------- parsing -----------------------------
def load_label_map(path):
    with open(path) as f:
        return json.load(f)["map"]


def parse_prediction(raw, label_map):
    """Extract [[label]] and map to canonical 7. No match -> bare-word fallback -> parse_fail.
    Fallback: models that ignore the [[label]] format and emit a bare word like 'sad' -> scan raw
    for the first canonical / label_map word. Does NOT affect [[label]]-emitting models, since
    LABEL_RE takes priority when present."""
    if raw is None:
        return "parse_fail"
    m = LABEL_RE.search(raw)
    if m:
        token = m.group(1).strip().lower()
        if token == "":
            return "parse_fail"
        if token in CANONICAL7:
            return token
        return label_map.get(token, "other")
    # no [[label]] -> bare-word fallback (whole string first, then scan words)
    low = (raw or "").strip().lower()
    if low in CANONICAL7:
        return low
    if low in label_map:
        return label_map[low]
    for w in re.findall(r"[a-z]+", low):
        if w in CANONICAL7:
            return w
        if w in label_map:
            return label_map[w]
    return "parse_fail"


# Abstention: model points out the image is missing / blank / faceless. Case-insensitive,
# whole-ish phrases to avoid catching "grey eyes" etc. Extend as needed.
ABSTENTION_PATTERNS = [
    r"no image", r"no (?:visible )?(?:person|face|human|subject|individual)",
    r"cannot see", r"can't see", r"can not see", r"unable to see",
    r"no visible", r"not visible", r"cannot determine", r"can't determine",
    r"unable to determine", r"cannot identify", r"can't identify",
    r"gr[ae]y (?:square|image|rectangle|box|background)", r"solid gr[ae]y",
    r"blank (?:image|square)?", r"is blank", r"no discernible",
    r"no (?:one|people)", r"there is no (?:person|face|image)",
    r"no facial", r"without.{0,10}(?:face|person|image)",
]
ABSTENTION_RE = re.compile("|".join(ABSTENTION_PATTERNS), re.IGNORECASE)


def classify_response(raw, label_map):
    """3-class response typing + pred_canonical.
    Returns (response_type, pred_canonical) where
      response_type in {valid_emotion, abstention, parse_fail};
      pred_canonical in CANONICAL7 + {other, abstention, parse_fail}.
    abstention wins even if a [[label]] is also present (model called out the missing image)."""
    if raw is not None and ABSTENTION_RE.search(raw):
        return "abstention", "abstention"
    pc = parse_prediction(raw, label_map)
    if pc == "parse_fail":
        return "parse_fail", "parse_fail"
    return "valid_emotion", pc


# ----------------------------- images -----------------------------
def make_blank():
    from PIL import Image
    return Image.new("RGB", (224, 224), (128, 128, 128))


def make_noise(seed):
    """Deterministic 224x224 Gaussian noise (given seed), clipped to uint8 RGB."""
    import numpy as np
    from PIL import Image
    rng = np.random.RandomState(seed & 0xFFFFFFFF)
    arr = rng.normal(128, 64, size=(224, 224, 3))
    arr = np.clip(arr, 0, 255).astype("uint8")
    return Image.fromarray(arr, mode="RGB")


def load_item_image(path):
    from PIL import Image
    return Image.open(path).convert("RGB")


# ----------------------------- data -----------------------------
def load_manifest(manifest_path, limit=None):
    """manifest.csv columns: item_id,gold_face,image_path (paths relative to manifest dir)."""
    base = Path(manifest_path).parent
    items = []
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            p = row["image_path"]
            p = p if os.path.isabs(p) else str(base / p)
            items.append({"item_id": row["item_id"], "gold_face": row["gold_face"], "image_path": p})
    if limit:
        items = items[:limit]
    return items


def dummy_items(n):
    """Synthetic items for --dry-run (no dataset needed)."""
    golds = CANONICAL7
    return [{"item_id": f"dummy_{i:04d}", "gold_face": golds[i % 7], "image_path": None}
            for i in range(n)]


# ----------------------------- model -----------------------------
class DryRunModel:
    """Stub predictor: cycles canonical labels so parsing/CSV get exercised without a GPU."""
    version = "dry-run-stub"

    def __init__(self):
        self._i = 0

    def generate(self, image, seed, prompt):
        # cycle valid labels + one abstention + one parse_fail to exercise classify_response
        samples = [f"[[{l}]]" for l in CANONICAL7] + [
            "The image is a solid gray square with no visible person. [[neutral]]",  # -> abstention
            "[]",  # -> parse_fail
        ]
        out = samples[self._i % len(samples)]
        self._i += 1
        return out


class HFImageTextModel:
    """AutoProcessor + AutoModelForImageTextToText families that share the messages API:
    Qwen-VL family (template_kwargs={'enable_thinking': False}) and
    Mistral3 / Pixtral / Llava (template_kwargs={}). Same frozen prompt for all."""

    def __init__(self, model_id, temperature=1.0, max_new_tokens=64, template_kwargs=None):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.torch = torch
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.template_kwargs = template_kwargs or {}
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="cuda"
        )
        self.model.eval()
        cfg = getattr(self.model, "config", None)
        self.version = getattr(cfg, "_name_or_path", model_id) if cfg else model_id

    def generate(self, image, seed, prompt):
        torch = self.torch
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **self.template_kwargs)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.model.device)
        torch.manual_seed(seed)
        with torch.no_grad():
            gen = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                do_sample=True, temperature=self.temperature)
        trimmed = gen[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]


# --- InternVL (internvl_chat): trust_remote_code + tile preprocessing + model.chat ---
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _internvl_transform(input_size=448):
    import torchvision.transforms as T
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB")),
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _internvl_dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    w, h = image.size
    ar = w / h
    ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
         if min_num <= i * j <= max_num}, key=lambda x: x[0] * x[1])
    best, best_diff = (1, 1), float("inf")
    for r in ratios:
        diff = abs(ar - r[0] / r[1])
        if diff < best_diff:
            best_diff, best = diff, r
    tw, th = image_size * best[0], image_size * best[1]
    blocks = best[0] * best[1]
    resized = image.resize((tw, th))
    tiles = []
    cols = tw // image_size
    for idx in range(blocks):
        box = ((idx % cols) * image_size, (idx // cols) * image_size,
               ((idx % cols) + 1) * image_size, ((idx // cols) + 1) * image_size)
        tiles.append(resized.crop(box))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


class InternVLModel:
    """InternVL (internvl_chat) via AutoModel(trust_remote_code) + model.chat()."""

    def __init__(self, model_id, temperature=1.0, max_new_tokens=64, max_tiles=12):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.max_tiles = max_tiles
        self.transform = _internvl_transform(448)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
        # InternVL's remote code calls .item() in __init__ -> breaks under device_map/meta-init.
        # Load real weights (low_cpu_mem_usage=False, no device_map) then move to the visible GPU.
        self.model = AutoModel.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, use_flash_attn=True,
            trust_remote_code=True, low_cpu_mem_usage=False).eval().cuda()
        self.gen_cfg = dict(max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature)
        self.version = model_id

    def generate(self, image, seed, prompt):
        torch = self.torch
        tiles = _internvl_dynamic_preprocess(image, max_num=self.max_tiles)
        pv = torch.stack([self.transform(t) for t in tiles]).to(torch.bfloat16).cuda()
        torch.manual_seed(seed)
        question = "<image>\n" + prompt
        return self.model.chat(self.tokenizer, pv, question, self.gen_cfg)


def build_model(family, model_id, temperature, max_new_tokens):
    if family == "qwen":
        return HFImageTextModel(model_id, temperature, max_new_tokens,
                                template_kwargs={"enable_thinking": False})
    if family in ("mistral3", "hf"):  # mistral3/pixtral/llava: plain messages template
        return HFImageTextModel(model_id, temperature, max_new_tokens, template_kwargs={})
    if family == "internvl":
        return InternVLModel(model_id, temperature, max_new_tokens)
    raise ValueError(f"unknown model family: {family}")


# ----------------------------- run -----------------------------
def image_for_condition(cond, item, seed):
    if cond == "image_present":
        return load_item_image(item["image_path"])
    if cond in UNRELATED_CONDITIONS:
        return load_item_image(_CVAR["_unrelated_abspath"])
    if cond in NOISE_CONDITIONS:
        return make_noise(seed)
    if cond in BLANK_CONDITIONS:
        return make_blank()
    raise ValueError(f"unsupported condition for open model: {cond}")


def run(args):
    label_map = load_label_map(args.label_map)
    conditions = args.conditions.split(",")
    seeds = parse_seeds(args.seeds)

    if args.dry_run:
        items = dummy_items(args.limit or 8)
        model = DryRunModel()
    else:
        items = load_manifest(args.manifest, args.limit)
        if args.num_shards > 1:
            items = items[args.shard::args.num_shards]  # strided data-parallel shard
            print(f"[run_exp1] shard {args.shard}/{args.num_shards}: {len(items)} items")
        model = build_model(args.model_family, args.model_path,
                            args.temperature, args.max_new_tokens)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # append: keep prior rows, only write header for a fresh file
    write_header = not (args.append and out_path.exists())
    mode = "a" if args.append and out_path.exists() else "w"
    n_rows = 0
    with open(out_path, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        for item in items:
            for seed in seeds:
                for cond in conditions:
                    prompt = CONDITION_PROMPT[cond]
                    img = None if args.dry_run else image_for_condition(cond, item, seed)
                    raw = model.generate(img, seed, prompt)
                    rtype, pcanon = classify_response(raw, label_map)
                    w.writerow({
                        "experiment": "exp1",
                        "model": args.model_name,
                        "model_version": model.version,
                        "dataset": args.dataset,
                        "item_id": item["item_id"],
                        "condition": cond,
                        "gold_face": item["gold_face"],
                        "gold_context": "",
                        "pred_raw": (raw or "").replace("\n", " ").strip(),
                        "pred_canonical": pcanon,
                        "seed": seed,
                        "response_type": rtype,
                    })
                    n_rows += 1
                    f.flush()  # per-row flush so a timeout keeps partial results
    print(f"[run_exp1] wrote {n_rows} rows ({mode}) -> {out_path}")
    if args.analyze:
        compute_metrics(out_path, args.dataset, args.model_name)
    if args.table:
        print_condition_table(out_path)
    if args.full:
        compute_full_metrics(out_path, args.dataset)


# ----------------------------- metrics -----------------------------
def _dist(labels):
    """Normalized 7-class histogram over canonical labels (pure python; no numpy)."""
    v = [0.0] * len(CANONICAL7)
    for lab in labels:
        if lab in CANONICAL7:
            v[CANONICAL7.index(lab)] += 1.0
    s = sum(v)
    return [x / s for x in v] if s > 0 else v


def _jsd(p, q):
    """Jensen-Shannon divergence (base 2), 0..1. Pure python."""
    import math
    m = [0.5 * (a + b) for a, b in zip(p, q)]
    def kl(a, b):
        return sum(ai * math.log2(ai / bi) for ai, bi in zip(a, b) if ai > 0 and bi > 0)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def compute_metrics(csv_path, dataset, model_name):
    import csv as _csv
    rows = list(_csv.DictReader(open(csv_path)))
    by_cond = {}
    for r in rows:
        by_cond.setdefault(r["condition"], []).append(r)

    def acc(rs):
        ok = sum(1 for r in rs if r["pred_canonical"] == r["gold_face"])
        return ok / len(rs) if rs else 0.0

    metrics = {"dataset": dataset, "model": model_name, "n_rows": len(rows), "by_condition": {}}
    for c, rs in by_cond.items():
        metrics["by_condition"][c] = {
            "n": len(rs),
            "acc": round(acc(rs), 4),
            "parse_fail_rate": round(sum(1 for r in rs if r["pred_canonical"] == "parse_fail") / len(rs), 4),
            "pred_dist": {CANONICAL7[i]: round(float(x), 4) for i, x in enumerate(_dist([r["pred_canonical"] for r in rs]))},
        }
    if "image_present" in by_cond and "blank" in by_cond:
        metrics["delta_acc_present_minus_blank"] = round(acc(by_cond["image_present"]) - acc(by_cond["blank"]), 4)
        metrics["jsd_present_vs_blank"] = round(_jsd(
            _dist([r["pred_canonical"] for r in by_cond["image_present"]]),
            _dist([r["pred_canonical"] for r in by_cond["blank"]])), 4)
    if "image_present" in by_cond and "noise" in by_cond:
        metrics["jsd_present_vs_noise"] = round(_jsd(
            _dist([r["pred_canonical"] for r in by_cond["image_present"]]),
            _dist([r["pred_canonical"] for r in by_cond["noise"]])), 4)

    mpath = Path(csv_path).with_suffix(".metrics.json")
    with open(mpath, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[run_exp1] metrics -> {mpath}")
    print(json.dumps(metrics, indent=2))


def parse_seeds(spec):
    """'0' -> [0]; '0,2,5' -> [0,2,5]; '0-29' -> [0..29]; mixes allowed."""
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if "-" in tok:
            a, b = tok.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif tok:
            out.append(int(tok))
    return out


def print_condition_table(csv_path, reference="image_present"):
    """Per-condition 7-class frequency + neutral share + JSD vs the reference condition."""
    rows = list(csv.DictReader(open(csv_path)))
    by = {}
    for r in rows:
        by.setdefault(r["condition"], []).append(r["pred_canonical"])
    ref = _dist(by[reference]) if reference in by else None
    order = [c for c in ["image_present", "blank", "noise", "blank_reworded", "blank_mirage_hint"] if c in by]
    order += [c for c in by if c not in order]

    ni = CANONICAL7.index("neutral")
    print(f"\n=== condition comparison (reference = {reference}) ===")
    header = f"{'condition':<20} {'n':>4}  " + " ".join(f"{c[:4]:>4}" for c in CANONICAL7) + f"  {'neut%':>6}  {'JSD':>6}"
    print(header)
    print("-" * len(header))
    for c in order:
        labs = by[c]
        counts = [sum(1 for x in labs if x == k) for k in CANONICAL7]
        neut_share = counts[ni] / len(labs) if labs else 0.0
        jsd = "" if ref is None else f"{_jsd(_dist(labs), ref):.3f}"
        row = f"{c:<20} {len(labs):>4}  " + " ".join(f"{n:>4}" for n in counts) + f"  {neut_share*100:>5.1f}%  {jsd:>6}"
        print(row)
    # note: parse_fail / other are not in the 7 columns; flag if any
    extra = sum(1 for r in rows if r["pred_canonical"] in ("parse_fail", "other"))
    if extra:
        print(f"(note: {extra} rows had pred_canonical in {{parse_fail, other}} - excluded from the 7 columns)")


DIST_CATS = CANONICAL7 + ["other", "abstention", "parse_fail"]
NO_VISION = ["blank", "noise", "blank_mirage_hint"]


def _dist_cats(labels, cats=DIST_CATS):
    v = [0.0] * len(cats)
    idx = {c: i for i, c in enumerate(cats)}
    for lab in labels:
        if lab in idx:
            v[idx[lab]] += 1.0
    s = sum(v)
    return [x / s for x in v] if s > 0 else v


def compute_full_metrics(csv_path, dataset):
    """Metrics organized as model x condition, with abstention/mirage rates. JSON + print.
    Structured so more models append under 'models' for later cross-model comparison."""
    rows = list(csv.DictReader(open(csv_path)))
    models = sorted(set(r["model"] for r in rows))
    out = {"dataset": dataset, "dist_categories": DIST_CATS, "models": {}}

    for model in models:
        mrows = [r for r in rows if r["model"] == model]
        by = {}
        for r in mrows:
            by.setdefault(r["condition"], []).append(r)

        def acc(rs):
            return sum(1 for r in rs if r["pred_canonical"] == r["gold_face"]) / len(rs) if rs else 0.0

        def rtype_counts(rs):
            c = {"valid_emotion": 0, "abstention": 0, "parse_fail": 0}
            for r in rs:
                c[r.get("response_type", "valid_emotion")] = c.get(r.get("response_type", "valid_emotion"), 0) + 1
            return c

        present = by.get("image_present", [])
        present_acc = acc(present)
        present_dist = _dist_cats([r["pred_canonical"] for r in present]) if present else None

        mblock = {}
        for cond, rs in by.items():
            n = len(rs)
            rtc = rtype_counts(rs)
            freq = {DIST_CATS[i]: int(x) for i, x in
                    enumerate([sum(1 for r in rs if r["pred_canonical"] == c) for c in DIST_CATS])}
            block = {
                "n": n,
                "acc": round(acc(rs), 4),
                "neutral_share": round(sum(1 for r in rs if r["pred_canonical"] == "neutral") / n, 4),
                "response_type_counts": rtc,
                "freq": freq,
            }
            if cond in NO_VISION:
                block["abstention_rate"] = round(rtc["abstention"] / n, 4)
                block["mirage_rate"] = round(rtc["valid_emotion"] / n, 4)  # confident emotion, no acknowledgement
                block["parse_fail_rate"] = round(rtc["parse_fail"] / n, 4)
                block["delta_acc_present_minus_cond"] = round(present_acc - acc(rs), 4) if present else None
                block["jsd_vs_present"] = (round(_jsd(_dist_cats([r["pred_canonical"] for r in rs]), present_dist), 4)
                                           if present_dist else None)
            mblock[cond] = block
        out["models"][model] = mblock

    mpath = Path(csv_path).with_suffix(".metrics.json")
    with open(mpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[run_exp1] full metrics -> {mpath}")
    print_full_table(out)
    return out


def print_full_table(metrics):
    order = ["image_present"] + NO_VISION
    for model, mblock in metrics["models"].items():
        conds = [c for c in order if c in mblock] + [c for c in mblock if c not in order]
        print(f"\n=== {model} on {metrics['dataset']} ===")
        hdr = (f"{'condition':<20}{'n':>5}{'acc':>7}{'neut%':>7}{'abst%':>7}"
               f"{'mirage%':>8}{'pfail%':>7}{'delta':>7}{'JSD':>7}")
        print(hdr); print("-" * len(hdr))
        for c in conds:
            b = mblock[c]
            n = b["n"]
            abst = b.get("abstention_rate")
            mir = b.get("mirage_rate")
            pf = b.get("parse_fail_rate")
            dl = b.get("delta_acc_present_minus_cond")
            jsd = b.get("jsd_vs_present")
            def pct(x): return f"{x*100:>6.1f}" if isinstance(x, (int, float)) else f"{'--':>6}"
            def num(x): return f"{x:>6.3f}" if isinstance(x, (int, float)) else f"{'--':>6}"
            print(f"{c:<20}{n:>5}{b['acc']:>7.3f}{b['neutral_share']*100:>6.1f}"
                  f"{pct(abst):>7}{pct(mir):>8}{pct(pf):>7}{num(dl):>7}{num(jsd):>7}")


# ----------------------------- cli -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Exp1: image-present vs absent emotion baseline")
    ap.add_argument("--manifest", help="path to manifest.csv (item_id,gold_face,image_path)")
    ap.add_argument("--model-path", default=os.environ.get("EXP1_MODEL_PATH", "Qwen/Qwen2.5-VL-7B-Instruct"))
    ap.add_argument("--model-name", default="qwen-vl")
    ap.add_argument("--model-family", default="qwen", choices=["qwen", "mistral3", "internvl", "hf"],
                    help="qwen(enable_thinking off) / mistral3 / internvl(trust_remote_code+chat) / hf(plain)")
    ap.add_argument("--dataset", default="fer2013")
    ap.add_argument("--conditions", default="image_present,blank,noise")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="cap number of items (smoke)")
    ap.add_argument("--shard", type=int, default=0, help="this worker's shard index [0..num_shards)")
    ap.add_argument("--num-shards", type=int, default=1, help="data-parallel shards (1 GPU each)")
    ap.add_argument("--label-map", default=str(Path(__file__).parent / "label_map.json"))
    ap.add_argument("--out", default="results/exp1_smoke.csv")
    ap.add_argument("--analyze", action="store_true", help="also write .metrics.json (acc/delta/JSD)")
    ap.add_argument("--full", action="store_true", help="after run, write model x condition full metrics (abstention/mirage)")
    ap.add_argument("--append", action="store_true", help="append rows to --out (keep prior results)")
    ap.add_argument("--table", action="store_true", help="after run, print per-condition comparison table")
    ap.add_argument("--table-only", action="store_true", help="no model: just print the table from --out and exit")
    ap.add_argument("--full-only", action="store_true", help="no model: recompute full metrics from --out and exit")
    ap.add_argument("--dry-run", action="store_true", help="no model/GPU; stub predictor to validate parse/CSV")
    args = ap.parse_args()

    if args.table_only:
        print_condition_table(args.out)
        return
    if args.full_only:
        compute_full_metrics(args.out, args.dataset)
        return
    if not args.dry_run and not args.manifest:
        ap.error("--manifest is required unless --dry-run")
    run(args)


if __name__ == "__main__":
    main()
