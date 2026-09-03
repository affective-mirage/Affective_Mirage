#!/usr/bin/env python3
"""
D4 - emotion uncertainty. One (model x dataset) run.

image-present only. For every image we produce a *legal posterior over the label set* by
teacher-forced scoring of the frozen answer format `[[label]]`, then compare that posterior
against the human multi-annotator vote distribution.

Per image:
  1. constrained scoring: one batched forward over all candidate answer strings
     `[[happy]] ... [[neutral]]` (a class may have several surface forms -> logsumexp over them),
     softmax over classes -> model_dist. Also a length-normalized variant (robustness column).
  2. greedy free generation + D1's [[label]] parser -> model_pred_free (consistency sanity).

No sampling anywhere. Reuses D1's prompt, label_map, parser and base-2 JSD.

Imports D1 helpers from `run_D1.py`, expected to sit next to this file.
"""
import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_D1 import PROMPT, LABEL_RE, load_label_map, _jsd  # noqa: E402

CONTRACT_FIELDS = [
    "experiment", "model", "model_version", "dataset", "item_id", "condition",
    "gold_face", "gold_context", "pred_raw", "pred_canonical", "seed",
]
D4_FIELDS = [
    "split", "taxonomy", "n_valid_votes",
    "human_dist", "human_entropy", "human_agreement", "human_majority_label",
    "model_dist", "model_dist_lennorm", "model_pmax", "model_entropy", "model_pred",
    "model_pred_free", "free_matches_constrained", "correct_vs_majority", "jsd_model_human",
]
CSV_FIELDS = CONTRACT_FIELDS + D4_FIELDS


# ----------------------------- prompt -----------------------------
def build_prompt(option_words):
    """Frozen prompt with the option word list swapped in.
    For canonical7 the words are the frozen ones, so the string is byte-identical to D1's."""
    frozen_words = ["happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"]
    frozen_line = "Choose exactly one: " + ", ".join(frozen_words) + "."
    new_line = "Choose exactly one: " + ", ".join(option_words) + "."
    assert frozen_line in PROMPT, "frozen prompt no longer contains the option line"
    return PROMPT.replace(frozen_line, new_line)


def parse_prediction_tax(raw, label_map, classes):
    """D1's `[[label]]` parser widened to the RUN's taxonomy. D1 validates against the
    canonical 7 only, which would demote a legal `[[contempt]]` answer (FER+) to `other`."""
    m = LABEL_RE.search(raw or "")
    if m:
        token = m.group(1).strip().lower()
        if token == "":
            return "parse_fail"
        if token in classes:
            return token
        mapped = label_map.get(token, "other")
        return mapped if mapped in classes else "other"
    # bare-word fallback: some models emit `sad` not `[[sad]]` (free-gen sanity only)
    low = (raw or "").strip().lower()
    if low in classes:
        return low
    for w in re.findall(r"[a-z]+", low):
        if w in classes:
            return w
        mp = label_map.get(w)
        if mp in classes:
            return mp
    return "parse_fail"


def build_candidates(classes, surface_forms, template):
    """[(class, answer_string)] in taxonomy order, a class may contribute several strings."""
    out = []
    for c in classes:
        for form in surface_forms[c]:
            out.append((c, template.format(form=form)))
    return out


# ----------------------------- math (base-2) -----------------------------
def entropy2(p):
    return -sum(x * math.log2(x) for x in p if x > 0)


def logsumexp(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def softmax_from_logs(logs):
    """logs are natural-log scores; returns a normalized distribution."""
    z = logsumexp(logs)
    return [math.exp(x - z) for x in logs]


def fmt_dist(p):
    return ";".join(f"{x:.6g}" for x in p)


# ----------------------------- model -----------------------------
class ScoringModel:
    """AutoProcessor + AutoModelForImageTextToText. Same families as D1's HFImageTextModel:
    qwen (enable_thinking off) / mistral3 / hf (plain template, incl. native InternVL)."""

    def __init__(self, model_id, family, max_new_tokens=64):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.template_kwargs = {"enable_thinking": False} if family == "qwen" else {}
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="cuda")
        self.model.eval()
        cfg = getattr(self.model, "config", None)
        self.version = getattr(cfg, "_name_or_path", model_id) if cfg else model_id
        self.tok = getattr(self.processor, "tokenizer", self.processor)
        if getattr(self.tok, "pad_token", None) is None:
            self.tok.pad_token = self.tok.eos_token
        # candidate answers differ in length; the shared prefix must stay at the FRONT,
        # so batched scoring requires right padding (InternVL's tokenizer defaults to left).
        self.tok.padding_side = "right"

    def _prefix_text(self, prompt):
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}]}]
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **self.template_kwargs)

    def score(self, image, prompt, answers):
        """Sum of token logprobs of each answer string, teacher-forced after (image, prompt).
        Returns [(logp, n_answer_tokens)] aligned with `answers`. One batched forward."""
        torch = self.torch
        prefix = self._prefix_text(prompt)
        k = len(answers)

        pre = self.processor(text=[prefix], images=[image], return_tensors="pt")
        prefix_len = int(pre["input_ids"].shape[1])

        # padding_side must be passed here, not only set on the tokenizer: VLM processors
        # carry their own default (InternVL's is "left"), which would move the shared prefix.
        inputs = self.processor(text=[prefix + a for a in answers], images=[image] * k,
                                return_tensors="pt", padding=True, padding_side="right")
        ids = inputs["input_ids"]
        # tokenizer must not merge across the prefix/answer boundary, else the split is wrong
        want = pre["input_ids"].expand(k, -1).to(ids.device)
        if not torch.equal(ids[:, :prefix_len], want):
            r, c = (ids[:, :prefix_len] != want).nonzero()[0].tolist()
            raise RuntimeError(
                "prefix tokenization changed when the answer was appended "
                f"(padding_side={self.tok.padding_side}); first mismatch row={r} col={c}: "
                f"got {int(ids[r, c])} ({self.tok.decode([int(ids[r, c])])!r}) "
                f"want {int(want[r, c])} ({self.tok.decode([int(want[r, c])])!r}); "
                f"prefix_len={prefix_len} full_shape={tuple(ids.shape)}")
        inputs = inputs.to(self.model.device)
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        with torch.no_grad():
            logits = self.model(**inputs).logits.float()
        logprobs = torch.log_softmax(logits, dim=-1)
        # token at position t is predicted by logits at t-1
        tgt = ids[:, prefix_len:]
        lp = logprobs[:, prefix_len - 1:-1, :].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        valid = mask[:, prefix_len:].bool()
        lp = lp.masked_fill(~valid, 0.0)
        totals = lp.sum(dim=-1).tolist()
        ntok = valid.sum(dim=-1).tolist()
        return list(zip(totals, [int(n) for n in ntok]))

    def free_generate(self, image, prompt):
        """Greedy, deterministic. Consistency sanity only."""
        torch = self.torch
        prefix = self._prefix_text(prompt)
        inputs = self.processor(text=[prefix], images=[image],
                                return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            gen = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                      do_sample=False)
        trimmed = gen[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]


# ----------------------------- data -----------------------------
def load_items(manifest_path, classes, limit=None):
    """manifest: item_id,gold_face,image_path[,votes_<class>...]. Vote columns are optional;
    when absent the human-side fields stay empty (single-label datasets)."""
    base = Path(manifest_path).parent
    vote_cols = [f"votes_{c}" for c in classes]
    items = []
    with open(manifest_path) as f:
        rdr = csv.DictReader(f)
        has_votes = all(c in rdr.fieldnames for c in vote_cols)
        for row in rdr:
            p = row["image_path"]
            it = {"item_id": row["item_id"], "gold_face": row.get("gold_face", ""),
                  "image_path": p if os.path.isabs(p) else str(base / p), "votes": None}
            if has_votes:
                it["votes"] = [float(row[c]) for c in vote_cols]
            items.append(it)
    if limit:
        items = items[:limit]
    return items, has_votes


def human_stats(votes, classes):
    total = sum(votes)
    p = [v / total for v in votes]
    return {"dist": p, "entropy": entropy2(p), "agreement": max(p),
            "majority": classes[p.index(max(p))], "n": total}


# ----------------------------- run -----------------------------
def run(args):
    cfg = yaml.safe_load(open(args.config))
    ds = cfg["datasets"][args.dataset]
    classes = cfg["taxonomies"][ds["taxonomy"]]
    prompt = build_prompt(cfg["prompt_option_words"][ds["taxonomy"]])
    cands = build_candidates(classes, cfg["surface_forms"], cfg["answer_template"])
    cand_class_idx = [classes.index(c) for c, _ in cands]
    answers = [a for _, a in cands]
    label_map = load_label_map(Path(__file__).resolve().parent / "label_map.json")
    seed = cfg["seed"]

    manifest = Path(args.manifest or ds["manifest"])
    if not manifest.exists():
        sys.exit(f"FATAL: manifest not found: {manifest}")
    items, has_votes = load_items(manifest, classes, args.limit)
    if args.num_shards > 1:
        items = items[args.shard::args.num_shards]
    print(f"[D4] {args.model_name} x {args.dataset}: {len(items)} items, "
          f"{len(classes)} classes, {len(answers)} candidate strings, votes={has_votes}")
    print(f"[D4] prompt:\n{prompt}")
    print(f"[D4] candidates: {answers}")

    from PIL import Image
    model = ScoringModel(args.model_path, args.model_family, cfg["max_new_tokens"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for item in items:
            image = Image.open(item["image_path"]).convert("RGB")
            scored = model.score(image, prompt, answers)
            # marginalize surface forms within a class, then normalize over classes
            per_class = [[] for _ in classes]
            per_class_ln = [[] for _ in classes]
            for (lp, ntok), ci in zip(scored, cand_class_idx):
                per_class[ci].append(lp)
                per_class_ln[ci].append(lp / max(ntok, 1))
            logs = [logsumexp(v) for v in per_class]
            logs_ln = [logsumexp(v) for v in per_class_ln]
            dist = softmax_from_logs(logs)
            dist_ln = softmax_from_logs(logs_ln)
            pmax = max(dist)
            pred = classes[dist.index(pmax)]

            raw = model.free_generate(image, prompt)
            pred_free = parse_prediction_tax(raw, label_map, classes)

            if item["votes"]:
                h = human_stats(item["votes"], classes)
                gold = h["majority"]
                jsd = _jsd(dist, h["dist"])
                hcols = {"n_valid_votes": h["n"], "human_dist": fmt_dist(h["dist"]),
                         "human_entropy": f"{h['entropy']:.6g}",
                         "human_agreement": f"{h['agreement']:.6g}",
                         "human_majority_label": h["majority"],
                         "jsd_model_human": f"{jsd:.6g}"}
            else:
                gold = item["gold_face"]
                hcols = {"n_valid_votes": "", "human_dist": "", "human_entropy": "",
                         "human_agreement": "", "human_majority_label": "",
                         "jsd_model_human": ""}

            row = {
                "experiment": "D4", "model": args.model_name,
                "model_version": model.version, "dataset": args.dataset,
                "item_id": item["item_id"], "condition": "image_present",
                "gold_face": gold, "gold_context": "",
                "pred_raw": (raw or "").replace("\n", " ").strip(),
                "pred_canonical": pred_free, "seed": seed,
                "split": ds["split"], "taxonomy": ds["taxonomy"],
                "model_dist": fmt_dist(dist), "model_dist_lennorm": fmt_dist(dist_ln),
                "model_pmax": f"{pmax:.6g}", "model_entropy": f"{entropy2(dist):.6g}",
                "model_pred": pred, "model_pred_free": pred_free,
                "free_matches_constrained": str(pred_free == pred),
                "correct_vs_majority": str(pred == gold),
            }
            row.update(hcols)
            w.writerow(row)
            f.flush()
            n += 1
            if args.verbose:
                print(json.dumps({k: row[k] for k in (
                    "item_id", "human_dist", "human_entropy", "human_agreement",
                    "human_majority_label", "model_dist", "model_pmax", "model_entropy",
                    "model_pred", "model_pred_free", "pred_raw", "jsd_model_human")}, indent=2))
            elif n % 100 == 0:
                print(f"[D4] {n}/{len(items)}", flush=True)
    print(f"[D4] wrote {n} rows -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="D4 per-image uncertainty inference")
    ap.add_argument("--config", default="configs/D4.yaml")
    ap.add_argument("--dataset", required=True, choices=["ferplus", "rafdb"])
    ap.add_argument("--manifest", help="override the config's manifest path")
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-family", required=True, choices=["qwen", "mistral3", "hf"])
    ap.add_argument("--out", required=True, help="output csv")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--verbose", action="store_true", help="dump every per-image field (dry-run)")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
