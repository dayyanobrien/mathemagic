"""Shared helpers for running and scoring evaluations."""

import glob
import json
import os
import re

from .benchmark import build_messages, load_jsonl
from .scoring import score_response, summarise

DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
DATASETS = {
    "benchmark": os.path.join(DATA_DIR, "benchmark"),
    "noise_ablation": os.path.join(DATA_DIR, "noise_ablation"),
}


def load_items(data, seeds=None):
    """Load test items from a named dataset, a directory, or JSONL file(s)."""
    paths = []
    for entry in ([data] if isinstance(data, str) else data):
        entry = DATASETS.get(entry, entry)
        if os.path.isdir(entry):
            paths += sorted(glob.glob(os.path.join(entry, "*.jsonl")))
        else:
            paths.append(entry)
    if not paths:
        raise FileNotFoundError(f"No test-set files found for {data!r}")
    items = [item for p in paths for item in load_jsonl(p)]
    if seeds:
        items = [it for it in items if it["seed"] in set(seeds)]
    return items


def make_job(item, setting, variant, shots):
    return {
        "key": f"{item['id']}|{setting}|v{variant}|{shots}",
        "item": item,
        "setting": setting,
        "instruction_variant": variant,
        "shots": shots,
        "messages": build_messages(item, shots, setting, variant),
    }


def parse_key(key):
    """Inverse of the job key: ``(item_id, setting, variant, shots)``."""
    item_id, setting, variant, shots = key.split("|")
    return item_id, setting, int(variant.lstrip("v")), int(shots)


def iter_jobs(items, settings, shots_list, variants):
    """Yield one job (dict) per (setting, variant, shots, item)."""
    for setting in settings:
        for variant in variants:
            for shots in shots_list:
                for item in items:
                    if shots <= len(item["examples"]):
                        yield make_job(item, setting, variant, shots)


def detail_row(job, model, output):
    item = job["item"]
    scored = score_response(output, item)
    answer = item["answer"]
    if isinstance(answer, float) and answer.is_integer():
        answer = int(answer)
    return {
        "key": job["key"],
        "model": model,
        "setting": job["setting"],
        "instruction_variant": job["instruction_variant"],
        "seed": item["seed"],
        "shots": job["shots"],
        "id": item["id"],
        "transformation_type": item["transformation_type"],
        "input_prompt": job["messages"][-1]["content"],
        "output_raw": output,
        "reference": answer,
        **scored,
    }


def sanitize(name):
    return re.sub(r"[^\w\-.]+", "_", str(name)).strip("_")


def read_details(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_details(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_summaries(rows, out_dir):
    """Write per-seed, seed-averaged and per-transformation summaries as CSV."""
    if not rows:
        return
    os.makedirs(out_dir, exist_ok=True)
    keys = ["model", "setting", "instruction_variant", "shots"]
    per_seed = summarise(rows, by=keys + ["seed"])
    per_seed.to_csv(os.path.join(out_dir, "summary_per_seed.csv"), index=False)

    metrics = [c for c in per_seed.columns if c not in keys + ["seed", "n"]]
    overall = per_seed.groupby(keys)[metrics].agg(["mean", "std"])
    overall.columns = [f"{m}_{s}" if s == "std" else m for m, s in overall.columns]
    overall.reset_index().to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    summarise(rows, by=keys + ["transformation_type"]).to_csv(
        os.path.join(out_dir, "summary_per_transformation.csv"), index=False)
    return overall.reset_index()
