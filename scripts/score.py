"""Score responses produced by your own inference code.

1. Dump prompts:   python scripts/evaluate.py --dump-prompts prompts.jsonl --shots 0 8 64
2. Run your model on each line's ``messages`` and write a JSONL file with one
   object per prompt: ``{"key": <key from prompts.jsonl>, "output": <model text>}``
3. Score:          python scripts/score.py responses.jsonl --model my-model

Writes ``details.jsonl`` and ``summary*.csv`` to ``--out-dir/<model>/``.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mathemagic.evaluation import DATASETS, detail_row, load_items, make_job, parse_key, sanitize, write_summaries  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("responses", help="JSONL with 'key' and 'output' fields.")
    p.add_argument("--model", required=True, help="Name to record for this model.")
    p.add_argument("--data", nargs="+", default=["benchmark"],
                   help=f"Dataset the prompts came from ({', '.join(DATASETS)}, a directory or JSONL files).")
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    with open(args.responses, encoding="utf-8") as f:
        responses = {r["key"]: r.get("output") for r in map(json.loads, filter(str.strip, f))}

    by_id = {item["id"]: item for item in load_items(args.data)}
    unknown = [k for k in responses if parse_key(k)[0] not in by_id]
    if unknown:
        sys.exit(f"{len(unknown)} response keys do not match any test item, e.g. {unknown[0]!r}. Check --data.")
    jobs = {}
    for key in responses:
        item_id, setting, variant, shots = parse_key(key)
        jobs[key] = make_job(by_id[item_id], setting, variant, shots)

    rows = [detail_row(jobs[k], args.model, out) for k, out in responses.items()]
    out_dir = os.path.join(args.out_dir, sanitize(args.model))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "details.jsonl"), "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    summary = write_summaries(rows, out_dir)
    cols = ["setting", "instruction_variant", "shots", "accuracy", "correct_wrt_orig"]
    if "refusal" in summary:  # noise ablation: there is no correct answer
        cols = ["setting", "shots", "correct_wrt_orig", "refusal", "wrong_other", "no_or_empty_box"]
    print(summary[cols].to_string(index=False, float_format="%.1f"))
    print(f"Wrote {len(rows)} scored rows and summaries to {out_dir}/")


if __name__ == "__main__":
    main()
