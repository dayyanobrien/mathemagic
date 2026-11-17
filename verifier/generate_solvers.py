"""Verifier step 1: turn each wrong answer's reasoning into a Python ``solver``.

For inductive responses whose boxed answer disagrees with the reference (and
that saw at least one example), the model's own explanation is given to a
code-generation model with ``codegen_prompt.txt`` (Appendix A.2), which
returns a function ``solver(problem)``. Step 2 (``check_solvers.py``) then
runs it on the examples shown and on the question.

The paper used Gemini 3.5 Flash through the Gemini Batch API (needs
``GEMINI_API_KEY`` and ``pip install google-genai``).

Usage:
    python verifier/generate_solvers.py results/<model>/benchmark/details.jsonl
    python verifier/generate_solvers.py results/*/benchmark/details.jsonl --limit 5   # smoke test
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mathemagic.evaluation import read_details  # noqa: E402

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "codegen_prompt.txt")


def build_prompt(reasoning):
    with open(PROMPT_PATH, encoding="utf-8") as f:
        few_shot = f.read().strip()
    return f"{few_shot}\n---\n**QUESTION**\n\n**REASONING:**\n{reasoning}\n\n**CODE:**"


def select_rows(rows, settings=("inductive",)):
    return [r for r in rows
            if r["setting"] in settings and r["shots"] > 0 and not r["correct"] and r.get("output_raw")]


def run_gemini_batch(prompts, model):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    request_file = "verifier_batch_requests.jsonl"
    with open(request_file, "w", encoding="utf-8") as f:
        for key, prompt in prompts.items():
            f.write(json.dumps({"key": key, "request": {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}}) + "\n")

    uploaded = client.files.upload(file=request_file,
                                   config=types.UploadFileConfig(display_name="mathemagic-verifier", mime_type="jsonl"))
    job = client.batches.create(model=model, src=uploaded.name, config={"display_name": "mathemagic-verifier"})
    print(f"Submitted batch job {job.name}; polling...")
    while job.state.name in ("JOB_STATE_PENDING", "JOB_STATE_RUNNING", "JOB_STATE_QUEUED"):
        time.sleep(30)
        job = client.batches.get(name=job.name)
        print(f"  {job.state.name}")
    if job.state.name != "JOB_STATE_SUCCEEDED":
        sys.exit(f"Batch job ended in state {job.state.name}")

    results = {}
    for line in client.files.download(file=job.dest.file_name).decode("utf-8").splitlines():
        res = json.loads(line)
        try:
            parts = res["response"]["candidates"][0]["content"]["parts"]
            text = parts[0].get("text", "").strip() if parts else ""
        except (KeyError, IndexError):
            text = ""
        results[res.get("key", "")] = text
    os.remove(request_file)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("details", nargs="+", help="details.jsonl file(s) from scripts/evaluate.py")
    p.add_argument("--out", default="verifier_solvers.jsonl")
    p.add_argument("--codegen-model", default="gemini-3.5-flash")
    p.add_argument("--limit", type=int, default=None, help="Only process the first N rows (smoke test).")
    args = p.parse_args()

    rows = select_rows([r for path in args.details for r in read_details(path)])
    if args.limit:
        rows = rows[:args.limit]
    print(f"{len(rows)} wrong inductive responses (shots > 0) to convert into solvers")
    if not rows:
        return

    # Keys combine model and job key so several models can share one batch.
    prompts = {f"{r['model']}||{r['key']}": build_prompt(r["output_raw"]) for r in rows}
    solvers = run_gemini_batch(prompts, args.codegen_model)

    with open(args.out, "w", encoding="utf-8") as f:
        for key in prompts:
            model, job_key = key.split("||", 1)
            f.write(json.dumps({"model": model, "key": job_key,
                                "generated_solver": solvers.get(key) or "None generated."}) + "\n")
    print(f"Wrote {len(prompts)} solvers to {args.out}")


if __name__ == "__main__":
    main()
