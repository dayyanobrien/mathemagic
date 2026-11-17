"""Compare a model on the noise ablation with the same model on the real benchmark.

If a model treated the benchmark's in-context examples as noise, it would fall
back to ordinary maths about as often on both. This prints, per shot count:

* noise ablation: how often the answer is ordinary maths (on all questions and
  with the matched overlap exclusion, see
  ``mathemagic.noise_ablation.matched_normal_math_rate``), refusals, other
  answers and format errors;
* benchmark (inductive, instruction variant 1): reversion to ordinary maths on
  questions whose transformed answer differs from the ordinary one, and format
  errors (missing box or unparsable answer).

Usage:
    python scripts/noise_ablation_report.py \\
        --noise results/<model>/noise_ablation/details.jsonl \\
        --benchmark results/<model>/benchmark/details.jsonl
"""

import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mathemagic.evaluation import load_items, read_details  # noqa: E402
from mathemagic.noise_ablation import matched_normal_math_rate, normal_answer  # noqa: E402
from mathemagic.scoring import is_match  # noqa: E402


def pct(part, whole):
    return 100 * part / whole if whole else float("nan")


def noise_table(rows, items):
    out = {}
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["shots"])].append(r)
    for (model, shots), group in sorted(groups.items()):
        n = len(group)
        # Seeded per model and shot count so each estimate is reproducible on its own.
        mean, lo, hi, kept = matched_normal_math_rate(group, items, shots,
                                                      rng_seed=f"{model.split('/')[-1]}|{shots}")
        out[(model, shots)] = {
            "n": n,
            "normal_all": pct(sum(r["correct_orig"] for r in group), n),
            "normal_matched": mean, "matched_lo": lo, "matched_hi": hi, "matched_kept": kept,
            "refusal": pct(sum(bool(r.get("refusal")) for r in group), n),
            "other": pct(sum(r["other_err"] for r in group), n),
            "format": pct(sum(r["format_err"] or r["parse_err"] for r in group), n),
        }
    return out


def benchmark_table(rows, items):
    by_id = {item["id"]: item for item in items}
    overlap = {}
    for item in items:
        try:
            overlap[item["id"]] = is_match(normal_answer(item["original"]), item["answer"])
        except (ArithmeticError, SyntaxError, TypeError, ValueError):
            overlap[item["id"]] = False
    groups = defaultdict(list)
    for r in rows:
        if r["setting"] == "inductive" and r["instruction_variant"] == 1 and r["id"] in by_id:
            groups[(r["model"], r["shots"])].append(r)
    out = {}
    for key, group in groups.items():
        kept = [r for r in group if not overlap[r["id"]]]
        out[key] = {
            "n": len(group),
            "nonoverlap": len(kept),
            "reversion": pct(sum(r["correct_orig"] and not r["correct"] for r in kept), len(kept)),
            "format": pct(sum(r["format_err"] or r["parse_err"] for r in group), len(group)),
        }
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--noise", nargs="+", required=True, help="details.jsonl from --data noise_ablation")
    p.add_argument("--benchmark", nargs="*", default=[], help="details.jsonl from --data benchmark")
    args = p.parse_args()

    noise = noise_table([r for path in args.noise for r in read_details(path)], load_items("noise_ablation"))
    bench = benchmark_table([r for path in args.benchmark for r in read_details(path)],
                            load_items("benchmark")) if args.benchmark else {}

    header = (f"{'model':30} {'shots':>5} | {'n':>4} {'normal':>7} {'matched':>8} {'(95% of draws)':>15} "
              f"{'refusal':>8} {'other':>6} {'format':>7} | {'bench reversion':>15} {'bench format':>12}")
    print("Noise ablation (left) vs benchmark (right), all values in %")
    print(header)
    print("-" * len(header))
    for (model, shots), ns in noise.items():
        b = bench.get((model, shots))
        right = f"{b['reversion']:>15.1f} {b['format']:>12.1f}" if b else f"{'-':>15} {'-':>12}"
        print(f"{model:30} {shots:>5} | {ns['n']:>4} {ns['normal_all']:>7.1f} {ns['normal_matched']:>8.1f} "
              f"{'[' + format(ns['matched_lo'], '.1f') + ', ' + format(ns['matched_hi'], '.1f') + ']':>15} "
              f"{ns['refusal']:>8.1f} {ns['other']:>6.1f} {ns['format']:>7.1f} | {right}")
    print("\nnormal: answer is ordinary maths, all questions. matched: same, after the matched overlap\n"
          "exclusion. bench reversion: ordinary-maths answers on benchmark questions whose answer\n"
          "differs from ordinary maths. format: no boxed answer or an unparsable one.")


if __name__ == "__main__":
    main()
