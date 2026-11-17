"""Generate MatheMagic test sets.

Examples
--------
Regenerate the paper's five test sets:

    python scripts/generate_benchmark.py

Make a fresh, unseen test set with a new seed:

    python scripts/generate_benchmark.py --seeds 2026 --out-dir data/my_sets

Check that the shipped files in data/ regenerate exactly:

    python scripts/generate_benchmark.py --check
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mathemagic.benchmark import (  # noqa: E402
    EXAMPLES_GENERATED_PER_ITEM,
    PAPER_ITEMS_PER_TRANSFORMATION,
    PAPER_SEEDS,
    create_test_set,
    load_gsm8k_expressions,
    load_jsonl,
    save_jsonl,
)
from mathemagic.transforms import TRANSFORMATIONS  # noqa: E402

DEFAULT_OUT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=PAPER_SEEDS)
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--items-per-transformation", type=int, default=PAPER_ITEMS_PER_TRANSFORMATION)
    p.add_argument("--max-examples", type=int, default=512, help="In-context examples stored per item (max shots).")
    p.add_argument("--transformations", nargs="+", default=TRANSFORMATIONS, choices=TRANSFORMATIONS)
    p.add_argument("--examples-per-item", type=int, default=EXAMPLES_GENERATED_PER_ITEM,
                   help="Examples generated per item (changes which examples items get). 512 matches the shipped "
                        "data; some paper runs (GPT-5-nano, Qwen2.5 14B/32B/72B, Qwen2.5-Math) used 2048.")
    p.add_argument("--check", action="store_true", help="Compare against existing files instead of writing.")
    args = p.parse_args()

    print("Loading GSM8K expressions...")
    train, test = load_gsm8k_expressions()
    print(f"  {len(train)} train / {len(test)} test expressions")

    jobs = []
    for seed in args.seeds:
        jobs.append((os.path.join(args.out_dir, "benchmark", f"seed_{seed}.jsonl"),
                     dict(seed=seed, transformations=args.transformations,
                          items_per_transformation=args.items_per_transformation,
                          max_examples=args.max_examples,
                          examples_per_item=args.examples_per_item)))

    failures = 0
    for path, kwargs in jobs:
        items = create_test_set(test, train, **kwargs)
        if args.check:
            ok = os.path.exists(path) and load_jsonl(path) == items
            failures += not ok
            print(f"{'OK  ' if ok else 'DIFF'} {path}")
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            save_jsonl(items, path)
            print(f"wrote {len(items)} items -> {path}")

    if args.check:
        print("All files match." if not failures else f"{failures} file(s) differ.")
        sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
