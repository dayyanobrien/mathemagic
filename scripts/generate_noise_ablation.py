"""Generate the noise-ablation set (examples that follow no single rule).

See ``mathemagic/noise_ablation.py`` for the design.

    python scripts/generate_noise_ablation.py           # write data/noise_ablation/seed_42.jsonl
    python scripts/generate_noise_ablation.py --check   # confirm the shipped file regenerates exactly
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mathemagic.benchmark import load_gsm8k_expressions, load_jsonl, save_jsonl  # noqa: E402
from mathemagic.noise_ablation import NOISE_MAX_EXAMPLES, NOISE_QUERIES, NOISE_SEED, create_noise_set  # noqa: E402

DEFAULT_OUT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "noise_ablation"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=NOISE_SEED)
    p.add_argument("--queries", type=int, default=NOISE_QUERIES)
    p.add_argument("--max-examples", type=int, default=NOISE_MAX_EXAMPLES)
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--check", action="store_true", help="Compare against the existing file instead of writing.")
    args = p.parse_args()

    train, test = load_gsm8k_expressions()
    items = create_noise_set(test, train, args.seed, args.queries, args.max_examples)
    path = os.path.join(args.out_dir, f"seed_{args.seed}.jsonl")
    if args.check:
        ok = os.path.exists(path) and load_jsonl(path) == items
        print(f"{'OK  ' if ok else 'DIFF'} {path}")
        sys.exit(0 if ok else 1)
    os.makedirs(args.out_dir, exist_ok=True)
    save_jsonl(items, path)
    print(f"wrote {len(items)} items -> {path}")


if __name__ == "__main__":
    main()
