"""
prepare_benign_pairs.py — Build benign paraphrase pair pool
============================================================
Generates 500 (or more) benign paraphrase pairs (q1, q2) drawn from
Natural Questions (KCA track) or MS-MARCO (SCP track), where q2 is a
GPT-paraphrased rewrite of q1. The pool is the shared calibration set
used by Assumption (i) and the Benign Hit Rate (BHR) measurement in
exp2 and exp5.

Cached to data/benign_pairs.jsonl (paths managed by common.py).

Usage:
  python prepare_benign_pairs.py                 # default: 500 pairs
  python prepare_benign_pairs.py --num 1000
  python prepare_benign_pairs.py --num 100 --output data/benign_pairs_small.jsonl

The output is a JSONL file with one object per line:
  {"q1": "...", "q2": "..."}

Downstream experiments call `common.load_benign_pairs(N)` to get the first
N pairs. If more are requested than exist in cache, the cache is auto-extended.
"""

import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm

from common import (
    load_nq_queries, gpt_call, OUTPUTS_DIR, DEFAULT_BENIGN_PAIRS_PATH,
)


PARAPHRASE_PROMPT = """Rewrite the following question in a semantically
equivalent but textually different form. Keep the same intent; just rephrase.
Return only the rewritten question, no preamble.

Question: {query}
Rewritten:"""


def generate_pairs(num_pairs, seed=42):
    """Generate (q1, q2) paraphrase pairs from NQ-Open queries."""
    # Fetch a larger pool than needed, since some may fail paraphrasing
    # or produce trivially-same output.
    pool = load_nq_queries(num=int(num_pairs * 1.3), seed=seed)
    pairs = []
    for q in tqdm(pool, desc="paraphrase"):
        p = gpt_call(PARAPHRASE_PROMPT.format(query=q), temperature=0.8)
        if p is None:
            continue
        p = p.strip().strip('"').strip()
        if p and p.lower() != q.lower() and len(p) > 10:
            pairs.append({"q1": q, "q2": p})
        if len(pairs) >= num_pairs:
            break
    return pairs


def load_existing(path):
    path = Path(path)
    if not path.exists():
        return []
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def append_pairs(path, new_pairs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for p in new_pairs:
            f.write(json.dumps(p) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=500,
                        help="Target number of pairs to have in cache")
    parser.add_argument("--output", type=str, default=str(DEFAULT_BENIGN_PAIRS_PATH),
                        help="Output JSONL path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true",
                        help="Regenerate all pairs (overwrites existing cache)")
    args = parser.parse_args()

    out_path = Path(args.output)

    if args.force and out_path.exists():
        print(f"[force] removing existing {out_path}")
        out_path.unlink()

    existing = load_existing(out_path)
    print(f"Cache: {len(existing)} existing pairs at {out_path}")

    needed = args.num - len(existing)
    if needed <= 0:
        print(f"Already have {len(existing)} >= {args.num} pairs. Nothing to do.")
        return

    print(f"Generating {needed} new pairs...")
    new_pairs = generate_pairs(needed, seed=args.seed + len(existing))
    append_pairs(out_path, new_pairs)

    total = len(existing) + len(new_pairs)
    print(f"\n[done] Cache now has {total} pairs at {out_path}")
    if total < args.num:
        print(f"[warn] Wanted {args.num} but only got {total}. "
              f"Some paraphrasing attempts may have failed.")


if __name__ == "__main__":
    main()
