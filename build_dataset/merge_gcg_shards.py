"""
merge_gcg_shards.py — Merge per-shard outputs of gcg_shard_ndss.py back
into a single canonical dataset, preserving the original record order.

Each shard JSONL produced by gcg_shard_ndss.py contains the slice of
records assigned to that shard, with the new white-box variant filled in
under r["ndss_attacks"]["..._white_box"] and r["gcg_white_box"][...].

This script:
  1. Reads every shard JSONL.
  2. Indexes shard records by their `id` field.
  3. Re-emits the original dataset in order, replacing each record by its
     freshly GCG-optimized version when present in the shard outputs.

Usage:
  python build_dataset/merge_gcg_shards.py \\
      --original data/scp.jsonl \\
      --shards    tmp_shards/scp.shard*.jsonl \\
      --output    data/scp.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict, List


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--original", type=str, required=True,
                   help="path to the canonical dataset whose order we preserve")
    p.add_argument("--shards", type=str, nargs="+", required=True,
                   help="shard JSONL files (glob patterns are expanded)")
    p.add_argument("--output", type=str, required=True,
                   help="output JSONL path (overwriting --original is OK)")
    args = p.parse_args()

    # Collect all updated records by id
    by_id: Dict[str, Dict[str, Any]] = {}
    expanded: List[str] = []
    for pat in args.shards:
        expanded.extend(sorted(glob.glob(pat)) or [pat])
    for path in expanded:
        if not Path(path).exists():
            print(f"[merge] missing shard {path}, skipping")
            continue
        recs = _read_jsonl(Path(path))
        for r in recs:
            rid = r.get("id")
            if rid is None:
                continue
            by_id[rid] = r
        print(f"[merge] {path}: {len(recs)} records")

    orig = _read_jsonl(Path(args.original))
    merged = [by_id.get(r["id"], r) for r in orig]
    n_replaced = sum(1 for r in orig if r["id"] in by_id)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[merge] wrote {len(merged)} records ({n_replaced} replaced from "
          f"{len(by_id)} shard records) -> {out_path}")


if __name__ == "__main__":
    main()
