"""Split a cross-set question slice into K client shards.

Each shard is written as `<out>/shard<k>.json` plus `<out>/shard<k>_details.json`, so
optimize.py's built-in identity check (it compares loaded question text against the adjacent
details ledger) runs per shard. That is what stops Mini-Dev indices from being silently applied
to the full BIRD dev set.

Round-robin over the source order keeps each shard's DB / difficulty mix close to the whole.
"""
import argparse
import json
import random
from pathlib import Path


def split(cross, details, k, seed=0):
    """-> [(cross_pairs, detail_records), ...] of length k, round-robin after a seeded shuffle."""
    order = list(range(len(cross)))
    random.Random(seed).shuffle(order)
    shards = [([], []) for _ in range(k)]
    for pos, idx in enumerate(order):
        pairs, recs = shards[pos % k]
        pairs.append(cross[idx])
        if details:
            recs.append(details[idx])
    return shards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cross-set", required=True, help="json {cross:[[db_id,idx],...]}")
    ap.add_argument("--clients", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="directory to write shard<k>.json into")
    args = ap.parse_args()

    src = Path(args.cross_set)
    cross = json.loads(src.read_text(encoding="utf-8"))["cross"]
    details_path = src.with_name(f"{src.stem}_details.json")
    details = json.loads(details_path.read_text(encoding="utf-8")) if details_path.exists() else None
    if details is not None and len(details) != len(cross):
        raise ValueError(f"{details_path} has {len(details)} records but {src} has {len(cross)} items")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for k, (pairs, recs) in enumerate(split(cross, details, args.clients, args.seed)):
        (out / f"shard{k}.json").write_text(json.dumps({"cross": pairs}, indent=2), encoding="utf-8")
        if details is not None:
            (out / f"shard{k}_details.json").write_text(json.dumps(recs, indent=2), encoding="utf-8")
        dbs = {}
        for db_id, _ in pairs:
            dbs[db_id] = dbs.get(db_id, 0) + 1
        print(f"shard{k}: {len(pairs)} questions  {dbs}")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
