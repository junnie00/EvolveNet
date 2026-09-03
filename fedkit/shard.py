"""Split a domain slice into K client shards (round-robin after a seeded shuffle).

    PYTHONPATH=. python -m fedkit.shard --domain ds1000 --slice ds1000/slices/hard50.json \
        --clients 5 --seed 0 --out ds1000/logs/shards_hard50
"""
import argparse
import random
from pathlib import Path

from .adapters import get_adapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--slice", required=True)
    ap.add_argument("--clients", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ad = get_adapter(args.domain)
    ids = ad.slice_ids(args.slice)
    order = list(range(len(ids)))
    random.Random(args.seed).shuffle(order)
    shards = [[] for _ in range(args.clients)]
    for pos, idx in enumerate(order):
        shards[pos % args.clients].append(ids[idx])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for k, part in enumerate(shards):
        ad.write_slice(part, out / f"shard{k}.json")
        print(f"shard{k}: {len(part)} problems")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
