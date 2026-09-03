"""Score ONE harness on ONE slice, as a standalone process. -> rows json + printed total.

Run by fed_loop as a subprocess (isolation: generated harness code is untrusted; a hang costs one
timeout). Also usable directly for held-out test evaluation:

    PYTHONPATH=. PYTHONHASHSEED=0 python -m fedkit.evaluate --domain ds1000 \
        --slice ds1000/slices/hard50.json --harness bare --out /tmp/rows.json
"""
import argparse
import json
from pathlib import Path

from .adapters import get_adapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--slice", required=True)
    ap.add_argument("--harness", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--solve-timeout", type=int, default=600)
    args = ap.parse_args()

    ad = get_adapter(args.domain)
    items = ad.load_items(ad.slice_ids(args.slice))
    rows = ad.solve_and_grade(args.harness, items, args.solve_timeout)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"{args.harness}: {sum(r['correct'] for r in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
