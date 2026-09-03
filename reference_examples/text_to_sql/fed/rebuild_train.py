import os
"""Rebuild the training slice from audited questions only, preserving the test set's mix.

Every question rejected by fed/audit.py is dropped and replaced by an audited-clean one from the same
database, matching difficulty where the pool allows and falling back to another difficulty in the SAME
database otherwise (the DB mix is what must match the test set; difficulty is second priority). Every
such fallback is printed and recorded.

Indices are written against dev.json, the same file test50.json is indexed against, so the whole
pipeline runs under one BIRD_DEV_FILE.
"""
import argparse
import collections
import json
import random
from pathlib import Path

from .. import bridge

DEV = Path(os.environ.get("BENCH_DIR", "data/bird") + "/dev_20240627/dev.json")


def strip_hint(t):
    return t.partition("\nHint:")[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    D = Path(args.split_dir)

    dev = {d["question_id"]: d for d in json.loads(DEV.read_text(encoding="utf-8"))}
    old = json.loads((D / "train50_details.json").read_text(encoding="utf-8"))

    ledger = {}
    for name in ("train50_audit.jsonl", "replacement_audit.jsonl", "extra_audit.jsonl"):
        p = D / name
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    e = json.loads(line)
                    ledger[e["question_id"]] = e

    kept = [r for r in old if ledger.get(r["question_id"], {}).get("verdict") == "accept"]
    dropped = [r for r in old if ledger.get(r["question_id"], {}).get("verdict") == "reject"]
    print(f"kept {len(kept)}, dropped {len(dropped)}")

    used = {r["question_id"] for r in old}
    used |= {x["question_id"] for x in json.loads((D / "test50_details.json").read_text(encoding="utf-8"))}
    clean = collections.defaultdict(list)
    for qid, e in ledger.items():
        if e["verdict"] == "accept" and qid not in used:
            clean[(e["db_id"], e["difficulty"])].append(qid)
    for v in clean.values():
        v.sort()

    rng = random.Random(args.seed)
    picked, deviations = [], []
    for r in dropped:
        db, diff = r["db_id"], r["difficulty"]
        pool = clean.get((db, diff))
        if pool:
            qid = pool.pop(rng.randrange(len(pool)))
        else:                                   # same DB, any difficulty — DB mix outranks difficulty
            alt = [(k, v) for k, v in clean.items() if k[0] == db and v]
            if not alt:
                raise SystemExit(f"no audited-clean replacement left for {db}")
            k, v = alt[0]
            qid = v.pop(rng.randrange(len(v)))
            deviations.append(f"{db}: wanted {diff}, took {k[1]}")
        picked.append(qid)

    final = [r["question_id"] for r in kept] + picked
    if len(final) != len(old):
        raise SystemExit(f"size drift: {len(final)} vs {len(old)}")
    if len(set(final)) != len(final):
        raise SystemExit("duplicate question in rebuilt slice")

    idx = {}
    for qid in final:
        db = dev[qid]["db_id"]
        if db not in idx:
            idx[db] = {strip_hint(q.question): i for i, q in enumerate(bridge.eval_questions(db))}
    pairs, recs = [], []
    for qid in final:
        d = dev[qid]
        i = idx[d["db_id"]].get(strip_hint(d["question"]))
        if i is None:
            raise SystemExit(f"qid {qid} not found in dev index")
        pairs.append([d["db_id"], i])
        recs.append({"question_id": qid, "db_id": d["db_id"],
                     "question": strip_hint(d["question"]), "difficulty": d.get("difficulty")})

    (D / "train50c.json").write_text(json.dumps({"cross": pairs}, indent=2), encoding="utf-8")
    (D / "train50c_details.json").write_text(json.dumps(recs, indent=2), encoding="utf-8")

    test = json.loads((D / "test50_details.json").read_text(encoding="utf-8"))
    print("\n            " + "  ".join(f"{k:>12s}" for k in ("db", "train", "test")))
    for db in sorted({r["db_id"] for r in recs} | {r["db_id"] for r in test}):
        a = sum(r["db_id"] == db for r in recs)
        b = sum(r["db_id"] == db for r in test)
        print(f"  {db:26s} train {a:2d}   test {b:2d}   {'' if a == b else '  <-- mismatch'}")
    print(f"\n  train difficulty {dict(collections.Counter(r['difficulty'] for r in recs))}")
    print(f"  test  difficulty {dict(collections.Counter(r['difficulty'] for r in test))}")
    print(f"  train/test overlap: {len(set(final) & {r['question_id'] for r in test})}")
    for d in deviations:
        print(f"  [deviation] {d}")
    print(f"\n[saved] {D}/train50c.json")


if __name__ == "__main__":
    main()
