import os
"""Build a train slice, and re-index the existing test slice, BOTH against full dev.json.

Why re-index the test set: a cross-set `[db_id, idx]` pair is only meaningful relative to the dataset
file that produced it. `genuine_hard50.json` was built against mini_dev.json; a train slice drawn from
the rest of BIRD has to be indexed against dev.json. Running both in one experiment with two different
BIRD_DEV_FILE values is exactly the mix-up that once invalidated a whole batch of runs (Mini-Dev
indices silently applied to full dev.json, 0/50 question-text agreement). Since mini_dev is a subset of
dev, indexing EVERYTHING against dev.json removes the hazard: one BIRD_DEV_FILE for the whole pipeline.

Train mirrors the test set's per-DB distribution and is drawn only from questions outside mini_dev, so
train and test are disjoint by construction. No bare screening: the training set deliberately keeps the
questions the frozen solver already answers, because a harness evolved only on failures gets no signal
about the behaviour it must not break.

Run with BIRD_DEV_FILE=dev.json.
"""
import argparse
import collections
import json
import random
from pathlib import Path

from .. import bridge
from ..evolve import PKG_DIR

DEV_DIR = Path(os.environ.get("BENCH_DIR", "data/bird") + "/dev_20240627")


def strip_hint(text):
    return text.partition("\nHint:")[0].strip()


def dev_index(db_ids):
    """{db_id: {question_text: idx}} for the CURRENTLY configured dataset file."""
    out = {}
    for db_id in db_ids:
        out[db_id] = {strip_hint(q.question): i for i, q in enumerate(bridge.eval_questions(db_id))}
    return out


def write_slice(path, pairs, records):
    path.write_text(json.dumps({"cross": pairs}, indent=2), encoding="utf-8")
    path.with_name(f"{path.stem}_details.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-slice", required=True, help="existing genuine_hard50.json (mini_dev-indexed)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train-size", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = json.loads((DEV_DIR / "dev.json").read_text(encoding="utf-8"))
    mini = json.loads((DEV_DIR / "mini_dev.json").read_text(encoding="utf-8"))
    mini_ids = {d["question_id"] for d in mini}

    test_details = json.loads(
        Path(args.test_slice).with_name(f"{Path(args.test_slice).stem}_details.json").read_text(encoding="utf-8"))
    quota = collections.Counter(d["db_id"] for d in test_details)
    print(f"test per-DB quota (mirrored by train): {dict(quota)}")

    idx = dev_index(sorted(quota))

    # --- test, re-indexed against dev.json ---
    test_pairs, test_recs, missing = [], [], []
    for d in test_details:
        text = strip_hint(d["question"])
        i = idx[d["db_id"]].get(text)
        if i is None:
            missing.append(d["question_id"])
            continue
        test_pairs.append([d["db_id"], i])
        test_recs.append({"question_id": d["question_id"], "db_id": d["db_id"],
                          "question": text, "difficulty": d.get("difficulty")})
    if missing:
        raise SystemExit(f"{len(missing)} test questions not found in dev.json: {missing[:5]}")

    # --- train: mirror test on BOTH db_id and difficulty, drawn only from OUTSIDE mini_dev ---
    # BIRD's own difficulty label is dataset metadata, so matching on it costs nothing (no solver runs)
    # and does not bias the set toward questions the solver already fails on.
    cell_quota = collections.Counter((d["db_id"], d.get("difficulty")) for d in test_details)
    rng = random.Random(args.seed)
    train_pairs, train_recs, deviations = [], [], []
    for db_id, n in sorted(quota.items()):
        avail = collections.defaultdict(list)
        for d in dev:
            if d["db_id"] == db_id and d["question_id"] not in mini_ids:
                avail[d.get("difficulty")].append(d)
        picked, short = [], 0
        for (cdb, diff), want in sorted(cell_quota.items(), key=lambda kv: str(kv[0])):
            if cdb != db_id:
                continue
            have = avail[diff]
            take = min(want, len(have))
            chosen = rng.sample(have, take)
            picked += chosen
            for c in chosen:
                have.remove(c)
            if take < want:
                short += want - take
                deviations.append(f"{db_id}/{diff}: wanted {want}, only {len(have) + take} available")
        if short:                                   # backfill within the SAME db to keep the DB quota exact
            rest = [d for diff in avail for d in avail[diff]]
            if len(rest) < short:
                raise SystemExit(f"{db_id}: cannot backfill {short} questions")
            picked += rng.sample(rest, short)
        if len(picked) != n:
            raise SystemExit(f"{db_id}: picked {len(picked)} but quota is {n}")
        for d in picked:
            i = idx[db_id].get(strip_hint(d["question"]))
            if i is None:
                raise SystemExit(f"train question not found in dev index: {db_id} {d['question_id']}")
            train_pairs.append([db_id, i])
            train_recs.append({"question_id": d["question_id"], "db_id": db_id,
                               "question": strip_hint(d["question"]), "difficulty": d.get("difficulty")})

    train_ids = {r["question_id"] for r in train_recs}
    test_ids = {r["question_id"] for r in test_recs}
    assert not (train_ids & test_ids), "train/test overlap"
    assert not (train_ids & mini_ids), "train leaked into mini_dev"

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_slice(out / "train50.json", train_pairs, train_recs)
    write_slice(out / "test50.json", test_pairs, test_recs)

    print(f"train {len(train_pairs)}  {dict(collections.Counter(r['db_id'] for r in train_recs))}")
    print(f"      difficulty {dict(collections.Counter(r['difficulty'] for r in train_recs))}")
    print(f"test  {len(test_pairs)}  {dict(collections.Counter(r['db_id'] for r in test_recs))}")
    print(f"      difficulty {dict(collections.Counter(r['difficulty'] for r in test_recs))}")
    print(f"overlap {len(train_ids & test_ids)}  |  both slices indexed against dev.json")
    for d in deviations:                            # DB quota is exact; difficulty may be backfilled
        print(f"  [deviation] {d} -> backfilled within the same DB")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
