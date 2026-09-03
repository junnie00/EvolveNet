import os
"""Build a 100-question training slice from audited-clean questions, mirroring the test set's mix.

Replaces train50c, which had two problems for ablation work:

  * It was built with the first, over-strict auditor (rejected 60% of a set the original project had
    already hand-audited as clean, and its "dirty" and "clean" groups solved at the same rate, i.e. no
    discriminative power). Twelve good questions were removed from training as a result.
  * At 50 questions it saturates: `bare` solves 31 and evolved harnesses reach 40, so only 9 questions
    separate any two methods. Two aggregation strategies that differ by 4 points on held-out test both
    scored exactly 40 here — the training set could not tell them apart.

Doubling to 100 restores headroom (bare misses roughly 38) and doubles each client's shard from 10 to 20,
which also reduces how much a client can overfit its own shard.

Distribution mirrors test50 on (db_id, difficulty) at 2x. Cells short of clean questions are backfilled
within the same database and the deviation is printed. Indexed against dev.json like every other slice.
"""
import argparse
import collections
import json
import random
from pathlib import Path

from .. import bridge

DEV = Path(os.environ.get("BENCH_DIR", "data/bird") + "/dev_20240627")
AUDITS = ("train50_audit_v2.jsonl", "replacement_audit.jsonl", "extra_audit.jsonl", "expand_audit.jsonl",
           "expand2_audit.jsonl")


def strip_hint(t):
    return t.partition("\nHint:")[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    D = Path(args.split_dir)

    dev = json.loads((DEV / "dev.json").read_text(encoding="utf-8"))
    mini = {d["question_id"] for d in json.loads((DEV / "mini_dev.json").read_text(encoding="utf-8"))}
    test = json.loads((D / "test50_details.json").read_text(encoding="utf-8"))
    testq = {x["question_id"] for x in test}

    verdict = {}
    for name in AUDITS:
        p = D / name
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                e = json.loads(line)
                verdict[e["question_id"]] = e["verdict"]

    mult = args.size / len(test)
    quota = {k: round(v * mult) for k, v in
             collections.Counter((x["db_id"], x["difficulty"]) for x in test).items()}

    clean = collections.defaultdict(list)
    for d in dev:
        if d["question_id"] in mini or d["question_id"] in testq:
            continue
        if verdict.get(d["question_id"]) == "accept":
            clean[(d["db_id"], d.get("difficulty"))].append(d)
    for v in clean.values():
        v.sort(key=lambda d: d["question_id"])

    rng = random.Random(args.seed)
    picked, deviations = [], []
    for (db, diff), want in sorted(quota.items(), key=lambda kv: str(kv[0])):
        avail = clean[(db, diff)]
        take = min(want, len(avail))
        chosen = rng.sample(avail, take)
        for c in chosen:
            avail.remove(c)
        picked += chosen
        if take < want:                        # backfill within the same DB; DB mix outranks difficulty
            short = want - take
            rest = [d for (db2, _), v in clean.items() if db2 == db for d in v]
            if len(rest) < short:
                raise SystemExit(f"{db}/{diff}: need {short} more but only {len(rest)} left in {db}")
            extra = rng.sample(rest, short)
            for e in extra:
                clean[(e["db_id"], e.get("difficulty"))].remove(e)
            picked += extra
            deviations.append(f"{db}/{diff}: wanted {want}, only {take} clean available")

    if len({d["question_id"] for d in picked}) != len(picked):
        raise SystemExit("duplicate question in slice")

    idx = {}
    for d in picked:
        if d["db_id"] not in idx:
            idx[d["db_id"]] = {strip_hint(q.question): i
                               for i, q in enumerate(bridge.eval_questions(d["db_id"]))}
    pairs, recs = [], []
    for d in picked:
        i = idx[d["db_id"]].get(strip_hint(d["question"]))
        if i is None:
            raise SystemExit(f"qid {d['question_id']} not found in dev index")
        pairs.append([d["db_id"], i])
        recs.append({"question_id": d["question_id"], "db_id": d["db_id"],
                     "question": strip_hint(d["question"]), "difficulty": d.get("difficulty")})

    name = f"train{len(recs)}"
    (D / f"{name}.json").write_text(json.dumps({"cross": pairs}, indent=2), encoding="utf-8")
    (D / f"{name}_details.json").write_text(json.dumps(recs, indent=2), encoding="utf-8")

    print(f"built {len(recs)} questions")
    print(f"  db mix   train {dict(collections.Counter(r['db_id'] for r in recs))}")
    print(f"           test  {dict(collections.Counter(r['db_id'] for r in test))}")
    print(f"  difficulty train {dict(collections.Counter(r['difficulty'] for r in recs))}")
    print(f"             test  {dict(collections.Counter(r['difficulty'] for r in test))}")
    print(f"  overlap with test: {len({r['question_id'] for r in recs} & testq)}")
    print(f"  all audited clean: {all(verdict.get(r['question_id']) == 'accept' for r in recs)}")
    for d in deviations:
        print(f"  [deviation] {d}")
    print(f"[saved] {D}/{name}.json")


if __name__ == "__main__":
    main()
