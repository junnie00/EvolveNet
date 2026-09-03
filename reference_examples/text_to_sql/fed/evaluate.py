"""Score frozen harnesses on a held-out slice with gold. Measurement only — nothing here feeds back.

This is the train-then-test half: fed_loop.py never sees gold and never touches this slice; here the
harness is already frozen and simply run. Pass several --harness values to get the comparison in one
pass (e.g. bare, the final global, a single client's harness), all on identical questions.

Each solve runs in its own killable process (generated harness code can hang), so a bad harness costs
one timeout instead of the whole evaluation.
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import bridge
from ..evolve import PKG_DIR
from ..optimize import isolated_solve


def score(harness, items, golds, run_dir, timeout, workers):
    """-> (n_correct, rows). One row per question; gold is compared here and nowhere else."""
    def one(j):
        db_id, q = items[j]
        out = isolated_solve(run_dir, harness, db_id, q.question, timeout, f"eval_{harness}_{j}")
        res = out.get("result") or {}
        ok = bool(res.get("ok")) and bridge.is_correct(res, golds[j])
        # See fed_loop.measure(): join these files on `q`, never on `i` — writers differ in row order.
        return {"i": j, "db_id": db_id, "q": q.question.partition("\nHint:")[0].strip()[:200],
                "status": out.get("status"), "exec_ok": res.get("ok"),
                "correct": ok, "sql": " ".join(str(out.get("sql") or "").split())[:300],
                "error": (res.get("error") or "")[:200]}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = sorted(pool.map(one, range(len(items))), key=lambda r: r["i"])
    return sum(r["correct"] for r in rows), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", required=True, help="cross-set json {cross:[[db_id,idx],...]}")
    ap.add_argument("--harness", action="append", required=True, help="repeatable")
    ap.add_argument("--run-name", default="eval1")
    ap.add_argument("--solve-timeout", type=int, default=180)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    slice_path = Path(args.slice).resolve()
    cross = json.loads(slice_path.read_text(encoding="utf-8"))["cross"]
    items = [(db_id, bridge.eval_questions(db_id)[i]) for db_id, i in cross]

    details_path = slice_path.with_name(f"{slice_path.stem}_details.json")
    if details_path.exists():                       # same identity guard optimize.py applies to a cross-set
        expected = json.loads(details_path.read_text(encoding="utf-8"))
        if len(expected) != len(items):
            raise SystemExit(f"{len(items)} items but {len(expected)} detail records")
        for pos, ((db_id, q), d) in enumerate(zip(items, expected)):
            if db_id != d["db_id"] or q.question.partition("\nHint:")[0].strip() != d["question"].strip():
                raise SystemExit(
                    f"dataset identity mismatch at {pos}: loaded [{db_id}] "
                    f"{q.question.partition(chr(10) + 'Hint:')[0].strip()[:80]!r}, expected "
                    f"[{d['db_id']}] {d['question'][:80]!r}. Set BIRD_DEV_FILE to the file this slice was built against.")
        print(f"[identity] {len(items)}/{len(items)} questions match {details_path.name}")

    golds = [bridge.gold_result(bridge.get_db(db_id), q.gold_sql) for db_id, q in items]
    run_dir = PKG_DIR / "logs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for h in args.harness:
        started = time.monotonic()
        n, rows = score(h, items, golds, str(run_dir), args.solve_timeout, args.workers)
        dur = round(time.monotonic() - started, 1)
        (run_dir / f"rows_{h}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        by_db = {}
        for r in rows:
            d = by_db.setdefault(r["db_id"], [0, 0])
            d[0] += r["correct"]
            d[1] += 1
        summary[h] = {"correct": n, "total": len(items), "seconds": dur,
                      "by_db": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_db.items())}}
        print(f"{h}: {n}/{len(items)}  ({dur}s)  {summary[h]['by_db']}", flush=True)

    out = {"slice": str(slice_path), "n": len(items), "harnesses": summary}
    (run_dir / "result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[saved] {run_dir}/result.json")


if __name__ == "__main__":
    main()
