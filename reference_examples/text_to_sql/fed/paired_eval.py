"""Paired, per-round comparison of aggregation strategies on IDENTICAL client harnesses.

Comparing two aggregation methods by their end-to-end scores does not work: each method's global
diverges after the first round, so the clients it trains next are different harnesses, and the
run-to-run spread in client quality is larger than the effect being measured. Directly observed:
one run's clients averaged 65 on the training set and another's averaged 68, while the two final
globals differed by 2 — the aggregation method was the smaller term.

Within a single round the inputs ARE identical: every aggregation strategy sees the same K client
harnesses produced from the same broadcast global. So the comparison is made there, once per round,
and the run supplies as many paired observations as it has rounds.

What each strategy would have produced from those same clients:
  merge        — the global the run actually committed (already on disk)
  select-best  — the client with the highest training score
  broadcast    — the global that was handed out, i.e. doing nothing (floor)

Scores are held-out test scores; training scores only decide what select-best picks. Divergence
across rounds is expected and not a confound: each row is self-contained.
"""
import argparse
import json
from pathlib import Path

from .. import bridge
from ..evolve import PKG_DIR
from .evaluate import score


def measure(harness, items, golds, run_dir, timeout, workers):
    n, _rows = score(harness, items, golds, run_dir, timeout, workers)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True, help="a completed fed_loop run")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--solve-timeout", type=int, default=180)
    args = ap.parse_args()

    run_dir = PKG_DIR / "logs" / args.run_name
    log = [json.loads(l) for l in (run_dir / "fed_log.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    def load(slice_path):
        cross = json.loads(Path(slice_path).read_text(encoding="utf-8"))["cross"]
        items = [(db, bridge.eval_questions(db)[i]) for db, i in cross]
        golds = [bridge.gold_result(bridge.get_db(db), q.gold_sql) for db, q in items]
        return items, golds

    tr_items, tr_golds = load(args.train)
    te_items, te_golds = load(args.test)
    out_dir = run_dir / "paired"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = {}

    def on(split, harness):
        key = (split, harness)
        if key not in cache:
            items, golds = (tr_items, tr_golds) if split == "train" else (te_items, te_golds)
            cache[key] = measure(harness, items, golds, str(out_dir), args.solve_timeout, args.workers)
        return cache[key]

    rows = []
    for d in log:
        broadcast = d.get("broadcast") or d.get("global_after")
        clients = [c["harness"] for c in d["clients"]]
        merged = d.get("merged_candidate") if d.get("accepted") else None
        if d.get("aggregate") == "select-best":
            merged = None                                  # that run committed a client, not a merge
        actual = d.get("global_after")

        train_scores = {h: on("train", h) for h in dict.fromkeys(clients)}
        # select-best: highest training score, ties to the first client (same rule the loop uses)
        pick = max(train_scores, key=lambda h: train_scores[h])
        row = {
            "round": d["round"],
            "broadcast": broadcast,
            "broadcast_test": on("test", broadcast),
            "client_train": train_scores,
            "select_best_pick": pick,
            "select_best_train": train_scores[pick],
            "select_best_test": on("test", pick),
            "committed": actual,
            "committed_test": on("test", actual) if actual else None,
        }
        if merged and merged != actual:
            row["merged_candidate"] = merged
            row["merged_test"] = on("test", merged)
        rows.append(row)
        print(f"round {d['round']}: broadcast={row['broadcast_test']}  "
              f"select-best({pick[-14:]}, train {row['select_best_train']})={row['select_best_test']}  "
              f"committed({str(actual)[-16:]})={row['committed_test']}", flush=True)

    (out_dir / "paired.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    wins = sum(1 for r in rows if r["committed_test"] is not None
               and r["committed_test"] > r["select_best_test"])
    losses = sum(1 for r in rows if r["committed_test"] is not None
                 and r["committed_test"] < r["select_best_test"])
    print(f"\npaired over {len(rows)} rounds — committed beats select-best in {wins}, "
          f"loses in {losses}, ties in {len(rows) - wins - losses}")
    print(f"[saved] {out_dir}/paired.json")


if __name__ == "__main__":
    main()
