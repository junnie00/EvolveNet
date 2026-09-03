"""Run a merge VARIANT on the client snapshots of a completed run, then score every variant's global
against select-best on held-out test — all on the SAME clients, per round.

This is how aggregation variants are compared without confounding them with client quality. A completed
fed_loop run recorded, for each round, the broadcast global and the K client harnesses it produced. Those
client harnesses are a fixed snapshot on disk. Here we replay ONLY the aggregation step: for each round we
hand the same snapshot to each variant's merger and to the select-best rule, and score the resulting
globals on test. Divergence across rounds is real (each round's clients came from that run's actual
trajectory) but every variant sees identical inputs within a round, so per-round differences are
attributable to the aggregation rule alone.

Note the round>0 caveat: the clients on disk were evolved from the ORIGINAL run's global, i.e. from
holistic-merge's trajectory. So this measures "given holistic's clients at round t, which aggregator makes
the best global" — a fair per-round comparison, not an independent end-to-end run of each variant. A true
end-to-end variant run is a separate, more expensive experiment.
"""
import argparse
import json
from pathlib import Path

from .. import bridge
from ..evolve import AGENTS_DIR, PKG_DIR
from . import merge as M
from .evaluate import score
from .fed_loop import client_gains


def test_score(harness, items, golds, run_dir, timeout, workers, cache):
    if harness not in cache:
        cache[harness] = score(harness, items, golds, run_dir, timeout, workers)[0]
    return cache[harness]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True, help="completed run whose client snapshots we reuse")
    ap.add_argument("--variants", default="holistic,quorum,ties")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--merge-timeout", type=int, default=2700)
    ap.add_argument("--solve-timeout", type=int, default=180)
    ap.add_argument("--budget-lines", type=int, default=150)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    variants = args.variants.split(",")
    run_dir = PKG_DIR / "logs" / args.run_name
    out_dir = run_dir / "variant_merge"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = [json.loads(l) for l in (run_dir / "fed_log.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    def load(p):
        cross = json.loads(Path(p).read_text(encoding="utf-8"))["cross"]
        items = [(db, bridge.eval_questions(db)[i]) for db, i in cross]
        return items, [bridge.gold_result(bridge.get_db(db), q.gold_sql) for db, q in items]

    tr_items, tr_golds = load(args.train)
    te_items, te_golds = load(args.test)
    tcache = {}

    def train_score(h):
        return score(h, tr_items, tr_golds, str(out_dir), args.solve_timeout, args.workers)[0]

    rows = []
    for d in log:
        rnd = d["round"]
        broadcast = d.get("broadcast") or d.get("global_after")
        clients = [{"client": c["client"], "harness": c["harness"], "trace_dir": c["trace_dir"],
                    "shard": c["shard"], "run": c.get("run")} for c in d["clients"]]
        gains = client_gains(d["clients"])

        result = {"round": rnd, "broadcast": broadcast,
                  "broadcast_test": test_score(broadcast, te_items, te_golds, str(out_dir),
                                               args.solve_timeout, args.workers, tcache)}

        # select-best on this snapshot
        cs = {c["harness"]: train_score(c["harness"]) for c in clients}
        pick = max(cs, key=lambda h: cs[h])
        result["select_best"] = {"pick": pick, "train": cs[pick],
                                 "test": test_score(pick, te_items, te_golds, str(out_dir),
                                                    args.solve_timeout, args.workers, tcache)}

        # each merge variant on the SAME snapshot
        for v in variants:
            name = f"vm_{args.run_name}_{v}_r{rnd}"
            wrote = M.merge(broadcast, clients, name, str(out_dir), args.model,
                            args.merge_timeout, args.budget_lines, gains=gains, variant=v)
            db0 = bridge.get_db(json.loads(Path(clients[0]["shard"]).read_text())["cross"][0][0])
            if wrote and M.loadable(name, db0):
                result[v] = {"train": train_score(name),
                             "test": test_score(name, te_items, te_golds, str(out_dir),
                                                args.solve_timeout, args.workers, tcache),
                             "lines": len((AGENTS_DIR / f"{name}.py").read_text().splitlines())}
            else:
                result[v] = {"error": "merge produced no loadable file"}
        rows.append(result)
        line = f"round {rnd}: broadcast={result['broadcast_test']}  select-best={result['select_best']['test']}"
        for v in variants:
            line += f"  {v}={result[v].get('test', 'ERR')}"
        print(line, flush=True)

    (out_dir / "variant_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\n=== test-score summary (per round) ===")
    hdr = f"{'round':>5} {'broadcast':>10} {'select-best':>12}" + "".join(f"{v:>10}" for v in variants)
    print(hdr)
    for r in rows:
        print(f"{r['round']:>5} {r['broadcast_test']:>10} {r['select_best']['test']:>12}"
              + "".join(f"{r[v].get('test', 'ERR'):>10}" for v in variants))
    print(f"[saved] {out_dir}/variant_results.json")


if __name__ == "__main__":
    main()
