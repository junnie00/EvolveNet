"""Incremental aggregation: absorb ONE client's delta at a time, gating each step.

Why: absorbing all K clients in a single merge is measurably net-negative on a representative slice
(+9 for a single client's own edits vs -3 for the 5-way merge), while each client's own changes are
net-positive. That points at composition, not at any one client — so absorb them one at a time and
keep only the steps that survive the gate.

Each step: merge(current_global, [client_k]) -> measure on the gate slice -> keep if it breaks no more
than it fixes, else discard and move on with the previous global. The result is a chain that never
regresses past its own gate, and a per-client record of which contributions actually helped.

Reuses an existing run's client snapshots — no client evolution is re-run.

    PYTHONPATH=. python -m reference_examples.text_to_sql.fed.incremental \
        --run-name abl_merge --round 0 --val logs/fed_split_v1/val100.json --out-name inc_r0
"""
import argparse
import json
from pathlib import Path

from ..evolve import AGENTS_DIR, PKG_DIR
from . import merge as M
from .fed_loop import compare, load_slice, measure, client_gains


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True, help="completed run whose client snapshots we reuse")
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out-name", required=True)
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--merge-timeout", type=int, default=2700)
    ap.add_argument("--solve-timeout", type=int, default=180)
    ap.add_argument("--anchor", choices=["broadcast", "best"], default="broadcast",
                    help="where the chain starts. 'best' anchors on the client scoring highest on the "
                         "gate, then absorbs the others' deltas — this makes the result >= select-best "
                         "BY CONSTRUCTION (the anchor IS select-best's answer) and lets aggregation only "
                         "add. Starting from the broadcast global instead has to climb to that level one "
                         "gated step at a time, and measurably fails to when one client is already far "
                         "ahead of the base.")
    ap.add_argument("--budget-lines", type=int, default=60,
                    help="per-STEP budget. Lower than the all-at-once budget on purpose: each client "
                         "gets its own increment, so the same total is spread over K gated steps.")
    ap.add_argument("--variant", default="holistic", choices=sorted(M.ADOPTION_RULES),
                    help="adoption rule used at every absorption step (see merge.ADOPTION_RULES)")
    args = ap.parse_args()

    run_dir = PKG_DIR / "logs" / args.run_name
    out_dir = PKG_DIR / "logs" / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log = [json.loads(l) for l in (run_dir / "fed_log.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rec = next(d for d in log if d["round"] == args.round)
    broadcast = rec.get("broadcast") or rec.get("global_after")
    clients = [{"client": c["client"], "harness": c["harness"], "trace_dir": c["trace_dir"],
                "shard": c["shard"], "run": c.get("run")} for c in rec["clients"]]

    items, golds = load_slice(Path(args.val).resolve())
    cur = broadcast
    cur_rows = measure(cur, items, golds, str(out_dir), args.solve_timeout)
    print(f"[inc] broadcast {cur} scores {sum(r['correct'] for r in cur_rows)}/{len(items)} on the gate",
          flush=True)

    anchor_client = None
    if args.anchor == "best":
        # Score every client on the gate and start from the strongest. This is exactly what select-best
        # would commit, so the chain can only improve on that baseline from here.
        scored = {}
        for c in clients:
            rows = measure(c["harness"], items, golds, str(out_dir), args.solve_timeout)
            scored[c["harness"]] = (sum(r["correct"] for r in rows), rows, c)
        best_h = max(scored, key=lambda h: scored[h][0])
        if scored[best_h][0] > sum(r["correct"] for r in cur_rows):
            cur, cur_rows, anchor_client = best_h, scored[best_h][1], scored[best_h][2]["client"]
        print("[inc] gate scores: "
              + ", ".join(f"c{v[2]['client']}={v[0]}" for v in scored.values()), flush=True)
        print(f"[inc] ANCHOR = {cur} ({sum(r['correct'] for r in cur_rows)}/{len(items)}) "
              f"— this is select-best's answer; aggregation may only add from here", flush=True)
        clients = [c for c in clients if c["client"] != anchor_client]

    # Absorb the biggest local contributor first: if order matters, the informative ordering is the one
    # that gives the strongest deltas the most room before the budget tightens.
    def gain(c):
        p = PKG_DIR / "logs" / c["run"] / "shard_verdicts_b0.json"
        if not p.exists():
            return 0
        v = json.loads(p.read_text(encoding="utf-8"))["questions"]
        return sum(q["delta"] == "fixed" for q in v) - sum(q["delta"] == "broke" for q in v)
    clients.sort(key=gain, reverse=True)
    print(f"[inc] absorption order by local net gain: "
          f"{[(c['client'], gain(c)) for c in clients]}", flush=True)

    steps = []
    for i, c in enumerate(clients):
        if c["harness"] == broadcast:
            # this client returned the broadcast unchanged — there is no delta to absorb
            print(f"[inc] step {i} (client {c['client']}): harness IS the broadcast, nothing to absorb "
                  f"— skipped", flush=True)
            steps.append({"step": i, "client": c["client"], "kept": False, "reason": "no delta"})
            continue
        name = f"{args.out_name}_s{i}c{c['client']}"
        g = client_gains([c])
        wrote = M.merge(cur, [c], name, str(out_dir), args.model, args.merge_timeout,
                        args.budget_lines, gains=g, variant=args.variant,
                        feedback=("This is an INCREMENTAL aggregation: you are folding in ONE client's "
                                  "delta on top of a global that already absorbed earlier clients. Keep "
                                  "the change minimal and additive — everything already in the base was "
                                  "accepted by a measured gate, so do not restructure or 'improve' it."))
        if not wrote:
            print(f"[inc] step {i} (client {c['client']}): merger produced no file — skipped", flush=True)
            steps.append({"step": i, "client": c["client"], "kept": False, "reason": "no file"})
            continue
        rows = measure(name, items, golds, str(out_dir), args.solve_timeout)
        fixed, broke = compare(rows, cur_rows)
        # STRICT improvement, not `broke <= fixed`. Anchored on select-best's answer, a step that
        # merely ties (measured: +6/-6) buys no accuracy while adding code and enlarging the surface for
        # later conflicts. Only steps that actually move the number are worth their cost.
        keep = len(fixed) > len(broke)
        print(f"[inc] step {i} (client {c['client']}): fixed {len(fixed)}, broke {len(broke)} "
              f"({sum(r['correct'] for r in rows)}/{len(items)}) -> {'KEEP' if keep else 'DISCARD'}",
              flush=True)
        steps.append({"step": i, "client": c["client"], "candidate": name, "kept": keep,
                      "fixed": len(fixed), "broke": len(broke),
                      "score": sum(r["correct"] for r in rows)})
        if keep:
            cur, cur_rows = name, rows

    out = {"broadcast": broadcast, "anchor_client": anchor_client, "final": cur, "round": args.round,
           "final_score": sum(r["correct"] for r in cur_rows), "n": len(items), "steps": steps}
    (out_dir / "incremental.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[inc] {broadcast} -> {cur}   gate {out['final_score']}/{len(items)}   "
          f"kept {sum(s['kept'] for s in steps)}/{len(steps)} clients")
    print(f"[saved] {out_dir}/incremental.json")


if __name__ == "__main__":
    main()
