"""Federated harness evolution, domain-blind: broadcast -> parallel local evolution -> aggregate -> repeat.

One federated round:
  1. BROADCAST  the current global harness name to every client
  2. LOCAL      each client runs the domain's optimize on its own shard (parallel), --supervised,
                whole shard as one batch so --max-rounds is the local epoch count E
  3. AGGREGATE  merge variant (or select-best) folds the K results into the next global
  4. VERIFY     supervised accept gate — PER-QUESTION, not totals: totals move by the measurement noise
                (±2-3 per 50 measured on SQL), while most questions agree between any two harnesses, so
                `broke <= fixed` on flipped questions is the robust criterion. A rejected merge gets ONE
                retry carrying the exact counterexamples (which questions it broke, old vs new artifact);
                after that the round rolls back to the broadcast global.

Rollback goes to the PREVIOUS GLOBAL, never to "the best client": picking a best client would need held-out
accuracy we refuse to touch during training, and it breaks the invariant every client starts one round from
one shared base. There is deliberately NO automatic post-merge repair pass — tried on SQL, it turned 4
regressions into 67; the accept gate plus the retry-with-counterexamples is the working design.

    PYTHONPATH=. python -m fedkit.fed_loop --domain ds1000 --shards ds1000/logs/shards_x \
        --train ds1000/slices/train.json --run-name dsfed1 --rounds 3 --local-rounds 3 --group 1
"""
import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .adapters import get_adapter, child_env
from . import merge as M


def run_eval(adapter, harness, slice_path, out_path, solve_timeout):
    """Score `harness` on `slice_path` in an isolated child (PYTHONHASHSEED pinned). -> rows or None."""
    cmd = [sys.executable, "-u", "-m", "fedkit.evaluate", "--domain", adapter.name,
           "--slice", str(slice_path), "--harness", harness, "--out", str(out_path),
           "--solve-timeout", str(solve_timeout)]
    r = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent), env=child_env(),
                       capture_output=True, text=True)
    if r.returncode != 0 or not Path(out_path).exists():
        print(f"   [eval] FAILED for {harness}: {(r.stderr or '')[-300:]}", flush=True)
        return None
    return json.loads(Path(out_path).read_text(encoding="utf-8"))


def compare(new_rows, old_rows):
    """(fixed_ids, broke_ids) by per-question flips — the noise-robust comparator (see module docstring)."""
    fixed = [n["id"] for n, o in zip(new_rows, old_rows) if n["correct"] and not o["correct"]]
    broke = [n["id"] for n, o in zip(new_rows, old_rows) if o["correct"] and not n["correct"]]
    return fixed, broke


def run_client(adapter, k, shard, global_name, run_name, args):
    client_run = f"{run_name}_r{args._round}_c{k}"
    cmd = adapter.client_cmd(shard, global_name, client_run, k, args)
    log_path = adapter.logs_dir / f"{client_run}.log"
    print(f"   [client {k}] shard={Path(shard).name} from {global_name} -> {client_run}", flush=True)
    started = time.monotonic()
    with open(log_path, "w", encoding="utf-8") as lf:
        proc = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent), env=child_env(),
                              stdout=lf, stderr=subprocess.STDOUT, text=True)
    dur = round(time.monotonic() - started, 1)
    res_path = adapter.logs_dir / client_run / "result.json"
    if proc.returncode != 0 or not res_path.exists():
        print(f"   [client {k}] FAILED rc={proc.returncode} after {dur}s — see {log_path}", flush=True)
        return None
    res = json.loads(res_path.read_text(encoding="utf-8"))
    print(f"   [client {k}] done in {dur}s -> {res['final_harness']}", flush=True)
    return {"client": k, "harness": res["final_harness"], "shard": str(shard),
            "trace_dir": str(adapter.logs_dir / client_run / "traces" / "b0"),
            "run": client_run, "seconds": dur}


def client_gains(adapter, results):
    """Measured per-client capability changes, from each client's shard_verdicts file. Costs nothing —
    the client already computed them — and names exactly what the merger must preserve."""
    blocks = []
    for r in results:
        vpath = adapter.logs_dir / r["run"] / "shard_verdicts_b0.json"
        if not vpath.exists():
            continue
        v = json.loads(vpath.read_text(encoding="utf-8"))
        fixed = [q for q in v["questions"] if q["delta"] == "fixed"]
        broke = [q for q in v["questions"] if q["delta"] == "broke"]
        if not fixed and not broke:
            continue
        lines = [f"- client {r['client']} (`{r['harness']}`):"]
        lines += [f"    NEWLY SOLVED  [{q.get('qid') or q.get('pid')}] {q['question'][:120]}" for q in fixed]
        lines += [f"    REGRESSED     [{q.get('qid') or q.get('pid')}] {q['question'][:120]}" for q in broke]
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--shards", required=True, help="directory holding shard<k> slice files")
    ap.add_argument("--train", required=True, help="the FULL training slice (union of shards)")
    ap.add_argument("--val", default=None,
                    help="slice for the ACCEPTANCE GATE. Must be REPRESENTATIVE of the workload, not "
                         "another hard slice: a gate that only sees items the base already fails can "
                         "measure what a merge fixes but not what it BREAKS. Measured on SQL: with the "
                         "gate on the (hard) training shards, three merges were accepted that a "
                         "representative gate rejects, costing 12 broken vs 9 fixed on held-out. "
                         "Defaults to --train, which is the wrong distribution unless --train is itself "
                         "representative.")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--local-rounds", type=int, default=3, help="E: generate rounds inside each client")
    ap.add_argument("--group", type=int, default=1, help="G: proposers per local round")
    ap.add_argument("--initial-harness", default="bare")
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--aggregate", default="merge",
                    choices=["merge", "select-best", "route"],
                    help="'merge' folds clients into one global; 'select-best' keeps the client that "
                         "scores highest on the training slice — the baseline merge must beat; 'route' "
                         "is delegation composition (the method SQL settled on, RESULTS.md §12): each "
                         "home domain dispatches to its own expert verbatim, everything else to the "
                         "gate-best client — zero merger sessions, zero rewrite loss")
    ap.add_argument("--specialist", action="store_true",
                    help="clients evolve as EXPERTS of their own slice — passed through to the "
                         "domain optimizer; prerequisite for route aggregation to gain")
    ap.add_argument("--warm-start", action="store_true",
                    help="round>0: each client starts from ITS OWN previous harness instead of the "
                         "committed global (behaviourally identical on its shard under route, but the "
                         "proposer sees an editable full program, not an opaque dispatcher)")
    ap.add_argument("--merge-variant", default="holistic", choices=sorted(M.ADOPTION_RULES))
    ap.add_argument("--propose-timeout", type=int, default=2700)
    ap.add_argument("--merge-timeout", type=int, default=2700)
    ap.add_argument("--solve-timeout", type=int, default=600)
    ap.add_argument("--budget-lines", type=int, default=150)
    ap.add_argument("--max-parallel-clients", type=int, default=5)
    args = ap.parse_args()

    ad = get_adapter(args.domain)
    shard_dir = Path(args.shards).resolve()
    shards = sorted(shard_dir.glob("shard*.json"), key=lambda p: p.stem)
    if not shards:
        raise SystemExit(f"no shard*.json in {shard_dir}")
    train_slice = Path(args.train).resolve()
    # Clients evolve on their (hard) shards; the gate measures on --val. The two distributions differ ON
    # PURPOSE: hard shards give the proposer room to learn, a representative gate notices when a merge
    # trades that learning for damage elsewhere.
    gate_slice = Path(args.val).resolve() if args.val else train_slice

    run_dir = ad.logs_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args) | {"shards": [str(s) for s in shards]},
                                                    indent=2, default=str), encoding="utf-8")
    hist = open(run_dir / "fed_log.jsonl", "w")
    sample_item = ad.load_items(ad.slice_ids(str(shards[0]))[:1])[0]

    global_name = args.initial_harness
    global_rows = run_eval(ad, global_name, gate_slice, run_dir / f"measure_{global_name}.json",
                           args.solve_timeout)
    if global_rows is None:
        raise SystemExit("cannot score the initial harness on the training slice")
    print(f"[fed:{args.domain}] {len(shards)} clients x {args.rounds} rounds, E={args.local_rounds} "
          f"G={args.group}, aggregate={args.aggregate}/{args.merge_variant}", flush=True)
    print(f"[fed] {global_name} solves {sum(r['correct'] for r in global_rows)}/{len(global_rows)} "
          f"of the gate slice", flush=True)
    lineage = [global_name]

    prev_harness = {}                     # client k -> harness it produced last round (for --warm-start)
    for t in range(args.rounds):
        args._round = t
        print(f"\n=== round {t}  broadcast global = {global_name} ===", flush=True)
        started = time.monotonic()

        def start_for(k):
            if args.warm_start and k in prev_harness:
                return prev_harness[k]
            return global_name

        with ThreadPoolExecutor(max_workers=args.max_parallel_clients) as pool:
            futs = [pool.submit(run_client, ad, k, s, start_for(k), args.run_name, args)
                    for k, s in enumerate(shards)]
            results = [f.result() for f in futs]
        results = [r for r in results if r]
        if not results:
            print(f"[fed] round {t}: every client failed — stopping", flush=True)
            break
        for r in results:
            prev_harness[r["client"]] = r["harness"]

        accepted, reason, merged_rows = False, "", None
        out_name = f"fedglobal_{args.run_name}_r{t + 1}"

        if args.aggregate == "route":
            # Delegation composition (see reference_examples/text_to_sql/fed/fed_loop.py for the measured
            # rationale). Gate rows are ASSEMBLED from per-client rows — zero extra solves.
            crows = {}
            for r in results:
                rows = run_eval(ad, r["harness"], gate_slice,
                                run_dir / f"measure_{r['harness']}.json", args.solve_timeout)
                if rows is not None:
                    crows[r["harness"]] = rows
            def dom_score(rows, dom):
                return sum(x["correct"] for x in rows if x.get("domain") == dom)
            pool_rows = dict(crows); pool_rows[global_name] = global_rows
            default = max(pool_rows, key=lambda h: sum(x["correct"] for x in pool_rows[h]))
            home = {}
            for r in results:
                if r["harness"] not in crows:
                    continue
                doms = {ad.item_domain(p) for p in ad.load_items(ad.slice_ids(r["shard"]))}
                for dom in doms:
                    cand = r["harness"]
                    if dom_score(crows[cand], dom) >= dom_score(pool_rows[default], dom):
                        if dom not in home or dom_score(crows[cand], dom) > dom_score(crows.get(home[dom], pool_rows[default]), dom):
                            home[dom] = cand
            route_rows = [pool_rows[home.get(row.get("domain"), default)][i]
                          for i, row in enumerate(global_rows)]
            fixed, broke = compare(route_rows, global_rows)
            score = sum(x["correct"] for x in route_rows)
            rname = f"fedroute_{args.run_name}_r{t + 1}"
            ok = bool(len(broke) <= len(fixed) and (fixed or broke))
            print(f"   [route] gate: {', '.join(f'{h}={sum(x['correct'] for x in v)}' for h, v in pool_rows.items())}", flush=True)
            print(f"   [route] home={home} default={default}", flush=True)
            print(f"   [route] {'ACCEPTED' if ok else 'REJECTED'} {rname} (fixed {len(fixed)}, "
                  f"broke {len(broke)}; {score}/{len(global_rows)} vs "
                  f"{sum(x['correct'] for x in global_rows)})", flush=True)
            hist.write(json.dumps({"round": t, "broadcast": global_name, "aggregate": "route",
                                   "home": home, "default": default, "accepted": ok,
                                   "gate_scores": {h: sum(x["correct"] for x in v) for h, v in pool_rows.items()},
                                   "fixed": len(fixed), "broke": len(broke), "gate_score": score,
                                   "global_after": rname if ok else global_name, "clients": results,
                                   "seconds": round(time.monotonic() - started, 1)}) + "\n")
            hist.flush()
            if ok:
                (ad.agents_dir / f"{rname}.py").write_text(ad.route_source(home, default),
                                                           encoding="utf-8")
                global_name, global_rows = rname, route_rows
                lineage.append(global_name)
            continue

        if args.aggregate == "select-best":
            pool_names = list(dict.fromkeys([r["harness"] for r in results] + [global_name]))
            scored = {}
            for h in pool_names:
                rows = (global_rows if h == global_name else
                        run_eval(ad, h, gate_slice, run_dir / f"measure_{h}.json", args.solve_timeout))
                if rows is not None:
                    scored[h] = rows
            totals = {h: sum(r["correct"] for r in rows) for h, rows in scored.items()}
            best = max(totals, key=lambda h: totals[h])
            print(f"   [select-best] {', '.join(f'{h}={n}' for h, n in totals.items())} -> {best}",
                  flush=True)
            hist.write(json.dumps({"round": t, "broadcast": global_name, "aggregate": "select-best",
                                   "scores": {h: sum(r["correct"] for r in rows) for h, rows in scored.items()},
                                   "picked": best, "global_after": best, "clients": results,
                                   "seconds": round(time.monotonic() - started, 1)}) + "\n")
            hist.flush()
            global_name, global_rows = best, scored[best]
            lineage.append(global_name)
            continue

        gains = client_gains(ad, results)
        feedback = ""
        fixed = broke = []
        for attempt in range(2):
            name = out_name if attempt == 0 else f"{out_name}_retry"
            print(f"   [merge:{args.merge_variant}] {len(results)} clients -> {name}"
                  f"{' (retry with failure evidence)' if attempt else ''}", flush=True)
            wrote = M.merge(ad, global_name, results, name, str(run_dir), args.model,
                            args.merge_timeout, args.budget_lines, gains=gains,
                            feedback=feedback, variant=args.merge_variant)
            if not wrote or not M.loadable(ad, name, sample_item):
                reason, feedback = "merger produced no loadable file", \
                    "Your previous attempt did not produce a loadable file. Produce one that imports cleanly."
                continue
            rows = run_eval(ad, name, gate_slice, run_dir / f"measure_{name}.json", args.solve_timeout)
            if rows is None:
                reason = "evaluation of merged harness failed"
                continue
            fixed, broke = compare(rows, global_rows)
            reason = (f"fixed {len(fixed)}, broke {len(broke)} "
                      f"({sum(r['correct'] for r in rows)}/{len(rows)} vs broadcast "
                      f"{sum(r['correct'] for r in global_rows)})")
            if len(broke) <= len(fixed):
                accepted, merged_rows, out_name = True, rows, name
                break
            print(f"   [merge] attempt {attempt} rejected — {reason}", flush=True)
            by_id = {r["id"]: r for r in rows}
            old_by_id = {r["id"]: r for r in global_rows}
            detail = "\n".join(
                f"  - problem {i}:\n      the broadcast global answered this CORRECTLY with:\n"
                f"        {old_by_id[i]['artifact'][:250]}\n"
                f"      your merge now produces this, which is WRONG:\n        {by_id[i]['artifact'][:250]}"
                for i in broke[:8])
            feedback = (f"YOUR PREVIOUS MERGE ATTEMPT WAS REJECTED. It fixed {len(fixed)} problems but "
                        f"BROKE {len(broke)} the broadcast global already answered correctly:\n{detail}\n\n"
                        f"Work out what you changed that caused these regressions and do not repeat it. "
                        f"The most likely cause is editing behaviour the base already had right — prompt "
                        f"rules apply to EVERY problem, unlike a code path that only fires on its "
                        f"condition. Prefer adopting fewer mechanisms over breaking working ones.")

        agents = ad.agents_dir
        base_lines = len((agents / f"{global_name}.py").read_text(encoding="utf-8").splitlines())
        new_lines = (len((agents / f"{out_name}.py").read_text(encoding="utf-8").splitlines())
                     if (agents / f"{out_name}.py").exists() else None)
        if accepted:
            print(f"   [merge] ACCEPTED {out_name} ({reason}); {base_lines} -> {new_lines} lines", flush=True)
            global_name, global_rows = out_name, merged_rows
        else:
            print(f"   [merge] REJECTED — {reason}; rolling back to {global_name}", flush=True)
        lineage.append(global_name)
        hist.write(json.dumps({"round": t, "broadcast": lineage[-2], "merged_candidate": out_name,
                               "aggregate": f"merge/{args.merge_variant}", "accepted": accepted,
                               "reason": reason, "fixed": fixed, "broke": broke,
                               "base_lines": base_lines, "merged_lines": new_lines,
                               "global_after": global_name, "clients": results,
                               "seconds": round(time.monotonic() - started, 1)}) + "\n")
        hist.flush()

    hist.close()
    (run_dir / "final.json").write_text(json.dumps({
        "final_global": global_name, "lineage": lineage, "rounds_run": len(lineage) - 1,
        "train_score": sum(r["correct"] for r in global_rows), "train_total": len(global_rows),
    }, indent=2), encoding="utf-8")
    print(f"\n[fed] final global = {global_name}  "
          f"(train {sum(r['correct'] for r in global_rows)}/{len(global_rows)})")
    print(f"[fed] lineage: {' -> '.join(lineage)}")
    print(f"[saved] {run_dir}/final.json")


if __name__ == "__main__":
    main()
