"""TEST-TIME harness optimization for DS-1000 (data-science coding domain) — the live loop (mirror of
livecodebench/lcb_optimize.py, agentic batch generate->judge).

ONE general DS1000Harness (arbitrary Python) starts from `bare` (thinking OFF) and ACCUMULATES across
batches. Per batch: OBSERVE (run each candidate, write a full trace = problem + every coder call (+thinking
choice) + final code + SELF-CHECK execution + back-translation) -> GENERATE (G agentic generators deep-read
all traces + write an improved harness) -> PICK (one agentic judge picks the harness whose solutions look
most likely correct from LABEL-FREE evidence) -> SCORE that batch with the chosen harness on the GOLD
code_context test (MEASUREMENT ONLY). The label-free signal is the self-check; the gold test never enters the
loop. Bare baseline cached per-pid in logs/bare_cache.json.

    cd <repo-root> && ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic ANTHROPIC_AUTH_TOKEN=... \
      TTHO_PROPOSER_MODEL=deepseek-v4-flash OPENAI_API_KEY=... PYTHONPATH=. \
      python -u -m ds1000.ds1000_optimize --pilot ds1000/logs/pilot.json \
      --batch-size 5 --group 2 --max-rounds 3 --run-name pilot
"""
import argparse
import json
import time
import os
from concurrent.futures import ThreadPoolExecutor

from . import ds1000_bridge as bridge
from . import ds1000_proposer as P
from .ds1000_common import load_harness, PKG_DIR, AGENTS_DIR
from audit_harness import audit_file

_SOLVE_POOL = ThreadPoolExecutor(max_workers=32)


def safe_solve(h, timeout):
    """Run a (proposer-written) harness's solve() under a hard wall-clock cap so a buggy harness can't hang."""
    try:
        return _SOLVE_POOL.submit(h.solve).result(timeout=timeout) or ""
    except Exception:
        return ""


def _loadable(name, problem):
    """A candidate is admissible only if it IMPORTS and passes the TTHE invariant audit.

    The audit was written but never wired in: all four domains' harness_base docstrings claim
    "audit_harness.py checks them", and nothing called it, so FROZEN-SOLVER and LABEL-FREE were honour-system
    only. That was tolerable while `selfcheck` was the harness's sole execution primitive; now that a harness
    can run arbitrary code it is not, because reading `problem.code_context` and grading against it would be
    both easy and invisible. A violating candidate is rejected here, which leaves its branch at its parent."""
    try:
        load_harness(name, problem)
    except Exception:
        return False
    try:
        bad = [v for v in audit_file(AGENTS_DIR / f"{name}.py") if v["rule"] != "PARSE"]
    except Exception:  # noqa: BLE001
        return True                       # auditor failure must not silently reject a valid candidate
    if bad:
        print(f"   [audit] REJECTED {name}: " +
              "; ".join(f"{v['rule']} line {v['line']}: {v['detail']}" for v in bad[:4]), flush=True)
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True, help="json: list of problem_ids OR {items:[{pid}...]}")
    ap.add_argument("--group", type=int, default=2, help="G agentic generators per GENERATE round")
    ap.add_argument("--max-rounds", type=int, default=3, help="GENERATE rounds per batch")
    ap.add_argument("--batch-size", type=int, default=5)
    ap.add_argument("--propose-timeout", type=int, default=600, help="hard cap per generator/judge claude session")
    ap.add_argument("--solve-timeout", type=int, default=600, help="hard cap per harness.solve (anti-hang)")
    ap.add_argument("--model", default=os.environ.get("TTHO_PROPOSER_MODEL", "deepseek-v4-flash"))
    ap.add_argument("--run-name", default="ds1000pilot")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--initial-harness", default="bare",
                    help="seed harness the evolution starts from (e.g. react)")
    ap.add_argument("--specialist", action="store_true",
                    help="clients evolve as EXPERTS of their own slice (see SQL fed_loop; "
                         "prerequisite for aggregation to have anything to combine")
    ap.add_argument("--supervised", action="store_true",
                    help="train-then-test mode: this slice is TRAINING data with usable labels. Traces state "
                         "each answer's graded verdict, and selection scores candidates by measured accuracy "
                         "on the batch instead of a judge's label-free investigation. The original "
                         "transductive protocol must leave this off.")
    ap.add_argument("--role-offset", type=int, default=0,
                    help="shift the proposer search role (federated: pass the client index so roles spread "
                         "ACROSS clients — with G=1 every client would otherwise get the same conservative "
                         "role and their deltas collapse into near-duplicates)")
    args = ap.parse_args()

    if args.fresh:
        for f in AGENTS_DIR.glob("cand_*.py"):   # only clear generated candidates; keep seed harnesses
            f.unlink()

    spec = json.load(open(args.pilot))
    spec = spec["items"] if isinstance(spec, dict) else spec
    pids = [str(it["pid"]) if isinstance(it, dict) else str(it) for it in spec]
    items = bridge.load_problems(ids=pids)

    # Timestamped run dir: reusing a --run-name must never let a PREVIOUS run's traces leak into
    # this one. Candidate names embed the run name, so a rerun of the same name produces IDENTICAL
    # trace filenames that would silently mix with the old ones — and the proposer, pointed at the
    # batch trace dir, would read a blend of two runs as if it were one.
    # Under --supervised the orchestrator NEEDS a predictable path to read result.json back, and it already
    # guarantees run-name uniqueness (one name per round x client), so the exact name is used there.
    run_dir = PKG_DIR / "logs" / (args.run_name if args.supervised
                                  else f"{args.run_name}_{time.strftime('%Y%m%d_%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    log = open(run_dir / "opt_log.jsonl", "w")
    print(f"\n######### harness optimization — DS-1000 — mode={'SUPERVISED train-then-test' if args.supervised else 'transductive test-time'} #########")
    libs = {}
    for p in items:
        libs[p.library] = libs.get(p.library, 0) + 1
    print(f"[stream] {len(items)} problems  by library={libs}  batch_size={args.batch_size} "
          f"group={args.group} rounds={args.max_rounds}", flush=True)

    def write_trace(trace_dir, name, j, problem, code, sc, steps, verdict=None):
        L = [f"# Trace — harness `{name}` — Q{j}  [{problem.pid} / {problem.library}]\n"]
        if verdict is not None:
            # Supervised (train-then-test) mode only: this slice is TRAINING data, so the graded verdict is
            # legitimately available and is the single most valuable line in the trace. The original
            # transductive protocol never writes this block.
            L.append(f"## GRADED OUTCOME (ground truth for this training problem)\n"
                     f"**This harness answered this problem {'CORRECTLY' if verdict else 'INCORRECTLY'}.**\n"
                     + ("Preserve whatever produced this.\n" if verdict else
                        "This is a problem to fix. Note it can be INCORRECT even though the self-check ran "
                        "cleanly — the self-check exercises the EXAMPLE input only; the graded verdict runs "
                        "the hidden test with different inputs. Running is necessary, not sufficient.\n"))
        L += [f"## PROBLEM\n{problem.prompt[:3500]}\n",
              "## WHAT THE HARNESS DID — every coder call + every self-check, in order:"]
        for i, st in enumerate(steps, 1):
            if st.get("step") == "coder_llm":
                # Show the HEAD and the TAIL of the prompt. A flat [:1200] cut kept only the problem statement
                # — which the proposer already knows — and always discarded the tail, where the harness's OWN
                # appended retry hints and error diagnoses live. A proposer writing those hints could therefore
                # never observe their effect in any trace, and was editing them blind.
                pr = str(st.get("prompt"))
                shown = pr if len(pr) <= 2600 else (pr[:900] + f"\n\n... [{len(pr) - 2600} chars of problem "
                                                    f"statement elided] ...\n\n" + pr[-1700:])
                L.append(f"\n### step {i} — coder call (thinking={st.get('thinking')})\nPROMPT:\n{shown}\n"
                         f"RESPONSE:\n{str(st.get('response'))[:4000]}")
            elif st.get("step") == "run":
                # The harness's SELF-MADE evidence. Show it in full-ish: this is the only place a proposer can
                # see whether a signal a candidate invented actually discriminated anything.
                L.append(f"\n### step {i} — harness-built check (rc={st.get('rc')})\nSCRIPT:\n"
                         f"{str(st.get('script'))[:1500]}\nSTDOUT:\n{str(st.get('stdout'))[:800]}\n"
                         f"STDERR:\n{str(st.get('stderr'))[:400]}")
            else:
                L.append(f"\n### step {i} — self-check: ran={st.get('ran')}  redefines_input={st.get('redefines')}  "
                         f"error={str(st.get('error'))[:200]!r}  output={str(st.get('output'))[:200]!r}")
        # FINAL CODE must be COMPLETE — the proposer diagnoses it; a mid-statement cut reads as a phantom bug.
        L.append(f"\n## FINAL CODE\n```python\n{str(code)}\n```")
        L.append(f"\n## SELF-CHECK (LABEL-FREE — the gold hidden test is NEVER shown; this is the only execution "
                 f"evidence):\n  checkable={sc.get('checkable', True)}  (False = NO example input exists in the "
                 f"prompt, so NOTHING could be executed — this is ABSENCE OF EVIDENCE, NOT a failure; do not "
                 f"treat it as a wrong answer)\n  ran={sc.get('ran')}\n  redefines_input={sc.get('redefines')}  "
                 f"(NON-EMPTY = the solution HARDCODES these input variables instead of using the provided ones "
                 f"-> runs here but FAILS the hidden test, which supplies different inputs; a near-certain WRONG)"
                 f"\n  error={str(sc.get('error'))[:600]!r}\n  output={str(sc.get('output'))[:600]!r}")
        L.append(f"\n## BACK-TRANSLATION — what the FINAL CODE literally computes, in plain English. COMPARE it to "
                 f"the PROBLEM above: if it computes something different from what the problem asks, the code is "
                 f"likely wrong (an intent-level check beyond the self-check).\n{bridge.back_translate(code)}")
        (trace_dir / f"{name}__q{j}.md").write_text("\n".join(L), encoding="utf-8")

    # (pid, code) -> hidden-test verdict. Different candidates routinely emit byte-identical code, and the
    # hidden test is the expensive part of a supervised round — dedupe it. Also makes verdicts consistent:
    # the same code can never be graded twice with different outcomes.
    gold_cache = {}

    def graded(code, p):
        key = (p.pid, code or "")
        if key not in gold_cache:
            gold_cache[key] = bool(bridge.is_correct(code, p)) if code else False
        return gold_cache[key]

    def observe(name, batch, trace_dir):
        """Run harness `name` on every batch problem (parallel); write each trace; return list of codes."""
        cls = type(load_harness(name, batch[0]))          # reload + class ONCE (reload not thread-safe)
        codes = [None] * len(batch)

        def one(jp):
            j, p = jp
            h = cls(p)
            code = safe_solve(h, args.solve_timeout)
            sc = bridge.selfcheck(code, p) if code else {"checkable": True, "ran": False, "error": "(no code)",
                                                         "output": "", "redefines": []}
            verdict = graded(code, p) if args.supervised else None
            write_trace(trace_dir, name, j, p, code, sc, getattr(h, "_trace", []), verdict)
            codes[j] = code
        with ThreadPoolExecutor(max_workers=min(len(batch), 8)) as ex:
            list(ex.map(one, list(enumerate(batch))))
        return codes

    H = args.initial_harness
    if not (AGENTS_DIR / f"{H}.py").exists():
        raise ValueError(f"--initial-harness not found: agents/{H}.py")
    B = args.batch_size
    batches = [items[i:i + B] for i in range(0, len(items), B)]
    tt_correct, tt_total, tt_log, ev_results = 0, 0, [], []
    for bi, batch in enumerate(batches):
        trace_dir = run_dir / "traces" / f"b{bi}"
        trace_dir.mkdir(parents=True, exist_ok=True)
        traced, cand_results = set(), {}
        branches = [H] * args.group
        print(f"\n===== BATCH {bi}/{len(batches)} ({len(batch)} q) — start from H={H} =====", flush=True)
        # GENERATE phase: G fixed branches from H. Every round all proposers see the same active
        # branch--trace pairs, but proposer gi edits ONLY branch gi. An invalid/unloadable child leaves
        # that branch at its own parent (per-branch fallback).
        for rnd in range(args.max_rounds):
            active = list(dict.fromkeys(branches))
            for c in active:
                if c not in traced and _loadable(c, batch[0]):
                    cand_results[c] = observe(c, batch, trace_dir)
                    traced.add(c)
            proposed = P.sample_branches(branches, trace_dir, run_dir, f"b{bi}r{rnd}", args.run_name,
                                         batch, args.model, args.propose_timeout,
                                         role_offset=args.role_offset, specialist=args.specialist)
            next_branches, advanced = [], 0
            for base, child in zip(branches, proposed):
                accepted = child if child and _loadable(child, batch[0]) else base
                next_branches.append(accepted)
                advanced += accepted != base
            branches = next_branches
            for c in dict.fromkeys(branches):
                if c not in traced and _loadable(c, batch[0]):
                    cand_results[c] = observe(c, batch, trace_dir)
                    traced.add(c)
            print(f"   batch{bi} gen-round{rnd}: {advanced}/{args.group} branches advanced", flush=True)
        # PICK phase — ROLLBACK GATE (ported from LCB, where it was measured). The judge chooses from EVERY
        # harness observed this batch (the incoming H plus every round's branches), not just the final round:
        # rounds routinely DEGRADE a good early branch, and offering only the last round silently discards it.
        # Keeping the incoming H in the pool is the gate itself — if nothing the batch produced beats H, the
        # judge keeps H and the accumulated harness never regresses. cand_results is insertion-ordered
        # (H first, then r0/r1/r2 branches), so use it directly as the pool.
        final_candidates = list(cand_results.keys())
        entering_H = H
        if args.supervised:
            # Selection by MEASURED accuracy on the batch's graded verdicts — codes are already computed, so
            # this costs only the (cached) hidden-test runs. The entering harness is in the pool (observed
            # first), so a batch can never hand back something measurably worse than what it was given.
            # On a TIE keep the NEW candidate: at equal measured accuracy it still carries this round's new
            # mechanisms — the raw material an aggregator needs. Falling back to the parent on ties strands
            # every client at the broadcast harness (small shards tie constantly; observed directly).
            scored = {c: sum(graded(code, p) for code, p in zip(cand_results[c], batch))
                      for c in final_candidates}
            best = max(scored.values())
            winners = [c for c in final_candidates if scored[c] == best]
            H = next((c for c in winners if c != entering_H), entering_H)
            print(f"   batch{bi}: SUPERVISED pick H={H} "
                  f"(scores {', '.join(f'{c}={sc}/{len(batch)}' for c, sc in scored.items())})", flush=True)
        else:
            # H is the incumbent: the judge must be told which candidate is currently in force, so that the
            # burden of proof sits on the challengers instead of all candidates being treated as symmetric.
            picked = P.pick_batch(final_candidates, trace_dir, run_dir, f"b{bi}", args.model,
                                  args.propose_timeout, incumbent=H)
            H = picked if picked in final_candidates else H     # judge failure -> keep the incoming harness
            print(f"   batch{bi}: final branches={branches} ({len(final_candidates)} unique) -> JUDGE picked H={H}",
                  flush=True)
        if args.supervised and cand_results.get(entering_H) is not None and cand_results.get(H) is not None:
            # Per-question record of what this batch CHANGED relative to the harness it was handed — in a
            # federated run that is the broadcast global, so this file names exactly which problems this
            # client newly solved (the capability an aggregator must preserve) and which it broke.
            verdicts = []
            for j, p in enumerate(batch):
                was = graded(cand_results[entering_H][j], p)
                now = graded(cand_results[H][j], p)
                verdicts.append({"q": j, "pid": p.pid, "question": p.prompt[:300],
                                 "entering_correct": was, "picked_correct": now,
                                 "delta": "fixed" if now and not was else
                                          "broke" if was and not now else "same"})
            (run_dir / f"shard_verdicts_b{bi}.json").write_text(
                json.dumps({"entering": entering_H, "picked": H, "questions": verdicts}, indent=2),
                encoding="utf-8")
        codes = cand_results.get(H)
        if codes is not None:
            bc = 0
            for code, p in zip(codes, batch):
                ok = bool(bridge.is_correct(code, p))
                ev_results.append({"pid": p.pid, "library": p.library, "correct": ok, "harness": H})
                bc += ok
            tt_correct += bc
            tt_total += len(batch)
            tt_log.append({"batch": bi, "harness": H, "correct": bc, "total": len(batch)})
            print(f"   [test-time] batch{bi} H={H}: {bc}/{len(batch)} (gold tests)", flush=True)
        log.write(json.dumps({"batch": bi, "harness": H, "branches": branches,
                              "candidates": final_candidates}) + "\n")
        log.flush()
    log.close()

    print(f"\n######### RESULT ({'supervised train-then-test' if args.supervised else 'test-time / transductive'}) — final H = {H} #########", flush=True)
    print(f"  test-time evolved = {tt_correct}/{tt_total}   (baseline = plain react, measured separately)")
    print("  by library (evolved):")
    for lib in sorted(libs):
        ev_l = sum(r["correct"] for r in ev_results if r["library"] == lib)
        ev_n = sum(1 for r in ev_results if r["library"] == lib)
        print(f"    {lib:12} evolved {ev_l}/{ev_n}")
    json.dump({"tt_correct": tt_correct, "tt_total": tt_total, "final_harness": H,
               "batches": tt_log, "per_problem": ev_results},
              open(run_dir / "result.json", "w"), indent=2)
    print(f"[saved] {run_dir}/result.json   [traces] {run_dir}/traces/")


if __name__ == "__main__":
    main()
