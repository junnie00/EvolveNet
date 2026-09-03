"""Server-side aggregation: fold K client harnesses back into one global harness.

This is the piece the reference implementation does NOT have. `propose_batch` lets G branches read
each other's source and traces, but every branch stays a descendant of its own assigned base and the
judge only ever PICKS one survivor — nothing merges them. Here the merger starts from the harness
that was broadcast to every client, reads what each client changed and the evidence for it, and
edits that one file in place.

Editing the broadcast global (rather than writing a fresh harness from K sources) is what keeps this
bounded: the merger reviews K small deltas instead of K full files, and can decline any of them.
Unbounded accumulation is a known failure mode here — a single lineage once grew 14 -> 1181 lines
over 9 rounds by stacking overlapping mechanisms.
"""
import argparse
import json
import shutil
from pathlib import Path

from .. import claude_wrapper
from ..evolve import AGENTS_DIR, MH_ROOT, PKG, PKG_DIR

MERGER_TOOLS = ["Read", "Glob", "Grep", "Write", "Edit", "Bash"]
MERGER_SYS = ("You are a Python+SQL engineer consolidating Text-to-SQL HARNESSES (classes wrapping a "
              "FROZEN weak SQL model). Output complete runnable code only.")


# The only thing that differs between aggregation variants is the ADOPTION RULE — how the merger decides
# which client mechanisms enter the global. Everything else (edit-the-broadcast-base, budget, load-bearing
# prompt rules, verification) is shared, so a difference in results is attributable to the rule alone. Each
# variant mirrors a classical federated-aggregation idea applied to code instead of weights.
ADOPTION_RULES = {
    # Holistic LLM judgement — the default. The merger adopts whatever the traces support.
    "holistic": (
        "4. For each candidate mechanism decide ADOPT / REJECT / MERGE-WITH-EXISTING. Adopt only what the "
        "traces actually support. REJECT a mechanism that encodes ONE question's quirk — a hardcoded "
        "answer, value, or column for a single question.\n"
        "   Do NOT reject a mechanism merely because it is specific to one DATABASE. Each database here is "
        "a large, recurring share of the workload, its schema conventions are stable, and a mechanism that "
        "reliably fixes one database's recurring confusion (which of two similar tables an entity lives "
        "in, which column holds a value) is exactly the kind of durable knowledge worth keeping. "
        "'Specific to one question' is the thing to reject; 'specific to one schema' is not.\n"
        "5. Where two clients solved the SAME problem differently, keep ONE — the simpler one with the "
        "better trace evidence. Do not keep both behind flags.\n"),
    # Quorum (FedAvg analogue): a mechanism must be independently proposed by at least two clients to be
    # adopted. Averaging weights keeps only the directions many clients agree on; here, agreement is that
    # two clients arrived at the same fix. A lone client's mechanism is adopted only if its own graded
    # evidence (a NEWLY SOLVED question) is undeniable.
    "quorum": (
        "4. ADOPT A MECHANISM ONLY IF AT LEAST TWO CLIENTS INDEPENDENTLY PROPOSED IT. Read the proposal "
        "cards and diffs and group mechanisms by the problem they address (their `observed_issue`), not by "
        "identical code. A fix that two or more clients converged on from separate shards is durable and "
        "goes in. A mechanism only ONE client proposed is REJECTED — UNLESS its proposal card cites a "
        "graded NEWLY SOLVED question that its mechanism is directly responsible for, in which case adopt "
        "it as a single-client exception. State the vote count for every mechanism in the report.\n"
        "5. When the quorum agrees on a problem but the clients coded it differently, adopt the ONE "
        "simplest implementation, not a union of all of them.\n"),
    # Scoped adoption — resolves the measured failure of holistic on specialist clients: the merger
    # globally REJECTED an expert's home-DB rules because they regressed on other DBs (fedglobal_spec1_r1:
    # formula_1 0/9 on the hard slice vs 3/9 for the very client whose rules were dropped). Scoping turns a
    # risky global change into a safe local one: the expert's rule fires only on its home DB, so it cannot
    # break anything elsewhere by construction.
    "scoped": (
        "4. SPLIT EVERY CANDIDATE MECHANISM INTO GLOBAL vs HOME-SCOPED before deciding.\n"
        "   - GLOBAL: mechanisms whose evidence spans databases or whose failure mode is universal (value "
        "probing, execution-error retry, SQL-dialect auto-fix, generic output-shape rules). Adopt ONE best "
        "implementation globally, exactly as a holistic merge would.\n"
        "   - HOME-SCOPED: a rule whose evidence comes from ONE client's home database (schema conventions, "
        "column-choice guidance, phrasing rules tied to that schema). Do NOT adopt it globally, and do NOT "
        "reject it for regressing on OTHER databases — adopt it CONDITIONALLY: the harness receives db_id "
        "at solve time, so wrap the rule so it only applies when db_id equals that client's home database. "
        "A home-scoped rule can only change behaviour at home, so off-home regressions are impossible by "
        "construction; judge it ONLY on its home evidence.\n"
        "   - Still REJECT single-question hacks (a hardcoded answer/value/column for one question) even "
        "inside a scope.\n"
        "5. CONFLICTS DISSOLVE UNDER SCOPING: when two clients changed the same behaviour in opposite "
        "directions, keep each side scoped to its own home database instead of picking a winner. Only if "
        "both sides claim GLOBAL evidence must you pick the one with stronger graded evidence. In the "
        "report, list every mechanism with its scope (GLOBAL or the db_id it is gated on).\n"),
    # Scoped + verbatim transfer — fixes the measured lossy-rewrite failure of plain scoped: the merger's
    # re-implementation of an expert's probe scored below the expert at its own home DB (toxicology 8 vs
    # 10, card_games 11 vs 12 on the 200-item test). Under db_id gating the experts' scopes are disjoint,
    # so their code cannot interact; the correct move is to COPY it, not to paraphrase it.
    "scoped-verbatim": (
        "4. SPLIT EVERY CANDIDATE MECHANISM INTO GLOBAL vs HOME-SCOPED before deciding.\n"
        "   - HOME-SCOPED: a mechanism whose evidence comes from ONE client's home database. Adopt it by "
        "COPYING THAT CLIENT'S CODE VERBATIM into a branch gated on its home db_id — the harness receives "
        "db_id at solve time. Do NOT re-implement, generalize, simplify, or 'improve' the client's code: "
        "every paraphrase is a chance to lose the behaviour that earned its evidence. Change only what is "
        "syntactically required to fit (indentation, variable capture). A home-scoped mechanism cannot "
        "affect other databases by construction, so judge it ONLY on its home evidence and do NOT reject "
        "it for off-home risk.\n"
        "   - GLOBAL: mechanisms whose evidence spans databases (value probing, execution-error retry, "
        "dialect auto-fix, generic output-shape rules). Pick the ONE client implementation with the best "
        "evidence and copy IT verbatim too; merge two implementations only when each demonstrably covers "
        "cases the other misses.\n"
        "   - REJECT only single-question hacks (a hardcoded answer/value/column for one question).\n"
        "5. CONFLICTS DISSOLVE UNDER SCOPING: two clients editing the same behaviour in opposite "
        "directions each keep their own version inside their own db_id branch. Only genuinely GLOBAL "
        "conflicts need a winner — pick the side with stronger graded evidence. In the report, list every "
        "mechanism with its scope and whether it was copied verbatim.\n"),
    # TIES-merging analogue: the failure mode we actually observed is two clients editing the SAME
    # behaviour in OPPOSITE directions (one made a join INNER, another kept it LEFT; both were right for
    # their own shard). TIES resolves sign conflicts before merging; here the merger must find such
    # conflicts and keep only the side with stronger graded evidence, never both.
    "ties": (
        "4. FIRST FIND CONFLICTS. Two clients are in conflict when they changed the SAME behaviour in "
        "OPPOSITE directions — e.g. one switches a join to INNER while another keeps it LEFT, one adds a "
        "DISTINCT another removes it, one adds a filter another drops it. Read the cards' `expected_effect` "
        "and the diffs to detect these. For each conflict, adopt ONLY the side whose graded evidence is "
        "stronger (more NEWLY SOLVED, fewer REGRESSED on that behaviour); discard the other side entirely. "
        "NEVER keep both behind a flag or condition — an unresolved conflict is the main way a merge breaks "
        "questions the base already answered.\n"
        "5. For non-conflicting mechanisms, adopt what the traces support and reject one-question quirks, "
        "keeping the simpler implementation where two clients agree. Report every conflict you found and "
        "which side you kept.\n"),
}


def merge(global_name, clients, out_name, run_dir, model, timeout, budget_lines=150,
          gains="", feedback="", variant="holistic", weights=None):
    """clients: [{"client": k, "harness": name, "trace_dir": path, "shard": path}, ...]

    `variant` selects the adoption rule (see ADOPTION_RULES); everything else is identical across variants.
    Returns True if the merger left a loadable file at agents/<out_name>.py.
    """
    if variant not in ADOPTION_RULES:
        raise ValueError(f"unknown merge variant {variant!r}; have {sorted(ADOPTION_RULES)}")
    base_path = AGENTS_DIR / f"{global_name}.py"
    out_path = AGENTS_DIR / f"{out_name}.py"
    if not base_path.exists():
        raise FileNotFoundError(base_path)
    out_path.unlink(missing_ok=True)
    shutil.copyfile(base_path, out_path)          # merger EDITS the broadcast global, never rewrites it
    base_lines = len(base_path.read_text(encoding="utf-8").splitlines())

    sample_db = "card_games"
    for c in clients:                              # a db_id that really occurs in the shards
        pairs = json.loads(Path(c["shard"]).read_text(encoding="utf-8"))["cross"]
        if pairs:
            sample_db = pairs[0][0]
            break

    # FedAvg aggregates by WEIGHT, not by dropping clients: w = sum_k (n_k/n) w_k. The code analogue is
    # to tell the merger how much each client has earned, measured on the gate slice, so a client that
    # generalises well carries more of the merge than one that does not. Handing over per-client
    # reliability is what an earlier incremental design bought with K sequential merger sessions -- at K
    # times the cost, with no cross-client comparison possible (each session saw one client) and an
    # arbitrary absorption order. One weighted session keeps the signal and drops both problems.
    wline = ""
    if weights:
        best = max(weights.values()); worst = min(weights.values())
        wline = ("\n\nCLIENT RELIABILITY — each client's harness was scored on the SAME held-out gate slice "
                 "the merged result will be judged on. Higher means its edits generalise better beyond its "
                 "own shard; this is measured, not asserted:\n"
                 + "\n".join(f"  - client {c['client']} (`{c['harness']}`): {weights.get(c['harness'], '?')}"
                              + ("   <- STRONGEST: prefer its mechanisms when clients disagree"
                                 if weights.get(c['harness']) == best else
                                 "   <- WEAKEST: adopt from it only on clear trace evidence"
                                 if weights.get(c['harness']) == worst else "")
                              for c in clients)
                 + "\n\nWeigh contributions accordingly. A high-scoring client's mechanism is the default when "
                   "two clients conflict; a low-scoring client's mechanism needs its own evidence to enter. "
                   "Do NOT simply copy the strongest client wholesale — its harness is already available and "
                   "adding nothing to it makes this merge pointless. The gain has to come from folding in "
                   "what the others got right that it did not.")

    rows = []
    for c in clients:
        # Each proposer is required to file a proposal card alongside its harness, recording what it
        # changed, which trace prompted it, and the DB evidence it checked. That is exactly what a merger
        # otherwise has to reverse-engineer from the diff, so hand it over instead.
        card = PKG_DIR / "logs" / c["run"] / "proposal_cards" / f"{c['harness']}.json" if c.get("run") else None
        card_note = f" ; proposal card: {card}" if card and card.exists() else ""
        dbs = sorted({p[0] for p in json.loads(Path(c["shard"]).read_text(encoding="utf-8"))["cross"]})
        rows.append(
            f"  - client {c['client']}: harness `{c['harness']}` "
            f"(code: {AGENTS_DIR}/{c['harness']}.py ; full traces: {c['trace_dir']}/{c['harness']}__q*.md ; "
            f"its shard: {c['shard']} ; home database(s): {', '.join(dbs)}{card_note})"
        )
    client_list = "\n".join(rows)
    report_path = Path(run_dir) / f"merge_report_{out_name}.md"

    prompt = (
        f"You are the SERVER in a federated harness-evolution run. A single GLOBAL harness "
        f"`{global_name}` was broadcast to {len(clients)} clients. Each client evolved it further on its "
        f"OWN private shard of questions and sent back the result. Your job: fold their improvements "
        f"back into ONE global harness.\n\n"
        + (f"You do NOT see gold answers. Judge every mechanism by trace evidence only.\n\n" if not gains else
           f"You do not see gold answers directly, but the per-question verdicts below WERE graded against "
           f"ground truth. Treat them as fact and judge everything else by trace evidence.\n\n")
        + f"BROADCAST GLOBAL (the base you must edit): {base_path} ({base_lines} lines)\n"
        f"CLIENT RESULTS:\n{client_list}{wline}\n\n"
        + (f"MEASURED CAPABILITY CHANGES — each client's harness was graded against ground truth on its "
           f"own shard, relative to the broadcast global. These are facts, not opinions:\n{gains}\n\n"
           f"Every question listed as NEWLY SOLVED is a capability this round actually gained. Your merged "
           f"harness must keep them solved — find the mechanism responsible in that client's code and "
           f"carry it over. A merge that drops these has lost the round's work. Questions listed as "
           f"REGRESSED are the opposite: whatever caused them is suspect, do not carry it over.\n\n"
           if gains else "")
        + (f"{feedback}\n\n" if feedback else "")
        + f"YOUR TARGET FILE: `{out_path}` — already copied byte-for-byte from the broadcast global. "
        f"EDIT IT IN PLACE. Do NOT rewrite it from scratch and do NOT replace it wholesale with any "
        f"single client's harness.\n\n"
        f"METHOD:\n"
        f"1. READ EACH CLIENT'S PROPOSAL CARD FIRST, where one is listed. The proposer that wrote the "
        f"harness filed it, and it states in its own words what changed and why: `behavior_changes` gives "
        f"per-change `observed_issue` / `change` / `db_evidence` / `expected_effect`, `preserved_behaviors` "
        f"names what it deliberately left alone, and `risks` lists regressions it already suspected. Start "
        f"from the card, then confirm against the diff — the card tells you the INTENT, the diff only shows "
        f"the edit. Two clients whose cards describe the same `observed_issue` have converged on one "
        f"problem; two whose `expected_effect` point in opposite directions are in conflict and you must "
        f"pick one rather than adopt both. A card is a claim, not proof: if the traces contradict it, the "
        f"traces win.\n"
        f"2. Diff each client harness against the broadcast global to see exactly what that client ADDED "
        f"or CHANGED. That delta is the client's contribution — work at that granularity, not whole files.\n"
        f"3. DEEP-READ the traces. Each `<harness>__q<j>.md` holds, for ONE harness on ONE question: the "
        f"QUESTION, the HINT (AUTHORITATIVE), the SCHEMA, EVERY step (each coder call's prompt+response, "
        f"each SQL run + its result), the FINAL SQL+result, and a BACK-TRANSLATION of the final SQL into "
        f"English. Compare the back-translation against the question+Hint: if the English says something "
        f"different from what was asked, that SQL is wrong however clean it looks.\n"
        + ADOPTION_RULES[variant]
        + f"6. TREAT THE BASE'S EXISTING PROMPT RULES AS LOAD-BEARING. You may APPEND to them. Do NOT "
        f"rewrite, renumber, reorder, compact, or delete rules the base already had, and do not restate "
        f"one of its rules in your own words. Prompt text applies to EVERY question the harness ever "
        f"answers, so editing a working rule silently changes behaviour everywhere, while adding a code "
        f"path only changes behaviour where its condition holds. If a base rule looks wrong to you, leave "
        f"it and say so in the report instead of 'fixing' it.\n\n"
        f"SIZE BUDGET — enforced: `{out_path}` may not exceed {base_lines + budget_lines} lines "
        f"(base {base_lines} + {budget_lines}). If adopting everything worthwhile would exceed that, you "
        f"MUST first consolidate or delete existing overlapping mechanisms to make room. Growth is not "
        f"free; overlapping half-working mechanisms are how this codebase has failed before. Check with: "
        f"wc -l {out_path}\n\n"
        f"KEEP IT GENERAL: never hardcode any question's value/column/answer. The Hint is authoritative. "
        f"Do NOT write a custom SQL parser or heavy/backtracking regex over SQL text (it HANGS the GIL); "
        f"inspect SQL by EXECUTING it. Harness API (NOT sqlite3): self.tables(); self.distinct(table,col); "
        f"self.column_types(table); self.execute(sql); self.llm(prompt,system='',temperature=0.0,n=1); "
        f"self.schema; bridge.extract_sql. Keep `from ..harness_base import SQLHarness` and "
        f"`from .. import bridge`.\n\n"
        f"Probe the DB freely, e.g.:\n"
        f"  PYTHONPATH={MH_ROOT} python -c \"from {PKG} import bridge; db=bridge.get_db('<db_id>'); "
        f"print(bridge.execute(db,'SELECT ...'))\"\n\n"
        f"VERIFY BEFORE YOU STOP: the file must import cleanly. Check with:\n"
        f"  PYTHONPATH={MH_ROOT} python -c \"from {PKG}.evolve import load_harness; from {PKG} import bridge; "
        f"load_harness('{out_name}', bridge.get_db('{sample_db}')); print('LOADS OK')\"\n"
        f"Fix any error and re-run until it prints LOADS OK.\n\n"
        f"FINALLY write `{report_path}` — one line per mechanism you considered: "
        f"`ADOPT|REJECT|MERGED  <client k>  <mechanism>  <the trace evidence, or why not>`. "
        f"This is the record of what aggregation actually did; be honest, including about what you dropped "
        f"for budget rather than for evidence.\n\n"
        f"Leave the final merged harness at EXACTLY `{out_path}`."
    )

    claude_wrapper.run(prompt=prompt, model=model, allowed_tools=MERGER_TOOLS, cwd=str(MH_ROOT),
                       log_dir=str(Path(run_dir) / "claude_sessions"), name=f"merge_{out_name}",
                       system_prompt=MERGER_SYS, timeout_seconds=timeout, progress=False)
    return out_path.exists()


REPAIR_TOOLS = ["Read", "Glob", "Grep", "Write", "Edit", "Bash"]


def repair(harness_name, counterexamples, run_dir, model, timeout):
    """Guard the rules that a merge broke, in place, without giving up what it fixed.

    Every regression observed so far has the same shape: a mechanism learned from one question is
    written as an UNCONDITIONAL prompt rule, and prompt text applies to every question the harness ever
    sees. One merge changed LEFT JOIN to INNER JOIN — correct for the question it was derived from,
    wrong for another that needed the outer rows; another concatenated first and last name into one
    column, correct for the question that asked for a full name, wrong for one that wanted two columns.

    So the fix is not to delete the rule (that discards the gain) and not to re-merge from scratch (that
    rerolls everything). It is to give the rule the precondition it was missing, using the two SQL
    statements as the counterexample.
    """
    path = AGENTS_DIR / f"{harness_name}.py"
    cases = "\n\n".join(
        f"REGRESSION {i + 1} — [{c['db_id']}] {c['question'][:200]}\n"
        f"  The PREVIOUS global answered this CORRECTLY with:\n    {c['old_sql'][:300]}\n"
        f"  YOUR merged harness now produces this, which is WRONG:\n    {c['new_sql'][:300]}"
        for i, c in enumerate(counterexamples))
    prompt = (
        f"You merged client harnesses into `{path}`. It improved several questions, but it also BROKE "
        f"{len(counterexamples)} questions that the previous global answered correctly. Your job is to keep "
        f"every improvement AND restore the broken behaviour.\n\n"
        f"{cases}\n\n"
        f"For each regression, compare the two SQL statements and identify what in your merged harness "
        f"caused the change — usually a rule you added to the system prompt. These rules are applied to "
        f"EVERY question, so a rule that is right for the question it came from will be wrong elsewhere.\n\n"
        f"FIX BY ADDING A PRECONDITION, NOT BY DELETING. The rule earned its place on some question; "
        f"deleting it trades one regression for another. State when it applies and when it does not — for "
        f"instance a rule about combining columns should say which phrasings ask for one value and which "
        f"ask for several; a rule about join type should say when rows without a match must still appear. "
        f"Derive the condition from the counterexample above, and keep it in the question's own terms "
        f"rather than naming a specific table or question.\n\n"
        f"Edit `{path}` in place. Do not restructure it and do not remove mechanisms unrelated to these "
        f"regressions. Verify it still imports:\n"
        f"  PYTHONPATH={MH_ROOT} python -c \"from {PKG}.evolve import load_harness; from {PKG} import "
        f"bridge; load_harness('{harness_name}', bridge.get_db('{counterexamples[0]['db_id']}')); print('LOADS OK')\""
    )
    claude_wrapper.run(prompt=prompt, model=model, allowed_tools=REPAIR_TOOLS, cwd=str(MH_ROOT),
                       log_dir=str(Path(run_dir) / "claude_sessions"), name=f"repair_{harness_name}",
                       system_prompt=MERGER_SYS, timeout_seconds=timeout, progress=False)
    return path.exists()


def loadable(name, db):
    """True if agents/<name>.py imports and instantiates."""
    from ..evolve import load_harness
    try:
        load_harness(name, db)
        return True
    except Exception as exc:
        print(f"   [merge] {name} not loadable: {type(exc).__name__}: {exc}", flush=True)
        return False


def smoke(name, run_dir, samples, timeout):
    """Run the merged harness on a few questions. -> (n_ok, n_total, rows).

    Gold is never consulted: this only asks whether the merged code still RUNS. A merge that
    crashes or errors out on most questions is worse than what was broadcast, and the caller
    rolls back.
    """
    from ..optimize import isolated_solve
    rows, ok = [], 0
    for i, (db_id, qtext) in enumerate(samples):
        out = isolated_solve(run_dir, name, db_id, qtext, timeout, f"smoke_{name}_{i}")
        status = out.get("status")
        res = out.get("result") or {}
        good = status == "ok" and res.get("ok")
        ok += bool(good)
        rows.append({"db_id": db_id, "status": status, "exec_ok": res.get("ok"),
                     "n_rows": len(res.get("rows") or []), "error": (res.get("error") or "")[:200]})
        print(f"   [smoke] {name} q{i} [{db_id}]: status={status} exec_ok={res.get('ok')}", flush=True)
    return ok, len(samples), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--global-harness", required=True)
    ap.add_argument("--clients", required=True, help="json list of {client,harness,trace_dir,shard}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--timeout", type=int, default=2700)
    ap.add_argument("--budget-lines", type=int, default=150)
    args = ap.parse_args()

    clients = json.loads(Path(args.clients).read_text(encoding="utf-8"))
    ok = merge(args.global_harness, clients, args.out, args.run_dir,
               args.model, args.timeout, args.budget_lines)
    print(f"[merge] wrote={ok} -> {AGENTS_DIR / (args.out + '.py')}")


if __name__ == "__main__":
    main()
