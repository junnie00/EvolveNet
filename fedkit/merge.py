"""Server-side aggregation for any domain: fold K client harnesses into one global, by editing the
broadcast base in place. Domain-blind except for the adapter's merge_blurb().

Everything here was learned the hard way on the SQL domain:
  * EDIT THE BROADCAST BASE, never rewrite from K sources — the merger reviews K small deltas, keeps a
    hard line budget, and unbounded accumulation is a measured failure mode (14 -> 1181 lines once,
    1457 -> 3182 lines with accuracy DROPPING on the unconstrained serial baseline);
  * the adoption rule is the ONLY thing aggregation variants change (ADOPTION_RULES), so variant
    differences are attributable to the rule alone;
  * the base's existing prompt rules are LOAD-BEARING: prompt text applies to every problem, so merging
    must append, never rewrite (a merger that "consolidated 17 rules into a tight format" broke questions
    the base already answered);
  * measured per-client gains (which problems each client NEWLY SOLVED) are handed to the merger as fact
    — without them it must reverse-engineer each client's worth and the union of client capabilities is
    routinely lost (union 29 vs best single 26, measured);
  * NO automatic post-merge "repair" pass: tried, and it turned 4 regressions into 67. The accept gate +
    one retry-with-counterexamples (fed_loop) is the working design.
"""
import json
import shutil
from pathlib import Path

import claude_wrapper

MERGER_TOOLS = ["Read", "Glob", "Grep", "Write", "Edit", "Bash"]
MERGER_SYS = ("You are a senior Python engineer consolidating agent HARNESSES (classes wrapping a FROZEN "
              "weak model). Output complete runnable code only.")

# The adoption rule is the aggregation variant. Same text as the SQL domain — it is domain-blind.
ADOPTION_RULES = {
    # GLOBAL-only ablation: the scope-typed operator with home-scoping switched OFF. Isolates how much
    # of FedHC's gain comes from promoting cross-domain mechanisms versus from conditioning
    # domain-specific ones. (An aggregator that may only adopt globally is exactly what "holistic" is
    # forced to be, but here the instruction is explicit rather than emergent.)
    "global-only": (
        "4. ADOPT ONLY GLOBAL MECHANISMS. A mechanism qualifies only if its evidence spans more than one "
        "client's domain, or its failure mode is plainly universal (execution-error retry, output-format "
        "discipline, extraction robustness, self-check logic). Adopt ONE best implementation of each such "
        "mechanism, applied to every input.\n"
        "   You may NOT introduce any condition on the item's domain (library / repository / difficulty): "
        "no `if self.library == ...`, no domain-labeled prompt sections, no per-domain branches of any "
        "kind. A mechanism whose evidence comes from a single client's domain must be REJECTED outright, "
        "however well it worked there.\n"
        "5. Where clients conflict, keep the single side with stronger graded evidence. Report every "
        "mechanism you adopted and the cross-domain evidence that justified it.\n"),
    # Scoped adoption — the SQL-measured fix for specialist clients: never judge a home-domain rule by
    # its off-home behaviour; scope it instead. (SQL evidence: holistic globally rejected an expert's
    # rules and lost the expert's home advantage entirely.)
    "scoped": (
        "4. SPLIT EVERY CANDIDATE MECHANISM INTO GLOBAL vs HOME-SCOPED before deciding.\n"
        "   - GLOBAL: mechanisms whose evidence spans clients' domains or whose failure mode is universal "
        "(execution-error retry, output-format discipline, generic self-checks). Adopt ONE best "
        "implementation globally.\n"
        "   - HOME-SCOPED: a mechanism whose evidence comes from ONE client's home domain (its library / "
        "repository / difficulty band — the slice its shard covers). Do NOT adopt it globally and do NOT "
        "reject it for risk elsewhere — adopt it CONDITIONALLY: the harness knows its item's domain at "
        "solve time (self.library / self.repo / problem difficulty), so gate the mechanism so it only "
        "fires on that client's home domain. A home-scoped mechanism cannot change behaviour elsewhere by "
        "construction; judge it ONLY on its home evidence, and prefer COPYING the client's code verbatim "
        "into the gated branch over re-implementing it.\n"
        "   - Still REJECT single-problem hacks (a hardcoded answer for one problem) even inside a scope.\n"
        "5. CONFLICTS DISSOLVE UNDER SCOPING: when two clients changed the same behaviour in opposite "
        "directions, keep each side gated on its own home domain instead of picking a winner. Only "
        "genuinely GLOBAL conflicts need a winner — take the side with stronger graded evidence. In the "
        "report, list every mechanism with its scope (GLOBAL or the domain it is gated on).\n"),
    "holistic": (
        "4. For each candidate mechanism decide ADOPT / REJECT / MERGE-WITH-EXISTING. Adopt only what the "
        "traces actually support. REJECT a mechanism that encodes ONE problem's quirk — a hardcoded "
        "answer or value for a single problem. Do NOT reject a mechanism merely because it targets a "
        "recurring CLASS of problems; durable class-level knowledge is exactly what is worth keeping. "
        "'Specific to one problem' is the thing to reject.\n"
        "5. Where two clients solved the SAME problem differently, keep ONE — the simpler one with the "
        "better trace evidence. Do not keep both behind flags.\n"),
    "quorum": (
        "4. ADOPT A MECHANISM ONLY IF AT LEAST TWO CLIENTS INDEPENDENTLY PROPOSED IT. Group mechanisms by "
        "the failure they address, not by identical code. A fix two clients converged on from separate "
        "shards is durable and goes in. A mechanism only ONE client proposed is REJECTED — unless the "
        "measured capability changes show a NEWLY SOLVED problem its mechanism is directly responsible "
        "for, in which case adopt it as a single-client exception. State the vote count for every "
        "mechanism in the report.\n"
        "5. When the quorum agrees on a problem but the clients coded it differently, adopt the ONE "
        "simplest implementation, not a union of all of them.\n"),
    "ties": (
        "4. FIRST FIND CONFLICTS. Two clients are in conflict when they changed the SAME behaviour in "
        "OPPOSITE directions (one enables thinking where another disables it, one adds a retry another "
        "removes it, one tightens a check another loosens it). For each conflict, adopt ONLY the side "
        "with stronger graded evidence (more NEWLY SOLVED, fewer REGRESSED on that behaviour); discard "
        "the other side entirely. NEVER keep both behind a flag — an unresolved conflict is the main way "
        "a merge breaks problems the base already answered.\n"
        "5. For non-conflicting mechanisms, adopt what the traces support and reject one-problem quirks, "
        "keeping the simpler implementation where two clients agree. Report every conflict you found and "
        "which side you kept.\n"),
}


def merge(adapter, global_name, clients, out_name, run_dir, model, timeout,
          budget_lines=150, gains="", feedback="", variant="holistic"):
    """clients: [{"client": k, "harness": name, "trace_dir": path, "shard": path}, ...].
    Returns True iff the merger left a file at <agents>/<out_name>.py."""
    if variant not in ADOPTION_RULES:
        raise ValueError(f"unknown merge variant {variant!r}; have {sorted(ADOPTION_RULES)}")
    agents = adapter.agents_dir
    base_path, out_path = agents / f"{global_name}.py", agents / f"{out_name}.py"
    if not base_path.exists():
        raise FileNotFoundError(base_path)
    out_path.unlink(missing_ok=True)
    shutil.copyfile(base_path, out_path)
    base_lines = len(base_path.read_text(encoding="utf-8").splitlines())

    client_list = "\n".join(
        f"  - client {c['client']}: harness `{c['harness']}` "
        f"(code: {agents}/{c['harness']}.py ; full traces: {c['trace_dir']}/{c['harness']}{adapter.trace_glob} ; "
        f"its shard: {c['shard']})" for c in clients)
    report_path = Path(run_dir) / f"merge_report_{out_name}.md"

    prompt = (
        f"You are the SERVER in a federated harness-evolution run. A single GLOBAL harness "
        f"`{global_name}` was broadcast to {len(clients)} clients; each evolved it further on its own "
        f"private shard of problems and sent back the result. Fold their improvements into ONE global.\n\n"
        f"{adapter.merge_blurb().replace('<NAME>', out_name)}\n\n"
        + (f"The per-problem verdicts below WERE graded against ground truth (this is supervised training "
           f"data). Treat them as fact; judge everything else by trace evidence.\n\n" if gains else
           f"You do NOT see gold answers. Judge every mechanism by trace evidence only.\n\n")
        + f"BROADCAST GLOBAL (the base you must edit): {base_path} ({base_lines} lines)\n"
        f"CLIENT RESULTS:\n{client_list}\n\n"
        + (f"MEASURED CAPABILITY CHANGES — each client graded on its own shard, relative to the broadcast "
           f"global:\n{gains}\n\n"
           f"Every problem listed as NEWLY SOLVED is a capability this round actually gained: find the "
           f"mechanism responsible in that client's code and carry it over — a merge that drops these has "
           f"lost the round's work. Problems listed as REGRESSED are the opposite: whatever caused them "
           f"is suspect, do not carry it over.\n\n" if gains else "")
        + (f"{feedback}\n\n" if feedback else "")
        + f"YOUR TARGET FILE: `{out_path}` — already copied byte-for-byte from the broadcast global. EDIT "
        f"IT IN PLACE. Do NOT rewrite it from scratch and do NOT replace it wholesale with any single "
        f"client's harness.\n\n"
        f"METHOD:\n"
        f"1. Diff each client harness against the broadcast global to see exactly what that client ADDED "
        f"or CHANGED. That delta is the client's contribution — work at that granularity, not whole "
        f"files.\n"
        f"2. DEEP-READ the traces. Each `<harness>{adapter.trace_glob.replace('*','<j>')}` holds, for ONE harness on ONE problem, every "
        f"model call and every execution the harness ran, plus (in supervised runs) the GRADED OUTCOME "
        f"header stating whether the answer was right.\n"
        f"3. Note which problems each client uniquely solved and what mechanism did it.\n"
        + ADOPTION_RULES[variant]
        + f"6. TREAT THE BASE'S EXISTING PROMPT RULES AS LOAD-BEARING. You may APPEND to them. Do NOT "
        f"rewrite, renumber, reorder, compact, or delete rules the base already had. Prompt text applies "
        f"to EVERY problem the harness ever answers, so editing a working rule silently changes behaviour "
        f"everywhere, while adding a code path only changes behaviour where its condition holds. If a "
        f"base rule looks wrong, leave it and say so in the report instead of 'fixing' it.\n\n"
        f"SIZE BUDGET — enforced: `{out_path}` may not exceed {base_lines + budget_lines} lines "
        f"(base {base_lines} + {budget_lines}). If adopting everything worthwhile would exceed that, "
        f"first consolidate or delete existing overlapping mechanisms to make room. Check: wc -l {out_path}\n\n"
        f"FINALLY write `{report_path}` — one line per mechanism considered: "
        f"`ADOPT|REJECT|MERGED  <client k>  <mechanism>  <evidence, or why not>`. Be honest, including "
        f"about what you dropped for budget rather than for evidence.\n\n"
        f"Leave the final merged harness at EXACTLY `{out_path}`, and verify it imports (command above) "
        f"before you stop."
    )
    claude_wrapper.run(prompt=prompt, model=model, allowed_tools=MERGER_TOOLS,
                       cwd=str(adapter.agents_dir.parent.parent),
                       log_dir=str(Path(run_dir) / "claude_sessions"), name=f"merge_{out_name}",
                       system_prompt=MERGER_SYS, timeout_seconds=timeout, progress=False)
    return out_path.exists()


def loadable(adapter, name, sample_item):
    try:
        adapter.load_harness(name, sample_item)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"   [merge] {name} not loadable: {type(exc).__name__}: {exc}", flush=True)
        return False
