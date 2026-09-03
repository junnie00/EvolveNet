import os
"""Audit a question slice for specification defects, following the project's own screening protocol.

Stage-2 of `logs/mini_dev500_bare_screen_v1/AUDIT_PROTOCOL.md`: a question is clean only if the gold
SQL is UNIQUELY determined by the question plus its Hint. Questions whose Hint contradicts the question,
whose gold silently drops a stated condition, or whose output shape is a free choice are rejected — a
solver cannot be expected to reproduce an arbitrary annotation decision.

This matters here because the training loop now reads gold (`--supervised`). Under the original
label-free protocol a defective gold only mis-scored a question offline; now it actively teaches the
proposer to chase an unreachable target.

Text heuristics do not work for this — a Hint glossary like "'cl' means Chlorine" looks exactly like a
Hint/question mismatch. So each item is judged by the model, one call per question, and every verdict
is written with its reason to a ledger in the same format the original screen used.
"""
import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import bridge

SYS = ("You audit Text-to-SQL benchmark items for SPECIFICATION DEFECTS. You are not judging whether the "
       "question is hard. You judge one thing: is the gold SQL uniquely determined by the question plus "
       "its Hint? Answer strictly in the requested JSON form.")

CRITERIA = """Your default is ACCEPT. Reject ONLY when a competent solver could not reach the gold
answer no matter how carefully it worked — not when the question is merely hard or under-specified in
some way a careful solver would resolve.

Reject if ANY of these hold:
1. The gold SQL fails to execute.
2. The Hint contradicts the question on a concrete VALUE, NUMBER, or NAMED ENTITY, so following the
   Hint and following the question give different answers. (Example: the question asks for mana cost
   10 while the Hint says 16. Example: the question names the set "Tenth Edition" while the Hint
   explains "Salvat 2011".)
3. The gold silently DROPS a condition the question states, or ADDS one it does not, so a solver that
   read the question correctly gets a different result. (Example: the question says "expansion
   commander type" but gold filters on 'commander' alone.)

Everything else is ACCEPT. In particular these are NOT defects — they are exactly the difficulties the
solver is supposed to overcome, and rejecting them removes the questions worth training on:

- CASE differences between the Hint and the stored data or gold (Hint says 'legal', gold uses 'Legal').
  SQLite string comparison is case-sensitive; discovering the stored form is the solver's job.
- Whether to use DISTINCT, when the question does not spell it out.
- Returning an id versus a name, a full name versus separate first/last columns, or any other
  reasonable projection choice.
- Row count, ordering, LIMIT or tie-breaking that the question implies rather than states.
- Needing an unstated join path, or schema knowledge a solver can obtain by inspecting the database.
- A Hint that supplies a glossary or a formula (e.g. "'cl' means Chlorine", "percentage = DIVIDE(a,b)").
- Any question that is simply hard, multi-step, or requires careful reading.

A useful test before rejecting: could a solver that probed the database and read the question and Hint
closely have produced the gold result? If yes — even if it would take real work — ACCEPT."""


def audit_one(rec, schema, gold_rows, gold_error):
    q, hint, sql = rec["question"], rec.get("evidence") or "(none)", rec["SQL"]
    prompt = (
        f"{CRITERIA}\n\n"
        f"DATABASE SCHEMA:\n{schema}\n\n"
        f"QUESTION: {q}\n\n"
        f"HINT (given to the solver, treated as authoritative): {hint}\n\n"
        f"GOLD SQL: {sql}\n\n"
        f"GOLD EXECUTION: {'ERROR: ' + str(gold_error) if gold_error else f'{len(gold_rows)} rows, sample {str(gold_rows[:3])[:300]}'}\n\n"
        'Reply with ONLY this JSON: {"verdict": "accept" | "reject", "criterion": <the number you '
        'applied, or 0 if accepting>, "reason": "<one or two sentences citing the specific mismatch>"}'
    )
    out = bridge.solver_llm(prompt, system=SYS, temperature=0.0)
    text = out if isinstance(out, str) else (out[0] if out else "")
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {"verdict": "accept", "criterion": 0, "reason": f"unparseable audit reply: {text[:150]}"}
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"verdict": "accept", "criterion": 0, "reason": f"unparseable audit JSON: {text[:150]}"}
    if d.get("verdict") not in ("accept", "reject"):
        d["verdict"] = "accept"
    return d


def audit_many(recs, workers=8):
    """recs: dev.json records. -> list of ledger entries, input order preserved."""
    schemas = {}

    def one(rec):
        db_id = rec["db_id"]
        if db_id not in schemas:
            db = bridge.get_db(db_id)
            schemas[db_id] = (db, bridge.get_db(db_id).schema_text())
        db, schema = schemas[db_id]
        res = bridge.execute(db, rec["SQL"])
        v = audit_one(rec, schema, res.get("rows") or [], None if res.get("ok") else res.get("error"))
        return {"question_id": rec["question_id"], "db_id": db_id,
                "difficulty": rec.get("difficulty"), "question": rec["question"],
                "verdict": v.get("verdict"), "criterion": v.get("criterion"),
                "reason": str(v.get("reason"))[:600]}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, recs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question-ids", required=True, help="json list of question_id to audit")
    ap.add_argument("--out", required=True, help="ledger jsonl path")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    dev = {d["question_id"]: d for d in json.loads(
        Path(os.environ.get("BENCH_DIR", "data/bird") + "/dev_20240627/dev.json").read_text(encoding="utf-8"))}
    qids = json.loads(Path(args.question_ids).read_text(encoding="utf-8"))
    entries = audit_many([dev[q] for q in qids], args.workers)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    n_rej = sum(e["verdict"] == "reject" for e in entries)
    print(f"audited {len(entries)}: accept {len(entries) - n_rej}, reject {n_rej}")
    for e in entries:
        if e["verdict"] == "reject":
            print(f"  REJECT qid={e['question_id']} [{e['db_id']}/{e['difficulty']}] crit={e['criterion']}")
            print(f"     {e['question'][:88]}")
            print(f"     {e['reason'][:200]}")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
