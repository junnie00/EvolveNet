"""Scoped merged harness: value probing + error/empty retry + output alignment + per-DB rules.

Consolidates mechanisms from 5 client evolutions of fedroute_main1_r1.
GLOBAL: output alignment, SQLite compatibility, execution-error retry,
        empty-result retry, question-shape guidance.
HOME-SCOPED: column-value probes for card_games, california_schools;
             output-category rules for toxicology.
"""
from ..harness_base import SQLHarness
from .. import bridge
import re

SYS = (
    "You are an expert Text-to-SQL system for SQLite. Read the schema carefully and output exactly "
    "one SQLite query that answers the question, inside a ```sql ... ``` block. "
    "Return ONLY the columns the question explicitly asks for — do NOT add extra columns "
    "such as sort keys, identifiers, or diagnostic labels that were not requested. "
    "If a column is only used for ORDER BY, do NOT include it in SELECT. "
    "If the question asks for a single value (e.g. a count, a date, a name) return only that."
)

# Home-scoped column probes: {db_id: [(table, column), ...]}
HOME_PROBES = {
    "card_games": [("legalities", "status")],
    "california_schools": [
        ("schools", "StatusType"), ("schools", "EILCode"),
        ("schools", "EdOpsCode"), ("schools", "FundingType"),
        ("frpm", "NSLP Provision Status"), ("schools", "County"),
        ("satscores", "rtype"),
    ],
}

TOXI_RULES = (
    "\n--- toxicology rules ---\n"
    "- molecule.label is ONLY '+' (carcinogenic) or '-' (non-carcinogenic),\n"
    "  NEVER a molecule ID. molecule_id is the identifier.\n"
    "- Category questions ('the bond type', 'the elements'):\n"
    "  return DISTINCT values in one single column.\n"
    "- 'X and their Y' / 'X and Y': return both columns.\n"
    "- 'What is the' (singular) -> exactly ONE row (use LIMIT 1).\n"
    "- 'Identify the X' -> return ONLY the X identifier column.\n"
    "- Bond element questions -> return element values as flat\n"
    "  DISTINCT rows, not atom1/atom2 paired columns.\n"
    "- 'List'/'enumerate' -> individual rows, not GROUP_CONCAT.\n"
    "- GROUP_CONCAT(DISTINCT col) must NOT take a separator argument."
)


def _extract_hint(q: str) -> tuple:
    """Split 'question\\nHint: hint' into (question_part, hint_part)."""
    if "\nHint:" not in q:
        return q.strip(), ""
    p = q.split("\nHint:", 1)
    return p[0].strip(), p[1].strip()


class ScopedHarness(SQLHarness):
    def solve(self, question: str) -> str:
        db_id = self.db.db_id

        # -- Value grounding (parsed from embedded hint, if any) -------
        hint_str = _extract_hint(question)[1]  # second element is hint text
        notes = []
        for tbl, col in HOME_PROBES.get(db_id, []):
            vals = self.distinct(tbl, col, limit=25)
            if vals:
                notes.append(f"  {tbl}.{col}: {vals[:12]}")

        # Hint-based consistency check (toxicology only)
        if hint_str and db_id == "toxicology":
            for seg in hint_str.replace(";", ",").split(","):
                seg = seg.strip()
                if "= '" not in seg:
                    continue
                i = seg.index("= '")
                cn = seg[:i].strip().split()[-1]
                hv = seg[i + 3:].split("'")[0] if "'" in seg[i + 3:] else ""
                if not cn or not hv:
                    continue
                for tn in self.tables():
                    av = self.distinct(tn, cn, limit=20)
                    if av and hv not in av:
                        notes.append(
                            f"  Hint says {cn}='{hv}' but actual values"
                            f" in {tn}.{cn}: {av[:10]}"
                        )

        vb = ""
        if notes:
            vb = "Actual stored values (use EXACT strings in filters):\n" + "\n".join(notes)

        # Output-shape guidance from question text (before Hint line)
        _q = question.split("\nHint:")[0] if "\nHint:" in question else question
        q_text = _q.lower()
        s_hints = []
        # Use word-boundary checks to avoid false substring matches (e.g. "rate" in "illustrated")
        if re.search(r'\b(how many|total number|count of|count the)\b', q_text):
            s_hints.append("Return a single count - one column, one row.")
        if re.search(r'\b(ratio|rate|percentage|proportion)\b', q_text):
            s_hints.append("Return ONLY the calculated ratio/rate - no extra columns.")
        if re.search(r'\b(revenue|paid|spend(ing)?|expense|cost|price)\b', q_text):
            s_hints.append("For spending/revenue use SUM(Amount * Price) not SUM(Price).")

        # -- Prompt assembly -------------------------------------------
        # Embed hint back into question text (matches client 0's proven pattern)
        parts = [f"Database schema:\n{self.schema}"]
        if vb:
            parts.append(vb)
        parts.append(f"Question: {question}")
        if s_hints:
            parts.append("---\n" + "\n".join(s_hints))
        if db_id == "toxicology":
            parts.append(TOXI_RULES)
        parts.append("\nWrite the SQLite query.")
        prompt = "\n\n".join(parts)

        # -- Execute-validate-retry loop --------------------------------
        last_sql = ""
        for attempt in range(3):
            resp = self.llm(prompt, system=SYS, temperature=0.0)
            sql = bridge.extract_sql(resp)
            if not sql:
                continue
            last_sql = sql
            result = self.execute(sql)

            if not result.get("ok"):
                prompt = (
                    f"Previous SQL errored:\n```sql\n{sql}\n```\n"
                    f"Error: {result.get('error', 'unknown')}\n\n"
                    f"Fix the error. Schema:\n{self.schema}\n"
                    + (f"Value hints:\n{vb}\n" if vb else "")
                    + f"Question: {question}\n\nWrite corrected SQLite query."
                )
                continue

            if result.get("rows") is not None and len(result["rows"]) == 0:
                prompt = (
                    f"Previous SQL returned 0 rows:\n```sql\n{sql}\n```\n\n"
                    f"The query executed successfully but returned NO data. This usually means:\n"
                    f"(1) A WHERE filter value does not exist in the column you used.\n"
                    f"    Try probing which table actually contains that value.\n"
                    f"(2) JOIN conditions are wrong or produce no matches.\n"
                    f"(3) A column name is wrong or belongs to a different table.\n\n"
                    f"IMPORTANT: If a filter value comes from the Hint, check which table's\n"
                    f"column actually contains that literal. Do NOT assume it is in the\n"
                    f"first table that has a matching column name.\n\n"
                    f"Schema:\n{self.schema}\n"
                    + (f"Value hints:\n{vb}\n" if vb else "")
                    + f"Question: {question}\n\nWrite corrected SQLite query."
                )
                continue

            return sql

        return last_sql
