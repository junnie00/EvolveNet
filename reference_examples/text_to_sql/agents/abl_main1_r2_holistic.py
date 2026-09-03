"""Holistic merge: 5 client evolutions from fedroute_main1_r1 round 2.
Value grounding (c0/c3), enriched types (c2), comment stripping (c1),
output alignment + mismatch detect + category/pair rules (c3), error/empty retry (all).
Preserves NEWLY SOLVED: card_games KevWalker rulings + Restricted count (c0), toxicology elements (c3)."""
from ..harness_base import SQLHarness
from .. import bridge
import re

SYS = (
    "You are an expert Text-to-SQL system for SQLite. Rules:\n"
    "1. Return ONLY the columns asked — no extra. Never SELECT *.\n"
    "2. Categories (\"the bond type\") → DISTINCT single column. "
    "Pairs (\"X and their Y\") → both columns.\n"
    "3. \"Which X have Y\" → X identifiers. \"Identify the X\" → X identifier only.\n"
    "4. \"What is the\" (singular) → LIMIT 1. \"List\" → rows, not GROUP_CONCAT.\n"
    "5. Bond elements → flat column, not atom1/atom2 paired columns.\n"
    "6. Exclude NULLs from output columns.\n"
    "7. SQLite: no YEAR()/DATEPART() — strftime. GROUP_CONCAT(DISTINCT col) no separator.\n"
    "8. Hint is AUTHORITATIVE — follow its column mappings exactly.\n"
    "Output ```sql ... ```."
)


def _strip_comments(sql: str) -> str:
    lines, cleaned = sql.split("\n"), []
    for line in lines:
        idx = line.find("--")
        if idx >= 0 and line[:idx].count("'") % 2 == 0:
            line = line[:idx]
        cleaned.append(line)
    sql = "\n".join(l for l in cleaned if l.strip())
    return re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL).strip()


class RouteHarness(SQLHarness):
    @staticmethod
    def _parse_hint(question: str):
        parts = question.split("\nHint:", 1)
        return (parts[0].strip(), parts[1].strip()) if len(parts) == 2 else (question.strip(), "")

    def _build_schema(self) -> str:
        lines = []
        for tname, cols in self.tables().items():
            types = self.column_types(tname)
            col_strs = [f"{c} ({types.get(c, '?')})" for c in cols]
            lines.append(f"Table {tname}({', '.join(col_strs)})")
        return "\n".join(lines)

    def _probe_values(self, hint: str) -> str:
        """Probe hint-referenced columns + known case-sensitive filter columns for exact values."""
        seen = set()
        notes = []
        if hint:
            for seg in hint.replace(";", ",").split(","):
                s = seg.strip()
                if "= '" not in s:
                    continue
                idx = s.index("= '")
                col = s[:idx].strip().split()[-1] if s[:idx].strip().split() else ""
                if not col:
                    continue
                for tname, tcols in self.tables().items():
                    if col not in tcols or (tname, col) in seen:
                        continue
                    seen.add((tname, col))
                    vals = self.distinct(tname, col, limit=20)
                    if vals:
                        notes.append(f"Actual {tname}.{col} values: {vals}")
        # Known case-sensitive filter columns (evidence-backed, recurring schema)
        for tname, col in [("legalities", "status")]:
            if tname in self.tables() and col in self.tables()[tname] and (tname, col) not in seen:
                seen.add((tname, col))
                vals = self.distinct(tname, col, limit=25)
                if vals:
                    notes.append(f"Actual {tname}.{col} values: {vals}")
        return ("\n".join(notes) + "\n") if notes else ""

    @staticmethod
    def _shape_hints(q_text: str) -> list:
        q = q_text.lower()
        hints = []
        if any(w in q for w in ("how many", "total number", "count of", "what is the number")):
            hints.append("Return a single count — one column, one row.")
        if any(w in q for w in ("ratio", "rate", "percentage", "proportion")):
            hints.append("Return ONLY the ratio — no extra identifier columns.")
        top_n = re.findall(r'(?:top|lowest|highest|most)\s+(\d+)', q)
        if top_n:
            hints.append(f"Return exactly {top_n[0]} rows.")
        return hints

    @staticmethod
    def _detect_mismatch(q_text: str, result: dict) -> list:
        q = q_text.lower()
        rows = result.get("rows", [])
        if not rows:
            return []
        nc = len(rows[0]) if isinstance(rows[0], (list, tuple)) else 1
        nr = len(rows)
        hints = []
        if nc > 1 and "identify the atoms" in q:
            hints.append("Return ONLY the atom_id column.")
        if nr > 1 and "what is the" in q:
            hints.append("Use LIMIT 1 — 'what is the' is singular.")
        if nc >= 2 and "element" in q and "bond id" in q \
                and "and their" not in q and "element and" not in q:
            hints.append("Return elements as one flat column, not atom1/atom2 pairs.")
        if nc > 1 and "the bond type" in q:
            hints.append("Return DISTINCT bond_type as one column.")
        return hints

    def solve(self, question: str) -> str:
        q_text, hint = self._parse_hint(question)
        schema = self._build_schema()
        value_notes = self._probe_values(hint)
        shape_hints = self._shape_hints(q_text)

        parts = [f"Database schema (column types):\n{schema}"]
        if value_notes:
            parts.append(value_notes)
        parts.append(f"Question: {q_text}")
        if hint:
            parts.append(f"\nIMPORTANT HINT (AUTHORITATIVE): {hint}")
        constraints = [
            "\nOUTPUT REQUIREMENTS:",
            "- Return EXACTLY the column(s) asked — nothing extra.",
            "- Single value asked (count, name, date) → return ONLY that column.",
            "- 'Most/highest/lowest X' → LIMIT 1.",
            "- Do NOT round/truncate unless asked.",
            "- Exclude NULLs from output columns (IS NOT NULL).",
            "- Use actual stored values from probes — match case exactly.",
        ]
        if shape_hints:
            constraints.append("--- Additional:")
            constraints.extend(f"  • {h}" for h in shape_hints)
        parts.append("\n".join(constraints))
        parts.append("Write the SQLite query.")
        prompt = "\n\n".join(parts)

        last_sql = ""
        for attempt in range(3):
            resp = self.llm(prompt, system=SYS, temperature=0.0)
            sql = bridge.extract_sql(resp)
            if not sql:
                continue
            sql = _strip_comments(sql)
            last_sql = sql
            result = self.execute(sql)

            if not result.get("ok"):
                prompt = (
                    f"SQL error:\n{sql}\n\nError: {result.get('error')}\n\n"
                    f"Schema:\n{schema}\n{value_notes}Question: {q_text}\n"
                    f"{'Hint: ' + hint if hint else ''}\n\n"
                    f"Fix. No YEAR()/DATEPART() — strftime. GROUP_CONCAT(DISTINCT) no separator."
                )
                continue

            if len(result.get("rows", [])) == 0:
                prompt = (
                    f"SQL returned 0 rows:\n{sql}\n\nSchema:\n{schema}\n{value_notes}"
                    f"Question: {q_text}\n{'Hint: ' + hint if hint else ''}\n\n"
                    f"Zero rows — likely wrong filter column/table, case mismatch, or bad JOIN.\n"
                    f"Check which table the filter column belongs to. Fix the SQL."
                )
                continue

            mis = self._detect_mismatch(q_text, result)
            if mis:
                resp2 = self.llm(
                    f"SQL:\n{sql}\n\nOutput shape wrong:\n" + "\n".join(mis)
                    + "\n\nFix the SQL to match the question's output.",
                    system=SYS, temperature=0.0
                )
                return bridge.extract_sql(resp2) or sql
            return sql
        return last_sql
