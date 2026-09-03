"""Merge of G0 harness improvements: enriched schema, value probing, output alignment, retry.

Derived from cand_main1_r0_c2_b0r0_g0 (column types, SQLite compat, output shape),
cand_main1_r0_c3_b0r0_g0 (hint value probing, entity-to-PK mapping, empty-result retry),
cand_main1_r0_c4_b0r0_g0 (date-format probing, execute-validate-retry loop).
Preserves base SYS with appended rules."""
from ..harness_base import SQLHarness
from .. import bridge

SYS = ("You are an expert Text-to-SQL system for SQLite. Read the schema carefully and output exactly "
       "one SQLite query that answers the question, inside a ```sql ... ``` block."
       "\n\nCRITICAL — SQLite compatibility:\n"
       "- YEAR() does NOT exist. Extract year with: CAST(strftime('%Y', date_column) AS INTEGER)\n"
       "- DATEPART() does NOT exist. Use strftime for all date extraction.\n\n"
       "OUTPUT PRECISION — return EXACTLY what the question asks for:\n"
       "- Include ONLY the columns mentioned in the question. Do NOT add extra columns.\n"
       "- When the question gives a specific entity ID (e.g. 'TR012 molecule'), filter on the\n"
       "  table's primary-key / ID column (e.g. molecule_id), NOT on string columns like label.\n"
       "- If the question asks for a single value ('the longest', 'the fastest', 'how much'),\n"
       "  return only that value column, not additional identifying columns.\n"
       "- When the Hint maps a concept to comma-separated columns (e.g. 'full name refers to\n"
       "  first_name, last_name'), keep those columns as SEPARATE SELECT items.\n"
       "- Use DISTINCT when the question asks for a list or set of items.\n"
       "- For GROUP_CONCAT with DISTINCT, do NOT pass a separator argument (SQLite limitation).\n"
       "- Do NOT round or truncate numeric results unless the question explicitly says to.\n"
       "- Use explicit JOINs with ON conditions. Do NOT use implicit joins.\n")

MAX_RETRIES = 2


class BareHarness(SQLHarness):

    def solve(self, question: str) -> str:
        q_text, hint = self._parse_question_hint(question)

        # Build enriched schema with column types from live DB introspection
        schema_lines = []
        for tname, cols in self.tables().items():
            types = self.column_types(tname)
            col_strs = [f"{c} ({types.get(c, '?')})" for c in cols]
            schema_lines.append(f"Table {tname}({', '.join(col_strs)})")
        enriched_schema = "\n".join(schema_lines)

        # Probe DB for value-guidance hints
        hint_notes = self._probe_hint_values(hint)
        date_notes = self._probe_date_formats()

        # Build initial prompt
        parts = [f"Database schema (column types in parentheses):\n{enriched_schema}",
                 f"Question: {q_text}"]
        if hint:
            parts.append(f"Hint: {hint}")
        if hint_notes:
            parts.append("--- Value notes ---\n" + "\n".join(hint_notes))
        if date_notes:
            parts.append("--- Date/time column samples ---\n" + "\n".join(date_notes))
        parts.append("Write the SQLite query.")
        prompt = "\n\n".join(parts)

        last_sql = ""
        for attempt in range(MAX_RETRIES + 1):
            resp = self.llm(prompt, system=SYS, temperature=0.0)
            sql = bridge.extract_sql(resp)
            if not sql:
                return ""
            last_sql = sql

            result = self.execute(sql)
            if result.get("ok") and len(result.get("rows", [])) > 0:
                return sql

            if not result.get("ok"):
                prompt = (f"Your SQL had an execution error:\n{result.get('error')}\n\n"
                          f"SQL:\n{sql}\n\nFix the SQL. Remember SQLite rules.\n\n"
                          f"Schema:\n{enriched_schema}\n\n"
                          f"Question: {q_text}\n"
                          f"{'Hint: ' + hint if hint else ''}\n"
                          f"Write the corrected SQLite query.")
            else:
                prompt = (f"Your SQL returned 0 rows:\n{sql}\n\n"
                          f"This likely means a filter column was mapped incorrectly. "
                          f"Use ID columns for entity filters, not label/name columns.\n\n"
                          f"Schema:\n{enriched_schema}\n\n"
                          f"Question: {q_text}\n"
                          f"{'Hint: ' + hint if hint else ''}\n"
                          f"Write the corrected SQLite query.")

        return last_sql

    # -----------------------------------------------------------------
    # helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _parse_question_hint(q: str):
        marker = "\nHint:"
        if marker in q:
            parts = q.split(marker, 1)
            return parts[0].strip(), parts[1].strip()
        return q, ""

    def _probe_hint_values(self, hint: str):
        """Parse column='value' patterns from hint, probe DB for stored-value mismatches."""
        notes = []
        if not hint:
            return notes
        for segment in hint.replace(";", ",").split(","):
            segment = segment.strip()
            if "= '" not in segment:
                continue
            idx = segment.index("= '")
            before = segment[:idx].strip()
            after = segment[idx + 2:].strip()
            if not after.startswith("'"):
                continue
            val = after[1:].split("'")[0] if "'" in after[1:] else ""
            col = before.split()[-1] if before.split() else ""
            if not val or not col:
                continue
            for tname, tcols in self.tables().items():
                if col not in tcols:
                    continue
                try:
                    actual = self.distinct(tname, col, limit=20)
                    if actual and val not in actual:
                        notes.append(
                            f"Note: Hint says {col}='{val}', but actual values "
                            f"in {tname}.{col} are: "
                            + ", ".join(repr(v) for v in actual)
                        )
                except Exception:
                    pass
        return notes

    def _probe_date_formats(self):
        """Sample date/time columns to reveal stored format."""
        hints = []
        for tname, cols in self.tables().items():
            for c in cols:
                if any(kw in c.lower() for kw in ("date", "time")):
                    try:
                        vals = self.distinct(tname, c, limit=8)
                        if vals:
                            hints.append(f"  {tname}.{c} samples: {vals}")
                    except Exception:
                        pass
        return hints
