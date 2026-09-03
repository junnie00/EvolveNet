"""G4 — value-grounding + execute-validate-retry + output-shape enforcement + column-type enrichment.

Strategy
--------
Three interacting mechanisms target the dominant failure modes observed across 36 incorrect
traces:

1. VALUE GROUNDING — `col = 'val'` patterns in the hint probe the live DB via self.distinct().
   If the hinted value does not appear in the actual column, the probe row explicitly lists the
   values that DO exist. This catches:
   - wrong spacing     (hint says bond_type = ' = ' but DB has '=')
   - wrong casing      (hint says status = 'restricted' but DB has 'Restricted')
   - wrong abbreviation (hint says 'Direct' but DB has 'Directly funded')
   - wrong literal     (hint says NSLP Provision Status = '2' but DB has 'Lunch Provision 2')
   See ├─ traces q22, q32, q33, q70, q73, q80.

2. EXECUTE-VALIDATE-RETRY — the generated SQL is immediately executed. On SQL error (not empty
   result, only a real error) the harness retries with the error message as feedback.
   Catches GROUP_CONCAT(DISTINCT …) misuse, missing year() function, CTE syntax issues.
   See ├─ traces q12, q56, q65.

3. OUTPUT-SHAPE + DATE-FORMAT ENFORCEMENT — the prompt warns never to add extra columns,
   and column types + sample values for any column named "date"/"Date" are injected into the
   enriched context so the coder sees the real format (YYYYMM vs YYYY-MM-DD).
   See ├─ traces q0, q26, q55, q69, q80, q82.
"""
from ..harness_base import SQLHarness
from .. import bridge

SYS = ("You are an expert Text-to-SQL system for SQLite. Follow the Hint exactly. "
       "Read the schema and the DB values carefully.")

OUTPUT_RULE = (
    "CRITICAL OUTPUT RULE: Return EXACTLY the columns the question asks for and NOTHING else. "
    "Never add extra columns (IDs, computed values, sort keys). "
    "If the question asks for a single column, return only that column. "
    "Do not round or truncate numbers. "
    "If a superlative (\"the most\", \"the highest\", \"the oldest\") is asked, return ONE row — not all ties."
)


class CandCentralpopR0B0R0G4Harness(SQLHarness):
    """Harness with value grounding, execute-validate-retry, and output-shape enforcement."""

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def solve(self, question: str) -> str:
        q_text, hint = self._split_hint(question)
        enriched_schema = self._enriched_schema()
        value_context = self._ground_hint_values(hint)
        date_context = self._probe_date_columns()
        sql = self._attempt_with_retry(q_text, hint, enriched_schema,
                                       value_context, date_context)
        return sql

    # ------------------------------------------------------------------
    # Hint parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _split_hint(question: str):
        """The question string may contain ``\\nHint: ...`` appended."""
        if "\nHint:" in question:
            parts = question.split("\nHint:", 1)
            return parts[0].strip(), parts[1].strip()
        return question.strip(), ""

    # ------------------------------------------------------------------
    # Enriched schema with column types
    # ------------------------------------------------------------------
    def _enriched_schema(self) -> str:
        """Return schema lines with TYPE annotations appended."""
        lines = []
        for table_name, cols in self.tables().items():
            types = self.column_types(table_name)
            annotated = []
            for c in cols:
                t = types.get(c, "")
                annotated.append(f"{c} [{t}]" if t else c)
            lines.append(f"Table {table_name}({', '.join(annotated)})")
        # FK info — reuse self.schema for FK lines
        for line in self.schema.splitlines():
            if line.startswith("FK "):
                lines.append(line)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Value grounding — probe `col='val'` patterns from the hint
    # ------------------------------------------------------------------
    def _ground_hint_values(self, hint: str) -> str:
        """Find ``col = 'value'`` patterns in the hint and verify against the live DB."""
        if not hint:
            return ""

        fragments = []
        # Normalise separators so each fragment is a short statement
        for sep in (";", ".", ","):
            hint = hint.replace(sep, "\n")
        lines = [ln.strip() for ln in hint.split("\n") if ln.strip()]

        tables_info = self.tables()

        for line in lines:
            pairs = self._extract_col_value(line)
            for col_name, hinted_val in pairs:
                for table_name, cols in tables_info.items():
                    if col_name not in cols:
                        continue
                    actual = self.distinct(table_name, col_name, limit=25)
                    if not actual:
                        continue
                    # CASE-SENSITIVE check: the hinted value *as a complete token*
                    if hinted_val not in actual:
                        # Try case-insensitive match to give a better hint
                        close = [v for v in actual if v.lower() == hinted_val.lower()]
                        note = f"[PROBE] {table_name}.{col_name} actual values: {actual}"
                        if close:
                            note += f"  ← hint says '{hinted_val}'; closest by case: '{close[0]}'"
                        else:
                            note += f"  ← hint says '{hinted_val}' NOT found"
                        fragments.append(note)
                    break  # found the table, no need to check others
        return "\n".join(fragments)

    @staticmethod
    def _extract_col_value(line: str):
        """From a line like '… refers to artist = \\'Jim Pavelec\\'' yield (col, val)."""
        if "'" not in line or "=" not in line:
            return []
        idx = line.index("=")
        col_part = line[:idx].strip()
        val_part = line[idx + 1:].strip()
        if not col_part:
            return []
        col_name = col_part.split()[-1]  # last word before =
        if val_part.startswith("'") and val_part.endswith("'"):
            return [(col_name, val_part[1:-1])]
        # also handle unquoted numeric values   isForeignOnly = 1
        stripped = val_part.strip()
        if stripped.isdigit() or (stripped.startswith("-") and stripped[1:].isdigit()):
            return [(col_name, stripped)]
        return []

    # ------------------------------------------------------------------
    # Date-format probe — sample values for columns whose name contains "date"
    # ------------------------------------------------------------------
    def _probe_date_columns(self) -> str:
        """Sample distinct values from any column with 'date' in its name (case-insensitive)."""
        hints = []
        for table_name, cols in self.tables().items():
            for c in cols:
                if "date" not in c.lower():
                    continue
                vals = self.distinct(table_name, c, limit=5)
                if vals:
                    hints.append(f"[FORMAT] {table_name}.{c} sample: {vals}")
        return "\n".join(hints)

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------
    def _build_prompt(self, q_text: str, hint: str,
                      enriched_schema: str,
                      value_context: str,
                      date_context: str,
                      feedback: str = "",
                      attempt: int = 0) -> str:
        parts = [f"Database schema (with types):\n{enriched_schema}"]
        if hint:
            parts.append(f"HINT (AUTHORITATIVE — follow exactly): {hint}")
        if value_context:
            parts.append(
                "Value verification (hinted values checked against real DB — note any discrepancies!):\n"
                + value_context)
        if date_context:
            parts.append(
                "Date column samples (use these to determine date format — do NOT guess the format):\n"
                + date_context)
        parts.append(f"Question: {q_text}")
        parts.append(OUTPUT_RULE)
        if feedback:
            parts.append(
                f"ATTEMPT {attempt + 1} — the previous SQL failed:\n{feedback}\nGenerate a corrected query.")
        parts.append("Write the SQLite query.")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Execute-validate-retry loop
    # ------------------------------------------------------------------
    def _attempt_with_retry(self, q_text: str, hint: str,
                            enriched_schema: str,
                            value_context: str,
                            date_context: str,
                            max_attempts: int = 3) -> str:
        last_sql = ""
        feedback = ""  # persists across iterations so attempt-1 error reaches attempt-2 prompt
        for attempt in range(max_attempts):
            prompt = self._build_prompt(q_text, hint, enriched_schema,
                                        value_context, date_context,
                                        feedback=feedback,
                                        attempt=attempt)
            resp = self.llm(prompt, system=SYS, temperature=0.0)
            sql = bridge.extract_sql(resp) or ""
            last_sql = sql
            if not sql:
                feedback = "No SQL found in response."
                continue
            result = self.execute(sql)
            if result.get("ok") is True:
                return sql
            err = result.get("error", "unknown error")
            feedback = f"SQL execution error: {err}"
        return last_sql
