"""Conservative repair of bare harness:
- Hint extraction + prominent AUTHORITATIVE display
- Output alignment emphasis in system prompt
- SQL comment stripping before execution (fixes Q24-style CTE comment errors)
- Execution error retry (fixes Q12-style GROUP_CONCAT DISTINCT separator errors)
- Empty-result retry with value probing (fixes Q22/Q32/Q33-style value mismatch)
- Value grounding via self.distinct() on columns mentioned in the hint
- On empty-result retry, probes all referenced tables for actual stored text values

Preserves every bare behavior that has sound trace evidence. Adds minimal general
mechanisms that fix concrete failures."""
from ..harness_base import SQLHarness
from .. import bridge

SYS = (
    "You are an expert Text-to-SQL system for SQLite. Read the schema carefully and output exactly "
    "one SQLite query that answers the question, inside a ```sql ... ``` block.\n\n"
    "STRICT OUTPUT RULES:\n"
    "- Return EXACTLY the columns the question asks for — no extra columns.\n"
    "- If the question asks for a single value (the most/highest/lowest, a count, a name, etc), "
    "return only that value/column, not additional identifiers or sorting columns.\n"
    "- Use ORDER BY + LIMIT 1 when the question asks for 'the most' / 'the highest' / 'the lowest' / "
    "'which' referring to one specific thing.\n"
    "- Preserve full numeric precision — do NOT round or truncate unless the question explicitly says to.\n"
    "- The HINT below is AUTHORITATIVE — follow it exactly, never override it.\n"
    "- SQLite does NOT support GROUP_CONCAT(DISTINCT x, separator) — remove the separator argument "
    "when using DISTINCT: use GROUP_CONCAT(DISTINCT x) without a separator."
)

import re


def _strip_sql_comments(sql: str) -> str:
    """Remove SQL single-line (--) and multi-line (/* */) comments."""
    # Remove multi-line comments
    sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
    # Remove single-line comments (but not in string literals)
    lines = sql.split('\n')
    result = []
    for line in lines:
        in_single = False
        in_double = False
        comment_pos = -1
        i = 0
        while i < len(line):
            if line[i] == "'" and not in_double:
                in_single = not in_single
            elif line[i] == '"' and not in_single:
                in_double = not in_double
            elif line[i] == '-' and i + 1 < len(line) and line[i + 1] == '-':
                if not in_single and not in_double:
                    comment_pos = i
                    break
            i += 1
        if comment_pos >= 0:
            line = line[:comment_pos]
        result.append(line)
    return '\n'.join(result)


def _extract_hint(question: str):
    """Split question into (question_text, hint_text)."""
    if "\nHint:" in question:
        qtext, _, hint = question.partition("\nHint:")
        return qtext.strip(), hint.strip()
    return question.strip(), ""


def _extract_table_names(sql: str):
    """Extract candidate table names from SQL using simple regex."""
    names = set()
    for m in re.finditer(r'\b(?:FROM|JOIN|UPDATE|INTO|TABLE)\s+["`]?(\w+)["`]?', sql, re.IGNORECASE):
        names.add(m.group(1))
    for m in re.finditer(r'\b(\w+)\s*\.', sql):
        names.add(m.group(1))
    return names


class BareHarness(SQLHarness):
    MAX_RETRIES = 2
    MAX_EMPTY_PROBES = 15  # limit DB probes on empty-result retry

    def _probe_hint_values(self, hint: str) -> str:
        """Scan the hint for ``column = 'value'`` patterns using known column names
        from the DB schema. When the hinted value doesn't match any stored value,
        report the actual distinct values to the LLM.
        Returns a context block or empty string."""
        snippets = []
        tables_map = self.tables()
        for table, cols in tables_map.items():
            for c in cols:
                # Column names can have spaces, parens etc. Build variants.
                name_variants = [c]
                name_variants.append(c.replace(" ", ""))
                name_variants.append(c.replace("(", "").replace(")", "").replace(" ", ""))
                # Also try individual words from multi-word column names
                for word in re.split(r'[\s()]+', c):
                    if len(word) >= 3 and word not in name_variants:
                        name_variants.append(word)

                for variant in name_variants:
                    # Look for: variant = 'value' or variant = number
                    escaped = re.escape(variant)
                    m = re.search(
                        escaped + r'\s*=\s*(' + "'" + r"[^']*" + "'" + r'|\d+)',
                        hint,
                        re.IGNORECASE,
                    )
                    if m:
                        raw_val = m.group(1).strip("'")
                        # Skip pure numeric assignments with non-string values
                        distinct_vals = self.distinct(table, c, limit=10)
                        if distinct_vals:
                            # Check case-insensitive first, then exact case
                            exact_match = any(str(v) == raw_val for v in distinct_vals if v is not None)
                            ci_match = any(str(v).lower() == raw_val.lower()
                                           for v in distinct_vals if v is not None)
                            if not exact_match:
                                displayed = [repr(v) for v in distinct_vals[:8] if v is not None]
                                if ci_match:
                                    snippets.append(
                                        f"  {table}.{c} stores: {', '.join(displayed)} "
                                        f"(note: '{raw_val}' has wrong case — SQLite uses case-sensitive =)"
                                    )
                                else:
                                    snippets.append(
                                        f"  {table}.{c} stores: {', '.join(displayed)} "
                                        f"(note: '{raw_val}' was not found — use one of the stored values)"
                                    )
                        break  # Found a match for this column; move to next column
        if not snippets:
            return ""
        return ("Actual database values (use these instead of guessed literals):\n"
                + "\n".join(snippets))

    def _probe_empty_result_context(self, sql: str) -> str:
        """When a query returns empty, probe all referenced tables for their
        text-column distinct values to help the LLM fix value mismatches."""
        tbl_names = _extract_table_names(sql)
        if not tbl_names:
            return ""
        tables_map = self.tables()
        lines = []
        probe_count = 0
        for tbl in sorted(tbl_names):
            if tbl not in tables_map:
                continue
            cols = tables_map[tbl]
            for c in cols:
                if probe_count >= self.MAX_EMPTY_PROBES:
                    break
                probe = self.distinct(tbl, c, limit=8)
                if not probe:
                    continue
                probe_count += 1
                # Show only if it's a text column with a few distinct values
                has_mixed = any(len(str(v)) > 40 for v in probe)
                is_numeric = all(
                    isinstance(v, (int, float)) or (isinstance(v, str) and v.replace('.', '', 1).isdigit())
                    for v in probe if v is not None
                )
                if len(probe) <= 20 and not is_numeric and not has_mixed and len(probe) > 1:
                    displayed = [repr(v) for v in probe if v is not None]
                    lines.append(f"  {tbl}.{c}: {', '.join(displayed)}")
            if probe_count >= self.MAX_EMPTY_PROBES:
                break
        if not lines:
            return ""
        return ("Distinct text values in referenced tables (check your WHERE clause literals):\n"
                + "\n".join(lines))

    def _build_prompt(self, qtext, hint, value_context="", history=""):
        prompt = f"Database schema:\n{self.schema}\n\nQuestion: {qtext}"
        if hint:
            prompt += f"\n\nAUTHORITATIVE HINT (must follow exactly):\n{hint}"
        if value_context:
            prompt += f"\n\n{value_context}"
        prompt += "\n\nWrite the SQLite query."
        if history:
            prompt += f"\n\n### PREVIOUS ATTEMPT FEEDBACK\n{history}\n\nPlease write a corrected query."
        return prompt

    def solve(self, question: str) -> str:
        qtext, hint = _extract_hint(question)

        # Probe the DB for actual values of columns mentioned in the hint
        value_context = self._probe_hint_values(hint) if hint else ""

        # First attempt
        prompt = self._build_prompt(qtext, hint, value_context)
        resp = self.llm(prompt, system=SYS, temperature=0.0)
        sql = bridge.extract_sql(resp)

        if not sql:
            return ""

        cleaned_sql = _strip_sql_comments(sql)

        for attempt in range(self.MAX_RETRIES + 1):
            result = self.execute(cleaned_sql)

            if result.get("ok") and result.get("rows"):
                return cleaned_sql

            # Build retry context
            history_parts = []
            if not result.get("ok"):
                history_parts.append(f"SQL execution error: {result.get('error', 'unknown')}")
                history_parts.append("Fix the SQL syntax. Avoid comments inside the SQL.")
            elif not result.get("rows"):
                history_parts.append(
                    "The query executed successfully but returned 0 rows (empty result). "
                    "The WHERE clause values may not match the database exactly."
                )
                # Probe tables for actual text values
                table_context = self._probe_empty_result_context(cleaned_sql)
                if table_context:
                    history_parts.append(table_context)

            if attempt < self.MAX_RETRIES:
                prompt = self._build_prompt(qtext, hint, value_context,
                                            "\n".join(history_parts))
                resp = self.llm(prompt, system=SYS, temperature=0.0)
                cleaned_sql = _strip_sql_comments(bridge.extract_sql(resp))
                if not cleaned_sql:
                    return ""
            else:
                return cleaned_sql

        return cleaned_sql
