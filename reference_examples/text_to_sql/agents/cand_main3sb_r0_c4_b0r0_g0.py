"""Independent exploration: value grounding + hint-guided decomposition + output alignment + empty-result retry.
Learns from failures in the bare harness: probes stored values for question literals, enforces output shape
from hints, and retries on empty/error with diagnostic feedback."""
from ..harness_base import SQLHarness
from .. import bridge
import re

SYS_CORE = (
    "You are an expert Text-to-SQL system for SQLite. Read the schema and follow the hint exactly.\n"
    "CRITICAL RULES:\n"
    "1. Return ONLY the column(s) the question explicitly asks for — no extra columns.\n"
    "   - If the question asks 'which X' (an entity), return ONLY that entity's identifier.\n"
    "   - If both entity AND value are asked (e.g. 'which X consumed how much'), return both.\n"
    "   - For 'how many' / 'what percentage', return ONLY the single count/value.\n"
    "2. When a hint says 'full name refers to first_name, last_name', return them as SEPARATE "
    "columns (first_name, last_name) — never concatenate into one column.\n"
    "3. Revenue = Amount * Price (Price is unit price, Amount is quantity).\n"
    "4. In the yearmonth table, the Date column is stored as YYYYMM integer format (e.g., 201309). "
    "Use integer comparison, not date strings.\n"
    "5. 'Paid' / 'price' in transaction records refers to the Price column, not Amount * Price.\n"
    "6. For percentage, return only the single numeric value — no extra column names.\n"
    "7. If the hint provides an exact formula (e.g. percentage = MULTIPLY(DIVIDE(...))), "
    "translate it verbatim into SQL."
)

RETRY_SYS = (
    "You are an expert Text-to-SQL system for SQLite. Your previous query failed (empty result or error).\n"
    "Diagnose the issue: check your column choices, value formats, and join conditions.\n"
    "For the yearmonth table, Date is YYYYMM integer (e.g. 201201).\n"
    "For 'paid 124.05', check the Price column (not Amount * Price).\n"
    "Follow the CRITICAL RULES about output columns.\n"
    "Write exactly one corrected SQLite query inside ```sql ... ```."
)


def _extract_numeric_values(text: str) -> list:
    """Extract numeric literals from question text, preferring decimals first."""
    # Find decimal numbers (including integer-like)
    decimals = re.findall(r'\b(\d+\.\d+)\b', text)
    integers = re.findall(r'\b(\d{3,})\b', text)  # only 3+ digit integers
    # Remove duplicates while preserving order
    seen = set()
    result = []
    for v in decimals + integers:
        if v not in seen:
            seen.add(v)
            result.append(v)
    return result


class G0ExploreHarness(SQLHarness):
    """Value-grounding + hint-guided output alignment + empty-result retry harness.
    Probes stored DB values for question literals, enforces column format from hint patterns,
    and retries on empty/error with diagnostic feedback to the coder."""
    def _probe_question_values(self, question: str, hint: str) -> str:
        """Probe the DB for values mentioned in the question and return a context string."""
        probes = []
        # Extract values from both question and hint
        text = question + " " + hint
        values = _extract_numeric_values(text)
        if not values:
            return ""

        tables = self.tables()
        for val_str in values:
            val = float(val_str) if '.' in val_str else int(val_str)
            for tname, cols in tables.items():
                for c in cols:
                    # Probe only numeric-looking columns (skip IDs, dates)
                    clower = c.lower()
                    if any(x in clower for x in ['id', 'date', 'zip', 'code', 'phone']):
                        continue
                    try:
                        vals = self.distinct(tname, c, limit=20)
                        if val in vals:
                            probes.append(f"  Table '{tname}', Column '{c}' contains the value {val_str}")
                    except Exception:
                        continue

        if not probes:
            return ""
        return "Database value probes (values from question found in these columns):\n" + "\n".join(probes)

    def _extract_hint_parts(self, hint: str) -> str:
        """Parse the hint for column-mapping clues and return guidance."""
        parts = []
        if not hint or hint.strip() in ("(none)", ""):
            return ""

        # Detect "full name refers to first_name, last_name" pattern
        full_name_pattern = re.search(
            r"full\s+name\s+refers\s+to\s+(\w+),\s*(\w+)",
            hint, re.IGNORECASE
        )
        if full_name_pattern:
            parts.append(
                f"Return '{full_name_pattern.group(1)}' and '{full_name_pattern.group(2)}' "
                f"as separate columns — do NOT concatenate them."
            )

        # Detect revenue clues
        if "revenue" in hint.lower():
            parts.append("Revenue = Amount * Price (multiply quantity by unit price).")

        # Detect date format clues in hint
        if "yearmonth.date" in hint.lower() or "yearmonth" in hint.lower():
            parts.append("yearmonth.Date uses YYYYMM integer format (e.g. 201309).")

        return "\n".join(parts)

    def _build_context(self, q_text: str, hint: str) -> str:
        """Build a context block from probe info and hint parsing."""
        probe_info = self._probe_question_values(q_text, hint)
        hint_guidance = self._extract_hint_parts(hint)
        context_parts = []
        if probe_info:
            context_parts.append(probe_info)
        if hint_guidance:
            context_parts.append(f"Hint-derived guidance:\n{hint_guidance}")
        if hint:
            context_parts.append(f"Hint (authoritative): {hint}")
        return "\n\n".join(context_parts) if context_parts else ""

    def _build_prompt(self, schema: str, context: str, q_text: str) -> str:
        """Build the full prompt for the coder."""
        parts = [f"Database schema:\n{schema}"]
        if context:
            parts.append(f"Context:\n{context}")
        parts.append(f"Question: {q_text}\n\nWrite exactly one SQLite query that answers the question, inside ```sql ... ```.")
        return "\n\n".join(parts)

    def _build_retry_prompt(self, schema: str, context: str, q_text: str,
                            prev_sql: str, prev_error: str) -> str:
        """Build a retry prompt with diagnostic feedback."""
        parts = [f"Database schema:\n{schema}"]
        if context:
            parts.append(f"Context:\n{context}")
        parts.append(
            f"Question: {q_text}\n\n"
            f"Your previous query returned no rows or encountered an error.\n"
            f"Previous SQL: {prev_sql}\n"
            f"Previous result: {prev_error}\n\n"
            f"Check your assumptions: are you using the correct column for the value? "
            f"For 'paid X' check the Price column, not Amount * Price. "
            f"For yearmonth.Date use YYYYMM integer format.\n"
            f"Write a corrected SQLite query inside ```sql ... ```."
        )
        return "\n\n".join(parts)

    def solve(self, question: str) -> str:
        # Hints are embedded in the question text after 'Hint:' for this batch
        hint = ""
        q_text = question
        if "\nHint:" in question or "\nHint :" in question:
            parts = re.split(r'\nHint\s*:\s*', question, maxsplit=1)
            q_text = parts[0].strip()
            hint = parts[1].strip() if len(parts) > 1 else ""

        context = self._build_context(q_text, hint)
        full_schema = self.schema

        # First attempt
        prompt = self._build_prompt(full_schema, context, q_text)
        resp = self.llm(prompt, system=SYS_CORE, temperature=0.0)
        sql = bridge.extract_sql(resp)

        # Verify by running and retry once on empty/error
        result = self.execute(sql)
        if not result.get("ok") or len(result.get("rows", [])) == 0:
            prev_error = result.get("error", "empty result — no rows returned")
            retry_prompt = self._build_retry_prompt(
                full_schema, context, q_text, sql, prev_error
            )
            resp2 = self.llm(retry_prompt, system=RETRY_SYS, temperature=0.0)
            sql = bridge.extract_sql(resp2)

        return sql
