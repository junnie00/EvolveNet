"""Federated merge: value-grounded probing + hint verification + retry on error/empty."""
import re
from ..harness_base import SQLHarness
from .. import bridge

SYS = (
    "You are an expert Text-to-SQL system for SQLite. Read the schema carefully and output exactly "
    "one SQLite query that answers the question, inside a ```sql ... ``` block.\n\n"
    "CRITICAL rules — never violate these:\n"
    "1. Select only the columns needed to answer the question. "
    "If asking for a person's name, give forename and surname as separate columns. "
    "When the question asks to identify or list people (e.g. 'which player', 'who is', "
    "'list their name'), return the stored name column as-is; only extract forename/surname "
    "when the question explicitly asks for first name, last name, or name components. "
    "Do not add extra columns like sums or IDs that the question doesn't ask for.\n"
    "2. Use ONLY native SQLite functions: strftime('%%Y', col) not YEAR(); "
    "GROUP_CONCAT(DISTINCT col) with ONE argument; CAST(... AS INTEGER).\n"
    "3. Do NOT compare to string values that have spaces inside quotes "
    "unless the actual data has spaces (check populated values in schema below).\n"
    "4. molecule_id identifies the molecule. The label column stores '+' or '-' "
    "(carcinogenic/non-carcinogenic), NOT the molecule ID.\n"
    "5. When computing a ratio between two groups named in order in the question "
    "(e.g. 'ratio between X and Y'), place the first-mentioned group in the numerator "
    "and the second-mentioned group in the denominator. Match group names from the "
    "question to column values case-insensitively by content, not by position in the schema."
)


def _build_rich_schema(harness):
    """Add column types + distinct values as annotations while preserving original format."""
    lines = harness.schema.split("\n")
    tables = harness.tables()
    result = []
    for line in lines:
        result.append(line)
        m = re.match(r"^Table (\w+)\((.+)\)\s*$", line.strip())
        if not m:
            continue
        tname = m.group(1)
        cols = [c.strip() for c in m.group(2).split(",")]
        ctypes = harness.column_types(tname)
        type_parts = [f"{c}({ctypes.get(c, '')})" if ctypes.get(c, "") else c for c in cols]
        result.append(f"  Column types: {', '.join(type_parts)}")
        for c in cols:
            if ctypes.get(c, "").upper() in ("TEXT", "VARCHAR", ""):
                vals = harness.distinct(tname, c, limit=20)
                if 2 <= len(vals) <= 20:
                    result.append(f"  Populated values in {c}: {', '.join(repr(v) for v in vals)}")
    return "\n".join(result)


def _verify_hint_values(harness, hint_text):
    """If a hint-quoted value doesn't match the DB, emit a space-mismatch correction."""
    if not hint_text or hint_text.strip() in ("", "(none)"):
        return []
    values = set(m.group(1) for m in re.finditer(r"'([^']+)'", hint_text))
    if not values:
        return []
    corrections = []
    for tname in harness.tables():
        for c, ct in harness.column_types(tname).items():
            if ct.upper() not in ("TEXT", "VARCHAR", ""):
                continue
            distinct = harness.distinct(tname, c, limit=100)
            if not distinct:
                continue
            for v in values:
                if v in distinct:
                    continue
                v_strip, v_nospace = v.strip(), v.replace(" ", "")
                if v_strip in distinct and v_strip != v:
                    corrections.append(
                        f"CORRECTION: Hint value {repr(v)} has leading/trailing spaces. "
                        f"Actual value in {tname}.{c} is {repr(v_strip)}.")
                if v_nospace in distinct and v_nospace not in (v, v_strip):
                    corrections.append(
                        f"CORRECTION: Hint value {repr(v)} contains spaces. "
                        f"Actual value in {tname}.{c} is {repr(v_nospace)}.")
    return corrections


def _retry_on_error(harness, question, hint_text, failed_sql, error_msg):
    """Retry SQL after execution error."""
    schema_str = _build_rich_schema(harness)
    prompt = f"Database schema:\n{schema_str}\n\nQuestion: {question}"
    if hint_text:
        prompt += f"\n\nHint: {hint_text}"
    prompt += (
        f"\n\nThe previous SQL attempt failed with an error:\n"
        f"```sql\n{failed_sql}\n```\n\nError: {error_msg}\n\n"
        f"Fix the SQL to use ONLY SQLite-compatible functions and syntax, "
        f"and output the corrected query in ```sql ... ```."
    )
    return bridge.extract_sql(harness.llm(prompt, system=SYS, temperature=0.0))


def _retry_on_empty(harness, question, hint_text, failed_sql):
    """Retry SQL after empty result, with targeted DB probes."""
    schema_str = _build_rich_schema(harness)
    tables = harness.tables()
    probe_parts = []
    for tn, col, lbl in [("molecule", "molecule_id", "molecule.molecule_id examples"),
                         ("molecule", "label", "molecule.label values"),
                         ("bond", "bond_type", "bond.bond_type values")]:
        if tn in tables:
            probe_parts.append(f"{lbl}: {harness.distinct(tn, col, limit=10)}")
    if "transactions_1k" in tables:
        probe_parts.append(f"transaction dates (sample): {harness.distinct('transactions_1k', 'Date', limit=5)}")
    prompt = f"Database schema:\n{schema_str}\n\nQuestion: {question}"
    if hint_text:
        prompt += f"\n\nHint: {hint_text}"
    if probe_parts:
        prompt += "\n\n" + "\n".join(probe_parts)
    prompt += (
        f"\n\nThe previous query returned zero rows:\n```sql\n{failed_sql}\n```\n\n"
        f"Possible reasons:\n"
        f"- Filter value doesn't match actual data (check populated values above)\n"
        f"- Wrong column referenced (e.g. molecule.label vs molecule.molecule_id)\n"
        f"- Wrong join path or missing join\n"
        f"Correct the query and output it in ```sql ... ```."
    )
    return bridge.extract_sql(harness.llm(prompt, system=SYS, temperature=0.0))


def _all_null_result(result):
    """True if result is a single row where every cell is None (e.g. SUM over empty set)."""
    rows = result.get("rows", [])
    return bool(rows and len(rows) == 1 and all(v is None for v in rows[0]))


def _looks_like_count_mismatch(question, result):
    """Detect COUNT+GROUP BY anti-pattern: 'how many' returns multiple rows of [1]."""
    if not any(w in question.lower() for w in ["how many", "count of", "number of", "total number"]):
        return False
    rows = result.get("rows", [])
    return len(rows) > 1 and all(len(r) >= 1 and r[0] == 1 for r in rows[:20])


def _has_suspicious_offset(sql, question):
    """Detect OFFSET > 0 on extreme-value questions (e.g. 'highest' + OFFSET 332)."""
    m = re.search(r'\bOFFSET\s+(\d+)', sql, re.IGNORECASE)
    if not m or int(m.group(1)) == 0:
        return False
    extreme = {"highest", "lowest", "most", "best", "worst", "top", "bottom",
               "maximum", "minimum", "largest", "smallest", "greatest", "least"}
    return bool(extreme & set(question.lower().split()))


def _has_all_null_column(result):
    """Check whether any result column is NULL in every row (data mismatch)."""
    rows = result.get("rows", [])
    if not rows:
        return False
    return any(all(row[c] is None for row in rows) for c in range(len(rows[0])))


def _retry_with_advice(harness, question, hint_text, failed_sql, advice):
    """Retry SQL with targeted advice about what went wrong (output-shape fix)."""
    schema_str = _build_rich_schema(harness)
    prompt = f"Database schema:\n{schema_str}\n\nQuestion: {question}"
    if hint_text:
        prompt += f"\n\nHint: {hint_text}"
    prompt += (
        f"\n\nThe previous SQL query had issues:\n"
        f"```sql\n{failed_sql}\n```\n\nIssues:\n"
        + "\n".join(f"- {a}" for a in advice)
        + "\n\nCorrect the query and output it in ```sql ... ```."
    )
    return bridge.extract_sql(harness.llm(prompt, system=SYS, temperature=0.0))


class Harness(SQLHarness):
    def solve(self, question: str) -> str:
        if "\nHint:" in question:
            question_clean, hint_text = question.split("\nHint:", 1)
            question_clean, hint_text = question_clean.strip(), hint_text.strip()
        else:
            question_clean, hint_text = question, ""

        corrections = _verify_hint_values(self, hint_text)
        schema_str = _build_rich_schema(self)

        prompt = f"Database schema:\n{schema_str}\n\nQuestion: {question_clean}"
        if hint_text:
            prompt += f"\n\nHint: {hint_text}"
        if corrections:
            prompt += "\n\n" + "\n".join(corrections)
        prompt += "\n\nWrite the SQLite query."

        sql = bridge.extract_sql(self.llm(prompt, system=SYS, temperature=0.0))
        if not sql:
            return sql

        result = self.execute(sql)
        if result["ok"] and result.get("rows"):
            # Post-execution quality checks (retry on detectable issues only)
            # 1) All-NULL single row — SUM/AVG over empty set
            if _all_null_result(result):
                sql_a = _retry_with_advice(
                    self, question_clean, hint_text, sql,
                    ["The query returned only NULL values. "
                     "Check that the join/filter produces matching rows "
                     "and that SUM/AVG has data to aggregate over."])
                if sql_a and sql_a != sql:
                    r_a = self.execute(sql_a)
                    if r_a["ok"] and r_a.get("rows") and not _all_null_result(r_a):
                        return sql_a
                return sql
            # 2) Count-shape mismatch: 'how many' returning multiple [1] rows
            if _looks_like_count_mismatch(question_clean, result):
                sql_b = _retry_with_advice(
                    self, question_clean, hint_text, sql,
                    ["This is a count question but returned multiple rows each with value 1.",
                     "Wrap the GROUP BY query in a subquery: "
                     "SELECT COUNT(*) FROM (SELECT ... GROUP BY ... HAVING ...) AS sub;"])
                if sql_b and sql_b != sql:
                    r_b = self.execute(sql_b)
                    if r_b["ok"] and r_b.get("rows") and not _looks_like_count_mismatch(question_clean, r_b):
                        return sql_b
                return sql
            # 3) OFFSET > 0 on extreme-value question
            if _has_suspicious_offset(sql, question_clean):
                sql_c = _retry_with_advice(
                    self, question_clean, hint_text, sql,
                    ["The query uses OFFSET > 0 but the question asks for an extreme value. "
                     "Remove the OFFSET clause entirely — LIMIT 1 (without OFFSET) is what you need."])
                if sql_c and sql_c != sql:
                    r_c = self.execute(sql_c)
                    if r_c["ok"] and r_c.get("rows"):
                        return sql_c
                return sql
            # 4) All-NULL result column (concrete data expected)
            if _has_all_null_column(result):
                sql_d = _retry_with_advice(
                    self, question_clean, hint_text, sql,
                    ["A result column is NULL in every row. "
                     "Add IS NOT NULL filters or COALESCE() to ensure real values are returned."])
                if sql_d and sql_d != sql:
                    r_d = self.execute(sql_d)
                    if r_d["ok"] and r_d.get("rows") and not _has_all_null_column(r_d):
                        return sql_d
                return sql
            return sql

        if not result.get("ok"):
            sql2 = _retry_on_error(
                self, question_clean, hint_text, sql, result.get("error", "Unknown error")
            )
            if sql2 and sql2 != sql:
                r2 = self.execute(sql2)
                if r2["ok"] and r2.get("rows") and not _all_null_result(r2):
                    return sql2
            return sql

        if not result.get("rows"):
            sql3 = _retry_on_empty(self, question_clean, hint_text, sql)
            if sql3 and sql3 != sql:
                r3 = self.execute(sql3)
                if r3["ok"] and r3.get("rows") and not _all_null_result(r3):
                    return sql3
            return sql

        return sql
