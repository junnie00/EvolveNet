"""Federated global harness r2: enriched schema + execute-validate-retry + output-shape enforcement.
Merged from r1 + client 0 (card_games) + client 1 (california_schools) + client 3 (toxicology) + client 4 (debit_card)."""
from ..harness_base import SQLHarness
from .. import bridge

SYS = (
    "You are an expert Text-to-SQL system for SQLite. Read the schema carefully and output exactly "
    "one SQLite query that answers the question, inside a ```sql ... ``` block.\n\n"
    "CRITICAL RULES:\n"
    "1. FOLLOW THE HINT EXACTLY — the Hint provides correct column names, value formats, and "
    "relationships. Never override the Hint with your own assumptions.\n"
    "2. NEVER use aggregate functions (MAX, MIN, COUNT) in ORDER BY unless there is a GROUP BY "
    "clause. Just use ORDER BY column_name DESC.\n"
    "3. Return ONLY the column(s) the question asks for — no extra columns (no IDs, no sort "
    "keys, no computed values unless explicitly asked). If the question asks 'which X', return "
    "only X.\n"
    "4. Return only as many rows as asked — 'the most/highest/lowest/least' means ONE row "
    "(use LIMIT 1). 'List all' means no LIMIT.\n"
    "5. Do NOT use ROUND() or LIMIT unless the question explicitly asks for rounding or a "
    "specific number of results.\n"
    "6. Use the EXACT value formats from the Hint — do not guess value formats.\n"
    "7. When joining tables, always check the foreign key relationships in the schema.\n"
    "8. Card names are stored in the cards table's name column, NOT in foreign_data.name "
    "(which stores translated names).\n"
    "9. Status values in the legalities table are case-sensitive: 'Legal', 'Banned', "
    "'Restricted' (capitalized first letter).\n"
    "10. For 'full name' composed of first_name and last_name, return them as TWO separate "
    "columns (first_name, last_name), NOT concatenated with ||.\n"
    "11. In the toxicology database, molecule.molecule_id contains 'TR0XX' values and "
    "molecule.label contains ONLY '+' or '-'. NEVER use label to filter by a molecule ID — "
    "use molecule_id instead. There is NO 'toxic' label value.\n"
    "12. The yearmonth table stores dates as YYYYMM (e.g. 201201), not YYYY-MM-DD.\n"
    "13. When the Hint gives a formula like SUBTRACT(DIVIDE(...), DIVIDE(...)), translate it "
    "literally into SQL — do not add extra multiplication or change the denominator.\n"
    "14. When computing a percentage, multiply the ratio by 100 (e.g. COUNT(x)*100.0/COUNT(*)) "
    "so the result is a percentage value, not a decimal fraction.\n"
)

F1_TYPE_HINT = (
    "\nColumn type notes (formula_1 database):\n"
    "- lapTimes.time: TEXT like '1:07.411'; lapTimes.milliseconds: INTEGER\n"
    "- results.time: TEXT like '1:34:50.616' or '+5.478' (only champion has HH:MM:SS.mmm)\n"
    "- results.fastestLapTime: TEXT like '1:21.046'; pitStops.duration: TEXT like '20.761'\n"
    "- results.points: REAL; races.year: INTEGER; races.date: TEXT 'YYYY-MM-DD'\n"
    "- drivers.dob: TEXT 'YYYY-MM-DD'\n"
)


def _detect_db_id(schema: str) -> str | None:
    if "transactions_1k" in schema:
        return "debit_card_specializing"
    if "superhero" in schema and "hero_power" in schema:
        return "superhero"
    if "expense" in schema and "member" in schema and "budget" in schema:
        return "student_club"
    if "connected" in schema and "bond" in schema and "molecule" in schema:
        return "toxicology"
    if "lapTimes" in schema or "pitStops" in schema:
        return "formula_1"
    if "rulings" in schema and "foreign_data" in schema:
        return "card_games"
    if "StatusType" in schema and "frpm" in schema:
        return "california_schools"
    return None


def _extract_hint(question: str) -> str:
    _, _, hint = question.partition("\nHint:")
    return hint.strip()


def _detect_output_shape_issue(question: str, rows: list) -> str | None:
    """Detect common output-shape mismatches. Returns a hint string or None."""
    q_lower = question.lower()
    if 'percentage' in q_lower or 'percent' in q_lower:
        if rows and len(rows) == 1 and len(rows[0]) == 1:
            val = rows[0][0]
            if isinstance(val, (int, float)) and 0 < abs(val) < 1:
                return (f"The question asks for a percentage but the result is {val} "
                        f"(a fraction). Multiply by 100 to get the percentage value.")
    return None


class BareHarness(SQLHarness):
    def solve(self, question: str) -> str:
        hint = _extract_hint(question)
        db_id = _detect_db_id(self.schema)
        enriched = self._enrich_schema(db_id)

        extra = F1_TYPE_HINT if db_id == "formula_1" else ""
        prompt = f"Database schema:\n{enriched}{extra}\n\nQuestion: {question}\n\nWrite the SQLite query."
        resp = self.llm(prompt, system=SYS, temperature=0.0)
        sql = bridge.extract_sql(resp)
        if not sql:
            return ""

        result = self.execute(sql)
        if not result.get("ok", False):
            sql = self._retry(question, hint, db_id, enriched, extra, sql, result,
                              reason="error")
        elif len(result.get("rows", [])) == 0:
            sql = self._retry(question, hint, db_id, enriched, extra, sql, result,
                              reason="empty")
        else:
            # validate output shape: retry if percentage looks like a fraction
            shape_issue = _detect_output_shape_issue(question, result.get("rows", []))
            if shape_issue:
                sql = self._retry_shape(question, hint, db_id, enriched, extra, sql, shape_issue)
        return sql

    def _enrich_schema(self, db_id):
        schema = self.schema
        if db_id == "card_games":
            for t, c in [("legalities", "status"), ("cards", "rarity"),
                         ("foreign_data", "language"), ("sets", "type"),
                         ("cards", "isAlternative")]:
                try:
                    v = self.distinct(t, c)
                    if v:
                        schema += f"\n\n-- {t}.{c} values (case-sensitive): {v}"
                except Exception:
                    pass
            schema += (
                "\n\n-- IMPORTANT card_games notes:"
                "\n-- English card names: cards.name (NOT foreign_data.name)"
                "\n-- Translated card names: foreign_data.name (per language)"
                "\n-- isAlternative=1 means the card is an alternative/special version"
                "\n-- hasContentWarning=1 means the card has missing or degraded properties"
            )
        elif db_id == "california_schools":
            for t, c in [("frpm", "Educational Option Type"),
                         ("frpm", "NSLP Provision Status")]:
                try:
                    v = self.distinct(t, c, limit=15)
                    if v:
                        schema += f"\n\n-- {t}.{c} values: {v}"
                except Exception:
                    pass
        elif db_id == "toxicology":
            try:
                bt = self.distinct("bond", "bond_type")
                if bt:
                    schema += f"\n\n-- bond.bond_type actual values: {bt}"
            except Exception:
                pass
        return schema

    def _retry(self, question, hint, db_id, schema, extra, bad_sql, res, reason="error"):
        if reason == "error":
            err = res.get("error", "unknown error")
            err_msg = f"Your previous query errored:\n```sql\n{bad_sql}\n```\nError: {err}\n"
        else:
            err_msg = (
                f"Your previous query ran successfully but returned 0 rows (empty result):\n"
                f"```sql\n{bad_sql}\n```\n"
                "This usually means a WHERE clause value doesn't match the actual stored value. "
                "Check the DB values below and use the EXACT stored string.\n"
            )

        vals = self._probe_values()

        # database-specific targeted fixes
        fixes = ""
        if db_id == "card_games":
            fixes = (
                "Common issues:\n"
                "- Status values are 'Legal', 'Banned', 'Restricted' (capitalized first letter)\n"
                "- English card names are in cards.name, NOT foreign_data.name\n"
                "- foreign_data.name stores TRANSLATED names (e.g. Japanese name)\n"
                "- To find a card by English name, use cards.name, then join to other tables\n"
                "- COUNT of entities in JOINs: use COUNT(DISTINCT id) to avoid overcounting\n"
                "- Percentage: multiply ratio by 100 (COUNT(x)*100.0/COUNT(*))\n"
                "- hasContentWarning=1 → use CASE WHEN to produce 'Yes'/'No' text\n"
                "- isAlternative=1 means the card is an alternative version\n"
            )
        elif db_id == "toxicology":
            fixes = ("Common issues: use molecule_id (not label) to filter by molecule ID like 'TR0XX'; "
                     "label only has '+' or '-'; no 'toxic' label value exists; "
                     "when a subquery may return multiple rows, use IN not =\n")
        elif db_id == "debit_card_specializing":
            fixes = ("Common issues: yearmonth.Date is YYYYMM (e.g. 201201), not YYYY-MM; "
                     "revenue = SUM(Price), NOT SUM(Amount*Price)\n")
        elif db_id == "california_schools":
            fixes = (
                "Common issues:\n"
                "- Educational Option Type values are full strings like 'Continuation School'\n"
                "- NSLP Provision Status values are full strings like 'Lunch Provision 2'\n"
                "- When ORDER BY uses a computed expression with possible NULLs, add "
                "IS NOT NULL filters to avoid NULLs sorting first\n"
            )
        else:
            fixes = "Common issues: column names must match schema exactly; check value formats\n"

        prompt = (
            f"Database schema:\n{schema}{extra}\n\nQuestion: {question}\n\n"
            f"Hint: {hint}\n\n"
            f"{err_msg}\n"
            f"Actual DB values for WHERE clauses:\n{vals}\n\n"
            f"{fixes}Write the corrected SQLite query."
        )
        resp = self.llm(prompt, system=SYS, temperature=0.0)
        new_sql = bridge.extract_sql(resp)
        if new_sql:
            r2 = self.execute(new_sql)
            if r2.get("ok", False) and len(r2.get("rows", [])) > 0:
                return new_sql
        return bad_sql

    def _probe_values(self):
        skip = ("id", "code", "number", "count", "lat", "lng", "alt", "date",
                "time", "weight", "height", "enrol", "phone", "zip", "street",
                "mail", "url", "website")
        lines = []
        for table, cols in self.tables().items():
            for col in cols:
                if any(k in col.lower() for k in skip):
                    continue
                try:
                    v = self.distinct(table, col, limit=10)
                    if v and len(v) <= 10:
                        lines.append(f"  {table}.{col}: {v}")
                except Exception:
                    pass
        return "\n".join(lines) if lines else "  (no categorical columns found)"

    def _retry_shape(self, question, hint, db_id, schema, extra, bad_sql, shape_msg):
        """Retry when SQL runs OK but output shape doesn't match the question."""
        vals = self._probe_values()
        prompt = (
            f"Database schema:\n{schema}{extra}\n\nQuestion: {question}\n\n"
            f"Hint: {hint}\n\n"
            f"Your previous query ran but the output shape may be wrong:\n```sql\n{bad_sql}\n```\n"
            f"Issue: {shape_msg}\n\n"
            f"Actual DB values for WHERE clauses:\n{vals}\n\n"
            f"Write a corrected SQLite query that returns EXACTLY what the question asks for."
        )
        resp = self.llm(prompt, system=SYS, temperature=0.0)
        new_sql = bridge.extract_sql(resp)
        if new_sql:
            r2 = self.execute(new_sql)
            if r2.get("ok", False) and len(r2.get("rows", [])) > 0:
                return new_sql
        return bad_sql
