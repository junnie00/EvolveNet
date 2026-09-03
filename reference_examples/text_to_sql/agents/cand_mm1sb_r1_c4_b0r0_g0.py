"""G0 harness (r1_c4): schema-enriched prompt + output-shape validation +
anti-pattern detection + improved retry with value-format guidance.

Key improvements over base (cand_mm1sb_r0_c0_b0r0_g0):
1. Schema enrichment: probe yearmonth.Date format, include in prompt
2. SYS rule: GROUP BY + HAVING with aggregate COUNT → wrap in subquery
3. SYS rule: Do NOT use ROUND() unless question explicitly asks
4. SYS rule: Output shape — only return the asked column(s), no extras
5. Post-execution output-shape validation: detect extra columns and retry
6. Improved retry: include DB-probed value-format hints when available
"""
from ..harness_base import SQLHarness
from .. import bridge

SYS = (
    "You are an expert Text-to-SQL system for SQLite. Read the schema carefully and output exactly "
    "one SQLite query that answers the question, inside a ```sql ... ``` block.\n\n"
    "CRITICAL RULES:\n"
    "1. FOLLOW THE HINT EXACTLY — the Hint maps question phrases to database columns/values. "
    "Use the exact column names and filter values it specifies. When the Hint uses SUM(condition) "
    "or COUNT(condition) syntax, treat the condition as a CASE WHEN expression, not as a WHERE filter.\n"
    "2. CASE-SENSITIVE text comparison: SQLite '=' is case-sensitive for ASCII. When matching "
    "text values (status, rarity, language, etc.), use LOWER(column) = LOWER(value) to avoid "
    "case mismatches (e.g. 'Legal' vs 'legal', 'Restricted' vs 'restricted').\n"
    "3. Do NOT use aggregate functions (MAX, MIN, COUNT, SUM) in ORDER BY without GROUP BY — "
    "use plain column ORDER BY instead.\n"
    "4. Return ONLY the columns the question asks for — do not add extra columns. If the question "
    "asks 'which X' return ONLY X. If it asks 'how much' return ONLY the number. No extra computed "
    "columns like Revenue, Score, etc.\n"
    "5. Use LEFT JOIN when you need rows from the primary table even if there is no match.\n"
    "6. When you need to COUNT groups after GROUP BY + HAVING, wrap in a subquery: "
    "SELECT COUNT(*) FROM (SELECT col FROM ... GROUP BY col HAVING ...) — do NOT put COUNT "
    "at the same level as GROUP BY + HAVING, as COUNT(DISTINCT col) in the outer SELECT will "
    "always return 1 per group.\n"
    "7. Do NOT use ROUND() unless the question explicitly asks for rounding or specific decimal places. "
    "Return full precision.\n"
    "8. Follow the Hint's formula literally. If the Hint says COUNT(condition) as a denominator, "
    "count ALL rows (don't filter WHERE first). Use the condition as a CASE WHEN expression."
)

SYS_RETRY = (
    "Your previous SQL had an issue. Fix it and output a corrected query in a ```sql block.\n"
    "Common fixes:\n"
    "- If error: check for aggregate misuse (no MAX/MIN in ORDER BY without GROUP BY), "
    "syntax errors, or wrong column/table references.\n"
    "- If empty result: the filter values may have wrong format. Check column formats by "
    "querying sample values (e.g. SELECT DISTINCT col FROM table LIMIT 5). "
    "For text comparisons, use LOWER() to avoid case mismatches.\n"
    "- If wrong columns: return ONLY what the question asks for, nothing extra.\n"
    "- If you have GROUP BY + HAVING and need to count the groups, wrap in a subquery.\n"
    "- Do NOT use ROUND() unless the question asks for it.\n"
    "- Follow the Hint exactly for column names, filter values, and formula structure."
)


class BareHarness(SQLHarness):
    def _schema_enrichment(self) -> str:
        """Probe DB for column value formats that commonly confuse the coder.
        Returns extra schema hints to append to the prompt."""
        extra = []
        try:
            # yearmonth.Date format: YYYYMM without separators
            vals = self.distinct("yearmonth", "Date", limit=5)
            if vals:
                extra.append(
                    f"NOTE: yearmonth.Date values are stored as 'YYYYMM' strings "
                    f"(e.g. '{vals[0]}'). For January 2012 use Date = '201201', "
                    f"not '2012-01' or '2012/01'."
                )
        except Exception:
            pass
        return "\n".join(extra)

    def _build_prompt(self, question: str, hint: str = "") -> str:
        enrichment = self._schema_enrichment()
        schema_section = f"Database schema:\n{self.schema}"
        if enrichment:
            schema_section += f"\n\n{enrichment}"
        hint_section = f"\nHint: {hint}" if hint else ""
        return f"{schema_section}\n\nQuestion: {question}{hint_section}\n\nWrite the SQLite query."

    def _detect_output_shape_issue(self, question: str, sql: str, rows: list) -> bool:
        """Check if the SQL returns more columns than the question asks for.
        Heuristic: 'which X' / 'what X' questions expect 1 column; 'list X and Y' expects 2."""
        q_lower = question.lower()
        # Detect "which" / "what" questions that want a single entity
        wants_single = (
            q_lower.startswith("which ") or q_lower.startswith("what ")
        ) and not any(
            w in q_lower for w in [" and ", " list", " name and", " describe"]
        )
        if not wants_single:
            return False
        # Check if we have > 1 column in result (skip aggregate-only results)
        if rows and len(rows[0]) > 1:
            return True
        return False

    def _strip_extra_column(self, sql: str, n_keep: int = 1) -> str:
        """Best-effort strip extra columns from SELECT. Only handles simple SELECT cases."""
        import re
        # Match SELECT ... FROM pattern
        m = re.match(r'(SELECT\s+)(.*?)(\s+FROM\s+.*)', sql, re.IGNORECASE | re.DOTALL)
        if not m:
            return sql
        prefix, cols_str, suffix = m.group(1), m.group(2), m.group(3)
        # Split columns by comma (respecting nested parens)
        cols = []
        depth = 0
        current = ""
        for ch in cols_str:
            if ch == '(':
                depth += 1
                current += ch
            elif ch == ')':
                depth -= 1
                current += ch
            elif ch == ',' and depth == 0:
                cols.append(current.strip())
                current = ""
            else:
                current += ch
        if current.strip():
            cols.append(current.strip())
        if len(cols) <= n_keep:
            return sql
        kept = cols[:n_keep]
        return f"{prefix}{', '.join(kept)}{suffix}"

    def solve(self, question: str) -> str:
        prompt = self._build_prompt(question)
        resp = self.llm(prompt, system=SYS, temperature=0.0)
        sql = bridge.extract_sql(resp)

        for attempt in range(2):
            res = self.execute(sql)

            if res["ok"] and len(res.get("rows", [])) > 0:
                # Output shape check: strip extra columns if needed
                rows = res.get("rows", [])
                if self._detect_output_shape_issue(question, sql, rows):
                    stripped = self._strip_extra_column(sql, n_keep=1)
                    if stripped != sql:
                        res2 = self.execute(stripped)
                        if res2["ok"] and len(res2.get("rows", [])) > 0:
                            return stripped
                return sql

            # Build retry feedback
            if not res["ok"]:
                err_msg = res.get("error", "unknown error")
                feedback = (
                    f"The SQL execution failed with error: {err_msg}\n"
                    f"Your SQL was:\n{sql}\n\n"
                    f"Fix the error. Remember: do not use MAX/MIN/COUNT in ORDER BY without GROUP BY. "
                    f"For text comparisons, use LOWER() to avoid case mismatches."
                )
            else:
                # Empty result — probe DB for value format hints
                format_hint = ""
                try:
                    # Check if yearmonth table is involved
                    sql_lower = sql.lower()
                    if "yearmonth" in sql_lower:
                        vals = self.distinct("yearmonth", "Date", limit=5)
                        if vals:
                            format_hint = (
                                f"\nyearmonth.Date format: values look like {vals[:3]} "
                                f"(YYYYMM, no separators). Use Date = '201201' not '2012-01'.\n"
                            )
                except Exception:
                    pass

                feedback = (
                    f"The SQL returned an empty result. Your SQL was:\n{sql}\n\n"
                    f"The query executed but found no matching rows. Possible causes:\n"
                    f"1. Text value case mismatch (e.g. 'Legal' vs 'legal'). "
                    f"Use LOWER(column) = LOWER(value) for text comparisons.\n"
                    f"2. Wrong join path — check that the tables you join actually connect "
                    f"the data the question asks about.\n"
                    f"3. Wrong filter values — follow the Hint exactly.\n"
                    f"4. Column format mismatch — check actual stored values."
                    f"{format_hint}"
                    f"Rewrite the query fixing the likely cause."
                )

            retry_prompt = f"{feedback}\n\nOriginal question: {question}\n\nWrite the corrected SQLite query."
            resp = self.llm(retry_prompt, system=SYS_RETRY, temperature=0.0)
            sql = bridge.extract_sql(resp)

        return sql
