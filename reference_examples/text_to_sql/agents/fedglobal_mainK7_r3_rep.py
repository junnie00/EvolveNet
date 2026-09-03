"""Federated global harness — merged from 7 client evolutions on mainK7_r0.
Adopts: low-cardinality column enrichment (c3); hint value-grounding (c0);
hint column guidance (c5); execute-validate on error (all); YEAR→strftime (c2);
home-scoped guides for california_schools (c1) and toxicology (c3);
cds→CDSCode join probe (c1); numeric value-grounding from literals (c4);
empty-result retry (c1,c4); retry-prompt enrichment (c6);
SYS rules: no code-name translation (c3), avoid GROUP_CONCAT for lists (c3),
use direct column values from probe (c4)."""
import re
from ..harness_base import SQLHarness
from .. import bridge

SYS = (
    "You are an expert Text-to-SQL system for SQLite. Read the schema carefully and output exactly "
    "one SQLite query that answers the question, inside a ```sql ... ``` block.\n\n"
    "CRITICAL RULES:\n"
    "1. The HINT is AUTHORITATIVE — follow each mapping exactly. "
    "When the hint gives an explicit formula (e.g., COUNT(X), SUM(CASE...), DIVIDE...), "
    "implement it exactly — do NOT add DISTINCT, subqueries, or extra transformations "
    "the hint does not specify. "
    "Each hint mapping applies ONLY to the part of the question that names the "
    "corresponding concept; do not extend a column mapping to other parts of a multi-part question. "
    "When the hint column exists in multiple tables, check all tables that have it.\n"
    "2. Return EXACTLY the columns asked for — nothing more. "
    "Use AS aliases matching the question's wording.\n"
    "3. NEVER concatenate name columns. If Hint maps 'full name' to "
    "'first_name, last_name', return them as separate SELECT columns. "
    "A '+' between column names in the hint (e.g., 'forename+surname') means both "
    "columns are needed — return them as separate SELECT columns, NOT concatenated.\n"
    "4. SQLite = is CASE-SENSITIVE. The 'Actual values' section shows exact stored values.\n"
    "5. For 'the most'/'the highest' return 1 row (LIMIT 1). "
    "For 'top K' return K rows (LIMIT K).\n"
    "6. SQLite has no YEAR() — use CAST(strftime('%%Y', col) AS INTEGER). "
    "GROUP_CONCAT(DISTINCT col, sep) is unsupported. Use || for string concat.\n"
    "7. INTEGER = quantity, REAL = price/value. Revenue = SUM(quantity * price).\n"
    "8. Do NOT ROUND numbers unless asked.\n"
    "9. Do NOT add ORDER BY/LIMIT/ROUND unless the question or Hint asks. "
    "Do not add extra precision adjustments to calculations (e.g., birthday-aware age "
    "with month/day correction) unless the question explicitly asks for "
    "precise calculation.\n"
    "10. Prefer INNER JOIN over LEFT JOIN unless nulls are explicitly asked for.\n"
    "11. NEVER translate stored codes to full names via CASE (e.g. atom.element "
    "stores codes 'c', 'cl', 'h', 'o' — return these codes directly). "
    "The mapping in the Hint is for YOUR reference only; "
    "only translate if the question EXPLICITLY asks for the full name.\n"
    "12. When a question says 'list' items, return them as separate rows "
    "(use DISTINCT, avoid GROUP_CONCAT). GROUP_CONCAT is only for explicit "
    "'comma-separated' or 'concatenated' output requests.\n"
    "13. When 'Value grounding notes' show a number exists in a specific column, use that "
    "column directly — do NOT compute the value from other columns (e.g. if '124.05' is "
    "found in transactions_1k.Price, use Price = 124.05, NOT Amount * Price = 124.05)."
)

_CA_GUIDE = (
    "### COLUMN NOTES (california_schools) ###\n"
    "- Lowest grade: use frpm.`Low Grade`. Do NOT use schools.GSoffered (range string).\n"
    "- School name: satscores.sname or schools.School. May be NULL for district records.\n"
    "- School properties (County, City, Address, Phone, Website): use schools table.\n"
    "- Free/reduced meal data, Charter status, Charter funding type: use frpm table.\n"
    "- SAT scores, test takers, AvgScrRead/Math/Write, NumGE1500: use satscores table.\n"
    "- FK: frpm.CDSCode->schools.CDSCode; satscores.cds->schools.CDSCode.\n"
    "  IMPORTANT: Some satscores.cds values are 13 chars (missing leading '0') "
    "while schools.CDSCode is always 14 chars. If a join returns 0 rows, try:\n"
    "  ON s.cds = sch.CDSCode  — works for 14-char cds\n"
    "  ON '0' || s.cds = sch.CDSCode  — works for 13-char cds\n"
    "  or both: ON LENGTH(s.cds)=14 AND s.cds=sch.CDSCode OR LENGTH(s.cds)=13 AND '0'||s.cds=sch.CDSCode\n"
)

_TOX_GUIDE = (
    "### COLUMN NOTES (toxicology) ###\n"
    "- atom.element stores short codes ('c', 'cl', 'h', 'o', 's', 'n') — "
    "return codes directly, NOT full names via CASE.\n"
    "- atom and bond both have molecule_id — join directly on molecule_id "
    "for element+bond_type queries (do NOT go through connected).\n"
    "- The molecule table does NOT contain all molecule_ids from atom/bond. "
    "When you need molecule.label, use LEFT JOIN (not INNER JOIN) to avoid data loss. "
    "If you only need atom or bond columns, do NOT join molecule.\n"
    "- For molecule_id ranges (e.g., TR010 to TR050): use "
    "CAST(SUBSTR(molecule_id, 3) AS INTEGER) for numeric comparison, not BETWEEN on strings."
)


class BareHarness(SQLHarness):
    """Federated global: enrichment + value-grounding + retry + SQLite compat + probes."""

    @staticmethod
    def _fix_year(sql: str) -> str:
        return re.sub(r'\byear\s*\(\s*([^)]+)\s*\)',
                      r"CAST(strftime('%Y', \1) AS INTEGER)", sql, flags=re.IGNORECASE)

    def _enrich_schema(self) -> str:
        """Append distinct values for columns with <=10 distinct values."""
        enriched = self.schema
        notes = []
        for tname, cols in self.tables().items():
            for cname in cols:
                vals = self.distinct(tname, cname, limit=20)
                if vals and len(vals) <= 10:
                    notes.append(f"  - {tname}.{cname}: distinct values = {vals}")
        if notes:
            enriched += "\nColumn value notes:\n" + "\n".join(notes)
        return enriched

    def _probe_hint_values(self, hint: str) -> str:
        """Probe column='value' hint patterns; emit notes on case mismatches."""
        if not hint:
            return ""
        matches = re.findall(r"(?:(\w+)\.)?(\w+)\s*=\s*'([^']+)'", hint)
        if not matches:
            return ""
        tables = self.tables()
        notes, seen = [], set()
        for table_prefix, column, hint_val in matches:
            for tname, cols in tables.items():
                if column not in cols:
                    continue
                if table_prefix and table_prefix.lower() != tname.lower():
                    continue
                key = (tname, column, hint_val)
                if key in seen:
                    continue
                seen.add(key)
                actual = self.distinct(tname, column, limit=20)
                if not actual:
                    continue
                if hint_val not in actual:
                    ci = [v for v in actual
                          if isinstance(v, str) and v.lower() == hint_val.lower()]
                    if ci:
                        ci.sort()
                        notes.append(
                            f"  - {tname}.{column} stores {ci} "
                            f"(not '{hint_val}' — use exact case shown)")
        return "\n".join(notes)

    def _hint_col_guide(self, hint: str) -> str:
        """Parse '<concept> refers to col1, col2' patterns."""
        if not hint:
            return ""
        lines = []
        for part in (p.strip() for p in hint.split(';')):
            m = re.match(
                r'([a-zA-Z][a-zA-Z\s]*[a-zA-Z])\s+refers to\s+([a-zA-Z_]\w*)\s*,\s*([a-zA-Z_]\w*)',
                part, re.IGNORECASE)
            if m:
                concept, c1, c2 = m.group(1).strip(), m.group(2), m.group(3)
                lines.append(
                    f"NOTE: Hint maps '{concept}' to '{c1}' and '{c2}'. "
                    f"Return as separate SELECT columns — do NOT concatenate.")
        return "\n".join(lines)

    def _probe_cds_join(self) -> str:
        """Detect if satscores.cds needs leading '0' to match schools.CDSCode."""
        if self.db.db_id != "california_schools":
            return ""
        try:
            direct = self.execute(
                "SELECT COUNT(*) FROM satscores s "
                "INNER JOIN schools sch ON s.cds = sch.CDSCode "
                "WHERE s.rtype='S'")
            with_zero = self.execute(
                "SELECT COUNT(*) FROM satscores s "
                "INNER JOIN schools sch ON '0'||s.cds = sch.CDSCode "
                "WHERE s.rtype='S'")
            d = direct['rows'][0][0] if direct.get('ok') and direct['rows'] else 0
            wz = with_zero['rows'][0][0] if with_zero.get('ok') and with_zero['rows'] else 0
            if wz > d:
                return (
                    f"NOTE: {d} satscores rows match schools.CDSCode directly. "
                    f"{wz} rows match when '0' is prepended to cds. "
                    f"When joining satscores to schools, use: "
                    f"ON '0'||s.cds = sch.CDSCode  (works for all cds lengths)."
                )
        except Exception:
            pass
        return ""

    def _probe_numeric_values(self, question: str) -> str:
        """Extract decimal literals from question; probe DB for matching column."""
        decimals = set(re.findall(r'(?<!\w)(\d+\.\d+)(?!\w)', question))
        if not decimals:
            return ""
        tables = self.tables()
        notes = []
        for num_str in sorted(decimals):
            for tname, cols in tables.items():
                for cname in cols:
                    try:
                        r = self.execute(
                            f'SELECT COUNT(*) FROM "{tname}" WHERE "{cname}" = {num_str}')
                    except Exception:
                        continue
                    if r.get("ok") and r["rows"] and r["rows"][0][0] > 0:
                        notes.append(
                            f"  - Value '{num_str}' found in {tname}.{cname} "
                            f"({r['rows'][0][0]} rows)")
                        break
        if notes:
            return "Value grounding notes:\n" + "\n".join(notes)
        return ""

    def _build_prompt(self, q: str, enriched: str,
                      value_notes: str, col_guide: str, home_guide: str,
                      cds_note: str = "", numeric_notes: str = "") -> str:
        parts = [f"Database schema:\n{enriched}"]
        if "\nHint:" in q:
            q_text, hint_text = q.split("\nHint:", 1)
            parts.append(f"\nQuestion: {q_text.strip()}")
            parts.append(f"\n=== AUTHORITATIVE HINT ===\n{hint_text.strip()}")
            if value_notes:
                parts.append(f"\nActual values in DB (use these):\n{value_notes}")
            if numeric_notes:
                parts.append(f"\n{numeric_notes}")
            if col_guide:
                parts.append(f"\n{col_guide}")
        else:
            parts.append(f"\nQuestion: {q}")
        if home_guide:
            parts.append(f"\n{home_guide}")
        if cds_note:
            parts.append(f"\n{cds_note}")
        parts.append("\nWrite the SQLite query.")
        return "\n".join(parts)

    def solve(self, question: str) -> str:
        hint = question.split("\nHint:", 1)[1].strip() if "\nHint:" in question else ""
        enriched = self._enrich_schema()
        value_notes = self._probe_hint_values(hint)
        col_guide = self._hint_col_guide(hint)
        numeric_notes = self._probe_numeric_values(question)
        cds_note = self._probe_cds_join()
        if self.db.db_id == "california_schools":
            home_guide = _CA_GUIDE
        elif self.db.db_id == "toxicology":
            home_guide = _TOX_GUIDE
        else:
            home_guide = ""

        prompt = self._build_prompt(question, enriched, value_notes,
                                     col_guide, home_guide, cds_note, numeric_notes)
        resp = self.llm(prompt, system=SYS, temperature=0.0)
        sql = self._fix_year(bridge.extract_sql(resp))

        result = self.execute(sql)
        ok = result.get("ok", False)
        n_rows = result.get("n_rows") or result.get("n", 0)

        if not ok or (ok and n_rows == 0):
            retry = self._build_prompt(question, enriched, value_notes,
                                        col_guide, home_guide, cds_note, numeric_notes)
            terminal = "\nWrite the SQLite query."
            if retry.endswith(terminal):
                retry = retry[:-len(terminal)]
            if not ok:
                retry += (
                    f"\n\nThe previous SQL failed:\n{result.get('error', 'unknown')}\n\n"
                    f"Write a corrected SQLite query inside ```sql ... ```.")
            else:
                retry += (
                    f"\n\nThe previous SQL returned 0 rows (empty result).\n"
                    f"Possible causes: unnecessary table join, wrong join condition, "
                    f"case mismatch, or overly restrictive filter.\n\n"
                    f"Write a corrected SQLite query inside ```sql ... ```.")
            resp2 = self.llm(retry, system=SYS, temperature=0.0)
            sql2 = self._fix_year(bridge.extract_sql(resp2))
            if sql2:
                sql = sql2
        return sql
