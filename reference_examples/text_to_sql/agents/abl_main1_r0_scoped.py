"""Merged harness from fed_evo round main1_r0.

Mechanisms by source:
- Client 2 (formula_1): SQLite compat guidance, exec-error retry, enriched schema with types.
- Client 3 (toxicology): hint-value probing, scoped entity-ID→PK rule.
- Client 4 (debit_card_specializing, student_club): numeric-value probing, output-alignment
  rules, pseudo-code formula note, empty-result retry.

Home-scoped rules: entity-ID→PK mapping only for toxicology."""
from ..harness_base import SQLHarness
from .. import bridge
import re

SYS = ("You are an expert Text-to-SQL system for SQLite. Read the schema carefully and output exactly "
       "one SQLite query that answers the question, inside a ```sql ... ``` block.\n\n"
       "Rules:\n"
       "1. Return ONLY the columns the question explicitly asks for — never add extra columns.\n"
       "2. When the Hint maps a concept to multiple columns (e.g. 'full name → first_name, last_name'),\n"
       "   keep them as SEPARATE SELECT columns — do NOT concatenate.\n"
       "3. Prefer direct FK joins over routing through intermediate tables.\n"
       "4. Use DISTINCT when the question asks for a list/set of items; avoid paired columns when\n"
       "   a simple distinct set is asked for.\n"
       "5. Do NOT use UNION or UNION ALL. For multi-table set queries, use SELECT DISTINCT\n"
       "   with IN subqueries instead.\n"
       "6. SQLite: YEAR() does NOT exist (use strftime('%%Y', col)).\n"
       "Output exactly one SQLite query inside a ```sql ... ``` block.")

MAX_RETRIES = 2


class BareHarness(SQLHarness):

    @staticmethod
    def _hint_text(question: str) -> str:
        return question.split("\nHint:", 1)[1].strip() if "\nHint:" in question else ""

    def _enriched_schema(self) -> str:
        """Incorporate column types (hide TEXT annotations to avoid spurious WHERE filters)."""
        lines = []
        for tname, cols in self.tables().items():
            types = self.column_types(tname)
            col_strs = []
            for c in cols:
                t = types.get(c, "")
                if t and t.upper() != "TEXT":
                    col_strs.append(f"{c} ({t})")
                else:
                    col_strs.append(c)
            lines.append(f"Table {tname}({', '.join(col_strs)})")
        for fk in self.db.schema.get("foreign_keys", []):
            lines.append(f"FK {fk['from_table']}.{fk['from_col']} -> {fk['to_table']}.{fk['to_col']}")
        return "\n".join(lines)

    def _hint_value_probes(self, hint: str) -> list:
        notes, seen = [], set()
        if not hint:
            return notes
        for seg in hint.replace(";", ",").split(","):
            seg = seg.strip()
            if "= '" not in seg:
                continue
            idx = seg.index("= '")
            col = seg[:idx].strip().split()[-1] if seg[:idx].strip().split() else ""
            after = seg[idx + 2:]
            val = after[1:].split("'")[0] if after.startswith("'") and "'" in after[1:] else ""
            if not val or not col or col in seen:
                continue
            seen.add(col)
            for tbl, tbl_cols in self.tables().items():
                if col not in tbl_cols:
                    continue
                actual = self.distinct(tbl, col, limit=20)
                if actual and val not in actual:
                    notes.append(
                        f"Note: Hint says {col}='{val}', but actual distinct "
                        f"values in {tbl}.{col}: {', '.join(repr(v) for v in actual)}"
                    )
        return notes

    def _numeric_value_probes(self, question: str) -> list:
        findings = []
        for raw in re.findall(r"(?<!\w)(\d+(?:\.\d+)?)(?!\w)", question):
            try:
                target = float(raw)
            except ValueError:
                continue
            for tbl, cols in self.tables().items():
                for col in cols:
                    ct = self.column_types(tbl).get(col, "")
                    if not any(k in ct.lower() for k in ("real", "int", "float", "numeric", "double")):
                        continue
                    try:
                        res = self.execute(
                            f'SELECT COUNT(*) AS _c FROM "{tbl}" WHERE ABS("{col}" - {target}) < 0.001'
                        )
                        if res["ok"] and res["rows"] and res["rows"][0][0] > 0:
                            findings.append(
                                f"  VALUE {raw} exists in {tbl}.{col} ({res['rows'][0][0]} rows)"
                            )
                    except Exception:
                        pass
        return findings

    def solve(self, question: str) -> str:
        hint = self._hint_text(question)
        schema = self._enriched_schema()
        notes_hint = self._hint_value_probes(hint)
        notes_num = self._numeric_value_probes(question)

        prompt = f"Database schema:\n{schema}\n\nQuestion: {question}"
        if notes_hint:
            prompt += "\n\n" + "\n".join(notes_hint)
        if notes_num:
            prompt += "\n\n--- Values found in columns ---\n" + "\n".join(notes_num)
        if self.db.db_id == "toxicology":
            prompt += ("\n\nNote: When the question gives a specific entity ID (e.g. 'TR012 molecule'), "
                       "filter on the table's primary-key / ID column (molecule_id), "
                       "NOT on other string columns like label.")
        if hint and any(fn in hint for fn in ("SUBTRACT(", "DIVIDE(", "SUM(", "COUNT(")):
            prompt += (
                "\n\nNote: The Hint shows a pseudocode formula. Inner expressions like "
                "COUNT(col = 'value') or SUM(col = 'value') are SQL boolean aggregates "
                "in the SELECT clause — do NOT add WHERE filters on those columns."
            )
        prompt += "\n\nWrite the SQLite query."

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

            retry = [f"The previous SQL did not produce a valid result:\n```sql\n{sql}\n```"]
            if not result.get("ok"):
                err = str(result.get("error", "Unknown error"))
                retry.append(f"Execution error: {err}")
                if "group_concat" in err.lower() or "distinct aggregates" in err.lower():
                    retry.append("GROUP_CONCAT with DISTINCT cannot take a second separator argument.")
            else:
                retry.append("The query returned zero rows. Possible fixes:")
                retry.append("- Verify filter columns match the actual stored values.")
                retry.append("- Check date/time column formats.")
                retry.append("- Make sure join conditions are correct.")
            prompt = "\n".join(retry)
            prompt += f"\n\nDatabase schema:\n{schema}\n\nQuestion: {question}\n\nWrite the corrected SQLite query."

        return last_sql
