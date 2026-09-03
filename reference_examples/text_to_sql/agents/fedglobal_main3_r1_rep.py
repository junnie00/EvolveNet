"""Merged global: value-grounding + NULL-safety + output alignment + comment-strip + exec/empty retry + toxicology rules."""
from ..harness_base import SQLHarness
from .. import bridge

_STOP = frozenset({'the','and','for','with','that','this','are','has','have','was','were','can','its','not','but','all','any','each','from','which','what','how','who','where','when','why','than','they','them','their','your','our','his','her'})
_KNOWN = {}

_SYS = ("You are an expert Text-to-SQL system for SQLite. "
    "CRITICAL RULES (follow each one exactly):\n"
    "1. Use ONLY the exact values shown in \"Verified exact values from the database\" — never guess or abbreviate string literals. "
    "Match string values exactly as shown, preserving original case — do NOT wrap column references with LOWER() or UPPER().\n"
    "2. Add IS NOT NULL filters ONLY on numeric columns used in arithmetic expressions (e.g. col1 * col2, a / b, strftime(...) - strftime(...)) that could produce NULL. "
    "Do NOT add IS NOT NULL to columns that appear only in the SELECT list, to columns used only in ORDER BY, or to columns compared only with = in WHERE.\n"
    "3. Return EXACTLY the columns the question asks for — no extra identifier columns (e.g. if asked for \"rates\", return only rates; if asked for \"email\", return only the email).\n"
    "4. Use LIMIT 1 only when the question asks for the entity with the most/least/highest/lowest/best/worst of some attribute "
    "(e.g. \"the most answers\", \"the lowest price\", \"which school has the highest enrollment\"). "
    "Do NOT add LIMIT 1 to questions phrased as \"How many\", \"State the\", or \"List the\" — these may return multiple matching rows.\n"
    "5. Do NOT use ROUND(), CEIL(), or FLOOR() unless the question explicitly asks for rounding.\n"
    "6. Do NOT include any comments (-- or /* */) in the SQL.\n"
    "7. Follow the Hint instructions exactly — they map natural-language terms to specific column names and value rules. "
    "If the hint says \"first_name, last_name\", return them as SEPARATE columns — NEVER concatenate.\n"
    "8. When joining, prefer explicit column references (table.column) to avoid ambiguity. "
    "Use the EXACT column names from the schema — preserve quoting (backticks or double-quotes) around column names that contain spaces or special characters.\n"
    "9. For questions involving multiple tables related by foreign keys, use explicit JOIN syntax — do NOT use scalar subqueries (SELECT ... WHERE ...) as substitutes for JOINs.\n"
    "Output exactly one SQLite query inside a ```sql ... ``` block. Keep the SQL concise and ensure it is syntactically complete.")

_SYS_TOX = ("\n\n--- TOXICOLOGY SCHEMA ---\n"
    "molecule(molecule_id, label): molecule_id is unique ID; label is '+' (carcinogenic) or '-' (non-carcinogenic). "
    "Never use label for molecule identification.\n"
    "atom(atom_id, molecule_id, element): element codes (c=Carbon, cl=Chlorine, h=Hydrogen, o=Oxygen, n=Nitrogen, s=Sulfur). Use short code.\n"
    "bond(bond_id, molecule_id, bond_type): single char NO spaces: '-' single, '=' double, '#' triple.\n"
    "connected(atom_id, atom_id2, bond_id): pairs of atoms in a bond.\n\n"
    "--- TOXICOLOGY RULES ---\n"
    "1. When listing items always use SELECT DISTINCT.\n"
    "2. Atoms' elements forming a bond: return just element values (one per row), NOT ordered pairs.\n"
    "3. \"atom ID\" = atom.atom_id; \"atom id2\"/\"atom_id2\" = connected.atom_id2.\n"
    "4. Molecule IDs are molecule_id VALUES (e.g. 'TR012'), NOT label. Filter on molecule_id column.\n"
    "5. For ranges like \"TR010 to TR050\", use CAST(substr(molecule_id,3,3) AS INTEGER).\n"
    "6. Do NOT add LIMIT unless the question explicitly asks for a specific number of results.\n"
    "7. GROUP_CONCAT(DISTINCT x) is fine; do NOT pass a second separator argument to DISTINCT.")


class BareHarness(SQLHarness):
    def _words(self, text):
        for ch in '()/\'",.':
            text = text.replace(ch, ' ')
        return {w for w in text.lower().split() if len(w) > 2 and w not in _STOP}

    def _probe_values(self, question, hint):
        combined = (question + " " + hint).lower()
        tables = self.tables()
        ctypes = {t: self.column_types(t) for t in tables}
        NUM = ('int','real','float','double','numeric','integer','decimal','number')
        result = {}
        for tbl, cols in tables.items():
            relevant = []
            for col in cols:
                ct = ctypes.get(tbl, {}).get(col, '').lower()
                if any(t in ct for t in NUM):
                    continue
                words = self._words(col)
                if not words or not any(w in combined for w in words):
                    continue
                key = (tbl, col)
                if key not in _KNOWN:
                    _KNOWN[key] = self.distinct(tbl, col, limit=20)
                vals = _KNOWN[key]
                if vals and len(vals) <= 15:
                    relevant.append((col, vals))
            if relevant:
                result[tbl] = relevant
        return result

    def _fmt_values(self, vg):
        if not vg:
            return ""
        lines = []
        for tbl in sorted(vg):
            for col, vals in vg[tbl]:
                display = [str(v) for v in vals[:8]]
                lines.append(f"  {tbl}.{col}: {', '.join(display)}")
        return ("\nVerified exact values from the database (use these strings "
                "verbatim in WHERE filters — never guess or abbreviate):\n" +
                "\n".join(lines)) if lines else ""

    @staticmethod
    def _strip_comments(sql):
        out = []; in_block = False
        for line in sql.split('\n'):
            if '/*' in line and not in_block:
                idx = line.find('/*')
                prefix = line[:idx]
                if prefix.count("'") % 2 == 0 and prefix.count('"') % 2 == 0:
                    rest = line[idx+2:]
                    if '*/' in rest:
                        line = prefix + rest[rest.find('*/')+2:]
                    else:
                        in_block = True; line = prefix
            elif in_block:
                if '*/' in line:
                    in_block = False; line = line[line.find('*/')+2:]
                else:
                    continue
            if '--' in line:
                idx = line.find('--')
                prefix = line[:idx]
                if prefix.count("'") % 2 == 0 and prefix.count('"') % 2 == 0:
                    line = prefix
            out.append(line)
        return '\n'.join(out)

    def solve(self, question: str) -> str:
        q_text = question; hint = ""
        if "\nHint:" in question:
            parts = question.split("\nHint:", 1)
            q_text = parts[0].strip(); hint = parts[1].strip()

        vg = self._probe_values(q_text, hint)
        vs = self._fmt_values(vg)
        sys_p = _SYS + (_SYS_TOX if self.db.db_id == 'toxicology' else "")

        parts = ["Database schema:", self.schema]
        if vs: parts.append(vs)
        parts += ["", f"Question: {q_text}"]
        if hint: parts.append(f"Hint: {hint}")
        parts += ["", "Write the SQLite query."]
        prompt = "\n".join(parts)

        resp = self.llm(prompt, system=sys_p, temperature=0.0)
        sql = self._strip_comments(bridge.extract_sql(resp))
        result = self.execute(sql)
        ok = result.get("ok", False)
        has_rows = result.get("rows") and len(result["rows"]) > 0

        if ok and has_rows:
            return sql

        # --- retry: exec error or empty ---
        if not ok:
            err = result.get("error", "Unknown error")
            fix = (f"The SQL you wrote failed with error:\n{err}\n\nSQL was:\n{sql}\n\n"
                   f"Please fix the error and write a corrected SQLite query. "
                   f"Remember: no comments, use exact DB values, return only the columns asked for.")
        else:
            tox = ""
            if self.db.db_id == 'toxicology':
                tox = ("\n  • bond_type values are single characters WITHOUT spaces: '-', '=', '#'\n"
                       "  • Molecule IDs belong to the molecule_id column, NOT the label column\n"
                       "  • Element codes are lowercase short forms (e.g. 'c', 'cl', 'h', 'o', 'n', 's')\n"
                       "  • Label values are '+' (carcinogenic) or '-' (non-carcinogenic)")
            fix = (f"Your earlier SQL executed but returned NO rows. Verify literal string values match "
                   f"the actual data format in the database:{tox}\n\n"
                   f"SQL was:\n{sql}\n\n"
                   f"If you used a computed expression like column_A * column_B = X, "
                   f"try filtering by each column individually instead of multiplying.\n"
                   f"Fix the query and output the corrected SQLite query.")

        resp2 = self.llm(fix, system=sys_p, temperature=0.0)
        sql = self._strip_comments(bridge.extract_sql(resp2))
        return sql
