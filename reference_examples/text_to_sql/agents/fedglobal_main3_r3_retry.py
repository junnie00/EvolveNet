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
    "10. BOOLEAN NEGATIONS: When the question uses \"not\", \"without\", \"no\", or \"except\", check whether the Hint "
    "maps a positive condition to a boolean flag. If the Hint says \"X refers to flag = 1\" and the question says \"not X\", "
    "use flag != 1 (or flag = 0) — NOT flag = 1.\n"
    "11. If \"Date format notes\" are provided below the schema, follow them strictly. "
    "Columns stored as 'YYYYMM' (6-digit, no hyphens) are NOT compatible with strftime(). "
    "Use direct string comparison or SUBSTR for year extraction.\n"
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
    "7. GROUP_CONCAT(DISTINCT x) is fine; do NOT pass a second separator argument to DISTINCT.\n"
    "8. Only JOIN the molecule table when filtering or sorting by the label column "
    "('+'/'-' for carcinogenic/non-carcinogenic). The molecule table may be INCOMPLETE — some molecule_ids "
    "that exist in bond or atom tables may have NO entry in the molecule table. "
    "For identifying molecules by bond_type or element, use the bond or atom table directly.\n"
    "9. When listing element and bond_type pairs for a molecule (e.g. \"List the element and bond type included "
    "in the molecule\"), JOIN atom and bond ON molecule_id directly — this gives all element-bond_type "
    "pairs present in that molecule. Do NOT involve the connected table.\n"
    "10. When asked to list bond types for molecules (e.g. \"List down the bond type for molecules\"), "
    "include both molecule_id and bond_type in the SELECT and ORDER BY molecule_id — show the bond types "
    "per molecule, not just the distinct bond types across all molecules.")

_SYS_F1 = ("\n\n--- FORMULA_1 SCHEMA TERM-TO-COLUMN MAPPING ---\n"
    "CRITICAL — Read these BEFORE writing SQL. These are term-to-column rules for THIS database.\n\n"
    "\"lap record\" / \"fastest lap\" / \"fastest lap time\":\n"
    "  → results.fastestLapTime (the fastest lap time recorded in each race, format M:SS.mmm). "
    "NEVER use the lapTimes table for \"lap records\" — lapTimes stores EVERY practice/qualifying/race lap, not records.\n\n"
    "\"lap time\" / \"average lap time\":\n"
    "  → lapTimes.milliseconds (numeric milliseconds) or lapTimes.time (text format).\n\n"
    "\"pit stop\" / \"pit stop duration\" / \"time spent at pit stop\":\n"
    "  → pitStops.duration (duration in seconds as text) or pitStops.milliseconds (duration in ms).\n\n"
    "\"ranked N\" / \"ranked N position\" / \"position in race\":\n"
    "  → results.position (race finish position — 1st, 2nd, 3rd...). "
    "Do NOT use driverStandings.position (that is championship standing, NOT race finish position).\n\n"
    "\"points scored\" / \"score\" in a race:\n"
    "  → results.points (points earned in that specific race). "
    "Do NOT use driverStandings.points (which is cumulative championship points).\n\n"
    "\"grid\" / \"grid formation\" / \"starting grid\":\n"
    "  → results.grid (the grid position at race start).\n\n"
    "\"full name\" of a driver:\n"
    "  → TWO SEPARATE columns: drivers.forename, drivers.surname — do NOT concatenate.\n\n"
    "\"date of birth\" / \"DOB\" / \"birthday\":\n"
    "  → drivers.dob (YYYY-MM-DD string).\n\n"
    "\"Wiki page\" / \"url\" / \"Wikipedia link\":\n"
    "  → drivers.url (for a driver), races.url (for a race), circuits.url (for a circuit).\n\n"
    "--- TIME FORMAT PARSING ---\n"
    "When converting fastestLapTime (format M:SS.mmm) to seconds, the format is variable-width "
    "(single-digit minutes possible, e.g. '1:27.452'). Use INSTR-dynamic parsing:\n"
    "  • minutes = CAST(SUBSTR(col, 1, INSTR(col, ':') - 1) AS REAL)\n"
    "  • seconds = CAST(SUBSTR(col, INSTR(col, ':') + 1, 2) AS REAL)\n"
    "  • milliseconds = CAST(SUBSTR(col, INSTR(col, '.') + 1) AS REAL)\n\n"
    "--- CIRCUIT/RACE RULES ---\n"
    "1. Circuits are in countries — filter with circuits.country.\n"
    "2. Race names in races.name DO NOT include the year (e.g., 'Turkish Grand Prix', NOT '2008 Turkish Grand Prix').\n"
    "3. Use races.year for year-based filtering (integer column).\n"
    "4. driverStandings.position stores cumulative championship standing per race — usually NOT what the question "
    "means by \"ranked\" or \"position\". Use results.position for race finish position.\n"
    "5. The lapTimes table stores ALL laps for ALL sessions — do NOT use it when the question asks about "
    "race results, race records, or fastest laps.")


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

    def _date_format_notes(self):
        """Detect date columns stored as YYYYMM and return guidance string."""
        notes = []
        tables = self.tables()
        for tbl, cols in tables.items():
            for col in cols:
                if 'date' not in col.lower():
                    continue
                key = (tbl, col)
                if key not in _KNOWN:
                    _KNOWN[key] = self.distinct(tbl, col, limit=5)
                vals = _KNOWN[key]
                if not vals:
                    continue
                sample = str(vals[0])
                if len(sample) == 6 and sample.isdigit():
                    notes.append(f"  {tbl}.{col}: stored as 'YYYYMM' (e.g., '{sample}'). "
                               f"NOT compatible with strftime(). "
                               f"Use direct string comparison or SUBSTR({col},1,4) for year extraction.")
        if notes:
            return ("\nDate format notes (follow these exactly when filtering by date):\n" +
                    "\n".join(notes))
        return ""

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
        df = self._date_format_notes()
        sys_p = _SYS + (_SYS_TOX if self.db.db_id == 'toxicology' else
                        (_SYS_F1 if self.db.db_id == 'formula_1' else ""))

        parts = ["Database schema:", self.schema]
        if vs: parts.append(vs)
        if df: parts.append(df)
        parts += ["", f"Question: {q_text}"]
        if hint: parts.append(f"Hint: {hint}")
        parts += ["", "First, list EXACTLY which columns the question asks for (one per line with the table source).",
                  "Then write the SQLite query using ONLY those columns."]
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
                   f"Remember: no comments, use exact DB values, return only the columns explicitly asked for.")
            if df: fix += f"\n\n{df}"
        else:
            f1 = ""
            if self.db.db_id == 'formula_1':
                f1 = ("\n  • \"lap record\" / \"fastest lap\" = results.fastestLapTime (NOT the lapTimes table)\n"
                      "  • \"ranked\" / \"position\" in races = results.position (race finish, NOT driverStandings.position)\n"
                      "  • \"full name\" = forename and surname as TWO SEPARATE columns (do NOT concatenate)\n"
                      "  • Race names DO NOT include the year — filter by races.year separately\n"
                      "  • When parsing fastestLapTime text (format M:SS.mmm), use INSTR to "
                      "find the colon and dot positions — do NOT assume fixed-width")
            tox = ""
            if self.db.db_id == 'toxicology':
                tox = ("\n  • bond_type values are single characters WITHOUT spaces: '-', '=', '#'\n"
                       "  • Molecule IDs belong to the molecule_id column, NOT the label column\n"
                       "  • Element codes are lowercase short forms (e.g. 'c', 'cl', 'h', 'o', 'n', 's')\n"
                       "  • Label values are '+' (carcinogenic) or '-' (non-carcinogenic)")
            fix = (f"Your earlier SQL executed but returned NO rows. Check these possible causes:\n"
                   f"1. Does the question use NEGATION ('not', 'without', 'no', 'except')? "
                   f"If so, verify you didn't invert the boolean condition.\n"
                   f"2. Verify literal string values match the actual data format in the database.{f1}{tox}\n"
                   f"3. Check JOIN conditions — they might be too restrictive and filtering out all rows.\n\n"
                   f"SQL was:\n{sql}\n\n"
                   f"Fix the query and output the corrected SQLite query.")
            if df: fix += f"\n\n{df}"

        resp2 = self.llm(fix, system=sys_p, temperature=0.0)
        sql = self._strip_comments(bridge.extract_sql(resp2))
        return sql
