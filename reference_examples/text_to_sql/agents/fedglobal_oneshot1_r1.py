"""Federated merge of 5 client evolutions of `bare` (oneshot1 r0c0..r0c4). GLOBAL: hint surfaced authoritatively;
value grounding (hint literals + stored date formats vs live DB); leading-comment strip; execute-validate-retry on
error/empty/all-NULL and targeted output defects; output-shape SYS rules. HOME-SCOPED (db_id-gated): card_games /
california_schools / formula_1 / toxicology schema conventions; schema-gated Price/Amount note for the card DBs."""
import re
from ..harness_base import SQLHarness
from .. import bridge

SYS = (
    "You are an expert Text-to-SQL system for SQLite. Read the schema carefully and output exactly "
    "one SQLite query that answers the question, inside a ```sql ... ``` block.\n\nRULES:\n"
    "1. HINT AUTHORITATIVE: follow every mapping exactly; 'X refers to Y' means use Y.\n"
    "2. Return EXACTLY the asked columns, in order (no extra helper columns, ids, sort keys, aggregates); "
    "'include the <id>' adds that id; an id that IS the answer returns only it.\n"
    "3. Singular extreme ('the most/highest/lowest/longest/average/fastest X') -> ONE row (ORDER BY key + LIMIT 1; "
    "LIMIT K for top K); 'list/name all/what are/how many' -> many rows; 'the set card X is in'/'the rulings for "
    "card X' return ALL rows, never LIMIT 1.\n"
    "4. Full precision (no ROUND unless asked). 5. 'percentage'/'percent' -> ratio * 100.0. 6. '=' is case-sensitive: "
    "use EXACT stored values. 7. 'how many <attr>' mapped to a column -> COUNT(DISTINCT col); 'how many <entity>' "
    "counts rows. 8. SQLite has NO YEAR(): use CAST(strftime('%Y', col) AS INTEGER). 9. No SQL comments. 10. Prefer "
    "a single-level correlated EXISTS over IN/NOT IN (SELECT ...); never EXISTS(EXISTS(...)).\n"
    "11. 'list/name all' of an entity -> raw rows, no DISTINCT (only for 'distinct values' and large single-column "
    "lists). 12. Extreme of a STORED key: no '<key> IS NOT NULL' (NULLs sort first in ASC); extreme of a COMPUTED key "
    "(rate/sum/avg): exclude NULL operands. 13. Listing an attribute (name/website/date): exclude NULL rows unless "
    "supplementary to a primary answer or a bulk phone list (keep the NULL-phone row).\n"
)

_NOTES = {
"card_games": "CARD_GAMES NOTES:\n- 'language' is in BOTH foreign_data (a CARD's text/name) and set_translations (a SET's translated name): a SET question (even 'cards with <lang> writing') -> set_translations.language; a CARD question -> foreign_data.language.\n- 'code of sets' -> set_translations.setCode (join sets.code = setCode).",
"california_schools": "CA NOTES:\n- grades: use frpm.`Low Grade`/`High Grade`; schools.GSoffered/GSserved are NULL for many rows.\n- schools.School/Website/ClosedDate/Ext are NULL for many rows (exclude NULL when that attribute is the listed answer).\n- satscores: cname=county, sname=school, rtype 'S'=school, 'D'=district aggregate; join satscores to schools ON cds=CDSCode (some cds drop the leading '0').\n- stored values: frpm.`Charter Funding Type`='Directly funded'; frpm.`NSLP Provision Status`='Lunch Provision 2'; schools.StatusType='Closed'; frpm.`Charter School (Y/N)`=1/0; EILCode 'HS'=high school.",
"formula_1": "F1 NOTES:\n- 'average points/score' = AVG(driverStandings.points) (cumulative season standings), NOT results.points.\n- 'ranked N' = results.rank = N (fastest-lap rank), NOT positionOrder.\n- Grand Prix names live in races.name ('Austrian Grand Prix'); circuits.name is the venue ('Red Bull Ring') — filter races.name.\n- lapTimes.time is TEXT 'M:SS.mmm': order by lapTimes.milliseconds, never MIN/MAX(time).\n- dob is TEXT 'YYYY-MM-DD'; 'full name'/'forename and surname' = two columns (forename, surname).",
"toxicology": "TOX NOTES:\n- bond.bond_type stores '-'/'='/'#': single/double/triple; molecule.label is '+'/'-'; return raw values ('c','cl','#'), never translated words.\n- 'atom id2'->connected.atom_id2; 'atom ID'->atom_id; 'bond ID'->bond_id; code-like IDs (TR012) go in *_id columns.\n- element+bond_type at molecule level: JOIN atom.molecule_id=bond.molecule_id directly, filter bond.bond_type; use `connected` only when a bond_id is named.\n- molecule is sparse (TR447 has atoms/bonds but no molecule row): a molecule's OWN label needs INNER JOIN to molecule; a child attribute comes from atom/bond.\n- id-ranged molecule set ('TR000 to TR050') ranges over molecule: FROM molecule JOIN bond, filter range on molecule.molecule_id.",
}


class BareHarness(SQLHarness):
    MAX_RETRIES = 2

    def _db_id(self):
        return getattr(self.db, "db_id", "")

    def _extract_hint(self, question):
        q, _, h = question.partition("\nHint:")
        return (q.strip(), h.strip()) if h else (question.strip(), "")

    def _clean(self, sql):
        return re.sub(r'/\*.*?\*/', '', "\n".join(l for l in sql.split("\n") if not l.lstrip().startswith("--")),
                      flags=re.DOTALL).strip().rstrip(";").strip()

    def _context(self, qtext, hint):
        db, tabs, parts = self._db_id(), self.tables(), []
        n = _NOTES.get(db)
        n and parts.append(n)
        q = qtext.lower()
        lows = [[c.lower() for c in cols] for cols in tabs.values()]
        if any("amount" in l and "price" in l for l in lows):
            if "revenue" in q:
                parts.append("NOTE: Price is the amount paid, Amount the quantity — revenue = SUM(Price), never "
                             "SUM(Amount * Price).")
            if re.search(r"paid\s*\d", q) and any("date" in l for l in lows):
                parts.append("NOTE: the amount actually paid is stored in Price (Amount is the quantity) — filter "
                             "'paid <amount>' on Price.")
        if hint and "full name refers to first_name, last_name" in hint:
            parts.append("NOTE: 'full name' maps to first_name + last_name as TWO separate columns.")
        v = []
        if db == "toxicology":
            for t in sorted(tabs):
                for c in sorted(tabs[t]):
                    vals = self.distinct(t, c, limit=50)
                    vals and v.append(f"- {t}.{c}: " + (", ".join(repr(x) for x in vals) if len(vals) <= 25
                                       else f"e.g. {[repr(x) for x in vals[:6]]} ... ({len(vals)}+ distinct)"))
            v and parts.append("Value profile (exact stored strings):\n" + "\n".join(v))
        elif db == "california_schools":
            for t, cs in (("frpm", ["Charter School (Y/N)", "Charter Funding Type", "NSLP Provision Status",
                                    "Educational Option Type", "School Type", "District Type"]),
                          ("schools", ["StatusType", "EdOpsCode", "EILCode", "Charter", "FundingType",
                                       "Magnet", "Virtual", "State", "MailState"])):
                for c in cs:
                    vals = self.distinct(t, c, limit=30)
                    vals and v.append(f"- {t}.{c}: {', '.join(repr(x) for x in vals if x is not None)}")
            v and parts.append("ACTUAL STORED VALUES (use verbatim):\n" + "\n".join(v))
        p = []
        for t, cols in tabs.items():
            for c in cols:
                if "date" in c.lower():
                    vals = self.distinct(t, c, limit=12)
                    vals and p.append(f"- {t}.{c} stores e.g. {vals[:4]!r}")
                for var in {c, c.replace(" ", ""), c.replace("(", "").replace(")", "").replace(" ", "")}:
                    m = re.search(re.escape(var) + r"\s*=\s*(?:'([^']*)'|(\d+))", hint or "", re.IGNORECASE)
                    if m:
                        raw = m.group(1) if m.group(1) is not None else m.group(2)
                        res = self.execute(f'SELECT COUNT(*) FROM "{t}" WHERE "{c}" = \'{raw.replace("'", "''")}\'')
                        if not (res.get("ok") and res["rows"] and res["rows"][0][0] > 0):
                            vals = self.distinct(t, c, limit=10)
                            if vals:
                                ci = any(str(x).lower() == raw.lower() for x in vals if x is not None)
                                p.append(f"- {t}.{c} stores {[repr(x) for x in vals[:6] if x is not None]} "
                                         f"('{raw}' {'has the wrong case' if ci else 'was not found'})")
                        break
        p and parts.append("Verified DB values (use verbatim):\n" + "\n".join(p))
        return "\n\n".join(parts)

    def _issue(self, qtext, hint, sql, res):
        rows = res.get("rows") or []
        if not res.get("ok"):
            return (f"SQL execution error: {res.get('error')}. Fix the SQL so it runs in SQLite — no YEAR(); "
                    "GROUP_CONCAT(DISTINCT col) takes no separator.")
        if not rows:
            return ("The query returned 0 rows — a WHERE literal may not match the stored value (wrong case/"
                    "spelling) or a join key is wrong (e.g. a leading '0' dropped from an id). Use the exact stored "
                    "values shown above.")
        if len(rows) == 1 and all(v is None for v in rows[0]):
            return "The query returned one all-NULL row — the WHERE matched no real data. Fix the filter literals."
        if "how many" in qtext.lower():
            m = re.search(r'([^;]+?)\s*refers to\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?=;|$)', hint)
            if m and (re.search(r'\bCOUNT\s*\(\s*(?:[A-Za-z0-9_]+\.)?' + m.group(2) + r'\s*\)', sql, re.IGNORECASE)
                      or re.search(r'\bCOUNT\s*\(\s*\*\s*\)', sql, re.IGNORECASE)):
                return (f"The question asks 'how many' of an attribute the HINT maps to column '{m.group(2)}'. Count "
                        f"DISTINCT values: COUNT(DISTINCT {m.group(2)}), NOT COUNT({m.group(2)}) and NOT COUNT(*).")
        n = len(rows)
        if n > 10 and len(rows[0]) == 1 and len({tuple(r) for r in rows}) < n:
            return ("The result is a large single-column list with repeated values — add SELECT DISTINCT so each "
                    "value appears once. Do NOT add a '<col> IS NOT NULL' filter: a bulk phone list keeps the NULL "
                    "row.")
        ent = re.search(r"\bthe\s+(customer|client|member|student)\s+(who|that)\b", qtext.lower())
        if ent:
            toks = {"customer": ["customerid"], "client": ["customerid", "clientid"],
                    "member": ["member_id", "memberid"], "student": ["member_id", "memberid", "studentid"]}[ent.group(1)]
            norm = re.sub(r"\s+", " ", sql).upper()
            sel = re.search(r"\bSELECT\b(.*?)\bFROM\b", norm, re.S)
            sel = sel.group(1) if sel else ""
            if len(rows) > 1 and ("IN (" in norm or "EXISTS (" in norm) and not any(t.upper() in sel for t in toks):
                return ("The result has several entities but the SELECT list lacks the entity identifier column — "
                        "add it (e.g. CustomerID, member_id). Keep the FROM/JOIN/WHERE clauses; change ONLY the "
                        "SELECT list.")
        return None

    def _prompt(self, qtext, hint, ctx, issue="", prev=""):
        s = f"Database schema:\n{self.schema}"
        s += f"\n\n{ctx}" if ctx else ""
        s += f"\n\nQuestion: {qtext}"
        s += f"\n\nAUTHORITATIVE HINT (must follow exactly):\n{hint}" if hint else ""
        s += "\n\nWrite the SQLite query."
        s += f"\n\n### PREVIOUS ATTEMPT FEEDBACK\nPrevious SQL:\n{prev}\n\nIssue: {issue}\n\nWrite a corrected " \
              "SQLite query inside ```sql ... ```." if issue else ""
        return s

    def solve(self, question: str) -> str:
        qtext, hint = self._extract_hint(question)
        ctx = self._context(qtext, hint)
        prompt = self._prompt(qtext, hint, ctx)
        sql = ""
        for _ in range(self.MAX_RETRIES + 1):
            resp = self.llm(prompt, system=SYS, temperature=0.0)
            sql = self._clean(bridge.extract_sql(resp))
            if not sql:
                return ""
            res = self.execute(sql)
            issue = self._issue(qtext, hint, sql, res)
            if issue is None:
                return sql
            prompt = self._prompt(qtext, hint, ctx, issue, sql)
        return sql
