"""The HARNESS interface (analogue of meta-harness's MemorySystem, for Text-to-SQL).

A harness is ARBITRARY PYTHON wrapping the FROZEN weak solver. Unlike a prompt + a fixed menu of
toggles, `solve()` can do anything the proposer writes: retrieve real DB values, decompose into
sub-steps, sample + execution-vote, self-verify against the schema, repair failed queries, etc.
This is the meta-harness representation — the proposer writes subclasses of SQLHarness.

LABEL-FREE: a harness may adapt ONLINE on the unlabeled test stream via observe(), which receives only
the question, its own SQL, and the EXECUTION result — never the gold. Gold is used outside, for
measurement only.
"""
from abc import ABC, abstractmethod

from . import bridge


class SQLHarness(ABC):
    """Subclass this. Available to every harness:
        self.db          — the database (has .schema_text(), .execute())
        self.schema      — db.schema_text() (string, precomputed)
        self.llm(prompt, system="", temperature=0.0, n=1)  — the frozen solver
        self.execute(sql)  -> {ok, rows, ...}              — run SQL on self.db
        bridge.extract_sql(text) -> str                    — pull SQL out of an LLM response
    """

    def __init__(self, db):
        self.db = db
        self.schema = db.schema_text()
        self._trace = []          # FULL execution trace of this solve: every coder call + every SQL run, in order
        self._call_seq = {}       # request signature -> times THIS instance already issued it

    # convenience wrappers (so harness code reads cleanly) — they also RECORD into self._trace so the
    # proposer can deep-read the harness's complete step-by-step behaviour (not a compressed summary).
    def llm(self, prompt, system="", temperature=0.0, n=1):
        # seq = how many times THIS instance already issued this exact request. Solver replies are cached
        # (see bridge.solver_llm), so without seq a harness that asks the same question three times in
        # order to vote would get one answer three times, silently deleting the mechanism it was built
        # around. With seq it draws three distinct replies, while a different harness making the same
        # asks receives the same three — identical behaviour scores identically.
        sig = (prompt, system, temperature, n)
        seq = self._call_seq.get(sig, 0)
        self._call_seq[sig] = seq + 1
        out = bridge.solver_llm(prompt, system=system, temperature=temperature, n=n, seq=seq)
        self._trace.append({"step": "coder_llm", "system": system, "prompt": prompt,
                            "response": out if isinstance(out, str) else list(out)})
        return out

    def execute(self, sql):
        res = bridge.execute(self.db, sql)
        self._trace.append({"step": "execute_sql", "sql": sql, "ok": res.get("ok"),
                            "error": res.get("error"), "rows": res.get("rows", [])[:5], "n_rows": len(res.get("rows", []))})
        return res

    def tables(self):
        """{table_name: [column_name, ...]} straight from the live DB. Use THIS — do NOT regex-parse
        self.schema (its format is `Table name(col1, col2, ...)`, not CREATE TABLE)."""
        return {t["name"]: [c["name"] for c in t["columns"]] for t in self.db.schema["tables"]}

    def column_types(self, table):
        """{column_name: type_string} for one table."""
        for t in self.db.schema["tables"]:
            if t["name"] == table:
                return {c["name"]: c.get("type", "") for c in t["columns"]}
        return {}

    def distinct(self, table, column, limit=50):
        """Distinct stored values of a column (exact strings) — for matching question literals to the
        real value. Returns [] on error."""
        res = self.execute(f'SELECT DISTINCT "{column}" FROM "{table}" WHERE "{column}" IS NOT NULL LIMIT {limit}')
        return [r[0] for r in res["rows"]] if res["ok"] else []

    @abstractmethod
    def solve(self, question: str) -> str:
        """Return a single SQLite query string answering `question` over self.db."""
        ...

    # ---- optional LABEL-FREE online adaptation on the test stream (NO gold, ever) ----
    def observe(self, question: str, sql: str, exec_result: dict) -> None:
        """Called after solve() with the execution result (no gold). Default: no adaptation."""
        return None

    def get_state(self) -> str:
        return ""

    def set_state(self, state: str) -> None:
        return None
