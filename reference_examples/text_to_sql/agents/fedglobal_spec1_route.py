"""Delegation-composed global harness (aggregation by expert delegation, zero-rewrite).

Each client's ENTIRE harness is preserved verbatim; aggregation only decides WHO answers: questions on a
client's home database go to that client's harness, everything else to the client that scored highest on
the server's gate slice (val99). Rule- or code-level transfer measurably loses expert behaviour in
rewriting (scoped merge: -2 at toxicology/card_games home vs the expert; verbatim rule-copying imported
poisonous rules along with good ones). Delegating the whole program is the zero-loss end of that spectrum,
costs no merger session, and cannot break any database another expert owns.
"""
from ..harness_base import SQLHarness


class RouteHarness(SQLHarness):
    HOME = {
        "card_games": "cand_spec1_r0_c0_b0r0_g0",
        "formula_1": "cand_spec1_r0_c2_b0r0_g0",
        "toxicology": "cand_spec1_r0_c3_b0r0_g0",
    }
    DEFAULT = "cand_spec1_r0_c3_b0r0_g0"   # highest gate (val99) score among clients

    def solve(self, question: str) -> str:
        import importlib
        name = self.HOME.get(self.db.db_id, self.DEFAULT)
        mod = importlib.import_module(f"{__package__}.{name}")
        cls = next(o for o in vars(mod).values()
                   if isinstance(o, type) and issubclass(o, SQLHarness) and o is not SQLHarness)
        inner = cls(self.db)
        try:
            return inner.solve(question)
        finally:
            self._trace = inner._trace     # expose the delegate's full trace as our own
