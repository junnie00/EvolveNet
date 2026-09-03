"""Phase B: the coder's harness is a composable PIPELINE, authored by the proposer.

Phase A let the proposer rewrite only the prompt TEXT — it could tell the model "match exact
values" but couldn't actually retrieve them. Phase B exposes real OPERATORS the proposer can turn
on/compose (value retrieval, multi-candidate execution-vote, self-repair, decomposition, join
graph) ON TOP of the rewritable prompt. The proposer still DISCOVERS the composition from the
coder's failures, label-free — it is not a human-picked menu.

NOTE on voting: num_candidates is a universal variance-reducer, not the method's credit. The eval
matches it on the baseline so the reported delta isolates the proposer's real contribution.
"""
from dataclasses import dataclass, asdict

from .harness import Harness, DEFAULT_HARNESS


@dataclass
class Pipeline:
    harness: Harness                      # system_prompt + user_template (Phase A power)
    use_value_retrieval: bool = False     # inject real stored column values
    use_join_graph: bool = False          # inject FK join graph
    num_candidates: int = 1               # >1 = sample N + pick majority-by-execution
    use_self_debug: bool = False          # auto-fix a query that errors
    use_decomposition: bool = False       # ask the model to reason step-by-step first

    def to_dict(self):
        return asdict(self)

    def flags(self):
        return (f"value={self.use_value_retrieval} join={self.use_join_graph} "
                f"nc={self.num_candidates} self_debug={self.use_self_debug} "
                f"decompose={self.use_decomposition}")


DEFAULT_PIPELINE = Pipeline(harness=DEFAULT_HARNESS)


def from_dict(d):
    h = d["harness"]
    return Pipeline(
        harness=Harness(system_prompt=h["system_prompt"], user_template=h["user_template"]),
        use_value_retrieval=bool(d.get("use_value_retrieval", False)),
        use_join_graph=bool(d.get("use_join_graph", False)),
        num_candidates=int(d.get("num_candidates", 1)),
        use_self_debug=bool(d.get("use_self_debug", False)),
        use_decomposition=bool(d.get("use_decomposition", False)),
    )
