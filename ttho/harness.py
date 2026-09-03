"""A 'harness' here is just an ase Pipeline (prompt + composable operators) wrapping the frozen solver.
TTHO evolves a POPULATION of these; this module defines the seed population and the execution-result
canonicalizer used to decide when two harnesses 'agree'."""
from dataclasses import replace

from ase.pipeline import DEFAULT_PIPELINE


def default_population():
    """A small DIVERSE seed population — different operators fail differently, which is exactly what
    CHAP needs (their agreement cancels systematic errors). Returns [(name, Pipeline)]."""
    base = DEFAULT_PIPELINE
    return [
        ("bare", base),
        ("value", replace(base, use_value_retrieval=True)),
        ("decompose", replace(base, use_decomposition=True)),
        ("join", replace(base, use_join_graph=True)),
        ("value+decompose", replace(base, use_value_retrieval=True, use_decomposition=True)),
    ]


def result_key(rows, ok, maxn=2000):
    """Canonical, hashable key for an execution result. Two SQLs AGREE iff same non-None key.
    Errors / empty results -> None (a failure, never counts as agreement). Matches the set-comparison
    semantics used for gold scoring (order-insensitive).

    Huge results (> maxn rows, almost always a cartesian/wrong answer) get a cheap size-based key
    instead of sorting 100k strings — that sort was a CPU hotspot. A BIG result won't match the small
    gold anyway, so it's correctly scored wrong."""
    if not ok or not rows:
        return None
    if len(rows) > maxn:
        return f"BIG:{len(rows)}"
    return str(sorted(str(x) for x in rows))[:4000]
