"""Domain adapters: the ONLY place fedkit knows one benchmark from another.

Everything the federated layer needs from a domain is behind this interface: how to read/write a slice
file, how to launch one client's local evolution, how to score a harness on a slice, and the text block
that tells the merger what a harness in this domain looks like. The orchestration itself (broadcast,
parallel clients, per-question acceptance, retry-with-feedback, rollback, aggregation variants) is
domain-blind and lives once in fedkit.

Ported pitfalls this interface deliberately preserves from the SQL build-out and TTHE_mono:
  * every child process runs under PYTHONHASHSEED=0 — set-iteration order otherwise differs per process,
    reaches prompts via probe text, changes solver-cache keys, and resurrects the sampling noise the
    cache exists to remove (measured on SQL: same harness scored 16/17/17 across three "identical" runs);
  * scoring runs in a SUBPROCESS per (harness, slice) — generated harness code is untrusted and a hang
    must cost one timeout, not the orchestrator;
  * per-problem verdict caching on (problem-id, code) — candidates routinely emit byte-identical code and
    the hidden suite is the expensive part of a supervised round.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


class Adapter:
    name = ""
    trace_glob = "__q*.md"   # per-item trace filename pattern under a client's traces/b0/
    pkg = ""                 # python package of the domain
    optimize_mod = ""        # module run as one client
    agents_dir = None        # where harness .py files live
    logs_dir = None
    cache_env = ""           # the domain bridge's solver-cache env var

    # -- slice handling -----------------------------------------------------------------
    def slice_ids(self, slice_path):
        raise NotImplementedError

    def write_slice(self, ids, out_path):
        raise NotImplementedError

    # -- scoring (runs inside fedkit.evaluate, already in its own process) ---------------
    def load_items(self, ids):
        """-> list of problem objects, in `ids` order."""
        raise NotImplementedError

    def solve_and_grade(self, harness_name, items, solve_timeout):
        """-> rows [{i, id, correct, artifact}] — shared implementation, domain hooks below."""
        from concurrent.futures import ThreadPoolExecutor
        cls = type(self.load_harness(harness_name, items[0]))
        cache, rows = {}, [None] * len(items)

        def one(j):
            p = items[j]
            try:
                from concurrent.futures import ThreadPoolExecutor as TP
                with TP(max_workers=1) as tp:
                    art = tp.submit(cls(p).solve).result(timeout=solve_timeout) or ""
            except Exception:  # noqa: BLE001 — untrusted generated code; a crash is a wrong answer
                art = ""
            key = (self.item_id(p), art)
            if key not in cache:
                cache[key] = bool(self.is_correct(art, p)) if art else False
            rows[j] = {"i": j, "id": self.item_id(p), "domain": self.item_domain(p),
                       "correct": cache[key], "artifact": str(art)[:400]}
        with ThreadPoolExecutor(max_workers=6) as ex:
            list(ex.map(one, range(len(items))))
        return rows

    def load_harness(self, name, problem):
        raise NotImplementedError

    def item_id(self, p):
        raise NotImplementedError

    def item_domain(self, p):
        """The natural client domain of an item (library / repo / difficulty) — the dispatch key that
        route aggregation gates on. Must be computable at solve time inside the harness too."""
        raise NotImplementedError

    def route_source(self, home, default):
        """Source text of a delegation-composed global harness (aggregation artifact of --aggregate
        route). Mechanical, no LLM: home domain -> that client's harness verbatim, else the gate-best."""
        raise NotImplementedError

    def is_correct(self, artifact, p):
        raise NotImplementedError

    # -- client launch -------------------------------------------------------------------
    def client_cmd(self, shard_path, global_name, run_name, k, args):
        """argv for one client's local evolution (the whole shard as ONE batch, E generate rounds)."""
        n = len(self.slice_ids(shard_path))
        return [sys.executable, "-u", "-m", self.optimize_mod,
                "--pilot", str(shard_path),
                "--initial-harness", global_name,
                "--run-name", run_name,
                "--batch-size", str(n),
                "--group", str(args.group),
                "--max-rounds", str(args.local_rounds),
                "--role-offset", str(k),
                "--propose-timeout", str(args.propose_timeout),
                "--solve-timeout", str(args.solve_timeout),
                "--model", args.model,
                "--supervised"] + (["--specialist"] if getattr(args, "specialist", False) else [])

    # -- merger-facing description -------------------------------------------------------
    def merge_blurb(self):
        """Domain-specific text for the merger prompt: what a harness is, its API, the invariants,
        and how to verify the merged file imports."""
        raise NotImplementedError


class LCBAdapter(Adapter):
    name, pkg = "lcb", "livecodebench"
    optimize_mod = "livecodebench.lcb_optimize"
    agents_dir = ROOT / "livecodebench" / "agents"
    logs_dir = ROOT / "livecodebench" / "logs"
    cache_env = "LCB_SOLVER_CACHE"

    def _bridge(self):
        from livecodebench import lcb_bridge
        return lcb_bridge

    def slice_ids(self, slice_path):
        spec = json.loads(Path(slice_path).read_text(encoding="utf-8"))
        items = spec["items"] if isinstance(spec, dict) else spec
        return [(it["qid"], it.get("difficulty", "?")) if isinstance(it, dict) else (it, "?")
                for it in items]

    def write_slice(self, ids, out_path):
        Path(out_path).write_text(json.dumps(
            {"items": [{"qid": q, "difficulty": d} for q, d in ids]}, indent=1), encoding="utf-8")

    def load_items(self, ids):
        b = self._bridge()
        allp = {p.qid: p for p in b.load_problems("test6", stdin_only=False)}
        out = []
        for q, d in ids:
            p = allp[q]
            p.difficulty = d
            out.append(p)
        return out

    def load_harness(self, name, problem):
        from livecodebench.lcb_common import load_harness
        return load_harness(name, problem)

    def item_id(self, p):
        return str(p.qid)

    def is_correct(self, code, p):
        return self._bridge().is_correct(code, p)

    def item_domain(self, p):
        return getattr(p, "difficulty", "?")

    def route_source(self, home, default):
        return (
            '"""Delegation-composed global harness (auto-generated, --aggregate route)."""\n'
            "from ..harness_base import CodeHarness\n\n\n"
            "class RouteHarness(CodeHarness):\n"
            f"    HOME = {home!r}\n"
            f"    DEFAULT = {default!r}\n\n"
            "    def solve(self) -> str:\n"
            "        from ..lcb_common import load_harness\n"
            "        name = self.HOME.get(getattr(self.problem, 'difficulty', '?'), self.DEFAULT)\n"
            "        inner = load_harness(name, self.problem)\n"
            "        try:\n"
            "            return inner.solve()\n"
            "        finally:\n"
            "            self._trace = inner._trace\n")

    def merge_blurb(self):
        return (
            "A harness is arbitrary Python wrapping a FROZEN weak coder for competitive programming "
            "(LiveCodeBench). Its solve() returns a complete program string. API available to it: "
            "self.content (problem text); self.public_tests; self.starter_code; self.llm(prompt, system='', "
            "thinking=False|'low'|'medium'|'high', n=1, max_tokens=None); self.run_public(code); "
            "self.stress(code); bridge.extract_code(text); bridge.back_translate(code). "
            "INVARIANTS (audited): FROZEN SOLVER — no new client/model/endpoint; LABEL-FREE — never touch "
            "hidden tests or is_correct, no per-problem hardcoding. The merged file must keep "
            "`from ..harness_base import CodeHarness` and `from .. import lcb_bridge as bridge`.\n"
            "Verify the merged file imports:\n"
            f"  PYTHONPATH={ROOT} python -c \"from livecodebench import lcb_bridge as b; "
            "from livecodebench.lcb_common import load_harness; "
            "p=b.load_problems('test6',stdin_only=False)[0]; load_harness('<NAME>', p); print('LOADS OK')\"")


class DS1000Adapter(Adapter):
    name, pkg = "ds1000", "ds1000"
    optimize_mod = "ds1000.ds1000_optimize"
    agents_dir = ROOT / "ds1000" / "agents"
    logs_dir = ROOT / "ds1000" / "logs"
    cache_env = "DS1000_SOLVER_CACHE"

    def _bridge(self):
        from ds1000 import ds1000_bridge
        return ds1000_bridge

    def slice_ids(self, slice_path):
        spec = json.loads(Path(slice_path).read_text(encoding="utf-8"))
        items = spec["items"] if isinstance(spec, dict) else spec
        return [str(it["pid"]) if isinstance(it, dict) else str(it) for it in items]

    def write_slice(self, ids, out_path):
        Path(out_path).write_text(json.dumps(list(ids), indent=1), encoding="utf-8")

    def load_items(self, ids):
        return self._bridge().load_problems(ids=list(ids))

    def load_harness(self, name, problem):
        from ds1000.ds1000_common import load_harness
        return load_harness(name, problem)

    def item_id(self, p):
        return str(p.pid)

    def item_domain(self, p):
        return p.library

    def route_source(self, home, default):
        return (
            '"""Delegation-composed global harness (auto-generated, --aggregate route)."""\n'
            "from ..harness_base import DS1000Harness\n\n\n"
            "class RouteHarness(DS1000Harness):\n"
            f"    HOME = {home!r}\n"
            f"    DEFAULT = {default!r}\n\n"
            "    def solve(self) -> str:\n"
            "        from ..ds1000_common import load_harness\n"
            "        inner = load_harness(self.HOME.get(self.library, self.DEFAULT), self.problem)\n"
            "        try:\n"
            "            return inner.solve()\n"
            "        finally:\n"
            "            self._trace = inner._trace\n")

    def is_correct(self, code, p):
        return self._bridge().is_correct(code, p)

    def merge_blurb(self):
        return (
            "A harness is arbitrary Python wrapping a FROZEN weak coder for DS-1000 (data-science snippets: "
            "pandas/numpy/scipy/sklearn/pytorch/tensorflow/matplotlib). Its solve() returns the solution "
            "snippet string (sets a `result` variable / fills the function body). API available to it: "
            "self.prompt (problem text); self.library; self.llm(prompt, system='', thinking=..., n=1, "
            "max_tokens=None); self.run(script, timeout=15) -> (rc, stdout, stderr) — label-free arbitrary "
            "execution; self.selfcheck(code) -> {checkable, ran, error, output, redefines} (checkable=False "
            "is ABSENCE OF EVIDENCE, never a failure signal); bridge.extract_code(text); "
            "bridge.back_translate(code). INVARIANTS (audited): FROZEN SOLVER; LABEL-FREE — never touch "
            "problem.code_context's hidden test or is_correct, no per-problem hardcoding. The merged file "
            "must keep `from ..harness_base import DS1000Harness` and `from .. import ds1000_bridge as "
            "bridge`.\n"
            "Verify the merged file imports:\n"
            f"  PYTHONPATH={ROOT} python -c \"from ds1000 import ds1000_bridge as b; "
            "from ds1000.ds1000_common import load_harness; p=b.load_problems(limit=1)[0]; "
            "load_harness('<NAME>', p); print('LOADS OK')\"")


class SWEAdapter(Adapter):
    """SWE-bench Verified. Three things set it apart and shape this adapter:
      * solving = a full mini-swe-agent rollout in the instance's Docker container (slow; concurrency 4,
        containers must be torn down even when solve() hangs);
      * grading = the official swebench harness, itself Docker-based and expensive — so it runs as ONE
        BATCH per (harness, slice) and is cached by patch hash in logs/gold_cache.json (43 verdicts
        already carried over from the TTHE runs);
      * the step budget (80) is FIXED inside the bridge and is deliberately not a knob here — a harness
        that merely raised it would buy its gain with money, not method."""
    name, pkg = "swe", "swe"
    trace_glob = "__i*.md"
    optimize_mod = "swe.swe_optimize"
    agents_dir = ROOT / "swe" / "agents"
    logs_dir = ROOT / "swe" / "logs"
    cache_env = "SWE_SOLVER_CACHE"

    def _bridge(self):
        from swe import swe_bridge
        return swe_bridge

    def slice_ids(self, slice_path):
        spec = json.loads(Path(slice_path).read_text(encoding="utf-8"))
        items = spec["items"] if isinstance(spec, dict) else spec
        return [it["instance_id"] if isinstance(it, dict) else it for it in items]

    def write_slice(self, ids, out_path):
        Path(out_path).write_text(json.dumps(list(ids), indent=1), encoding="utf-8")

    def load_items(self, ids):
        return self._bridge().load_instances(ids=list(ids))

    def load_harness(self, name, instance):
        from swe.swe_common import load_harness
        return load_harness(name, instance)

    def item_id(self, inst):
        return inst["instance_id"]

    def item_domain(self, inst):
        return inst["repo"]

    def route_source(self, home, default):
        return (
            '"""Delegation-composed global harness (auto-generated, --aggregate route)."""\n'
            "from ..harness_base import SWEHarness\n\n\n"
            "class RouteHarness(SWEHarness):\n"
            f"    HOME = {home!r}\n"
            f"    DEFAULT = {default!r}\n\n"
            "    def solve(self) -> str:\n"
            "        from ..swe_common import load_harness\n"
            "        inner = load_harness(self.HOME.get(self.repo, self.DEFAULT), self.instance)\n"
            "        try:\n"
            "            return inner.solve()\n"
            "        finally:\n"
            "            self._trace = inner._trace\n")

    def solve_and_grade(self, harness_name, items, solve_timeout):
        """Override: rollouts at low concurrency with guaranteed container teardown, then ONE batch gold
        call (dedupes by patch hash against the shared cache)."""
        from concurrent.futures import ThreadPoolExecutor
        b = self._bridge()
        cls = type(self.load_harness(harness_name, items[0]))
        patches = [None] * len(items)

        def one(j):
            h = cls(items[j])
            try:
                with ThreadPoolExecutor(max_workers=1) as tp:
                    patches[j] = tp.submit(h.solve).result(timeout=solve_timeout) or ""
            except Exception:  # noqa: BLE001
                patches[j] = ""
            finally:
                try:
                    h.cleanup()
                except Exception:  # noqa: BLE001
                    pass
        with ThreadPoolExecutor(max_workers=4) as ex:
            list(ex.map(one, range(len(items))))
        gold = b.is_correct_batch(list(zip(items, patches)), run_id=f"fedkit_{harness_name}_eval")
        return [{"i": j, "id": items[j]["instance_id"], "domain": items[j]["repo"],
                 "correct": bool(gold.get(items[j]["instance_id"], False)),
                 "artifact": str(patches[j])[:400]} for j in range(len(items))]

    def merge_blurb(self):
        return (
            "A harness is arbitrary Python wrapping a FROZEN weak model for SWE-bench Verified (fix a real "
            "GitHub issue in the instance's real repo, running in Docker). Its solve() returns a git-diff "
            "PATCH string. API available to it: self.instance / self.problem / self.repo; "
            "self.llm(messages) -> str (ONE frozen-solver call); self.exec(command, timeout=None) -> "
            "{output, returncode} (bash in the REAL repo container — reproduction scripts, the repo's OWN "
            "tests, git apply are all label-free and fair game); self.run_agent(system_template=None, "
            "instance_template=None) -> patch (the stock mini-swe-agent loop — keep it, wrap it with "
            "verify->repair, or replace it with an own llm+exec loop; the 80-step budget is FIXED and not "
            "a harness parameter). INVARIANTS (audited): FROZEN SOLVER; LABEL-FREE — never run the gold "
            "FAIL_TO_PASS/PASS_TO_PASS suite as a verdict, never read the reference patch or resolved "
            "status, no bridge.is_correct. Traces here are named `<harness>__i<j>.md`. The merged file "
            "must keep `from ..harness_base import SWEHarness` and `from .. import swe_bridge as bridge`.\n"
            "Verify the merged file imports:\n"
            f"  PYTHONPATH={ROOT} python -c \"from swe import swe_bridge as b; "
            "from swe.swe_common import load_harness; i=b.load_instances(limit=1)[0]; "
            "load_harness('<NAME>', i); print('LOADS OK')\"")


ADAPTERS = {"lcb": LCBAdapter, "ds1000": DS1000Adapter, "swe": SWEAdapter}


def get_adapter(name):
    if name not in ADAPTERS:
        raise SystemExit(f"unknown domain {name!r}; have {sorted(ADAPTERS)}")
    return ADAPTERS[name]()


def child_env():
    """Environment for every spawned child: pinned hash seed (determinism — see module docstring)."""
    return dict(os.environ, PYTHONHASHSEED="0", PYTHONPATH=str(ROOT))
