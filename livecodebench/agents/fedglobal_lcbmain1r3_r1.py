"""Merged harness: output-aware retry with domain-adaptive reasoning (r3).

Folds r3 clients on r2: client-0 prev-code-verbatim on retry prompts (global, retry-only);
client-1 verification layer + MAX_ATTEMPTS=4 (layer HOME-SCOPED to medium, client 1's shard);
client-2 unchanged. Non-medium all-pass returns immediately, byte-identical to base.
Stress-TIMEOUT fixed abc400_c, differential fixed abc387_c, MAX_ATTEMPTS=4 fixed arc191_a.
Stress gates TIMEOUT only (CRASH/EMPTY are generator false-positive traps)."""
from ..harness_base import CodeHarness
from .. import lcb_bridge as bridge

MAX_ATTEMPTS = 4
MAX_FEEDBACK_CHARS = 2000
VERIFY_TIMEOUT = 6
DIFF_CASES = 12
MAX_DIFF_INPUT_CHARS = 2000

SYS = ("You are an expert competitive programmer. Read the problem and output ONE complete, self-contained "
       "Python 3 program that reads from standard input and prints the answer to standard output, inside a "
       "single ```python ... ``` block. No explanation outside the code block.")
RETRY_SYS = ("Your previous solution failed the sample tests. Debug the logic carefully: trace through "
             "your algorithm on the failing inputs, identify the root cause, then write a completely "
             "correct solution. Output ONE complete self-contained Python 3 program (with `import sys`, "
             "a `solve()` function, and `if __name__ == '__main__': solve()`) inside a ```python ... ``` block."
             " No explanation outside the code block.")
REASON_RETRY_SYS = ("Your previous solution failed the sample tests. "
                    "First reason through the problem and design a solution. "
                    "Then output the complete Python 3 solution inside a ```python ... ``` block.")
VERIFY_SYS = ("You write two small Python functions for a competitive-programming problem. "
              "Be exact and simple; prefer an obviously-correct brute force over cleverness.")

def _syntax_ok(code):
    try:
        compile(code, '<harness>', 'exec')
        return True
    except SyntaxError:
        return False

def _verify_prompt(content):
    return (f"{content[:2500]}\n\n"
            "Write a single Python snippet that defines exactly two functions:\n"
            "1) `gen_case()` -> returns ONE SMALL random valid input string for this problem (the exact stdin text, ending with a newline). Use SMALL sizes (values roughly 1..10). "
            "It must ALWAYS return a valid input without error (no dead or broken lines).\n"
            "2) `brute_force(inp)` -> returns the EXACT expected stdout string for that input, computed by a simple NAIVE algorithm that tries all possibilities directly. Include any trailing newline.\n"
            "The snippet must import cleanly and both functions must run without error.\n"
            "Do not print anything at import time. Output the whole snippet in one ```python``` block.")

def _outs_equal(a, b):
    a, b = a.strip(), b.strip()
    if a == b:
        return True
    try:
        fa, fb = float(a), float(b)
    except ValueError:
        return False
    return abs(fa - fb) <= 1e-9 * max(1.0, abs(fa), abs(fb))

class FedGlobalHarness(CodeHarness):
    def __init__(self, problem):
        super().__init__(problem)
        self._verifiers = []            # validated brute-force verifier snippets (per problem)
    def _ask_verifier(self, note=""):
        prompt = _verify_prompt(self.content) + (("\n\n" + note) if note else "")
        resp = self.llm(prompt, system=VERIFY_SYS, thinking=False)
        snip = bridge.extract_code(resp)
        if not snip:
            return None
        g = {}
        try:
            exec(compile(snip, '<verify>', 'exec'), g)
        except Exception:
            return None
        if 'brute_force' not in g or 'gen_case' not in g:
            return None
        wrap = snip + "\nimport sys\nsys.stdout.write(brute_force(sys.stdin.read()))"
        # Trust the verifier only if it reproduces the public sample outputs.
        validated = 0
        for t in self.public_tests:
            if not t.get("input"):
                continue
            try:
                r = bridge.run_code(wrap, [{"input": t["input"], "output": ""}], timeout=4)["results"][0]
                validated += r["rc"] == 0 and _outs_equal(r["stdout"], t.get("output", ""))
            except Exception:
                pass
        if validated == 0:
            return None
        try:                              # gen_case must actually produce a small usable input
            c = g["gen_case"]()
            if not isinstance(c, str) or not (0 < len(c) <= MAX_DIFF_INPUT_CHARS):
                return None
        except Exception:
            return None
        return {"wrap": wrap, "g": g}
    def _get_verifier(self, note=""):
        v = self._ask_verifier(note)
        if v is None and not note:
            v = self._ask_verifier("Your previous reply had no usable code block. "
                                   "Output ONLY the Python snippet in one ```python``` block.")
        if v is not None:
            self._verifiers.append(v)
        return v
    def _diff_check(self, code):
        v1 = self._get_verifier()
        if v1 is None:
            return None
        inputs = []
        try:
            inputs = [i for i in bridge.gen_stress_inputs(self.problem, k=3) if len(i) <= MAX_DIFF_INPUT_CHARS]
        except Exception:
            pass
        if self._verifiers:
            g = self._verifiers[-1]["g"]
            for _ in range(2 * DIFF_CASES):
                if len(inputs) >= DIFF_CASES:
                    break
                try:
                    c = g["gen_case"]()
                    if isinstance(c, str) and 0 < len(c) <= MAX_DIFF_INPUT_CHARS:
                        inputs.append(c)
                except Exception:
                    continue
        if len(inputs) < 3:
            return None
        for inp in inputs:
            bres = bridge.run_code(v1["wrap"], [{"input": inp, "output": ""}], timeout=VERIFY_TIMEOUT)["results"][0]
            if bres["rc"] != 0 or not bres["stdout"].strip():
                continue
            bf_out = bres["stdout"].strip()
            cres = bridge.run_code(code, [{"input": inp, "output": ""}], timeout=VERIFY_TIMEOUT)["results"][0]
            cand_out = cres["stdout"].strip()
            if _outs_equal(cand_out, bf_out):
                continue
            v2 = self._get_verifier("Now write a SECOND, independently-written brute-force "
                                    "using a different approach/algorithm from the first.")
            if v2 is None:
                return None
            bres2 = bridge.run_code(v2["wrap"], [{"input": inp, "output": ""}], timeout=VERIFY_TIMEOUT)["results"][0]
            if bres2["rc"] == 0 and _outs_equal(bres2["stdout"].strip(), bf_out):
                return (inp, bf_out, cand_out)
            return None
        return None
    def _verify(self, code):
        try:
            st = self.stress(code)
            for r in st["results"]:
                if r["status"] == "TIMEOUT":   # CRASH/EMPTY are generator false-positive traps
                    return (False, {
                        "kind": "stress",
                        "detail": (f"A maximum-constraint input caused TIMEOUT "
                                   f"(out={r.get('out', '')[:60]!r} err={r.get('err', '')[:100]!r})."),
                        "advice": ("Your program passes the samples but is NOT robust on maximum-size inputs "
                                   "(it times out on the largest constraints). Rewrite it with an efficient algorithm that "
                                   "handles the largest constraints within the time limit and always prints an answer. Output "
                                   "the COMPLETE CORRECTED solution inside a ```python ... ``` block."),
                    })
        except Exception:
            pass                           # stress unavailable -> don't fail the code on it
        if getattr(self, "starter_code", ""):
            return (True, None)            # functional problems: the stdin verifier doesn't apply
        try:
            mm = self._diff_check(code)
            if mm:
                inp, exp, cand = mm
                return (False, {
                    "kind": "diff",
                    "detail": (f"Additional test case:\n{inp!r}\nYour output:     {cand!r}\nCorrect output:  {exp!r}"),
                    "advice": ("Your program passes the samples but gives a WRONG ANSWER on the additional test case "
                               "above (two independent brute-force verifiers agree on the correct answer). Find the edge "
                               "case or logic bug and fix it, keeping every sample output correct. Output the COMPLETE "
                               "CORRECTED solution inside a ```python ... ``` block."),
                })
        except Exception:
            pass
        return (True, None)
    @staticmethod
    def _retry_sections(prev_response, prev_code, code_first=False):
        crit = ""
        if prev_response:
            crit = (f"\n\nYOUR PREVIOUS RESPONSE (with your reasoning and code):\n"
                    f"---\n{prev_response[-3000:]}\n---\n\n"
                    f"Review your previous reasoning above. Identify the logical flaw "
                    f"in your approach, then write a corrected solution.")
        cod = ""
        if prev_code:
            cod = (f"\n\nYOUR PREVIOUS CODE (verbatim, the program you submitted last):\n"
                   f"```python\n{prev_code}\n```\n")
        return (cod + crit) if code_first else (crit + cod)
    def solve(self) -> str:
        task = "Write the complete Python 3 solution (read stdin, print stdout)."
        prompt = f"{self.content}\n\n{task}"
        best_code = ""
        best_score = -1                        # best n_pass seen; -1 means no valid result yet
        is_hard = getattr(self.problem, 'difficulty', None) == 'hard'
        is_medium = getattr(self.problem, 'difficulty', None) == 'medium'  # client-1 home scope
        prev_response = ""                     # store previous full response for self-critique
        prev_code = ""                         # store previous extracted code for retry prompts

        for attempt in range(MAX_ATTEMPTS):
            sys_prompt = SYS if attempt == 0 else (REASON_RETRY_SYS if is_hard else RETRY_SYS)
            # Hard problems get reasoning budget from first attempt (client 2 home-scoped);
            # all problems get it on retry (client 1 global).
            use_thinking = attempt > 0 or is_hard
            resp = self.llm(prompt, system=sys_prompt, thinking=('low' if use_thinking else False))
            code = bridge.extract_code(resp)
            if not code:
                prompt = (f"{self.content}\n\n{task}\n\n"
                          "Your previous reply contained no code block. "
                          "Output exactly one ```python ... ``` block with the complete solution.")
                continue
            if not _syntax_ok(code):
                prompt = (f"{self.content}\n\n{task}\n\n"
                          "Your previous reply contained a Python code block with a syntax error. "
                          "Fix all syntax errors and output the COMPLETE CORRECTED solution "
                          "inside a ```python ... ``` block.")
                continue

            res = self.run_public(code)
            all_pass = res["n_total"] == 0 or res["n_pass"] == res["n_total"]
            if all_pass:
                if not is_medium:
                    return code               # non-medium: byte-identical to base
                # HOME-SCOPED (medium): verify beyond the samples before declaring victory.
                vok, vfail = self._verify(code)
                if vok:
                    return code
                if res["n_total"] > 0 and res["n_pass"] > best_score:
                    best_code = code
                    best_score = res["n_pass"]
                fb = f"VERIFICATION FAILED ({vfail['kind']}):\n{vfail['detail']}"
                prompt = (f"{self.content}\n\n{task}\n\n"
                          f"Your previous solution passed {res['n_pass']}/{res['n_total']} public tests "
                          f"but FAILED an additional check.\n\n{fb}\n\n{vfail['advice']}"
                          f"{self._retry_sections(prev_response, prev_code)}")
                prev_response = resp
                prev_code = code
                continue
            if res["n_pass"] > best_score:
                best_code = code
                best_score = res["n_pass"]
            parts = []
            for i, r in enumerate(res["results"]):
                if r.get("rc", 0) != 0:
                    parts.append(f"Test {i}: CRASH (rc={r['rc']}) — {str(r.get('stderr', ''))[:200]}")
                elif not r.get("ok", False):
                    inp_s = str(r.get("input", "")).strip()[:80]
                    parts.append(f"Test {i}: input={inp_s!r} expected="
                                 f"{str(r.get('expected', '')).strip()!r} got={str(r.get('stdout', '')).strip()!r}")
            fb = "\n".join(parts[:8])
            if len(fb) > MAX_FEEDBACK_CHARS:
                fb = fb[:MAX_FEEDBACK_CHARS] + "..."
            if res["n_pass"] == 0:
                advice = ("Your algorithm is fundamentally incorrect — all sample tests failed. "
                          "Do NOT patch it. Start from scratch: re-read the problem, think about "
                          "the optimal strategy, verify against the sample cases, then implement. "
                          "Output the COMPLETE CORRECTED solution inside a ```python ... ``` block.")
            else:
                advice = ("Carefully trace through your algorithm on each failing test case "
                          "to find the bug. Then output the COMPLETE CORRECTED solution "
                          "inside a ```python ... ``` block.")
            # Public-failure retry prompt: base's cached prompt, plus client-0's full
            # previous code verbatim (code first, then self-critique).
            prompt = (f"{self.content}\n\n{task}\n\n"
                      f"Your previous solution failed {res['n_pass']}/{res['n_total']} public tests.\n"
                      f"Failures:\n{fb}\n\n"
                      f"{advice}"
                      f"{self._retry_sections(prev_response, prev_code, code_first=True)}")
            prev_response = resp
            prev_code = code
        return best_code if best_code else code
