"""Merged harness: output-aware retry with domain-adaptive reasoning.

Combines mechanisms evolved independently on three difficulty shards:
- Client 0 (easy): public-test validation + single retry (foundational — folded in).
- Client 1 (medium): multi-attempt output-aware retry with adaptive advice,
  empty-code guard, best-score fallback, detailed per-test feedback.
- Client 2 (hard): reasoning-allowed retry for hard problems.

HOME-SCOPED: hard problems get a reasoning-first retry prompt (client 2's domain);
easy/medium get the standard trace-through-debug approach (client 1's domain).
"""
from ..harness_base import CodeHarness
from .. import lcb_bridge as bridge

MAX_ATTEMPTS = 3
MAX_FEEDBACK_CHARS = 2000

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


class FedGlobalHarness(CodeHarness):
    def solve(self) -> str:
        task = "Write the complete Python 3 solution (read stdin, print stdout)."
        prompt = f"{self.content}\n\n{task}"
        best_code = ""
        best_score = -1                        # best n_pass seen; -1 means no valid result yet
        is_hard = getattr(self.problem, 'difficulty', None) == 'hard'

        for attempt in range(MAX_ATTEMPTS):
            sys_prompt = SYS if attempt == 0 else (REASON_RETRY_SYS if is_hard else RETRY_SYS)
            resp = self.llm(prompt, system=sys_prompt)
            code = bridge.extract_code(resp)

            if not code:
                prompt = (f"{self.content}\n\n{task}\n\n"
                          "Your previous reply contained no code block. "
                          "Output exactly one ```python ... ``` block with the complete solution.")
                continue

            res = self.run_public(code)
            if res["n_total"] == 0 or res["n_pass"] == res["n_total"]:
                return code

            if res["n_pass"] > best_score:
                best_code = code
                best_score = res["n_pass"]

            parts = []
            for i, r in enumerate(res["results"]):
                if r.get("rc", 0) != 0:
                    parts.append(f"Test {i}: CRASH (rc={r['rc']}) — {str(r.get('stderr', ''))[:200]}")
                elif not r.get("ok", False):
                    inp_s = str(r.get("input", "")).strip()[:80]
                    exp_s = str(r.get("expected", "")).strip()
                    got_s = str(r.get("stdout", "")).strip()
                    parts.append(f"Test {i}: input={inp_s!r} expected={exp_s!r} got={got_s!r}")
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

            prompt = (f"{self.content}\n\n{task}\n\n"
                      f"Your previous solution failed {res['n_pass']}/{res['n_total']} public tests.\n"
                      f"Failures:\n{fb}\n\n"
                      f"{advice}")

        return best_code if best_code else code
