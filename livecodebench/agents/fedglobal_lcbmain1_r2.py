"""Merged harness: output-aware retry with domain-adaptive reasoning (r2).

Round 2 merges mechanisms from three difficulty-shard clients:
- Client 0 (easy): no substantive changes adopted (one-text tweak declined).
- Client 1 (medium): self-critique retry (prev_response + thinking=low on retry)
- Client 2 (hard): thinking=low on first call for hard problems; syntax pre-validation.

HOME-SCOPED: hard problems get thinking='low' from first attempt (client 2's domain);
easy/medium get thinking='low' only on retry.
Syntax validation and self-critique retry apply globally.
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


def _syntax_ok(code: str) -> bool:
    """Quick syntax check -- catches unclosed brackets, stray fences, etc."""
    try:
        compile(code, '<harness>', 'exec')
        return True
    except SyntaxError:
        return False


class FedGlobalHarness(CodeHarness):
    def solve(self) -> str:
        task = "Write the complete Python 3 solution (read stdin, print stdout)."
        prompt = f"{self.content}\n\n{task}"
        best_code = ""
        best_score = -1                        # best n_pass seen; -1 means no valid result yet
        is_hard = getattr(self.problem, 'difficulty', None) == 'hard'
        prev_response = ""                     # store previous full response for self-critique

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

            # Syntax pre-check -- reject obviously broken code early (client 2)
            if not _syntax_ok(code):
                prompt = (f"{self.content}\n\n{task}\n\n"
                          "Your previous reply contained a Python code block with a syntax error. "
                          "Fix all syntax errors and output the COMPLETE CORRECTED solution "
                          "inside a ```python ... ``` block.")
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

            # Self-critique: feed back previous response for the model to identify its own
            # logical flaws (client 1).  Skips on first attempt since prev_response is empty.
            prev_section = ""
            if prev_response:
                prev_section = (f"\n\nYOUR PREVIOUS RESPONSE (with your reasoning and code):\n"
                                f"---\n{prev_response[-3000:]}\n---\n\n"
                                f"Review your previous reasoning above. Identify the logical flaw "
                                f"in your approach, then write a corrected solution.")

            prompt = (f"{self.content}\n\n{task}\n\n"
                      f"Your previous solution failed {res['n_pass']}/{res['n_total']} public tests.\n"
                      f"Failures:\n{fb}\n\n"
                      f"{advice}"
                      f"{prev_section}")

            # Store current response for next retry's self-critique
            prev_response = resp

        return best_code if best_code else code
