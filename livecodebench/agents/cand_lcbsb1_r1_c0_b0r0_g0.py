"""Execute-public-tests + retry-on-failure: one-shot generation validated against public sample tests; if
any test fails, retry once with the exact failure details (input, expected, got) fed back as a correction
prompt. The retry uses low thinking so the model can reason about the discrepancy. No per-problem code,
no hidden-test access — purely the label-free signal of public-test execution."""
from ..harness_base import CodeHarness
from .. import lcb_bridge as bridge

SYS = ("You are an expert competitive programmer. Read the problem and output ONE complete, self-contained "
       "Python 3 program that reads from standard input and prints the answer to standard output, inside a "
       "single ```python ... ``` block. No explanation outside the code block.")

RETRY_SYS = ("You are an expert competitive programmer debugging a wrong solution. Below are the problem, "
             "your previous (incorrect) code, and the sample test failures. "
             "Fix the code so ALL sample tests pass. "
             "Output ONE complete Python 3 program (read stdin, print stdout) inside a ```python``` block.")

BLOCK_SYS = ("You are an expert competitive programmer. Read the problem and output ONE complete, self-contained "
             "Python 3 program that reads from standard input and prints the answer to standard output. "
             "Put your entire solution inside a SINGLE ```python ... ``` code block with no surrounding explanation.")


class BareHarness(CodeHarness):
    def _gen_code(self, prompt: str, sys_prompt: str = SYS, **kwargs) -> str:
        """Generate code, retrying once if extract_code returns empty."""
        resp = self.llm(prompt, system=sys_prompt, **kwargs)
        code = bridge.extract_code(resp)
        if not code.strip():
            resp = self.llm(prompt, system=BLOCK_SYS, **kwargs)
            code = bridge.extract_code(resp)
        return code

    def solve(self) -> str:
        prompt = f"{self.content}\n\nWrite the complete Python 3 solution (read stdin, print stdout)."
        code = self._gen_code(prompt)                            # thinking=False (default)

        # No public tests → no validation possible; return as-is.
        if not self.public_tests:
            return code

        result = self.run_public(code)

        # All passed → return immediately (most problems, no regression).
        if result["n_pass"] == result["n_total"]:
            return code

        # Some tests failed → assemble error report and retry once.
        failed = [r for r in result["results"] if not r["ok"]]
        lines = []
        for i, r in enumerate(failed):
            inp = r.get("input", "")
            exp = r.get("expected", "")
            got = r.get("stdout", "")
            err = r.get("stderr", "")
            lines.append(
                f"--- failing test {i} ---\n"
                f"input:\n{inp}\n"
                f"expected output:\n{exp}\n"
                f"your output:\n{got}\n"
                f"stderr:\n{err}"
            )
        error_report = "\n".join(lines)

        retry_prompt = (
            f"{self.content}\n\n"
            f"=== YOUR PREVIOUS (incorrect) CODE ===\n"
            f"```python\n{code}\n```\n\n"
            f"=== SAMPLE TEST FAILURES ===\n"
            f"{error_report}\n\n"
            f"Fix the code. Write the complete Python 3 solution (read stdin, print stdout)."
        )
        code2 = self._gen_code(retry_prompt, sys_prompt=RETRY_SYS, thinking="low")

        # Guard: identical or empty retry — no point re-running tests.
        if not code2.strip() or code2.strip() == code.strip():
            return code

        # For safety, validate the retry; return whichever passes more public tests.
        result2 = self.run_public(code2)
        if result2["n_pass"] >= result["n_pass"]:
            return code2
        return code
