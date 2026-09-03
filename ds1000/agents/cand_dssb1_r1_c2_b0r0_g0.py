"""G0: custom extraction + targeted prompt guidance + crash-retry mechanism.

CHANGES from base (cand_dssb1_r0_c1_b0r0_g0):
  1. CUSTOM EXTRACTION: preserves original indentation; avoids bridge.extract_code's
     textwrap.dedent that destroys function-body indentation.
  2. SYSTEM PROMPT: five targeted guards (input redefinition, dtype=, library reuse,
     variable scope, function wrapping).
  3. CRASH RETRY: if the first solution crashes with a genuine execution error
     (ran=False, checkable=True, Traceback), retry once with the error message
     appended to the prompt.  Only adopt the retry if it runs clean.

  Trace evidence for the retry mechanism:
  - Q16 [pid 638]: sns.pairplot(..., legend=False) → TypeError (legend not a valid
    pairplot parameter).  The error message tells the model the exact API issue.
  - Q17 [pid 646]: ax.errorbar(..., ecolor=c, ...) where c is a list of colors →
    ValueError: Invalid RGBA argument during savefig rendering.  The fix is to
    iterate over positions and call errorbar once per bar with a single color.
  - No CORRECT problem has ran=False, so the retry never fires on a problem the
    base already handles.  Asymmetry principle: a crashed solution is known-wrong,
    replacing it with any untested solution cannot be a regression.

SAFETY — each CORRECT problem analysed: identical to base (no retry fires since
all ran=True)."""
from ..harness_base import DS1000Harness
from .. import ds1000_bridge as bridge
import re


SYS = ("You are an expert Python data-science programmer. Read the problem and output ONLY the solution code "
       "that should be INSERTED to compute the required `result` variable, USING the input variables already "
       "defined in the problem's context (e.g. df, a, X). "
       "Do NOT redefine or re-create those input variables — if they are given in the code context, "
       "use them exactly as provided; never write `x = np.array(x)` or similar reassignments. "
       "Do NOT assume any variable exists unless it is explicitly assigned in the code context. "
       "Do NOT add your own example or test data. "
       "Do NOT wrap the answer in a function unless the problem explicitly asks you to define one. "
       "Avoid unnecessary type specifications (like `dtype=int` in `np.array()`) — let the library "
       "infer types naturally unless the problem explicitly demands a specific dtype. "
       "When a problem mentions a specific library function by name, use it directly — "
       "trust the library to handle edge cases (unequal array sizes, NaN, etc.) correctly. "
       "Do not waste time reimplementing what the library already provides. "
       "Put the solution in a single ```python ... ``` block, no prose.")


def _extract_code(text):
    """Extract Python code from model response, preserving original indentation.

    Like bridge.extract_code but WITHOUT the first-line-padding + uniform-dedent step
    that destroys function body indentation (the root cause of Q11's IndentationError).
    Only the code block body is returned; stray fence artefacts are removed.

    SAFETY: For code where all lines start at column 0 (the vast majority of DS-1000
    solutions), the output is byte-identical to bridge.extract_code.
    """
    # Match the opening fence: ```python or ```py or just ```
    fence = re.search(r"```[ \t]*(?:python|py)?[ \t]*\r?\n", text, re.I)
    if fence:
        # Use the LAST ``` in the reply as the closing fence
        end = text.rfind("```")
        if end > fence.end():
            raw = text[fence.end():end].strip()
        else:
            # No closing fence — take everything after opening
            raw = text[fence.end():].strip()
    else:
        # Fallback: non-greedy between paired ```
        m = re.search(r"```(.*?)```", text, re.S)
        raw = m.group(1).strip() if m else ""
    if not raw:
        return ""
    # Remove stray fence markers that sometimes leak inside the block
    keep = [ln for ln in raw.splitlines()
            if ln.strip() not in ("```", "```python", "python", "py")]
    return "\n".join(keep).rstrip("\n") if keep else ""


class BareHarness(DS1000Harness):
    def solve(self) -> str:
        # First attempt: one greedy shot
        resp = self.llm(self.prompt, system=SYS)
        code = _extract_code(resp)
        if not code:
            return code

        # Label-free self-check — does it execute?
        sc = self.selfcheck(code)

        # Retry ONLY on a genuine execution crash.  Three gates:
        #   1. checkable=True  — the probe had enough info; False means
        #      "no example input in prompt" = absence of evidence.
        #   2. ran=False       — the solution itself failed to execute.
        #   3. error starts with 'Traceback' — a real Python crash.
        #      (skips TIMEOUT and empty-string errors.)
        if (not sc['ran']
                and sc.get('checkable', False)
                and (sc.get('error') or '').startswith('Traceback')):
            err = sc['error'][:1000]
            retry_prompt = (
                self.prompt
                + "\n\nYour previous solution produced an error when run:\n```\n"
                + err
                + "\n```\n"
                "Fix the error and output ONLY the NEW lines that would be INSERTED "
                "at the `# SOLUTION START` marker.\n"
                "Do NOT repeat the problem setup (imports, variable definitions, "
                "plot commands already in the problem).\n"
                "Use only the input variables already defined in the problem's context — "
                "do NOT redefine them.\n"
                "Put the corrected snippet in a single ```python ... ``` block."
            )
            resp2 = self.llm(retry_prompt, system=SYS)
            code2 = _extract_code(resp2)
            if code2:
                sc2 = self.selfcheck(code2)
                if sc2['ran']:
                    code = code2

        return code
