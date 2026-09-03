"""Merged global harness: function-body handling, retry-on-error with domain hints, marker stripping.

Derived from 5 client evolutions of the G0 bare harness:
  - client 0: function-body detection + reindent + return-result insertion + safety-gated retry
  - client 2: Traceback retry + Matplotlib/Seaborn debugging hints (home-scoped on self.library)
  - client 3: template-marker (### BEGIN/END SOLUTION) stripping
  - client 4: retry context (error + output) + indentation guidance for function-body retries
"""
import re as _re
from ..harness_base import DS1000Harness
from .. import ds1000_bridge as bridge

SYS = ("You are an expert Python data-science programmer. Read the problem and output ONLY the solution code "
       "that should be INSERTED to compute the required `result` variable, USING the input variables already "
       "defined in the problem's context (e.g. df, a, X). Do NOT redefine or re-create those input variables, "
       "do NOT add your own example/test data, do NOT wrap the answer in a function — just the lines that compute "
       "`result` from the given inputs. Put it in a single ```python ... ``` block, no prose.")

# Matplotlib/Seaborn hints injected only during a retry on matplotlib problems.
_RETRY_SYS_HINTS = (
    "\n\nMatplotlib/Seaborn debugging notes:\n"
    "- sns.pairplot() returns a PairGrid object. It does NOT accept a `legend` parameter.\n"
    "  To hide the pairplot legend, call g._legend.remove() on the returned PairGrid.\n"
    "- plt.xticks(va=...) / plt.yticks(rotation=...) apply visual properties to tick labels;\n"
    "  use ax.set_xticklabels(labels, va=...) or plt.setp(ax.get_xticklabels(), va='top').\n"
    "- To invert an axis: ax.invert_yaxis() or plt.gca().invert_yaxis().\n"
    "- Title position is controlled with set_title(..., y=1.02) not with `pad` alone.\n"
    "- plt.pie() returns (patches, texts, autotexts); assign to result if the problem expects it."
)


def _is_function_body(prompt: str) -> bool:
    """Detect DS-1000 function-body problems where the solution goes inside a function."""
    return bool(_re.search(r'def\s+\w+\s*\([^)]*\)\s*:.*?BEGIN SOLUTION', prompt, _re.DOTALL))


def _reindent(code: str, spaces: int = 4) -> str:
    """Add uniform leading indentation to every non-empty line."""
    prefix = " " * spaces
    return prefix + ("\n" + prefix).join(code.splitlines())


def _strip_markers(code: str) -> str:
    """Remove DS-1000 template markers that sometimes leak (### BEGIN/END SOLUTION)."""
    lines = [ln for ln in code.splitlines()
             if ln.strip() not in ("### END SOLUTION", "### BEGIN SOLUTION")]
    return "\n".join(lines)


def _retry_context(sc):
    """Build execution-failure summary from self-check result — error AND output."""
    parts = []
    error = (sc.get("error") or "").strip()
    output = (sc.get("output") or "").strip()
    if error:
        parts.append(f"ERROR:\n{error}")
    if output:
        parts.append(f"OUTPUT produced:\n{output[:500]}")
    return "\n\n".join(parts) if parts else ""


class BareHarness(DS1000Harness):
    def solve(self) -> str:
        # --- FIRST ATTEMPT ---
        resp = self.llm(self.prompt, system=SYS)
        code = bridge.extract_code(resp)
        if not code:
            return ""

        # Strip template markers that sometimes leak from the model response
        code = _strip_markers(code)

        # Function-body problems: the exec_context template has def f...: [insert]
        # so the solution must be indented and end with return result.
        is_fn_body = _is_function_body(self.prompt)
        if is_fn_body:
            code = _reindent(code, 4)
            code += "\n    return result"

        # --- LABEL-FREE SELF-CHECK ---
        sc = self.selfcheck(code)

        # Fast path: code runs clean
        if sc.get("checkable") and sc.get("ran"):
            return code

        # --- RETRY PATH: concrete execution failure ---
        # Trigger: checkable=True (example input exists) AND ran=False (crashed) AND real error
        if sc.get("checkable") and not sc.get("ran") and sc.get("error"):
            ctx = _retry_context(sc)

            # Home-scoped hints: matplotlib debugging guidance on retries
            retry_sys = SYS
            if self.library == "Matplotlib":
                retry_sys += _RETRY_SYS_HINTS

            for _ in range(2):  # up to 2 retries
                retry_prompt = (
                    self.prompt
                    + "\n\n[The previous solution attempt failed with this execution result:\n"
                    + ctx
                    + "\n\nFix the error and output ONLY the corrected code in a ```python block.]"
                )
                resp2 = self.llm(retry_prompt, system=retry_sys)
                code2 = bridge.extract_code(resp2)
                if not code2:
                    break
                code2 = _strip_markers(code2)
                if is_fn_body:
                    code2 = _reindent(code2, 4)
                    code2 += "\n    return result"
                sc2 = self.selfcheck(code2)
                if sc2.get("checkable") and sc2.get("ran"):
                    code = code2
                    break
                # Feed new error into next retry
                if sc2.get("checkable") and not sc2.get("ran") and sc2.get("error"):
                    ctx = _retry_context(sc2)
                else:
                    break

        return code
