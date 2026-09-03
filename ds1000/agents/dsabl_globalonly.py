"""Federated merge: output-var detection, indent-safe extraction, scatter/pie hints,
missing-result detection, template-run validation, redefine rejection."""
import ast as _ast
import re as _re
import textwrap as _textwrap
from ..harness_base import DS1000Harness
from .. import ds1000_bridge as bridge

SYS = ("You are an expert Python data-science programmer. Read the problem and output ONLY the solution code "
       "that should be INSERTED to compute the required `result` variable, USING the input variables already "
       "defined in the problem's context (e.g. df, a, X). Do NOT redefine or re-create those input variables, "
       "do NOT add your own example/test data, do NOT wrap the answer in a function — just the lines that compute "
       "`result` from the given inputs. Put it in a single ```python ... ``` block, no prose.")

# Client 1: initial Matplotlib guidance to prevent common silent-error patterns.
# Enhanced by client 2 with scatter edgecolors and pie return-value clarification.
_SYS_HINTS = (
    "\n\nMatplotlib/Seaborn guidance:\n"
    "- Axis inversion: use ax.invert_yaxis() or plt.gca().invert_yaxis(), never negate the data.\n"
    "- To set tick label vertical alignment: ax.set_xticklabels(labels, va='top') or\n"
    "  plt.setp(ax.get_xticklabels(), va='top').  plt.xticks(va='...') is NOT valid —\n"
    "  that keyword is silently ignored.\n"
    "- Title position in subplots: use set_title(label, y=float) to raise/lower the title.\n"
    "  The `pad` parameter controls distance from the axes edge, NOT relative height.\n"
    "- plt.pie() WITHOUT autopct returns (patches, texts) — only 2 values.\n"
    "  WITH autopct it returns (patches, texts, autotexts) — 3 values.\n"
    "  Unpack to match the actual number of return values.\n"
    "- plt.scatter() uses `c` (or `color`) for marker face color and `edgecolors` for edge color.\n"
    "  `facecolors` and `facecolor` are NOT valid parameters for scatter."
)

# Client 2: Sklearn-specific guidance.
_SKLEARN_HINTS = (
    "\n\nSklearn usage notes:\n"
    "- The <code> block defines the INPUT VARIABLES (e.g. X, y, df, clf, model).\n"
    "  Variables created only during data loading (pd.read_csv, load_data) are NOT available\n"
    "  as inputs in the hidden test — do NOT reference them in your solution.\n"
    "- Name your output variable to match what the problem asks for after the <code> block\n"
    "  (e.g. `predict`, `X_train`, `transformed_df`, `model_name`). That variable is what\n"
    "  the grader evaluates.\n"
    "- Avoid hardcoding example-specific values (step names, column names, indices, thresholds).\n"
    "  The hidden test uses different data; write code that works for any input of the same form."
)

# Matplotlib/Seaborn debugging hints injected during retries.
# Enhanced by client 2 with pie return-value clarification.
_RETRY_SYS_HINTS = (
    "\n\nMatplotlib/Seaborn debugging notes:\n"
    "- sns.pairplot() returns a PairGrid object. It does NOT accept a `legend` parameter.\n"
    "  To hide the pairplot legend, call g._legend.remove() on the returned PairGrid.\n"
    "- plt.xticks(va=...) / plt.yticks(rotation=...) apply visual properties to tick labels;\n"
    "  use ax.set_xticklabels(labels, va=...) or plt.setp(ax.get_xticklabels(), va='top').\n"
    "- To invert an axis: ax.invert_yaxis() or plt.gca().invert_yaxis().\n"
    "- Title position is controlled with set_title(..., y=1.02) not with `pad` alone.\n"
    "- plt.pie() WITHOUT autopct returns (patches, texts); WITH autopct returns (patches, texts, autotexts).\n"
    "  Unpack to match: 2 values without autopct, 3 with autopct."
)

# Client 2: NameError-specific hint on retry.
_NAMEERROR_HINT = (
    "\n\nNote: The error 'NameError: name X is not defined' means X is NOT available as an input"
    " variable in the execution context. Only use variables that are explicitly set up in the"
    " problem's <code> block — variables created during data loading are not available."
)

# Client 1: hint for function-body problems where the coder over-parameterizes the function.
_TYPE_ERROR_HINT = (
    "\n\nNote: A TypeError with 'missing required positional arguments' means the function"
    " signature has more parameters than the calling code provides. The test harness calls"
    " the function with fewer arguments than your definition specifies. Reduce the parameter"
    " list — use any needed context variables (like x_min, x_max, a, b) directly from the"
    " enclosing scope rather than adding them as parameters or hardcoding their values."
)

# Client 4: redefine-variables warning.
_REDEFINE_HINT = (
    "\n\nWARNING: The solution redefined input variables that are ALREADY defined in the"
    " problem's <code> block: {vars}. The hidden test provides those variables with"
    " DIFFERENT values, so hardcoding them will FAIL the test."
    " Remove ALL assignments to these variables from your solution —"
    " they are already available with the correct values for each test case."
)


def _reindent(code: str, spaces: int = 4) -> str:
    prefix = " " * spaces
    return prefix + ("\n" + prefix).join(code.splitlines())


def _fn_body_apply(code: str) -> str:
    """Indent + add return result for function-body problems."""
    return _reindent(code, 4) + "\n    return result"


def _is_fn_body_error(sc) -> bool:
    """True if the selfcheck error signals an un-indented block inside a function wrapper."""
    return "expected an indented block after function definition" in (sc.get("error") or "").lower()


def _is_function_body_regex(prompt: str) -> bool:
    """Regex fn-body detection for when checkable=False (no execution feedback available).

    The ``</code>`` guard (client 3) rejects false positives where ``def`` is in prose
    rather than the template.
    """
    m = _re.search(r'def\s+\w+\s*\([^)]*\)\s*:.*?BEGIN SOLUTION', prompt, _re.DOTALL)
    return bool(m) and '</code>' not in m.group()


def _strip_markers(code: str) -> str:
    lines = [ln for ln in code.splitlines()
             if ln.strip() not in ("### END SOLUTION", "### BEGIN SOLUTION")]
    return "\n".join(lines)


def _retry_context(sc):
    parts = []
    error = (sc.get("error") or "").strip()
    output = (sc.get("output") or "").strip()
    if error:
        parts.append(f"ERROR:\n{error}")
    if output:
        parts.append(f"OUTPUT produced:\n{output[:500]}")
    return "\n\n".join(parts) if parts else ""


def _extract_code_safe(text):
    """Extract code from ```python block, preserving indentation structure.

    Unlike bridge.extract_code, this avoids the dedent bug that destroys
    function-body indentation when the entire code block is uniformly indented.
    """
    m = _re.search(r'```\w*\s*\n(.*?)```', text, _re.DOTALL)
    if not m:
        return bridge.extract_code(text)
    raw = m.group(1)
    lines = [ln for ln in raw.splitlines()
             if ln.strip() not in ("```", "```python", "python", "py",
                                    "### BEGIN SOLUTION", "### END SOLUTION")]
    if not lines:
        return ""
    non_empty = [ln for ln in lines if ln.strip()]
    if non_empty:
        indents = [len(ln) - len(ln.lstrip()) for ln in non_empty]
        if len(set(indents)) == 1 and indents[0] > 0:
            return _textwrap.dedent("\n".join(lines)).strip()
    return "\n".join(lines).strip()


def _expected_output_var(prompt: str) -> str:
    """Parse the expected output variable from the problem's assignment hint.

    DS-1000 problems after the <code> block say either
    ``result = ... # put solution in this variable`` or
    ``df = ... # put solution in this variable``. Returns the LHS name.
    """
    m = _re.search(r'^\s*(\w+)\s*=\s*\.\.\.\s*#\s*put solution in this variable',
                   prompt, _re.MULTILINE)
    return m.group(1) if m else 'result'


def _modifies_target(code: str, target: str) -> bool:
    """AST-based check: does the code reassign, subscript-assign, or attribute-assign *target*?

    Covers ``df = …``, ``df['col'] = …``, ``df.loc[…] = …``, ``df.attr = …``,
    and any attribute chain.  Returns True on SyntaxError so we never force a
    retry for a non-parseable snippet.
    """
    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return True
    for node in _ast.walk(tree):
        if not isinstance(node, _ast.Assign):
            continue
        for t in node.targets:
            if isinstance(t, _ast.Name) and t.id == target:
                return True
            root = t
            while isinstance(root, (_ast.Subscript, _ast.Attribute)):
                root = root.value
            if isinstance(root, _ast.Name) and root.id == target:
                return True
    return False


def _has_result_assignment(code: str) -> bool:
    """True when any line in *code* assigns to ``result``."""
    return bool(_re.search(r'^result\s*=(?!=)', code, _re.MULTILINE))


class BareHarness(DS1000Harness):
    def _template_run(self, code):
        """Extract <code> template, prepend, execute; return selfcheck-like dict or None."""
        m = _re.search(r'<code>(.*?)</code>', self.prompt, _re.DOTALL)
        if not m:
            return None
        template = m.group(1).strip()
        if not template:
            return None
        template_nlines = template.count("\n") + 1
        try:
            rc, out, err = self.run(template + "\n" + code, timeout=15)
        except Exception:
            return None
        out = (out or "").strip()
        err = (err or "").strip()
        if rc == 0:
            return {"checkable": True, "ran": True, "error": "", "output": out[:500]}
        if not err:
            return None
        lm = _re.search(r'File "<script>", line (\d+)', err)
        if lm and int(lm.group(1)) <= template_nlines:
            return None  # error in template, not our code
        return {"checkable": True, "ran": False, "error": err[:500], "output": out[:500]}

    def _maybe_fn_body(self, code):
        """If prompt matches fn-body pattern (regex fallback), apply transformation."""
        if _is_function_body_regex(self.prompt):
            return _fn_body_apply(code)
        return code

    def solve(self) -> str:
        target_var = _expected_output_var(self.prompt)

        # --- DOMAIN-SCOPED SYSTEM PROMPT ---
        sys_prompt = SYS
        if self.library == "Matplotlib":
            sys_prompt += _SYS_HINTS
        elif self.library == "Sklearn":
            sys_prompt += _SKLEARN_HINTS

        # --- FIRST ATTEMPT ---
        resp = self.llm(self.prompt, system=sys_prompt)
        code = _extract_code_safe(resp)
        if not code:
            return ""
        code = _strip_markers(code)

        # --- EXECUTION-GUIDED FN-BODY DETECTION (client 0) ---
        sc = self.selfcheck(code)
        redefines = sc.get("redefines") or []
        if sc.get("checkable") and sc.get("ran"):
            if redefines:
                pass  # fall through — reject redefines (client 4)
            elif target_var != 'result' and not _modifies_target(code, target_var):
                pass  # fall through — wrong output variable (client 0)
            elif target_var == 'result' and not _has_result_assignment(code):
                pass  # fall through — missing result assignment (client 3)
            else:
                return code
        is_fn_body = _is_fn_body_error(sc)
        if is_fn_body:
            code_fb = _fn_body_apply(code)
            sc_fb = self.selfcheck(code_fb)
            if sc_fb.get("checkable") and sc_fb.get("ran"):
                return code_fb

        # --- BUILD RETRY CONTEXT ---
        ctx = None
        if sc.get("checkable") and not sc.get("ran") and sc.get("error"):
            # Client 3: validate error via template_run — if template+code runs
            # despite selfcheck failure, the error is in the probe, not the code.
            fb = self._template_run(code)
            if fb is not None:
                if fb.get("ran"):
                    return code  # template+code works; selfcheck probe was faulty
                ctx = _retry_context(fb)
        elif redefines:
            ctx = _REDEFINE_HINT.format(vars=redefines)
        elif target_var != 'result' and not _modifies_target(code, target_var):
            ctx = (f"The problem expects `{target_var}` to be the output variable "
                   f"(it says `{target_var} = ... # put solution in this variable`). "
                   f"Your code should modify or reassign `{target_var}` rather than "
                   f"assigning to `result`. The `result` variable is derived automatically "
                   f"from `{target_var}`.")
        elif not sc.get("checkable"):
            # Selfcheck can't build a probe. Try regex fn-body fallback (client 3)
            # for problems where the template wraps in def f(): but has load_data().
            if _is_function_body_regex(self.prompt):
                code_fb = _fn_body_apply(code)
                sc_fb = self.selfcheck(code_fb)
                if sc_fb.get("checkable") and sc_fb.get("ran"):
                    return code_fb
                if sc_fb.get("checkable") and not sc_fb.get("ran") and sc_fb.get("error"):
                    is_fn_body = True
                    ctx = _retry_context(sc_fb)
                else:
                    code = code_fb  # apply transformation even if we can't validate
            else:
                # Client 4: template-run fallback to detect concrete execution errors.
                fb = self._template_run(code)
                if fb is not None:
                    if fb.get("ran"):
                        return code
                    ctx = _retry_context(fb)

        if ctx is None:
            return code

        # --- RETRY PATH ---
        retry_sys = sys_prompt
        if self.library == "Matplotlib":
            retry_sys += _RETRY_SYS_HINTS
        if "NameError" in ctx:
            retry_sys += _NAMEERROR_HINT
        if "missing" in ctx and "required positional arguments" in ctx:
            retry_sys += _TYPE_ERROR_HINT
        if target_var != 'result' and "The problem expects" in (ctx or ""):
            retry_sys += (f"\n\nIMPORTANT: For this problem, the expected output variable "
                          f"to compute is `{target_var}`, NOT `result`. Do NOT assign to "
                          f"`result`. Instead, modify or reassign `{target_var}` directly. "
                          f"The `result` variable will be captured automatically.")

        for _ in range(2):
            retry_prompt = (
                self.prompt
                + "\n\n[The previous solution attempt failed with this execution result:\n"
                + ctx
                + "\n\nFix the error and output ONLY the corrected code in a ```python block."
                + "\nDo NOT redefine the input variables already provided in the problem context.]"
            )
            resp2 = self.llm(retry_prompt, system=retry_sys)
            code2 = _extract_code_safe(resp2)
            if not code2:
                break
            code2 = _strip_markers(code2)

            sc2 = self.selfcheck(code2)
            redefines2 = sc2.get("redefines") or []
            if sc2.get("checkable") and sc2.get("ran") and not redefines2:
                if target_var == 'result' or _modifies_target(code2, target_var):
                    code = code2
                    break
                # fall through — still setting wrong output variable

            # Client 4: redefines on retry — reject and retry again
            if redefines2:
                ctx = _REDEFINE_HINT.format(vars=redefines2)
                if "NameError" in ctx:
                    retry_sys += _NAMEERROR_HINT
                continue

            # Fn-body fallback for retries (exec-guided)
            if _is_fn_body_error(sc2):
                code2_fb = _fn_body_apply(code2)
                sc2_fb = self.selfcheck(code2_fb)
                if sc2_fb.get("checkable") and sc2_fb.get("ran"):
                    code = code2_fb
                    break
                if sc2_fb.get("checkable") and sc2_fb.get("error"):
                    ctx = _retry_context(sc2_fb)
                    if "NameError" in ctx:
                        retry_sys += _NAMEERROR_HINT
                    if "missing" in ctx and "required positional arguments" in ctx:
                        retry_sys += _TYPE_ERROR_HINT
                    continue
                break

            # Client 0: malformed-probe fallback — selfcheck fails but template_run succeeds
            if (not sc2.get("ran") and target_var != 'result'
                    and _modifies_target(code2, target_var)):
                fb2 = self._template_run(code2)
                if fb2 is not None and fb2.get("ran"):
                    code = code2
                    break

            # Template-run fallback for retries (checkable=False)
            if not sc2.get("checkable"):
                if _is_function_body_regex(self.prompt):
                    code2 = _fn_body_apply(code2)
                    sc2 = self.selfcheck(code2)
                    if sc2.get("checkable") and sc2.get("ran"):
                        code = code2
                        break
                    if sc2.get("checkable") and sc2.get("error"):
                        ctx = _retry_context(sc2)
                        if "NameError" in ctx:
                            retry_sys += _NAMEERROR_HINT
                        if "missing" in ctx and "required positional arguments" in ctx:
                            retry_sys += _TYPE_ERROR_HINT
                        continue
                    else:
                        code = code2
                        break
                fb2 = self._template_run(code2)
                if fb2 is not None:
                    if fb2.get("ran"):
                        code = code2
                        break
                    ctx2 = _retry_context(fb2)
                    if ctx2:
                        ctx = ctx2
                        if "NameError" in ctx:
                            retry_sys += _NAMEERROR_HINT
                        if "missing" in ctx and "required positional arguments" in ctx:
                            retry_sys += _TYPE_ERROR_HINT
                        continue
                break

            # Feed new error
            if sc2.get("checkable") and not sc2.get("ran") and sc2.get("error"):
                if (target_var != 'result'
                        and not _modifies_target(code2, target_var)):
                    pass  # keep target-var hint from pre-retry ctx
                else:
                    ctx = _retry_context(sc2)
                    if "NameError" in ctx:
                        retry_sys += _NAMEERROR_HINT
                    if "missing" in ctx and "required positional arguments" in ctx:
                        retry_sys += _TYPE_ERROR_HINT
            else:
                break

        return code
