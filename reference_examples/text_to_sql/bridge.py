"""Bridge from the EvolveNet harness framework to the BIRD/ASE infrastructure.

Reuses our working stack: BIRD loading (ase.dataset), SQLite execution (ase.db), and the FROZEN weak
solver through the configured OpenAI-compatible endpoint (ase.llm). Importing this module:
  - puts the repository and configured work directory on sys.path,
  - uses config.yaml (or TTHE_CONFIG) for the model and dataset,
  - builds one shared LLM + dataset.

Set EVO_DIR to override the work directory. Credentials are read only from the environment variable named
by ``llm.api_key_env`` in the configuration.
"""
import os
import sys

import yaml

_THIS = os.path.abspath(__file__)
_MH_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))
EVO = os.path.abspath(os.environ.get("EVO_DIR", _MH_ROOT))
for p in (_MH_ROOT, EVO):                                                     # absolute paths, chdir-safe
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(EVO)                                                                 # ase dataset paths are relative

from ase.llm import LLM, LLMConfig, extract_sql          # noqa: E402
from ase.dataset import build_dataset                    # noqa: E402
from ase.db import compare_results                       # noqa: E402

from .solver_cache import SolverCache                    # noqa: E402

EXEC_TIMEOUT, EXEC_LIMIT = 30.0, 20000

_config_path = os.path.abspath(os.environ.get("TTHE_CONFIG", os.path.join(EVO, "config.yaml")))
_cfg = yaml.safe_load(open(_config_path, encoding="utf-8"))
# --- solver model-ablation overrides (env only; config.yaml left untouched). When SOLVER_MODEL is set we
# point BOTH the solver and controller roles at that one model, so a model-ablation run is pure (no
# a second provider dependency). Unset -> config.yaml is used as-is. ---
_llmcfg = dict(_cfg["llm"])
if os.environ.get("SOLVER_BASE_URL"):
    _llmcfg["base_url"] = os.environ["SOLVER_BASE_URL"]
if os.environ.get("SOLVER_MODEL"):
    _llmcfg["solver_model"] = _llmcfg["controller_model"] = os.environ["SOLVER_MODEL"]
if os.environ.get("SOLVER_API_KEY_ENV"):
    _llmcfg["api_key_env"] = _llmcfg["controller_api_key_env"] = os.environ["SOLVER_API_KEY_ENV"]
_LLM = LLM(LLMConfig(**_llmcfg))
_DS = build_dataset(_cfg["dataset"], _cfg["output_dir"])

extract_sql = extract_sql        # re-export
compare_results = compare_results


def get_db(db_id):
    return _DS.get_database(db_id)


def eval_questions(db_id):
    return _DS.eval_questions(db_id)


import threading as _threading
_tls = _threading.local()


def set_temp_override(t):
    """Thread-local temperature override for solver_llm (None clears). Lets us run a harness at T>0 for
    SELF-CONSISTENCY without touching the harness's own temperature=0 calls."""
    _tls.temp_override = t


_CACHE = SolverCache(os.environ.get("SQL_SOLVER_CACHE",
                                   os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                "logs", "solver_cache.json")))


def solver_llm(prompt, system="", temperature=0.0, n=1, seq=0):
    """Call the FROZEN weak solver (deepseek-v4-flash). n=1 -> str, n>1 -> list[str].

    Replies are CACHED on (prompt, system, effective temperature, n, seq). Without this, two harnesses
    that build byte-identical prompts still receive different answers — thinking mode is nondeterministic
    even at temperature 0 — so a score difference cannot be attributed to a behavioural difference. That
    sampling spread was measured here at +-2..3 questions per 50, which is the size of the effects this
    loop is trying to detect. `seq` (see harness_base.llm) keeps deliberate resampling intact: the same
    harness asking the same question three times still draws three distinct replies.
    """
    ov = getattr(_tls, "temp_override", None)
    t = ov if ov is not None else temperature
    # Some model APIs constrain temperature to a single value (for example,
    # kimi-k2.6 accepts only 1). Keep the default harness behavior unchanged
    # unless an isolated model-ablation process explicitly opts in.
    if os.environ.get("SOLVER_TEMPERATURE_OVERRIDE"):
        t = float(os.environ["SOLVER_TEMPERATURE_OVERRIDE"])
    outs = _CACHE.get_or_call((prompt, system, t, n, seq),
                              lambda: _LLM.chat("harness", system, prompt, model_role="solver",
                                                n=n, temperature=t))
    return outs[0] if n == 1 else outs


def execute(db, sql):
    """Run SQL; returns {ok, rows, ...}. Never raises."""
    return db.execute(sql, timeout=EXEC_TIMEOUT, limit=EXEC_LIMIT)


def gold_result(db, gold_sql):
    return db.execute(gold_sql, timeout=EXEC_TIMEOUT, limit=EXEC_LIMIT)


def is_correct(pred_result, gold):
    """MEASUREMENT ONLY. True iff the predicted rows match the gold rows (set semantics)."""
    return bool(pred_result["ok"] and gold["ok"] and compare_results(pred_result["rows"], gold["rows"]))
