# EvolveNet: Collaborative Harness Evolution for Agent Self-Improvement

Code release for *EvolveNet: Collaborative Harness Evolution for Agent Self-Improvement*.

EvolveNet evolves executable agent harnesses at several data-local clients, aggregates the resulting
program adaptations, applies an acceptance gate, and redistributes the shared harness. Model weights are
not fine-tuned.

## Release scope

This repository contains the implementation for BIRD, DS-1000, LiveCodeBench, and SWE-bench Verified,
together with the final and baseline harness programs selected in the reported runs. It intentionally
excludes benchmark data, experiment split files, model replies and caches, logs, result files, model
weights, Docker images, figures, and manuscript files.

```text
fedkit/                         shared federated loop, adapters, aggregation and evaluation
ase/                            shared LLM, cache and BIRD database support
ds1000/                         DS-1000 bridge, proposer, evaluator and harnesses
livecodebench/                  LiveCodeBench bridge, proposer, evaluator and harnesses
swe/                            SWE-bench bridge, proposer, evaluator and harnesses
reference_examples/text_to_sql/ BIRD implementation and harnesses
```

The selected final harnesses are:

- BIRD: `reference_examples/text_to_sql/agents/fedglobal_main3_r3_retry.py`
- DS-1000: `ds1000/agents/fedglobal_dsmain1_r3.py`
- LiveCodeBench: `livecodebench/agents/fedglobal_lcbmain1r3_r1.py`
- SWE-bench Verified: `swe/agents/fedglobal_swemain1r3_r1.py`

## 1. Install the runtime

Run commands from the repository root in a Linux environment with Git installed. Python 3.11 or newer
is recommended for BIRD, LiveCodeBench, SWE-bench, and the shared orchestration code.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

SWE-bench additionally needs a working Docker daemon and its pinned Python packages. Its benchmark images
are external and are pulled by the evaluation tooling on first use:

```bash
python -m pip install -r requirements-swe.txt
docker info
```

DS-1000 snippets must run with the benchmark's scientific packages, not only the four packages in
`requirements.txt`. The checked-in Conda file reproduces the official DS-1000 environment at commit
[`b39aab7`](https://github.com/xlang-ai/DS-1000/blob/b39aab71da6d23ef8d3cac59a7c5f834516ab334/environment.yml):

```bash
conda env create -f environment-ds1000.yml
conda activate ds1000-3.10
python -m pip install -r requirements.txt
```

Generating or merging new harnesses also requires an authenticated `claude` executable because
`claude_wrapper.py` invokes Claude Code in non-interactive mode. The experiments used Claude Code
`2.1.245`; confirm the executable before starting an evolution run:

```bash
claude --version
```

The proposer/merger model is separate from the solver configured below. Pass a model name accepted by
your Claude Code setup through `--model`. When using an Anthropic-compatible endpoint instead of Claude
Code's account authentication, configure that subprocess only through its standard environment variables:

```bash
export ANTHROPIC_BASE_URL=https://your-anthropic-compatible-endpoint
export ANTHROPIC_AUTH_TOKEN=...
```

The reported runs passed `--model deepseek-v4-flash`. Evaluating an already released harness does not
invoke Claude Code.

## 2. Configure the models without storing a key

`config.yaml` is an offline mock configuration and contains no credential. For a real BIRD, DS-1000, or
LiveCodeBench run, copy it to the ignored `config.local.yaml`, change `provider`, `base_url`, model names,
and thinking settings, but keep `api_key_env` as the *name* of an environment variable:

```bash
cp config.yaml config.local.yaml
export TTHE_CONFIG="$PWD/config.local.yaml"
export EVOLVENET_API_KEY=...
```

For example, the corresponding part of `config.local.yaml` should contain
`api_key_env: EVOLVENET_API_KEY`; the key itself must not be written into YAML. The solver and controller
are configured independently by `solver_model` and `controller_model`.

SWE-bench reads its solver configuration directly from the environment:

```bash
export SWE_SOLVER_BASE_URL=https://your-openai-compatible-endpoint/v1
export SWE_SOLVER_MODEL=openai/your-model-id
export SWE_SOLVER_API_KEY=...
export SWE_THINKING_STYLE=deepseek  # use "none" for endpoints without this extension
```

To reproduce a model-dependent comparison, keep the endpoint, model identifier, reasoning settings, and
token limits fixed across the baseline and EvolveNet runs.

## 3. Obtain benchmark data

The code pins remote datasets where the dataset is loaded automatically:

| Benchmark | Source consumed by the code | Pinned revision / release |
|---|---|---|
| DS-1000 | [`xlangai/DS-1000`](https://huggingface.co/datasets/xlangai/DS-1000) | `4416080ac5cb80bdf7576aefb8f9a0b4d5426a44`, split `test` |
| LiveCodeBench | [`livecodebench/code_generation_lite`](https://huggingface.co/datasets/livecodebench/code_generation_lite) | `0fe84c3912ea0c4d4a78037083943e8f0c4dd505`, file `test6.jsonl` |
| SWE-bench Verified | [`princeton-nlp/SWE-Bench_Verified`](https://huggingface.co/datasets/princeton-nlp/SWE-Bench_Verified) | `c104f840cc67f8b6eec6f759ebc8b2693d585d4a`, split `test` |

These three datasets are downloaded and cached by `datasets` or `huggingface_hub` on first use.

For BIRD, download the official development set from the [BIRD benchmark](https://bird-bench.github.io/)
and configure an absolute directory containing

```text
/absolute/path/to/bird-dev/
├── dev.json
└── dev_databases/
    └── <db_id>/<db_id>.sqlite
```

Then set `dataset.name: bird` and `dataset.bird_root: /absolute/path/to/bird-dev` in
`config.local.yaml`. If a slice was indexed against another question file, set `BIRD_DEV_FILE` to that
filename; the default is `dev.json`.

## 4. Prepare split files

Experiment splits are deliberately external to this code-only release. The accepted JSON formats are:

```jsonc
// DS-1000: a list of problem IDs
["0", "1", "2"]

// LiveCodeBench: qid plus the routing label used by scoped aggregation
{"items": [{"qid": "abc123_a", "difficulty": "hard"}]}

// SWE-bench Verified: a list of instance IDs
["astropy__astropy-12907"]

// BIRD: each pair is [database ID, zero-based index in BIRD_DEV_FILE]
{"cross": [["california_schools", 0]]}
```

For DS-1000, LiveCodeBench, and SWE-bench, create deterministic client shards with:

```bash
python -m fedkit.shard \
  --domain ds1000 --slice /path/to/train.json \
  --clients 5 --seed 0 --out runs/shards
```

`--domain` accepts `ds1000`, `lcb`, or `swe`. Create BIRD shards with its format-aware command:

```bash
python -m reference_examples.text_to_sql.fed.shard \
  --cross-set /path/to/bird-train.json \
  --clients 5 --seed 0 --out runs/bird-shards
```

## 5. Run EvolveNet

Run the shared loop on DS-1000, LiveCodeBench, or SWE-bench:

```bash
python -m fedkit.fed_loop \
  --domain ds1000 --shards runs/shards \
  --train /path/to/train.json --val /path/to/validation.json \
  --run-name my_run --rounds 3 --local-rounds 1 --group 1 \
  --specialist --merge-variant scoped --model your-claude-code-model-id
```

The comparison settings use the same command with `--aggregate select-best`, `--aggregate route`,
`--merge-variant global-only`, or `--rounds 1 --local-rounds 3`.

Evaluate a released harness on an explicit split:

```bash
python -m fedkit.evaluate \
  --domain ds1000 --slice /path/to/test.json \
  --harness fedglobal_dsmain1_r3 --out runs/ds1000-test.json
```

BIRD uses its original loop:

```bash
python -m reference_examples.text_to_sql.fed.fed_loop \
  --shards /path/to/bird-shards --run-name bird_run \
  --rounds 3 --local-rounds 1 --group 1 \
  --specialist --merge-variant scoped --model your-claude-code-model-id
```

## Reproducibility boundary

This repository is sufficient to inspect the method, run it on user-supplied splits, and evaluate the
released harnesses after installing the stated public benchmarks and model runtime. Dataset revisions and
Python packages are pinned to prevent silent dependency drift.

It is not, by itself, sufficient to reproduce the paper's exact numerical tables: the requested
code-only release excludes the paper's split JSON files and cached model replies, and LLM-driven evolution
remains stochastic even at temperature zero. Exact score reproduction requires those same split
definitions, the same model endpoint/version, and either the original response cache or new repeated runs
with uncertainty reported. The checked-in final harness files are deterministic artifacts from the
reported runs; regenerating an identical harness is not guaranteed.

Source-level checks that do not download data or call an API:

```bash
python -m fedkit.fed_loop --help
python -m fedkit.evaluate --help
python -m fedkit.shard --help
python -m reference_examples.text_to_sql.fed.fed_loop --help
python -m compileall -q ase fedkit ds1000 livecodebench swe reference_examples
```

## Citation

```bibtex
@article{nie2026evolvenet,
  title  = {EvolveNet: Collaborative Harness Evolution for Agent Self-Improvement},
  author = {Nie, Jun and Zhang, Yonggang and Cai, Qianshu and Cheung, Yiu-ming and Tian, Xinmei and Han, Bo},
  year   = {2026}
}
```
