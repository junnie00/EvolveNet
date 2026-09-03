# EvolveNet: Collaborative Harness Evolution for Agent Self-Improvement

Code release for *EvolveNet: Collaborative Harness Evolution for Agent Self-Improvement*.

EvolveNet evolves executable agent harnesses at several data-local clients, aggregates the resulting
program adaptations, applies an acceptance gate, and redistributes the shared harness. Model weights are
not fine-tuned.

## Release scope

This repository contains the implementation for BIRD, DS-1000, LiveCodeBench, and SWE-bench Verified,
together with the final and baseline harness programs and the ID-only splits used in the reported runs.
It intentionally excludes benchmark question content, reference answers, model replies and caches, logs,
result files, model weights, Docker images, figures, and manuscript files.

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
`config.local.yaml`. The released BIRD splits are indexed against `dev.json`; its expected SHA-256 is
recorded in `reference_examples/text_to_sql/slices/fed_split_v1/meta.json` and can be checked with:

```bash
sha256sum /absolute/path/to/bird-dev/dev.json
# 630272f2b1c44d8cef2c3b246f623355cf0bbc1e832c81061df895530dfc2f06
```

## 4. Prepare split files

The fixed seed-0 splits and client shards used for the reported results are included. They contain only
public benchmark identifiers and routing labels, not questions, answers, model outputs, or evaluation
results:

| Benchmark | Client training shards | Full train / validation / held-out test |
|---|---|---|
| BIRD | `reference_examples/text_to_sql/slices/fed_split_v1/shard{0..4}.json` | `train100.json` / `val100.json` / `test150.json` |
| DS-1000 | `ds1000/slices/fed_split_v1/shard{0..4}.json` | `train100.json` / `val100.json` / `test200.json` |
| LiveCodeBench | `livecodebench/slices/fed_split_v1/shard{0..2}.json` | `train60.json` / `val20.json` / `test30.json` |
| SWE-bench Verified | `swe/slices/fed_split_v1/shard{0..4}.json` | `train30.json` / `val10.json` / `test40.json` |

BIRD entries are `[db_id, index]` pairs, where `index` is the zero-based position among that database's
records in the pinned `dev.json`. The released `test150.json` is the exact 150-question held-out set used
for the paper result, re-indexed against full `dev.json`. Its ordered `question_id` checksum is recorded
in the adjacent `meta.json`. DS-1000 and SWE-bench files are ID lists. LiveCodeBench records contain
`qid`, `difficulty`, and `testtype`, because scoped aggregation uses the difficulty label.

Use the checked-in `shard*.json` files for exact replication. `fedkit.shard` and the BIRD-specific
`reference_examples.text_to_sql.fed.shard` command remain available for constructing new experiments,
but running them is not part of reproducing the reported split.

## 5. Run EvolveNet

The following commands reproduce the reported evolution configurations. They assume the environment and
credentials from Sections 1--3 are already active.

BIRD and DS-1000 each ran as one three-round job:

```bash
PYTHONHASHSEED=0 BIRD_DEV_FILE=dev.json \
python -m reference_examples.text_to_sql.fed.fed_loop \
  --shards reference_examples/text_to_sql/slices/fed_split_v1 \
  --val reference_examples/text_to_sql/slices/fed_split_v1/val100.json \
  --run-name main3 --rounds 3 --local-rounds 1 --group 1 \
  --initial-harness bare --aggregate merge --specialist --supervised \
  --merge-variant scoped --model deepseek-v4-flash \
  --propose-timeout 2700 --merge-timeout 2700 --solve-timeout 180 \
  --budget-lines 150 --max-parallel-clients 5

PYTHONHASHSEED=0 \
python -m fedkit.fed_loop \
  --domain ds1000 --shards ds1000/slices/fed_split_v1 \
  --train ds1000/slices/fed_split_v1/train100.json \
  --val ds1000/slices/fed_split_v1/val100.json \
  --run-name dsmain1 --rounds 3 --local-rounds 1 --group 1 \
  --initial-harness bare --aggregate merge --specialist \
  --merge-variant scoped --model deepseek-v4-flash \
  --propose-timeout 2700 --merge-timeout 2700 --solve-timeout 60 \
  --budget-lines 150 --max-parallel-clients 5
```

LiveCodeBench and SWE-bench were run for two rounds and then resumed for the third round. Keeping the two
stages preserves the released harness names and lineage:

```bash
PYTHONHASHSEED=0 python -m fedkit.fed_loop \
  --domain lcb --shards livecodebench/slices/fed_split_v1 \
  --train livecodebench/slices/fed_split_v1/train60.json \
  --val livecodebench/slices/fed_split_v1/val20.json \
  --run-name lcbmain1 --rounds 2 --local-rounds 1 --group 1 \
  --initial-harness bare --aggregate merge --specialist \
  --merge-variant scoped --model deepseek-v4-flash \
  --propose-timeout 2700 --merge-timeout 2700 --solve-timeout 300 \
  --budget-lines 150 --max-parallel-clients 5

PYTHONHASHSEED=0 python -m fedkit.fed_loop \
  --domain lcb --shards livecodebench/slices/fed_split_v1 \
  --train livecodebench/slices/fed_split_v1/train60.json \
  --val livecodebench/slices/fed_split_v1/val20.json \
  --run-name lcbmain1r3 --rounds 1 --local-rounds 1 --group 1 \
  --initial-harness fedglobal_lcbmain1_r2 --aggregate merge --specialist \
  --merge-variant scoped --model deepseek-v4-flash \
  --propose-timeout 2700 --merge-timeout 2700 --solve-timeout 300 \
  --budget-lines 150 --max-parallel-clients 5

PYTHONHASHSEED=0 python -m fedkit.fed_loop \
  --domain swe --shards swe/slices/fed_split_v1 \
  --train swe/slices/fed_split_v1/train30.json \
  --val swe/slices/fed_split_v1/val10.json \
  --run-name swemain1 --rounds 2 --local-rounds 1 --group 1 \
  --initial-harness bare --aggregate merge --specialist \
  --merge-variant scoped --model deepseek-v4-flash \
  --propose-timeout 2700 --merge-timeout 2700 --solve-timeout 1800 \
  --budget-lines 150 --max-parallel-clients 5

PYTHONHASHSEED=0 python -m fedkit.fed_loop \
  --domain swe --shards swe/slices/fed_split_v1 \
  --train swe/slices/fed_split_v1/train30.json \
  --val swe/slices/fed_split_v1/val10.json \
  --run-name swemain1r3 --rounds 1 --local-rounds 1 --group 1 \
  --initial-harness fedglobal_swemain1_r2 --aggregate merge --specialist \
  --merge-variant scoped --model deepseek-v4-flash \
  --propose-timeout 2700 --merge-timeout 2700 --solve-timeout 1800 \
  --budget-lines 150 --max-parallel-clients 5
```

To evaluate the released final harnesses without re-running evolution:

```bash
mkdir -p runs

PYTHONHASHSEED=0 BIRD_DEV_FILE=dev.json \
python -m reference_examples.text_to_sql.fed.evaluate \
  --slice reference_examples/text_to_sql/slices/fed_split_v1/test150.json \
  --harness fedglobal_main3_r3_retry --run-name paper_bird_test \
  --solve-timeout 180 --workers 5

PYTHONHASHSEED=0 \
python -m fedkit.evaluate \
  --domain ds1000 --slice ds1000/slices/fed_split_v1/test200.json \
  --harness fedglobal_dsmain1_r3 --out runs/ds1000-test.json \
  --solve-timeout 60

PYTHONHASHSEED=0 \
python -m fedkit.evaluate \
  --domain lcb --slice livecodebench/slices/fed_split_v1/test30.json \
  --harness fedglobal_lcbmain1r3_r1 --out runs/lcb-test.json \
  --solve-timeout 300

PYTHONHASHSEED=0 \
python -m fedkit.evaluate \
  --domain swe --slice swe/slices/fed_split_v1/test40.json \
  --harness fedglobal_swemain1r3_r1 --out runs/swe-test.json \
  --solve-timeout 1800
```

The SWE-bench adapter runs each released instance in its benchmark Docker image and grades the resulting
patch with the official SWE-bench evaluator. The other adapters execute their benchmark-provided tests or
gold query evaluator. Generated logs, caches, patches, and result JSON remain under ignored paths.

The comparison settings use the same evolution commands with `--aggregate select-best`,
`--aggregate route`, `--merge-variant global-only`, or `--rounds 1 --local-rounds 3` as appropriate.

## Reproducibility boundary

The repository now fixes the sample identities, client assignments, validation gates, dataset revisions,
run arguments, and final harness source. This is sufficient to run the released artifacts on the exact
reported splits after installing the public benchmark data and configuring the model endpoint.

The response caches are intentionally not published because they contain model outputs derived from
benchmark questions. Consequently, an external API service can still return different samples or change
the implementation behind the same model name; LLM-driven evolution can vary even at temperature zero.
Recreating a byte-identical evolved harness or bit-identical score therefore additionally requires the
original model service version and response cache. For a fresh reproduction, keep the endpoint and model
settings fixed, run the released split, and report repeated-run uncertainty where the service is
nondeterministic.

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
