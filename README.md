# RouteSQL

RouteSQL is the current public-facing name for the project developed in this repository, previously organized locally as `DAIL-SQL2`.

The project started from the original DAIL-SQL codebase and evolved into a path-aware, subspace-aware, multi-candidate Text-to-SQL system with structured repair and final-stage arbitration. The current code focuses on Spider / Spider-style evaluation, with `dataset_min` used for fast iteration and `spider` full runs used for final validation.

## What This Repository Is

This repository is not a fresh reimplementation of the original DAIL-SQL paper. It is a continued engineering and research branch built on top of that baseline, with the goal of improving LLM-based Text-to-SQL through:

- schema and value linking
- join-path discovery and ranking
- schema/path subspace construction
- question rewriting under structured subspace hints
- two-stage SQL framework generation and filling
- multi-route candidate generation
- structured repair
- final merge and reranking

## Evolution History

### Stage 0: Original DAIL-SQL baseline

The project began from the original DAIL-SQL prompt-engineering style pipeline:

- few-shot example selection
- prompt generation
- one-pass or self-consistency SQL generation

### Stage 1: First local evolution in `DAIL-SQL`

The first major extension line was developed in the older local repository `DAIL-SQL`:

- `V1`: introduced `stage1 + stage2`, framework generation, fill, and repair
- `V2`: introduced `stage1/stage2 merge`
- `V3`: softened the framework from a hard scaffold into a soft constraint and added `schema-first` / `direct` routes
- `V4`: added execution-result statistics, framework confidence, candidate trace, and stronger merge

This line is now kept mainly for history and comparison.

### Stage 2: Method-level redesign in `DAIL-SQL2`

The current repository contains the second major evolution line:

- `V1`: path candidates, path scoring, path-aware repair
- `V2`: path-aware route selection and typed repair
- `V3`: multi-route support bonuses and finer typed repair
- `V4`: schema/path subspace, graph consistency, structured repair, path-conditioned route selection
- `V5`: added question rewriting with subspace hints, path-graph subspace construction, graph-consistency scoring, join-path graph repair, pairwise candidate preference, and EM-oriented normalization
- `V6`: current active line, used to continue improving the repair / merge / verifier layer on top of the V5 backbone

## Current Repository Status

Current version marker:

- `VERSION.txt`: `V6_qaware_simplemerge`

Current practical meaning:

- `V5` is the stable backbone for generation
- `V6` is the current iteration on top of `V5`, focused mainly on the final decision layer rather than a full redesign
- full Spider performance is still an active optimization target

## Repository Layout

Key directories:

- `scripts/python_tools/`
  Main pipeline logic, including SQL generation, repair, reranking, and merge.
- `utils/schema_path_utils.py`
  Schema graph construction, join-path enumeration, path scoring, and graph consistency helpers.
- `utils/linking_process.py`
  Linking preprocessing and schema expansion utilities.
- `utils/linking_utils/`
  Schema linking and value linking support code.
- `prompt/`
  Prompt templates and example selector logic.
- `scripts/server/`
  Reproducible server-side pipeline scripts for dataset-min and full Spider runs.
- `results/`
  Only `latest` and archived `runs/` results should be kept here in a clean repo state.

## Core Modules

### 1. Linking Layer

Responsible for grounding the question into candidate tables, columns, and values.

Main files:

- `utils/linking_process.py`
- `utils/linking_utils/spider_match_utils.py`

### 2. Schema / Path Reasoning Layer

Responsible for building schema graphs, enumerating possible join paths, and deriving schema/path subspaces.

Main file:

- `utils/schema_path_utils.py`

### 3. LLM Control Layer

Responsible for:

- SQL normalization
- framework generation and parsing
- question rewriting
- route planning
- candidate generation
- structured repair
- candidate ranking
- verifier-style diagnostics

Main file:

- `scripts/python_tools/ask_llm.py`

### 4. Final Merge Layer

Responsible for stage1/stage2 arbitration and final output selection.

Main file:

- `scripts/python_tools/merge_predictions.py`

### 5. Runtime / Pipeline Layer

Responsible for reproducible command-line execution and archived run management.

Main files:

- `scripts/server/run_datasetmin_pipeline_V5.sh`
- `scripts/server/run_datasetmin_pipeline_V6.sh`
- `scripts/server/run_spider_full_pipeline_v5.sh`
- `utils/runtime_setup.py`

## What Is Intentionally Not Tracked

For a clean public repository, large local artifacts should not be committed:

- local Python environments such as `venv/` and `.conda/`
- model and embedding caches in `.cache/` and `vector_cache/`
- local datasets under `dataset/`
- logs under `logs/`
- experiment outputs under `results/` except documentation and archived summaries you intentionally keep
- private environment files such as `scripts/server/server.env`
- downloaded third-party runtime bundles such as Stanford CoreNLP jars

See `.gitignore` for the expected public-repo policy.

## Setup Notes

This repository expects the following external resources to be prepared locally before running experiments:

- Spider dataset under `dataset/`
- Stanford CoreNLP under `third_party/`
- a local environment configured from `requirements.txt`
- `scripts/server/server.env` created from `scripts/server/server.env.example`

## Recommended GitHub Repository Name

Recommended public repo name:

- `RouteSQL`

Reason:

- short and easier to remember than `DAIL-SQL2`
- still clearly preserves the `SQL` identity
- matches the project's current emphasis on path / route / subspace reasoning

If you want a slightly more descriptive variant, a good fallback is:

- `RouteSQL-Text2SQL`

## Current Goal

The near-term goal is not to add many new heuristics at once, but to stabilize the generation, repair, and merge stack so that:

- `dataset_min` experiments are consistently reproducible
- full Spider runs do not regress relative to earlier strong baselines
- the final system is clean enough to modularize further
