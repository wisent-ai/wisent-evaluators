# wisent-evaluators

<!-- wisent-readme-signals:start -->
[![CI](https://github.com/wisent-ai/wisent-evaluators/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/wisent-ai/wisent-evaluators/actions/workflows/tests.yml)
[![Release](https://img.shields.io/github/v/release/wisent-ai/wisent-evaluators?display_name=tag&sort=semver)](https://github.com/wisent-ai/wisent-evaluators/releases)
[![Downloads](https://img.shields.io/github/downloads/wisent-ai/wisent-evaluators/total)](https://github.com/wisent-ai/wisent-evaluators/releases)
[![License](https://img.shields.io/github/license/wisent-ai/wisent-evaluators)](https://github.com/wisent-ai/wisent-evaluators)
[![Discord](https://img.shields.io/badge/Discord-Join%20Wisent-5865F2?logo=discord&logoColor=white)](https://discord.gg/qRjpkthq54)
<!-- wisent-readme-signals:end -->


Benchmark evaluators split out of the wisent monorepo. Contains the `BaseEvaluator`
metaclass registry and ~130 benchmark-specific evaluator classes (math, code,
hallucination, safety, multilingual, reasoning). Ships the 380 JSON config files
under `wisent/support/parameters/evaluator_methodologies/` as package data.

## Install

```
pip install wisent-evaluators
```

Pulls `wisent-extractors` transitively (needed by `rotator.py` for task-name
dispatch) and heavy ML deps (torch, transformers) for judge-model evaluators.

## Usage

```python
from wisent.core.reading.evaluators.core.atoms import BaseEvaluator
from wisent.core.reading.evaluators.rotator import EvaluatorRotator

# Auto-select an evaluator by task name
ev = EvaluatorRotator.for_task("mmlu_abstract_algebra")
```

## Namespace packaging

Namespace-style (PEP 420) — no top-level `wisent/__init__.py`. Side-by-side with
`wisent` (core) and `wisent-extractors`.
