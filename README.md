# wisent-evaluators

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
