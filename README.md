<!-- wisent-banner:start -->
<p align="center">
  <img src="assets/readme-banner.webp" alt="wisent-evaluators by Wisent" width="100%">
</p>
<!-- wisent-banner:end -->

<!-- wisent-readme-signals:start -->
[![Source](https://img.shields.io/badge/GitHub-Source-181717?logo=github)](https://github.com/wisent-ai/wisent-evaluators) [![Issues](https://img.shields.io/badge/GitHub-Issues-181717?logo=github)](https://github.com/wisent-ai/wisent-evaluators/issues) [![Wisent](https://img.shields.io/badge/Wisent-Website-0B0B0B)](https://wisent.ai) [![Discord](https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white)](https://discord.gg/qRjpkthq54) [![LinkedIn](https://img.shields.io/badge/LinkedIn-Follow-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/company/wisent-ai/) [![X](https://img.shields.io/badge/X-Follow-000000?logo=x&logoColor=white)](https://x.com/wisentai) [![Enterprise](https://img.shields.io/badge/Enterprise-Book%20a%20call-0B0B0B?logo=calendly)](https://calendly.com/lbartoszcze)
<!-- wisent-readme-signals:end -->

# wisent-evaluators

Monitor and Control Your AI Agent Brain.

You look at what your model says. But what was it actually thinking? Wisent shows
you how to use information from AI activations, intermediate steps within its
layers, to your advantage. Wisent is a full toolkit for representation
engineering, activation steering and mechanistic interpretability. Cut
hallucination rates, decensor your model or stop it from being detected by
AI-generated text detectors. Your Models — Yours to Control. Better than
fine-tuning. Better than analysing the outputs directly.

Deploy the latest research in your stack. This is where the benchmark evaluators
live — the registry, ~130 evaluator classes and their 380 configs.

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
