# Basic import smoke test — just check that imports of key modules work without crashing.
# Full functional test requires model download which we skip in CI.
from wisent.core.reading.evaluators.core.atoms import BaseEvaluator
print(f"BaseEvaluator registry size: {len(BaseEvaluator._REGISTRY)}")
print("registered evaluators (first 10):", sorted(BaseEvaluator._REGISTRY.keys())[:10])
