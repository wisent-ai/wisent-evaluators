from setuptools import setup, find_namespace_packages

setup(
    name="wisent-evaluators",
    version="0.1.3",
    author="Lukasz Bartoszcze and the Wisent Team",
    author_email="lukasz.bartoszcze@wisent.ai",
    description="Benchmark evaluators (metric computation, oracle/judge logic) for the wisent package family",
    url="https://github.com/wisent-ai/wisent-evaluators",
    # Namespace-package discovery so wisent/core/reading/evaluators ships even
    # though wisent/core/__init__.py and wisent/core/reading/__init__.py don't
    # exist in this repo (they live in wisent-core; this repo only contributes
    # the evaluators subtree under the wisent.* namespace).
    packages=find_namespace_packages(include=["wisent", "wisent.*"]),
    python_requires=">=3.9",
    include_package_data=True,
    package_data={
        "wisent": [
            "support/parameters/evaluator_methodologies/**/*.json",
        ],
    },
    install_requires=[
        "wisent>=0.10.0",
        "wisent-extractors>=0.1.0",  # rotator.py dispatches to extractors
        "sympy>=1.12",
        "latex2sympy2_extended>=1.0.0",
        "word2number",
        "evaluate",
        "torch>=2.0",
        "transformers>=4.30",
        "scipy",
        "pebble",
        "tqdm",
        "regex",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
