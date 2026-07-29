"""Print this package's public surface: the evaluator names it registers.

What a caller of this package depends on is not its Python symbols — the exported
classes are reached through a registry, not by import — but **which evaluators it
offers by name**. Every evaluator here subclasses `BaseEvaluator`, and
`BaseEvaluator.__init_subclass__` files the subclass under its class attribute
`name`. That string is the whole handle a user has:

    EvaluatorRotator(evaluator="math")      # or BaseEvaluator.get("math")

Adding a name is a capability; removing or renaming one breaks whoever evaluated
with it yesterday, with an `EvaluatorError: Unknown evaluator`. So the registered
names are the public contract, and this prints them for the shared versioning rule
to compare.

Read with `ast`, never by importing. Importing this package pulls in `wisent`,
`wisent-extractors`, `torch`, `transformers` and `sympy`, and a release decision must
not depend on a machine having them. It also means this runs unchanged against an
unpacked sdist, so the surface of an already published version can be recovered
exactly rather than assumed.

Finding the registry without importing means finding the subclasses. A class counts
as an evaluator when its base chain reaches `BaseEvaluator` through classes defined
in this repository, and its name is the literal assigned to `name` in the class body:

    class MathEvaluator(BaseEvaluator):
        name = "math"

Bases are matched on their final segment (`BaseEvaluator`, `atoms.BaseEvaluator` and
`BaseEvaluator[T]` all count), so two same-named classes in different modules share a
verdict. That errs towards reporting a name, never towards dropping one. Classes in
the parallel `CustomEvaluator` hierarchy and the coding `Provider` registry are not
part of this surface: they are not in the evaluator registry and cannot be asked for
by name through it.

Usage:
    python3 scripts/surface.py [root]     # root defaults to the repository
"""

from __future__ import annotations

import ast
import json
import pathlib
import sys

ROOT_CLASS = "BaseEvaluator"
NAME_ATTRIBUTE = "name"


def base_name(node: ast.expr) -> str:
    """The final segment of a base class expression, or "" if it has none."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return base_name(node.value)
    return ""


def registered_name(node: ast.ClassDef) -> str:
    """The literal assigned to `name` in a class body, or "" if there is none."""
    found = ""
    for statement in node.body:
        if isinstance(statement, ast.Assign):
            targets = [t.id for t in statement.targets if isinstance(t, ast.Name)]
            value = statement.value
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            targets, value = [statement.target.id], statement.value
        else:
            continue
        if NAME_ATTRIBUTE not in targets:
            continue
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            found = value.value
    return found


def classes(source: pathlib.Path) -> list:
    """Every class in one module, as (class name, base names, registered name)."""
    try:
        tree = ast.parse(source.read_text(), filename=str(source))
    except OSError as error:
        raise SystemExit(f"{source}: {error}") from error
    except SyntaxError as error:
        # Refuse rather than skip. A module that does not parse cannot be imported
        # either, so its evaluators never reach the registry; skipping it would
        # report a smaller surface, and the rule would read that as a removed
        # capability. The surface is unknown here, not shrunk.
        raise SystemExit(
            f"{source}: does not parse, so the surface is unknown: {error}"
        ) from error

    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            bases = [base_name(base) for base in node.bases]
            found.append((node.name, [b for b in bases if b], registered_name(node)))
    return found


def surface(root: pathlib.Path, tolerant: bool = False) -> tuple:
    """The surface, and the modules that had to be skipped to produce it.

    `tolerant` exists for one job: recovering the surface of an artifact that was
    already published with a module that does not parse. Such a module cannot be
    imported by whoever installed it either, so its evaluators were never really on
    offer and leaving them out is the truthful reading. Skipped modules are always
    reported, never swallowed.
    """
    package = root / "wisent"
    if not package.is_dir():
        raise SystemExit(f"{package} is not a directory; is {root} the repository root?")

    declared = []
    skipped = []
    for source in sorted(package.rglob("*.py")):
        try:
            declared.extend(classes(source))
        except SystemExit:
            if not tolerant:
                raise
            skipped.append(str(source.relative_to(root)))

    # Subclassing is transitive, and a subclass may be declared before its base is
    # read, so grow the set of evaluator classes until it stops growing.
    evaluators = {ROOT_CLASS}
    growing = True
    while growing:
        growing = False
        for class_name, bases, _ in declared:
            if class_name not in evaluators and any(b in evaluators for b in bases):
                evaluators.add(class_name)
                growing = True

    names = set()
    anonymous = []
    for class_name, _, name in declared:
        if class_name not in evaluators or class_name == ROOT_CLASS:
            continue
        if name:
            names.add(name)
        else:
            anonymous.append(class_name)

    if anonymous:
        # Registration reads `name` off the class. A subclass without a literal one
        # either fails to import or silently inherits its parent's, and either way
        # what this package offers can no longer be read off the source.
        raise SystemExit(
            "these evaluator classes register under a name this cannot read, so the "
            f"surface is unknown: {', '.join(sorted(anonymous))}"
        )
    if not names:
        raise SystemExit(
            f"no evaluator names found under {package}. Either the evaluators moved, "
            f"or they stopped subclassing {ROOT_CLASS} with a literal `name` — both "
            "change what this package promises, so refusing rather than reporting an "
            "empty surface"
        )
    return sorted(names), skipped


def main(argv: list) -> int:
    tolerant = "--tolerant" in argv
    positional = [arg for arg in argv if not arg.startswith("-")]
    root = (
        pathlib.Path(positional[int(False)])
        if positional
        else pathlib.Path(__file__).resolve().parent.parent
    )
    names, skipped = surface(root, tolerant)
    document = {"surface": names}
    if skipped:
        document["unparseable"] = skipped
    print(json.dumps(document, indent=int(True) + int(True)))
    return int(False)


if __name__ == "__main__":
    sys.exit(main(sys.argv[int(True) :]))
