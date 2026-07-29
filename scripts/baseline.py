"""Regenerate released-surface.json from the artifact the registry actually serves.

The baseline is the surface of the version people can install today, so it is not
written by hand and not read out of setup.py. This asks PyPI which version it serves,
downloads that artifact, and runs `scripts/surface.py` against the unpacked tree.

Two things this deliberately does not do:

* It never looks up the version setup.py declares. The moment someone bumps ahead of
  a release, that lookup 404s, and a generator that fell back to the working tree
  would throw away the real published baseline and measure every later change against
  something nobody ever installed.
* It never falls back a tier when a better one exists, and never invents a tier it
  cannot recover. An unknown situation is a loud failure, because a baseline that
  quietly describes the wrong artifact is worse than no baseline.

The first whitespace-delimited token of the `source` field is a marker naming the
tier the baseline came from, so the workflow can assert the two agree in both
directions — a `pypi-*` baseline must be served by PyPI, and a non-registry baseline
must exist only because PyPI serves nothing. The markers are constants here and the
workflow matches on them, so the coupling is a token and not prose.

This package's newest release is wheel-only, so the wheel reader carries the baseline
alone. A wheel contains only what `packages=` managed to list, and a file it drops is
indistinguishable from a removed capability — the rule would call it breaking. So the
wheel reader is not taken on trust: `--cross-check` points both readers at a version
that published an sdist *and* a wheel and requires the same set out of each.

Usage:
    python3 scripts/baseline.py                    # rewrite released-surface.json
    python3 scripts/baseline.py --cross-check 0.1.1  # sdist and wheel must agree
"""

from __future__ import annotations

import io
import json
import pathlib
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile

FIRST = int(False)

sys.path.insert(FIRST, str(pathlib.Path(__file__).resolve().parent))

from surface import surface  # noqa: E402

PROJECT = "wisent-evaluators"
INDEX = "https://pypi.org/pypi"

# Baseline tiers, best first. The workflow reads the marker back out of `source`.
SDIST_MARKER = "pypi-sdist"
WHEEL_MARKER = "pypi-wheel"

# A wheel is only a faithful source tree when it is pure Python and not platform- or
# interpreter-specific; anything else would hide code behind a build.
PURE_PYTHON_WHEEL = "-py3-none-any.whl"

REPOSITORY = pathlib.Path(__file__).resolve().parent.parent
BASELINE = REPOSITORY / "released-surface.json"


def fetch(url: str) -> bytes:
    """The bytes at a URL, or a loud failure."""
    try:
        with urllib.request.urlopen(url) as response:
            return response.read()
    except urllib.error.URLError as error:
        raise SystemExit(f"{url}: {error}") from error


def latest_published() -> dict:
    """What PyPI serves for this project right now."""
    return json.loads(fetch(f"{INDEX}/{PROJECT}/json"))


def candidates(version: str) -> dict:
    """Every recoverable artifact for one published version, keyed by marker."""
    release = json.loads(fetch(f"{INDEX}/{PROJECT}/{version}/json"))
    found = {}
    for entry in release["urls"]:
        if entry["packagetype"] == "sdist":
            found.setdefault(SDIST_MARKER, entry)
        elif entry["filename"].endswith(PURE_PYTHON_WHEEL):
            found.setdefault(WHEEL_MARKER, entry)
    return found


def artifact(version: str) -> tuple:
    """The best recoverable artifact for one published version, as (marker, entry)."""
    found = candidates(version)
    for marker in (SDIST_MARKER, WHEEL_MARKER):
        if marker in found:
            return marker, found[marker]
    raise SystemExit(
        f"{PROJECT} {version} publishes no sdist and no pure-Python wheel, so its "
        "surface cannot be recovered from the registry"
    )


def recover(marker: str, entry: dict) -> tuple:
    """The surface of one published artifact, and the modules it could not parse."""
    with tempfile.TemporaryDirectory() as scratch:
        root = unpack(marker, entry, pathlib.Path(scratch))
        # Tolerant: this describes an artifact that is already out there. A module in
        # it that does not parse cannot be imported by whoever installed it either, so
        # its evaluators were never really on offer; the skipped modules are recorded.
        return surface(root, tolerant=True)


def unpack(marker: str, entry: dict, destination: pathlib.Path) -> pathlib.Path:
    """Unpack a published artifact and return the root that holds the `wisent` tree."""
    payload = io.BytesIO(fetch(entry["url"]))
    if marker == SDIST_MARKER:
        with tarfile.open(fileobj=payload, mode="r:gz") as archive:
            try:
                archive.extractall(destination, filter="data")
            except TypeError:  # filter= arrived in Python 3.12
                archive.extractall(destination)
    else:
        with zipfile.ZipFile(payload) as archive:
            archive.extractall(destination)

    if (destination / "wisent").is_dir():
        return destination
    inner = [child for child in destination.iterdir() if (child / "wisent").is_dir()]
    if len(inner) != int(True):
        raise SystemExit(
            f"{entry['filename']}: expected exactly one tree containing `wisent`, "
            f"found {len(inner)}"
        )
    return inner[FIRST]


def cross_check(version: str) -> int:
    """Assert the wheel reader and the sdist reader agree on one version.

    The baseline tier is chosen per version, so a wheel-only release is read by a
    path no sdist ever validated. A wheel omits whatever `packages=` failed to list,
    and that omission is indistinguishable from a removed capability: the rule would
    read it as breaking. The only way to know the wheel reader is honest is to point
    both readers at a version that published BOTH artifacts and require the same set.
    """
    found = candidates(version)
    missing = [m for m in (SDIST_MARKER, WHEEL_MARKER) if m not in found]
    if missing:
        raise SystemExit(
            f"{PROJECT} {version} publishes no {' and no '.join(missing)}, so the two "
            "readers cannot be compared on it; pick a version that has both"
        )

    refused = {}
    surfaces = {}
    for marker, entry in found.items():
        try:
            surfaces[marker], _ = recover(marker, entry)
        except SystemExit as error:
            # Say which artifact refused. "One of them is unreadable" is the answer
            # that sends someone auditing the reader when the artifact is the problem.
            refused[marker] = f"{entry['filename']}: {error}"
    if refused:
        raise SystemExit(
            f"cannot compare the readers on {PROJECT} {version}; "
            + "; ".join(f"{marker} {why}" for marker, why in sorted(refused.items()))
        )

    from_sdist = surfaces[SDIST_MARKER]
    from_wheel = surfaces[WHEEL_MARKER]
    only_sdist = sorted(set(from_sdist) - set(from_wheel))
    only_wheel = sorted(set(from_wheel) - set(from_sdist))
    if only_sdist or only_wheel:
        raise SystemExit(
            f"the two readers disagree on {PROJECT} {version}: "
            f"sdist only {only_sdist or 'nothing'}, wheel only {only_wheel or 'nothing'}. "
            "A wheel that drops names reads as removed capability, so the wheel tier "
            "cannot be trusted until this is explained"
        )
    print(
        f"{version}: {found[SDIST_MARKER]['filename']} and "
        f"{found[WHEEL_MARKER]['filename']} agree on {len(from_sdist)} names"
    )
    return int(False)


def identity() -> tuple:
    """What the registry serves right now: (version, marker, entry, source string).

    Two small JSON requests. Nothing is downloaded and nothing is parsed, which is
    all the staleness check needs to know whether the committed baseline still names
    the best artifact.
    """
    published = latest_published()
    version = published["info"]["version"]
    marker, entry = artifact(version)

    # No punctuation between the marker and the prose: the marker is the first
    # whitespace-delimited token, so a trailing comma would end up inside it.
    tail = f"{entry['filename']} unpacked and read by scripts/surface.py"
    if marker == WHEEL_MARKER:
        tail = f"{tail}; that release publishes no sdist"
    return version, marker, entry, f"{marker}:{tail}"


def best_available() -> dict:
    """The baseline document for the best artifact the registry serves right now."""
    version, marker, entry, source = identity()
    names, skipped = recover(marker, entry)
    document = {"version": version, "source": source, "surface": names}
    if skipped:
        document["unparseable"] = skipped
    return document


def main(argv: list) -> int:
    if "--cross-check" in argv:
        positional = [arg for arg in argv if not arg.startswith("-")]
        if not positional:
            raise SystemExit("--cross-check needs the version to compare the readers on")
        return cross_check(positional[FIRST])

    if "--best" in argv:
        # For the workflow's staleness check, which needs only the version and the
        # tier. No surface is computed at all here — not computed and discarded, but
        # never produced — so a recomputed surface cannot reach the decision even by
        # accident, and CI does not download an artifact to learn a filename.
        version, _marker, _entry, source = identity()
        print(json.dumps({"version": version, "source": source}, indent=int(True) + int(True)))
        return int(False)

    document = best_available()
    BASELINE.write_text(json.dumps(document, indent=int(True) + int(True)) + "\n")
    print(
        f"{BASELINE.name}: {document['source'].split(':')[FIRST]} "
        f"{document['version']}, {len(document['surface'])} names"
    )
    return int(False)


if __name__ == "__main__":
    sys.exit(main(sys.argv[int(True) :]))
