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

Usage:
    python3 scripts/baseline.py           # rewrite released-surface.json
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


def artifact(version: str) -> tuple:
    """The best recoverable artifact for one published version, as (marker, entry)."""
    release = json.loads(fetch(f"{INDEX}/{PROJECT}/{version}/json"))
    files = release["urls"]
    for entry in files:
        if entry["packagetype"] == "sdist":
            return SDIST_MARKER, entry
    for entry in files:
        if entry["filename"].endswith(PURE_PYTHON_WHEEL):
            return WHEEL_MARKER, entry
    kinds = ", ".join(sorted({e["filename"] for e in files})) or "nothing"
    raise SystemExit(
        f"{PROJECT} {version} publishes no sdist and no pure-Python wheel ({kinds}), "
        "so its surface cannot be recovered from the registry"
    )


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


def main() -> int:
    published = latest_published()
    version = published["info"]["version"]
    marker, entry = artifact(version)

    with tempfile.TemporaryDirectory() as scratch:
        root = unpack(marker, entry, pathlib.Path(scratch))
        # Tolerant: this describes an artifact that is already out there. A module in
        # it that does not parse cannot be imported by whoever installed it either, so
        # its evaluators were never really on offer; the skipped modules are recorded.
        names, skipped = surface(root, tolerant=True)

    # No punctuation between the marker and the prose: the marker is the first
    # whitespace-delimited token, so a trailing comma would end up inside it.
    tail = f"{entry['filename']} unpacked and read by scripts/surface.py"
    if marker == WHEEL_MARKER:
        tail = f"{tail}; that release publishes no sdist"
    document = {
        "version": version,
        "source": f"{marker}:{tail}",
        "surface": names,
    }
    if skipped:
        document["unparseable"] = skipped
    BASELINE.write_text(json.dumps(document, indent=int(True) + int(True)) + "\n")
    print(f"{BASELINE.name}: {marker} {version}, {len(names)} names")
    return int(False)


if __name__ == "__main__":
    sys.exit(main())
