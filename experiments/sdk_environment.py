"""Explicit offline dependency gate for the separately supplied research SDK.

No SDK or dataset is downloaded, copied into the repository, or substituted.
Set RFQ_SDK_ROOT to the directory containing rfq/, examples/ and demo/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
SDK_PREFIX = "inputs/official/kit/IMC/"
KIT = Path(os.environ.get("RFQ_SDK_ROOT", ROOT / "inputs/official/kit/IMC")).expanduser().absolute()
EXPECTED = ROOT / "data/sdk-contract-hashes.json"
MESSAGE = (
    "Optional SDK research is disabled: provide the separately supplied SDK, "
    "set RFQ_SDK_ROOT to its IMC directory, install requirements-sdk.txt, "
    "and run python -m experiments.sdk_environment. See docs/sdk-research.md."
)


def sdk_available():
    return (KIT / "rfq/rfq/engine.py").is_file()


def source_key(path):
    """Stable relative archive key even when the SDK lives outside this tree."""
    path = Path(path).absolute()
    try:
        return SDK_PREFIX + str(path.relative_to(KIT))
    except ValueError:
        return str(path.relative_to(ROOT))


def source_path(key):
    key = str(key)
    if key.startswith(SDK_PREFIX):
        return KIT / key[len(SDK_PREFIX):]
    return ROOT / key


def complete_source_hashes(hashes):
    """Include the dependency gate and contract map in every new study seal."""
    result = dict(hashes)
    for path in (Path(__file__), EXPECTED):
        result[source_key(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def require_sdk():
    if not sdk_available():
        raise ImportError(MESSAGE)
    expected = json.loads(EXPECTED.read_text())
    failures = []
    for name, digest in expected["sha256"].items():
        path = KIT / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if actual != digest:
            failures.append(name)
    if failures:
        raise ImportError(
            "The supplied SDK differs from the audited contract snapshot: "
            + ", ".join(failures)
            + ". Re-audit this SDK before using these experiments."
        )
    # A notebook may already have imported a different installation. Matching
    # files on disk does not establish which cached modules Python will use.
    # This catches accidental context mixing; it is not a security sandbox.
    package_root = (KIT / "rfq/rfq").resolve()
    for name, module in tuple(sys.modules.items()):
        if name != "rfq" and not name.startswith("rfq."):
            continue
        origins = []
        filename = getattr(module, "__file__", None)
        if filename is not None:
            origins.append(Path(filename).resolve())
        namespace = getattr(module, "__path__", ())
        origins.extend(Path(path).resolve() for path in namespace)
        if not origins or any(not path.is_relative_to(package_root) for path in origins):
            raise ImportError(
                f"Already imported {name} has an origin outside the audited SDK "
                f"package {package_root}. Restart this Python process with the "
                "configured RFQ_SDK_ROOT before importing rfq."
            )
    import_root = (KIT / "rfq").resolve()
    location = str(import_root)
    # Move, rather than merely add: an existing later entry must not leave a
    # different installation ahead of the audited source tree.
    sys.path[:] = [location] + [
        entry for entry in sys.path
        if not isinstance(entry, (str, os.PathLike)) or Path(entry).resolve() != import_root
    ]
    return KIT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        require_sdk()
        import numpy
        import websockets
        from rfq.engine import Engine  # noqa: F401
        from rfq.runner import DealerRunner  # noqa: F401
    except ImportError as error:
        print(json.dumps({"available": False, "reason": str(error), "sdk_root": str(KIT)}, indent=2))
        raise SystemExit(2)
    print(json.dumps({
        "available": True, "contract_hashes_match": True, "sdk_root": str(KIT),
        "python": sys.version, "numpy": numpy.__version__,
        "websockets": websockets.__version__, "platform": platform.platform(),
        "scope": "Offline research only; no network or external trade is initiated.",
    }, indent=2))


if __name__ == "__main__":
    main()
