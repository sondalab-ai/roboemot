#!/usr/bin/env python3
"""Vendor a pinned copy of @sondalab/ui-kit for the offline web UI (SPEC 12, web UI).

The robot serves the operator page without internet, so the kit's CSS, effects runtime and two
fonts are copied verbatim under ``webui/static/vendor/sondalab-ui-kit/`` together with
``VENDORED.json`` (version, tarball integrity, sha256 of every file). Nothing is rewritten.

    python scripts/sync_ui_kit.py              # download the pinned tarball, verify, copy
    python scripts/sync_ui_kit.py --from DIR   # use an unpacked package/ directory instead
    python scripts/sync_ui_kit.py --check      # offline: files on disk match the manifest

To upgrade, change ``UI_KIT_VERSION`` and ``UI_KIT_INTEGRITY`` (npm ``dist.integrity``), run the
script and review the diff. ``tests/test_webui_vendor.py`` fails if disk, manifest and pin drift.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

UI_KIT_VERSION = "0.37.0"
UI_KIT_INTEGRITY = (
    "sha512-Jx4zz27Wjf2KixfUZ69nXBtBOTb5+wt3KAfKjFZDMs+L2bMo3wT6727QhHFSjelg/"
    "TCPTUbaiMm8piGuvNwSWA=="
)
TARBALL_URL = f"https://registry.npmjs.org/@sondalab/ui-kit/-/ui-kit-{UI_KIT_VERSION}.tgz"
VENDOR_DIR = (
    Path(__file__).resolve().parent.parent
    / "packages/roboemot-core/src/roboemot/webui/static/vendor/sondalab-ui-kit"
)
MANIFEST = "VENDORED.json"
# No glass, lens, goo or halftone: the page uses the aurora, press and glitch families only.
FILES = (
    "package.json",
    "tokens.css",
    "components.css",
    "effects.css",
    "fonts.css",
    "themes/light.css",
    "themes/dark.css",
    "themes/high-contrast.css",
    "themes/products.css",
    "assets/fonts/geist-latin.woff2",
    "assets/fonts/geist-mono-latin.woff2",
    "assets/fonts/OFL-Geist.txt",
    "assets/fonts/OFL-GeistMono.txt",
    "effects.js",
    "geom.js",
    "colour.js",
    "live-light.js",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_tarball() -> bytes:
    with urllib.request.urlopen(TARBALL_URL, timeout=60) as resp:  # noqa: S310 (fixed https URL)
        data = resp.read()
    got = "sha512-" + base64.b64encode(hashlib.sha512(data).digest()).decode()
    if got != UI_KIT_INTEGRITY:
        sys.exit(f"tarball integrity mismatch: got {got}")
    return data


def read_from_tarball(data: bytes) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for name in FILES:
            member = tar.extractfile(f"package/{name}")
            if member is None:
                sys.exit(f"missing in tarball: {name}")
            out[name] = member.read()
    return out


def read_from_dir(root: Path) -> dict[str, bytes]:
    return {name: (root / name).read_bytes() for name in FILES}


def write(files: dict[str, bytes], integrity: str | None) -> None:
    pkg = json.loads(files["package.json"])
    if pkg.get("version") != UI_KIT_VERSION:
        sys.exit(f"package is {pkg.get('version')}, pinned {UI_KIT_VERSION}")
    for name, body in files.items():
        target = VENDOR_DIR / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
    manifest = {
        "package": "@sondalab/ui-kit",
        "version": UI_KIT_VERSION,
        "license": pkg.get("license"),
        "integrity": integrity or UI_KIT_INTEGRITY,
        "files": {name: sha256(files[name]) for name in FILES},
    }
    (VENDOR_DIR / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")


def check() -> int:
    manifest = json.loads((VENDOR_DIR / MANIFEST).read_text())
    problems = []
    if manifest["version"] != UI_KIT_VERSION or manifest["integrity"] != UI_KIT_INTEGRITY:
        problems.append("manifest does not match the pinned version")
    if set(manifest["files"]) != set(FILES):
        problems.append("manifest file list differs from FILES")
    for name, digest in manifest["files"].items():
        path = VENDOR_DIR / name
        if not path.is_file() or sha256(path.read_bytes()) != digest:
            problems.append(f"changed or missing: {name}")
    on_disk = {p.relative_to(VENDOR_DIR).as_posix() for p in VENDOR_DIR.rglob("*") if p.is_file()}
    for extra in sorted(on_disk - set(manifest["files"]) - {MANIFEST}):
        problems.append(f"not in manifest: {extra}")
    print("\n".join(problems) or "vendored kit matches the manifest")
    return 1 if problems else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from", dest="source", type=Path, help="unpacked package/ directory")
    ap.add_argument("--check", action="store_true", help="verify disk against the manifest")
    args = ap.parse_args()
    if args.check:
        return check()
    if args.source:
        write(read_from_dir(args.source), None)
    else:
        write(read_from_tarball(fetch_tarball()), UI_KIT_INTEGRITY)
    return check()


if __name__ == "__main__":
    sys.exit(main())
