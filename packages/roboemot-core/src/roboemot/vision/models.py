"""Model manifest and fetcher (SPEC 5, 13; downloads are untrusted data).

``models.yaml`` lists each model file with URL, size, sha256 and licence. ``fetch`` downloads
into the models folder only when asked (``roboemot models fetch``), never at import:

- Redirects are followed by hand and every hop must be https; the size from the manifest caps
  the download; bytes go to a fresh ``.part`` file (created exclusively, never through a
  symlink) while the sha256 is computed; the file is fsynced and renamed into place only if
  size and digest match; any failure deletes the ``.part`` file.
- A verified file gets a ``<file>.verified`` marker (sha256, size, mtime). The backends refuse
  files whose size or mtime no longer match the marker (``check_file``), so a swapped file is
  caught without rehashing 343 MB at every start. ``roboemot models fetch`` on a folder that
  already holds the right files (for example the spike cache) writes the markers after hashing.
- Folders created by the fetcher are private (0o700), markers 0o600.

Downloaded files are only ever loaded as ONNX weights. The folder is ``--cache-dir``,
``vision.models_dir`` or, by default, ``~/.cache/roboemot/models``.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import yaml
from pydantic import BaseModel, ConfigDict

MANIFEST = Path(__file__).with_name("models.yaml")
ENV_DIR = "ROBOEMOT_MODELS_DIR"
CHUNK_BYTES = 1 << 20
TIMEOUT_S = 60.0
MAX_REDIRECTS = 5
PRIVATE_DIR_MODE = 0o700
PRIVATE_FILE_MODE = 0o600
MARKER_SUFFIX = ".verified"
PART_SUFFIX = ".part"


class ModelError(RuntimeError):
    """A model is unknown, a download failed or a file did not verify."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelFile(_Strict):
    """One downloadable file: path under the models folder, https URL, size and sha256."""

    path: str
    url: str
    sha256: str
    size_bytes: int | None = None


class ModelSpec(_Strict):
    """A model: role, weight licence, source page, files and (classifiers) the label order.

    ``labels`` is the output order of a classifier, read from the ``id2label`` of the model's
    ``config.json`` (``labels_source``, pinned by ``labels_sha256``), so index meaning is part
    of the pinned manifest and not assumed in code.
    """

    name: str
    role: str
    licence: str
    source: str
    notes: str = ""
    files: list[ModelFile]
    labels: list[str] | None = None
    labels_source: str | None = None
    labels_sha256: str | None = None


class Manifest(_Strict):
    """The parsed ``models.yaml``."""

    models: list[ModelSpec]


def load_manifest(path: Path = MANIFEST) -> dict[str, ModelSpec]:
    """Parse the manifest into ``{name: spec}``.

    The manifest is ``models.yaml`` next to this module unless ``path`` is given. Unknown keys
    are rejected, so a typo in a digest or URL field fails loudly.
    """
    manifest = Manifest.model_validate(yaml.safe_load(path.read_text()))
    return {m.name: m for m in manifest.models}


def default_dir() -> Path:
    """``$ROBOEMOT_MODELS_DIR`` or ``~/.cache/roboemot/models``.

    This is where ``roboemot models fetch`` writes and where the backends look unless
    ``--cache-dir`` or ``vision.models_dir`` says otherwise.
    """
    env = os.environ.get(ENV_DIR)
    return Path(env) if env else Path.home() / ".cache" / "roboemot" / "models"


def models_dir(configured: str = "") -> Path:
    """The folder for ``vision.models_dir`` (empty means the default).

    A configured path has ``~`` expanded. The folder is not created here; ``fetch`` creates
    the subfolders it needs.
    """
    return Path(configured).expanduser() if configured else default_dir()


def sha256_of(path: Path) -> str:
    """Hex sha256 of a file, read in chunks.

    Used both to skip a file already downloaded and to check a fresh download. Large model
    files never need to fit in memory.
    """
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _target(root: Path, file: ModelFile) -> Path:
    target = (root / file.path).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ModelError(f"{file.path}: path leaves the models folder")
    return target


def _marker(target: Path) -> Path:
    return target.with_name(target.name + MARKER_SUFFIX)


def _write_marker(target: Path, digest: str) -> None:
    info = target.stat()
    data = {"sha256": digest, "size": info.st_size, "mtime_ns": info.st_mtime_ns}
    marker = _marker(target)
    marker.unlink(missing_ok=True)
    fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, PRIVATE_FILE_MODE)
    with os.fdopen(fd, "w") as out:
        json.dump(data, out)


def check_file(root: Path, file: ModelFile) -> Path:
    """Path of a verified model file; ``ModelError`` if missing, unmarked or changed.

    Compares size and mtime with the ``.verified`` marker written after a hash check, and the
    marker's digest with the manifest. It does not rehash; use ``status(..., verify=True)``.
    """
    target = _target(root, file)
    try:
        marker = json.loads(_marker(target).read_text())
        info = target.stat()
    except (OSError, ValueError) as exc:
        raise ModelError(
            f"{file.path}: not verified in {root}; run `roboemot models fetch` for it"
        ) from exc
    ok = (
        marker.get("sha256") == file.sha256
        and marker.get("size") == info.st_size
        and marker.get("mtime_ns") == info.st_mtime_ns
    )
    if not ok:
        raise ModelError(f"{file.path}: changed since it was verified; run `roboemot models fetch`")
    return target


def status(spec: ModelSpec, root: Path, *, verify: bool = False) -> str:
    """``ok``, ``unverified`` (right files, no valid marker), ``corrupt`` or ``missing``.

    ``missing`` means none of the files exists. By default a file is trusted when its marker
    matches (cheap); ``verify=True`` rehashes every file (slow for the 343 MB models).
    """
    targets = [_target(root, f) for f in spec.files]
    if not any(t.exists() for t in targets):
        return "missing"
    if not all(t.exists() for t in targets):
        return "corrupt"
    if verify and any(sha256_of(t) != f.sha256 for t, f in zip(targets, spec.files, strict=True)):
        return "corrupt"
    return "ok" if all(_marked(root, f) for f in spec.files) else "unverified"


def _marked(root: Path, file: ModelFile) -> bool:
    try:
        check_file(root, file)
    except ModelError:
        return False
    return True


@dataclass
class FetchReport:
    """Per file: ``downloaded`` or ``present`` (digest already right)."""

    files: dict[str, str] = field(default_factory=dict)


def _make_dirs(root: Path, target: Path) -> None:
    missing = []
    d = target.parent
    while d != root.parent and not d.exists():
        missing.append(d)
        d = d.parent
    try:
        for d in reversed(missing):
            d.mkdir(mode=PRIVATE_DIR_MODE)
    except OSError as exc:
        raise ModelError(f"cannot create {d}: {exc.strerror}") from exc


def _stream(client: httpx.Client, url: str, file: ModelFile, part: Path) -> str:
    """Download ``url`` (https hops only) into ``part``; returns the hex digest."""
    for _ in range(MAX_REDIRECTS + 1):
        if not url.startswith("https://"):
            raise ModelError(f"{file.path}: only https URLs are fetched")
        with client.stream("GET", url, follow_redirects=False) as response:
            if response.is_redirect:
                url = str(response.url.join(response.headers.get("location", "")))
                continue
            response.raise_for_status()
            digest = hashlib.sha256()
            done = 0
            fd = os.open(
                part, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, PRIVATE_FILE_MODE
            )
            with os.fdopen(fd, "wb") as out:
                for chunk in response.iter_bytes(CHUNK_BYTES):
                    done += len(chunk)
                    if file.size_bytes is not None and done > file.size_bytes:
                        raise ModelError(f"{file.path}: larger than the manifest size")
                    out.write(chunk)
                    digest.update(chunk)
                out.flush()
                os.fsync(out.fileno())
            if file.size_bytes is not None and done != file.size_bytes:
                raise ModelError(f"{file.path}: size {done} differs from the manifest")
            return digest.hexdigest()
    raise ModelError(f"{file.path}: too many redirects")


def fetch_file(file: ModelFile, root: Path, client: httpx.Client) -> str:
    """Download one file with verification; returns ``present`` or ``downloaded``.

    A file already there with the right digest is only (re)marked. Otherwise bytes stream into
    a fresh ``<file>.part``; only a matching size and sha256 is renamed into place and marked.
    A mismatch, HTTP, redirect or OS error raises ``ModelError``, the partial file is deleted
    and an existing file is left untouched. Only https URLs are fetched.
    """
    if not file.url.startswith("https://"):
        raise ModelError(f"{file.path}: only https URLs are fetched")
    target = _target(root, file)
    try:
        if target.exists() and sha256_of(target) == file.sha256:
            if not _marked(root, file):
                _write_marker(target, file.sha256)
            return "present"
        _make_dirs(root, target)
        part = target.with_name(target.name + PART_SUFFIX)
        part.unlink(missing_ok=True)
        try:
            digest = _stream(client, file.url, file, part)
            if digest != file.sha256:
                raise ModelError(f"{file.path}: sha256 mismatch, download discarded")
            part.replace(target)
        finally:
            part.unlink(missing_ok=True)
        _write_marker(target, file.sha256)
    except httpx.HTTPError as exc:
        raise ModelError(f"{file.path}: download failed ({type(exc).__name__})") from exc
    except OSError as exc:
        raise ModelError(f"{file.path}: {exc.strerror or type(exc).__name__}") from exc
    return "downloaded"


def fetch(
    names: list[str],
    root: Path,
    *,
    client: httpx.Client | None = None,
) -> FetchReport:
    """Fetch the named models (all files of each) into ``root``.

    Raises ``ModelError`` for an unknown name before downloading anything. Pass ``client`` to
    control transport (tests use ``httpx.MockTransport``). Never called at import time.
    """
    specs = load_manifest()
    unknown = [n for n in names if n not in specs]
    if unknown:
        raise ModelError(f"unknown model(s) {unknown}; known: {sorted(specs)}")
    report = FetchReport()
    own = client is None
    http = client or httpx.Client(timeout=TIMEOUT_S)
    try:
        for name in names:
            for f in specs[name].files:
                report.files[f.path] = fetch_file(f, root, http)
    finally:
        if own:
            http.close()
    return report
