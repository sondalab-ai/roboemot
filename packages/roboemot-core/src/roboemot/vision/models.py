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

- Derived files (INT8 FairFace, SPEC 5, 13) are not downloaded: ``fetch`` first downloads and
  verifies the FP32 source, then quantizes it with ONNX Runtime ``quantize_dynamic`` (the same
  call and settings as ``scripts/bench/quantize.py``; needs the ``vision`` extra). Their marker
  records the source sha256, the weight type, the ORT version and the produced file's sha256,
  size and mtime, and stands alone: ``check_file`` accepts a produced file without its source
  present, and ``prune_sources`` (``roboemot models prune --sources``) deletes the FP32 sources
  once every file derived from them is verified (about 690 MB freed; only about 174 MB stay).
- ORT version policy: the quantized ops are stable across ONNX Runtime versions, so a file made
  by another ORT version still loads. ``status`` reports it as ``stale-ort`` and ``fetch``
  re-quantizes only on ``requantize=True`` (``--requantize``), which needs the FP32 source again;
  a failed re-quantization leaves the old file and marker untouched.

Downloaded files are only ever loaded as ONNX weights. The folder is ``--cache-dir``,
``vision.models_dir`` or, by default, ``~/.cache/roboemot/models``.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Literal

import httpx
import yaml
from pydantic import BaseModel, ConfigDict, model_validator

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


class Quantize(_Strict):
    """Dynamic quantization of another manifest file (``source`` is its path)."""

    source: str
    weight_type: Literal["QInt8", "QUInt8"]


class ModelFile(_Strict):
    """One file under the models folder: downloaded (https URL, size, sha256) or derived.

    A derived file has ``quantize`` instead of ``url`` and ``sha256``: its digest depends on the
    ONNX Runtime version, so it is recorded in the marker when the file is produced, not pinned.
    """

    path: str
    url: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    quantize: Quantize | None = None

    @model_validator(mode="after")
    def _one_origin(self) -> ModelFile:
        downloaded = (self.url is not None, self.sha256 is not None)
        if self.quantize is None and not all(downloaded):
            raise ValueError(f"{self.path}: give url and sha256 together, or quantize")
        if self.quantize is not None and (any(downloaded) or self.size_bytes is not None):
            raise ValueError(f"{self.path}: a quantize file takes no url, sha256 or size_bytes")
        return self


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


@cache
def load_manifest(path: Path = MANIFEST) -> dict[str, ModelSpec]:
    """Parse the manifest into ``{name: spec}``.

    The manifest is ``models.yaml`` next to this module unless ``path`` is given. Unknown keys
    are rejected, so a typo in a digest or URL field fails loudly. The result is cached per
    path (treat it as read-only), so the manifest is parsed once per process.
    """
    manifest = Manifest.model_validate(yaml.safe_load(path.read_text()))
    sources = {f.path for m in manifest.models for f in m.files if f.quantize is None}
    for m in manifest.models:
        for f in m.files:
            if f.quantize is not None and f.quantize.source not in sources:
                raise ValueError(f"{f.path}: quantize source {f.quantize.source} is not a file")
    return {m.name: m for m in manifest.models}


def resolve_precision(configured: str, profile: str | None) -> str:
    """``int8`` or ``fp32`` for the age and gender models.

    ``vision.attribute_precision`` is ``auto`` by default: INT8 on the ``cm4`` profile of the
    robot manifest, FP32 on ``npu`` or when the profile is not known. ``int8``/``fp32`` force it.
    """
    if configured == "auto":
        return "int8" if profile == "cm4" else "fp32"
    return configured


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


def _write_marker(target: Path, digest: str, extra: dict[str, str] | None = None) -> None:
    info = target.stat()
    data = {"sha256": digest, "size": info.st_size, "mtime_ns": info.st_mtime_ns, **(extra or {})}
    marker = _marker(target)
    marker.unlink(missing_ok=True)
    fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, PRIVATE_FILE_MODE)
    with os.fdopen(fd, "w") as out:
        json.dump(data, out)


def _source_of(file: ModelFile) -> ModelFile | None:
    """The manifest file a derived file is quantized from (None for downloaded files)."""
    if file.quantize is None:
        return None
    for spec in load_manifest().values():
        for f in spec.files:
            if f.path == file.quantize.source:
                return f
    raise ModelError(f"{file.path}: quantize source {file.quantize.source} not in the manifest")


def check_file(root: Path, file: ModelFile) -> Path:
    """Path of a model file whose ``.verified`` marker is present and consistent.

    The guarantee is marker-based: the file's size and mtime must equal the ones the marker
    recorded when the file was hashed, and the marker's digest must be the manifest's (downloaded
    files) or name the manifest's source digest and weight type (derived files). The content is
    not rehashed at load, so a same-size, same-mtime swap or an edited marker is not detected;
    ``roboemot models list --verify`` (``status(..., verify=True)``) rehashes. A derived file is
    judged by its own marker; its FP32 source does not have to be present.
    """
    target = _target(root, file)
    try:
        marker = json.loads(_marker(target).read_text())
        info = target.stat()
    except (OSError, ValueError) as exc:
        raise ModelError(
            f"{file.path}: not verified in {root}; run `roboemot models fetch` for it"
        ) from exc
    source = _source_of(file)
    if source is None:
        digest_ok = marker.get("sha256") == file.sha256
    else:
        assert file.quantize is not None
        digest_ok = (
            marker.get("source_sha256") == source.sha256
            and marker.get("weight_type") == file.quantize.weight_type
            and isinstance(marker.get("sha256"), str)
        )
    ok = (
        digest_ok
        and marker.get("size") == info.st_size
        and marker.get("mtime_ns") == info.st_mtime_ns
    )
    if not ok:
        raise ModelError(f"{file.path}: changed since it was verified; run `roboemot models fetch`")
    return target


def status(spec: ModelSpec, root: Path, *, verify: bool = False) -> str:
    """``ok``, ``stale-ort``, ``unverified`` (no valid marker), ``corrupt`` or ``missing``.

    ``stale-ort`` is a verified derived file made by another ONNX Runtime version: it loads, and
    ``fetch --requantize`` rebuilds it.

    ``missing`` means none of the files exists. By default a file is trusted when its marker
    matches (cheap); ``verify=True`` rehashes every file (slow for the 343 MB models).
    """
    targets = [_target(root, f) for f in spec.files]
    if not any(t.exists() for t in targets):
        return "missing"
    if not all(t.exists() for t in targets):
        return "corrupt"
    if verify and any(
        sha256_of(t) != _expected_sha(root, f) for t, f in zip(targets, spec.files, strict=True)
    ):
        return "corrupt"
    if not all(_marked(root, f) for f in spec.files):
        return "unverified"
    installed = ort_version()
    stale = any(
        f.quantize is not None
        and installed != "unknown"
        and _recorded(_target(root, f), "ort_version") != installed
        for f in spec.files
    )
    return "stale-ort" if stale else "ok"


def _expected_sha(root: Path, file: ModelFile) -> str | None:
    """Pinned digest, or for a derived file the digest its marker recorded when produced."""
    if file.sha256 is not None:
        return file.sha256
    try:
        return json.loads(_marker(_target(root, file)).read_text()).get("sha256")
    except (OSError, ValueError):
        return None


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
    if file.url is None or file.sha256 is None:
        raise ModelError(f"{file.path}: derived file, produced by fetch from its source")
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


Quantizer = Callable[[Path, Path, str], None]


def quantize_dynamic_file(source: Path, dest: Path, weight_type: str) -> None:
    """Write a dynamically quantized copy of ``source`` (ONNX Runtime ``quantize_dynamic``).

    Same call as ``scripts/bench/quantize.py``: weights to INT8, activations quantized per call,
    no calibration data. ``ModelError`` if onnxruntime or onnx is missing (the ``vision`` extra).
    """
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:
        raise ModelError("INT8 models need the `vision` extra (onnxruntime and onnx)") from exc
    quantize_dynamic(source, dest, weight_type=QuantType[weight_type])


def ort_version() -> str:
    """Installed ONNX Runtime version (``unknown`` without it), recorded in derived markers."""
    try:
        import onnxruntime
    except ImportError:
        return "unknown"
    return str(onnxruntime.__version__)


def _recorded(target: Path, key: str) -> object:
    try:
        return json.loads(_marker(target).read_text()).get(key)
    except (OSError, ValueError):
        return None


def produce_file(file: ModelFile, root: Path, quantizer: Quantizer = quantize_dynamic_file) -> str:
    """Quantize a derived file from its already verified source; returns ``produced``.

    The quantizer writes a fresh ``.part`` file that replaces the target only on success, so a
    failure leaves an existing file and its marker as they were. The marker records the source
    sha256, the settings, the ORT version and the produced file's sha256, size and mtime.
    """
    assert file.quantize is not None
    source = _source_of(file)
    assert source is not None
    src_path, target = _target(root, source), _target(root, file)
    extra = {
        "source_sha256": str(source.sha256),
        "weight_type": file.quantize.weight_type,
        "ort_version": ort_version(),
    }
    try:
        _make_dirs(root, target)
        part = target.with_name(target.name + PART_SUFFIX)
        part.unlink(missing_ok=True)
        try:
            quantizer(src_path, part, file.quantize.weight_type)
            part.chmod(PRIVATE_FILE_MODE)
            part.replace(target)
        finally:
            part.unlink(missing_ok=True)
        _write_marker(target, sha256_of(target), extra)
    except OSError as exc:
        raise ModelError(f"{file.path}: {exc.strerror or type(exc).__name__}") from exc
    return "produced"


def prune_sources(root: Path) -> list[Path]:
    """Delete FP32 source files whose derived files are all verified; returns what was deleted.

    A source is removed (with its marker) only if at least one file derives from it and every
    derived file passes ``check_file``. Sources nobody derives from, or with an unverified
    derived file, are kept. Re-fetching a source later downloads it again.
    """
    derived: dict[str, list[ModelFile]] = {}
    for spec in load_manifest().values():
        for f in spec.files:
            if f.quantize is not None:
                derived.setdefault(f.quantize.source, []).append(f)
    deleted = []
    for source_path, files in derived.items():
        if not all(_marked(root, f) for f in files):
            continue
        source = next(f for s in load_manifest().values() for f in s.files if f.path == source_path)
        target = _target(root, source)
        if target.exists():
            target.unlink()
            _marker(target).unlink(missing_ok=True)
            deleted.append(target)
    return deleted


def fetch(
    names: list[str],
    root: Path,
    *,
    client: httpx.Client | None = None,
    quantizer: Quantizer = quantize_dynamic_file,
    requantize: bool = False,
) -> FetchReport:
    """Fetch the named models (all files of each) into ``root``.

    Raises ``ModelError`` for an unknown name before downloading anything. Downloads come
    first (each verified source is hashed once per call, even if two files derive from it),
    then derived files are produced. A derived file that is already verified is left alone,
    and its source is not even needed, unless ``requantize`` asks to rebuild it. Pass ``client``
    to control transport (tests use ``httpx.MockTransport``) and ``quantizer`` to replace the
    ONNX Runtime quantization. Never called at import time.
    """
    specs = load_manifest()
    unknown = [n for n in names if n not in specs]
    if unknown:
        raise ModelError(f"unknown model(s) {unknown}; known: {sorted(specs)}")
    files = [f for n in names for f in specs[n].files]
    todo = [f for f in files if f.quantize is None]
    pending = [f for f in files if f.quantize is not None and (requantize or not _marked(root, f))]
    for f in pending:
        source = _source_of(f)
        assert source is not None
        if source not in todo:
            todo.append(source)
    report = FetchReport()
    own = client is None
    http = client or httpx.Client(timeout=TIMEOUT_S)
    try:
        for f in todo:
            report.files[f.path] = fetch_file(f, root, http)
    finally:
        if own:
            http.close()
    for f in files:
        if f.quantize is not None:
            report.files[f.path] = produce_file(f, root, quantizer) if f in pending else "present"
    return report
