"""Model manifest and fetcher (SPEC 5, 13; downloads are untrusted data).

``models.yaml`` lists each model file with URL, size, sha256 and licence. ``fetch`` downloads
into the models folder only when asked (``roboemot models fetch``), never at import:

- Redirects are followed by hand and every hop must be https; the size from the manifest caps
  the download; bytes go to a fresh ``.part`` file (created exclusively, never through a
  symlink) while the sha256 is computed; the file is fsynced and renamed into place only if
  size and digest match; any failure deletes the ``.part`` file.
- A verified file gets a ``<file>.verified`` marker (sha256, size, mtime). The backends refuse
  files whose size or mtime no longer match the marker (``check_file``), so a swapped file is
  caught without rehashing every file at every start. ``roboemot models fetch`` on a folder that
  already holds the right files (for example the spike cache) writes the markers after hashing.
- Folders created by the fetcher are private (0o700), markers 0o600.

- Derived files (``quantize``; none in the shipped manifests since the FairFace INT8 pair was
  retired, SPEC 5, 13) are not downloaded: ``fetch`` first downloads and
  verifies the FP32 source, then quantizes it with ONNX Runtime ``quantize_dynamic`` (the same
  call and settings as ``scripts/bench/quantize.py``; needs the ``vision`` extra). Their marker
  records the source sha256, the weight type, the ORT version and the produced file's sha256,
  size and mtime, and stands alone: ``check_file`` accepts a produced file without its source
  present, and ``prune_sources`` (``roboemot models prune --sources``) deletes the FP32 sources
  once every file derived from them is verified.
- Retired files (``retired`` in the manifest: files of replaced or removed models, such as the
  FairFace pair and the yolo11n-face-age age model) are deleted by ``prune_retired``
  (``roboemot models prune``), and nothing else is.
- ORT version policy: the quantized ops are stable across ONNX Runtime versions, so a file made
  by another ORT version still loads. ``status`` reports it as ``stale-ort`` and ``fetch``
  re-quantizes only on ``requantize=True`` (``--requantize``), which needs the FP32 source again;
  a failed re-quantization leaves the old file and marker untouched.

- Archives (voice models, SPEC 8 stage 2): a file with ``archive: zip`` is downloaded and verified
  like any other, then unpacked into a fresh folder next to its target and renamed into place;
  ``path`` names the unpacked folder. Entries that are absolute, climb out with ``..``, are
  symlinks or other non-regular files, or exceed ``unpacked_bytes`` in total reject the whole
  archive (zip-slip, zip bombs). The folder's marker (``<folder>.verified``) holds the archive
  sha256 and is written last; the archive itself is not kept. The voice registry lives in
  ``roboemot.voice.models`` and reuses this code.

Downloaded files are only ever loaded as ONNX weights (or, for archives, read by Vosk).
The folder is ``--cache-dir``,
``vision.models_dir`` or, by default, ``models/`` in the app data folder (``roboemot.paths``,
``~/.local/share/roboemot``), so uninstalling the app removes the models with it.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import stat
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Literal

import httpx
import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from .. import paths

MANIFEST = Path(__file__).with_name("models.yaml")
ENV_DIR = "ROBOEMOT_MODELS_DIR"
CHUNK_BYTES = 1 << 20
TIMEOUT_S = 60.0
MAX_REDIRECTS = 5
# A dropped connection resumes from the bytes already written; waits double from RETRY_WAIT_S.
DOWNLOAD_ATTEMPTS = 5
RETRY_WAIT_S = 2.0
RETRIED = (httpx.NetworkError, httpx.TimeoutException, httpx.RemoteProtocolError)
PRIVATE_DIR_MODE = 0o700
PRIVATE_FILE_MODE = 0o600
MARKER_SUFFIX = ".verified"
PART_SUFFIX = ".part"
EXTRACT_SUFFIX = ".extract"
OLD_SUFFIX = ".old"
# Entry cap for an archive: the Vosk models have about 20 entries; this only bounds inode use.
MAX_ARCHIVE_ENTRIES = 10_000
ZIP_MODE_SHIFT = 16  # a zip entry keeps its Unix st_mode in the high 16 bits of external_attr


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
    An archive (``archive: zip``) has ``url``, ``sha256`` and ``size_bytes`` of the zip itself;
    ``path`` is the folder it unpacks to and ``unpacked_bytes`` caps the total unpacked size.
    """

    path: str
    url: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    quantize: Quantize | None = None
    archive: Literal["zip"] | None = None
    unpacked_bytes: int | None = None

    @model_validator(mode="after")
    def _one_origin(self) -> ModelFile:
        if (self.archive is None) != (self.unpacked_bytes is None):
            raise ValueError(f"{self.path}: archive and unpacked_bytes go together")
        if self.archive is not None and self.quantize is not None:
            raise ValueError(f"{self.path}: an archive cannot also be quantized")
        downloaded = (self.url is not None, self.sha256 is not None)
        if self.quantize is None and not all(downloaded):
            raise ValueError(f"{self.path}: give url and sha256 together, or quantize")
        if self.quantize is not None and (any(downloaded) or self.size_bytes is not None):
            raise ValueError(f"{self.path}: a quantize file takes no url, sha256 or size_bytes")
        return self


class ModelSpec(_Strict):
    """A model: role, weight licence, source page, files and (classifiers) the label order.

    ``labels`` is the output order of a classifier or detector, read from the ``id2label`` of the
    model's ``config.json``, the class list of its training config (``labels_source``, pinned
    by ``labels_sha256``) or the metadata of the pinned file itself, so index meaning is part of
    the pinned manifest and not assumed in code.
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
    """The parsed ``models.yaml``; ``retired`` lists files earlier versions fetched."""

    models: list[ModelSpec]
    retired: list[str] = []


@cache
def _parse(path: Path) -> Manifest:
    manifest = Manifest.model_validate(yaml.safe_load(path.read_text()))
    sources = {f.path for m in manifest.models for f in m.files if f.quantize is None}
    for m in manifest.models:
        for f in m.files:
            if f.quantize is not None and f.quantize.source not in sources:
                raise ValueError(f"{f.path}: quantize source {f.quantize.source} is not a file")
    current = {f.path for m in manifest.models for f in m.files}
    if current & set(manifest.retired):
        raise ValueError(f"retired paths still in use: {sorted(current & set(manifest.retired))}")
    return manifest


@cache
def load_manifest(path: Path = MANIFEST) -> dict[str, ModelSpec]:
    """Parse the manifest into ``{name: spec}``.

    The manifest is ``models.yaml`` next to this module unless ``path`` is given. Unknown keys
    are rejected, so a typo in a digest or URL field fails loudly. The result is cached per
    path (treat it as read-only), so the manifest is parsed once per process.
    """
    return {m.name: m for m in _parse(path).models}


def retired_paths(path: Path = MANIFEST) -> list[str]:
    """Files (relative to the models folder) that earlier manifests fetched and nothing loads."""
    return list(_parse(path).retired)


def default_dir(data_dir: str = "") -> Path:
    """``$ROBOEMOT_MODELS_DIR`` or ``models/`` in the app data folder.

    ``data_dir`` is the ``data_dir`` config value (empty: ``$ROBOEMOT_DATA_DIR`` or
    ``~/.local/share/roboemot``). This is where ``roboemot models fetch`` writes and where the
    backends look unless ``--cache-dir`` or ``vision.models_dir`` says otherwise.
    """
    env = os.environ.get(ENV_DIR)
    return Path(env) if env else paths.models_subdir(data_dir)


def legacy_dir() -> Path:
    """The default models folder before it moved under the app data folder."""
    return Path.home() / ".cache" / "roboemot" / "models"


def models_dir(configured: str = "", data_dir: str = "") -> Path:
    """The folder for ``vision.models_dir`` (empty means the default under ``data_dir``).

    A configured path has ``~`` expanded. The folder is not created here; ``fetch`` creates
    the subfolders it needs.
    """
    return Path(configured).expanduser() if configured else default_dir(data_dir)


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


def prune_retired(root: Path, manifest: Path = MANIFEST) -> list[Path]:
    """Delete the manifest's ``retired`` files and their markers; returns the files deleted.

    Only the listed paths are touched, never other files in the folder. A listed path that
    resolves outside ``root`` (for example a symlink pointing out) is left alone with its
    marker; an in-root symlink is unlinked, not followed. Folders left empty by a deletion are
    removed up to ``root``. On the bench cache (``--cache-dir ~/.cache/roboemot-bench/models``)
    this deletes the FairFace files that ``scripts/bench`` and ``scripts/eval`` still use.
    """
    base = root.resolve()
    deleted = []
    for rel in retired_paths(manifest):
        path = root / rel
        if not path.resolve().is_relative_to(base):
            continue
        for suffix in (MARKER_SUFFIX, PART_SUFFIX):
            path.with_name(path.name + suffix).unlink(missing_ok=True)
        if path.is_symlink() or path.is_file():
            path.unlink()
            deleted.append(path)
        parent = path.parent
        while parent.resolve() != base and parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent
    return deleted


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
    if file.archive is not None:
        if (
            not target.is_dir()
            or marker.get("sha256") != file.sha256
            or not _unpacked_ok(target, marker)
        ):
            raise ModelError(
                f"{file.path}: changed since it was verified; run `roboemot models fetch`"
            )
        return target
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
    matches (cheap); ``verify=True`` rehashes every file (slow for the large models), and for an
    unpacked archive every file recorded in its marker.
    """
    targets = [_target(root, f) for f in spec.files]
    if not any(t.exists() for t in targets):
        return "missing"
    if not all(t.exists() for t in targets):
        return "corrupt"
    if verify and any(_rehash_bad(root, t, f) for t, f in zip(targets, spec.files, strict=True)):
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


def _rehash_bad(root: Path, target: Path, file: ModelFile) -> bool:
    """True when rehashing shows the file (or, for an archive, any unpacked file) changed."""
    if file.archive is None:
        return sha256_of(target) != _expected_sha(root, file)
    try:
        marker = json.loads(_marker(target).read_text())
    except (OSError, ValueError):
        return False
    return not _unpacked_ok(target, marker, rehash=True)


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
    """Create the models folder (parents get the default mode) and the folders below it, 0o700."""
    d = root
    try:
        if not root.exists():
            root.mkdir(mode=PRIVATE_DIR_MODE, parents=True)
        for part in target.parent.relative_to(root.resolve()).parts:
            d = d / part
            if not d.exists():
                d.mkdir(mode=PRIVATE_DIR_MODE)
    except OSError as exc:
        raise ModelError(f"cannot create {d}: {exc.strerror}") from exc


@dataclass
class _Progress:
    done: int = 0
    digest: hashlib._Hash = field(default_factory=hashlib.sha256)


def _stream(client: httpx.Client, url: str, file: ModelFile, part: Path) -> str:
    """Download ``url`` (https hops only) into a fresh ``part``; returns the hex digest.

    A connection dropped mid-body is retried up to ``DOWNLOAD_ATTEMPTS`` times with doubling
    waits, resuming with a ``Range`` request from the bytes already written; a server that
    answers the range with the whole file restarts it. HTTP status, size and redirect errors
    are not retried. The caller owns ``part`` and deletes it on any failure.
    """
    fd = os.open(part, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, PRIVATE_FILE_MODE)
    state = _Progress()
    with os.fdopen(fd, "wb") as out:
        for attempt in range(DOWNLOAD_ATTEMPTS):
            try:
                _get_into(client, url, file, out, state)
                break
            except RETRIED:
                if file.size_bytes is not None and state.done == file.size_bytes:
                    break  # dropped after the last byte: a resume gets a 416; the digest decides
                if attempt == DOWNLOAD_ATTEMPTS - 1:
                    raise
                time.sleep(RETRY_WAIT_S * 2**attempt)
        out.flush()
        os.fsync(out.fileno())
    if file.size_bytes is not None and state.done != file.size_bytes:
        raise ModelError(f"{file.path}: size {state.done} differs from the manifest")
    return state.digest.hexdigest()


def _get_into(
    client: httpx.Client, url: str, file: ModelFile, out: io.BufferedWriter, state: _Progress
) -> None:
    """One GET of ``url`` appended to ``out`` from ``state.done`` (each redirect re-checked)."""
    headers = {"Range": f"bytes={state.done}-"} if state.done else {}
    for _ in range(MAX_REDIRECTS + 1):
        if not url.startswith("https://"):
            raise ModelError(f"{file.path}: only https URLs are fetched")
        with client.stream("GET", url, headers=headers, follow_redirects=False) as response:
            if response.is_redirect:
                url = str(response.url.join(response.headers.get("location", "")))
                continue
            response.raise_for_status()
            if state.done and response.status_code == httpx.codes.PARTIAL_CONTENT:
                if not response.headers.get("content-range", "").startswith(f"bytes {state.done}-"):
                    raise ModelError(f"{file.path}: resumed at the wrong offset")
            elif state.done:
                out.seek(0)
                out.truncate()
                state.done, state.digest = 0, hashlib.sha256()
            for chunk in response.iter_bytes(CHUNK_BYTES):
                if file.size_bytes is not None and state.done + len(chunk) > file.size_bytes:
                    raise ModelError(f"{file.path}: larger than the manifest size")
                out.write(chunk)
                state.digest.update(chunk)
                state.done += len(chunk)
            return
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
    if file.archive is not None:
        return _fetch_archive(file, root, client, target)
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


def _unpacked_ok(folder: Path, marker: dict, *, rehash: bool = False) -> bool:
    """Every file the marker recorded at unpack time is still there with its size (and sha256)."""
    files = marker.get("files")
    if not isinstance(files, dict) or not files:
        return False
    for rel, (size, digest) in files.items():
        path = folder / rel
        try:
            if not path.is_file() or path.is_symlink() or path.stat().st_size != size:
                return False
            if rehash and sha256_of(path) != digest:
                return False
        except OSError:
            return False
    return True


def _entry_parts(file: ModelFile, info: zipfile.ZipInfo) -> list[str]:
    """Path components of a zip entry; raises for anything that could leave the folder."""
    name = info.filename
    parts = name.rstrip("/").split("/")
    if (
        name.startswith("/")
        or "\\" in name
        or ":" in parts[0]
        or any(p in ("", ".", "..") for p in parts)
    ):
        raise ModelError(f"{file.path}: unsafe path in archive")
    kind = stat.S_IFMT(info.external_attr >> ZIP_MODE_SHIFT)
    if kind and kind not in (stat.S_IFREG, stat.S_IFDIR):
        raise ModelError(f"{file.path}: archive holds a link or special file")
    if len(parts) == 1 and not info.is_dir():
        raise ModelError(f"{file.path}: archive holds a file at the top level")
    return parts


def _unpack_zip(archive: Path, dest: Path, file: ModelFile) -> tuple[str, dict[str, list]]:
    """Unpack ``archive`` into the empty folder ``dest``.

    Returns the single top-level folder and ``{relative path: [size, sha256]}`` of every file
    written (recorded in the marker). Every entry is checked before anything is written: no
    absolute, drive, ``.``, empty or ``..`` components, no symlinks or special files, no file at
    the top level, one top-level folder, at most ``MAX_ARCHIVE_ENTRIES`` entries, and the
    declared sizes must sum to at most ``unpacked_bytes``. The bytes actually written are
    counted too, since the declared sizes are only claims. Files are created 0o600, folders 0o700.
    """
    assert file.unpacked_bytes is not None
    try:
        zf = zipfile.ZipFile(archive)
    except zipfile.BadZipFile as exc:
        raise ModelError(f"{file.path}: not a zip archive") from exc
    recorded: dict[str, list] = {}
    with zf:
        entries = zf.infolist()
        if len(entries) > MAX_ARCHIVE_ENTRIES:
            raise ModelError(f"{file.path}: too many entries in archive")
        tops = {_entry_parts(file, info)[0] for info in entries}
        if len(tops) != 1:
            raise ModelError(f"{file.path}: archive must hold exactly one top-level folder")
        if sum(i.file_size for i in entries) > file.unpacked_bytes:
            raise ModelError(f"{file.path}: unpacked size above the manifest limit")
        written = 0
        for info in entries:
            out = dest / info.filename
            if not out.resolve().is_relative_to(dest.resolve()):
                raise ModelError(f"{file.path}: unsafe path in archive")
            if info.is_dir():
                out.mkdir(parents=True, exist_ok=True)
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
            digest = hashlib.sha256()
            size = 0
            with (
                zf.open(info) as src,
                os.fdopen(os.open(out, flags, PRIVATE_FILE_MODE), "wb") as dst,
            ):
                while chunk := src.read(CHUNK_BYTES):
                    written += len(chunk)
                    if written > file.unpacked_bytes:
                        raise ModelError(f"{file.path}: unpacked size above the manifest limit")
                    dst.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            rel = "/".join(info.filename.rstrip("/").split("/")[1:])
            recorded[rel] = [size, digest.hexdigest()]
    for folder in [dest, *dest.rglob("*")]:
        if folder.is_dir():
            folder.chmod(PRIVATE_DIR_MODE)
    return tops.pop(), recorded


def _write_archive_marker(file: ModelFile, target: Path, files: dict[str, list]) -> None:
    fd = os.open(
        _marker(target), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, PRIVATE_FILE_MODE
    )
    with os.fdopen(fd, "w") as out:
        json.dump({"sha256": file.sha256, "archive": file.archive, "files": files}, out)


def _fetch_archive(file: ModelFile, root: Path, client: httpx.Client, target: Path) -> str:
    """Download, verify and unpack a zip; returns ``present`` or ``downloaded``.

    The zip is hashed again right before unpacking. The unpacked folder is swapped in by moving
    an existing folder and its marker aside first; any failure puts them back, so a failed
    update leaves the old model usable. The marker (archive sha256 plus a manifest of the
    unpacked files) is written last.
    """
    assert file.url is not None and file.sha256 is not None
    if target.is_dir() and _marked(root, file):
        return "present"
    scratch = target.with_name(target.name + EXTRACT_SUFFIX)
    part = target.with_name(target.name + ".zip" + PART_SUFFIX)
    old = target.with_name(target.name + OLD_SUFFIX)
    marker, old_marker = _marker(target), _marker(old)
    try:
        _make_dirs(root, target)
        part.unlink(missing_ok=True)
        shutil.rmtree(scratch, ignore_errors=True)
        shutil.rmtree(old, ignore_errors=True)
        old_marker.unlink(missing_ok=True)
        try:
            digest = _stream(client, file.url, file, part)
            if digest != file.sha256 or sha256_of(part) != file.sha256:
                raise ModelError(f"{file.path}: sha256 mismatch, download discarded")
            scratch.mkdir(mode=PRIVATE_DIR_MODE)
            top, recorded = _unpack_zip(part, scratch, file)
            had_old = target.exists() or target.is_symlink()
            if had_old:
                target.rename(old)
                if marker.exists():
                    marker.rename(old_marker)
            try:
                (scratch / top).replace(target)
                _write_archive_marker(file, target, recorded)
            except BaseException:
                shutil.rmtree(target, ignore_errors=True)
                marker.unlink(missing_ok=True)
                if had_old:
                    old.rename(target)
                    if old_marker.exists():
                        old_marker.rename(marker)
                raise
            shutil.rmtree(old, ignore_errors=True)
            old_marker.unlink(missing_ok=True)
        finally:
            part.unlink(missing_ok=True)
            shutil.rmtree(scratch, ignore_errors=True)
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
    manifest: Path = MANIFEST,
) -> FetchReport:
    """Fetch the named models (all files of each) into ``root``.

    Raises ``ModelError`` for an unknown name before downloading anything. Downloads come
    first (each verified source is hashed once per call, even if two files derive from it),
    then derived files are produced. A derived file that is already verified is left alone,
    and its source is not even needed, unless ``requantize`` asks to rebuild it. Pass ``client``
    to control transport (tests use ``httpx.MockTransport``) and ``quantizer`` to replace the
    ONNX Runtime quantization. ``manifest`` selects the registry (the voice one by
    ``roboemot.voice.models``). Never called at import time.
    """
    specs = load_manifest(manifest)
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
