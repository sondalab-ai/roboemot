"""A network audio stream decoded to PCM (SPEC 11, Voice tools, Media player; SPEC 12).

``open_stream(url, rate_hz, timeout_s)`` gives a :class:`PcmStream`: ``read(frames)`` returns
that many mono float32 samples at ``rate_hz``, or None once the stream ended, stalled for
``timeout_s`` or failed; ``close`` ends it. It raises :class:`StreamError` when no audio came
within ``timeout_s``.

The decoder is a ``gst-launch-1.0`` subprocess that writes 16-bit PCM to its stdout. GStreamer is
already on the robot for the Reachy Mini SDK (checked read-only on 2026-11-17: gst-launch-1.0
1.26.2; no ffmpeg; no PyAV in the system Python or the SDK's venvs), so the radio adds no
dependency. ``decodebin`` picks the MP3, AAC or Ogg decoder. The subprocess gets an argv list,
never a shell. gst-launch joins its argv into a pipeline description, so the URL goes in as a
double-quoted ``location`` value; the station URL validator (``config.schema.Station``) has
already refused anything that could end that value (a quote, a space, a backslash) or link
elements (``!``). Like core's HTTP clients the source ignores proxy settings from the
environment (they are removed from the child's environment) and follows no redirect, and it asks
for no ICY metadata (``iradio-mode=false``), so no stream title ever reaches RoboEmot. stderr is
discarded (GStreamer's errors quote the URL), and :class:`StreamError` never holds the URL or a
decoder message (SPEC 12). ``close`` kills the subprocess; so does a stall.
"""

from __future__ import annotations

import contextlib
import os
import select
import shutil
import subprocess
import time
from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np

GST_LAUNCH = "gst-launch-1.0"
SAMPLE_BYTES = 2  # S16LE
PCM16_FULL_SCALE = 32768.0
KILL_WAIT_S = 1.0  # how long ``close`` waits for the killed decoder to exit
READ_BYTES = 65536  # most bytes one read from the decoder's pipe takes
# Proxy settings the child must not inherit (SPEC 12: no proxy from the environment).
PROXY_VARS = frozenset(
    {
        "http_proxy", "https_proxy", "all_proxy", "no_proxy",
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    }
)  # fmt: skip
UNREACHABLE = "unreachable"
NO_DECODER = "no_decoder"


class StreamError(Exception):
    """The stream could not start: ``unreachable`` (no audio in time, a network error, an
    error status, a redirect) or ``no_decoder``. Never holds the URL."""


class PcmStream(Protocol):
    """Mono float32 PCM from one stream (see the module docstring)."""

    def read(self, frames: int) -> np.ndarray | None: ...
    def close(self) -> None: ...


def available() -> bool:
    """Whether the decoder (``gst-launch-1.0``) is on this machine's PATH."""
    return shutil.which(GST_LAUNCH) is not None


def http_source(url: str) -> list[str]:
    """The pipeline's source for an http(s) ``url`` already checked by the station validator:
    no redirect followed, no ICY metadata asked for."""
    return [
        "souphttpsrc",
        f'location="{url}"',
        "automatic-redirect=false",
        "iradio-mode=false",
    ]


def pipeline(source: Sequence[str], rate_hz: int) -> list[str]:
    """The gst-launch argv: ``source`` decoded by ``decodebin`` to mono S16LE at ``rate_hz``
    on stdout. ``-q`` keeps gst-launch's own messages off stdout (they would corrupt the PCM)."""
    caps = f"audio/x-raw,format=S16LE,channels=1,rate={int(rate_hz)}"
    return [
        GST_LAUNCH,
        "-q",
        *source,
        "!",
        "decodebin",
        "!",
        "audioconvert",
        "!",
        "audioresample",
        "!",
        caps,
        "!",
        "fdsink",
        "fd=1",
    ]


def child_env() -> dict[str, str]:
    """This process's environment without the proxy settings (``PROXY_VARS``)."""
    return {k: v for k, v in os.environ.items() if k not in PROXY_VARS}


class GstStream:
    """A running decoder subprocess read as :class:`PcmStream`. ``timeout_s`` bounds every
    read: no audio for that long ends the stream (None) and kills the decoder."""

    def __init__(
        self,
        proc: subprocess.Popen[bytes],
        timeout_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._proc = proc
        self._timeout_s = timeout_s
        self._clock = clock
        self._buf = bytearray()
        self._ended = False

    def wait_audio(self) -> bool:
        """Wait at most ``timeout_s`` for the first bytes of PCM; whether they came."""
        return self._fill(SAMPLE_BYTES)

    def read(self, frames: int) -> np.ndarray | None:
        """``frames`` samples in [-1, 1), or None when the stream ended, stalled for
        ``timeout_s`` or was closed (the decoder is then killed)."""
        want = max(1, int(frames)) * SAMPLE_BYTES
        if not self._fill(want):
            self.close()
            return None
        data = bytes(self._buf[:want])
        del self._buf[:want]
        return np.frombuffer(data, "<i2").astype(np.float32) / np.float32(PCM16_FULL_SCALE)

    def _fill(self, want: int) -> bool:
        """Read from the pipe until ``want`` bytes are buffered; False at the end of the
        stream, a pipe error, a close, or ``timeout_s`` without a byte."""
        out = self._proc.stdout
        if out is None:
            return False
        fd = out.fileno()
        deadline = self._clock() + self._timeout_s
        while len(self._buf) < want:
            if self._ended:
                return False
            left = deadline - self._clock()
            if left <= 0:
                return False
            try:
                ready, _, _ = select.select([fd], [], [], left)
                if not ready:
                    return False
                chunk = os.read(fd, READ_BYTES)
            except (OSError, ValueError):  # closed meanwhile
                return False
            if not chunk:
                self._ended = True
                return False
            self._buf += chunk
            deadline = self._clock() + self._timeout_s  # a stall counts from the last byte
        return True

    def close(self) -> None:
        """Kill the decoder and release its pipe; safe to call twice and from any thread."""
        self._ended = True
        proc = self._proc
        with contextlib.suppress(OSError):
            proc.kill()
        with contextlib.suppress(subprocess.TimeoutExpired, OSError):
            proc.wait(KILL_WAIT_S)
        if proc.stdout is not None:
            with contextlib.suppress(OSError):
                proc.stdout.close()


def open_pipeline(source: Sequence[str], rate_hz: int, timeout_s: float) -> GstStream:
    """Start the decoder on ``source`` and wait at most ``timeout_s`` for its first audio.
    :class:`StreamError` when there is no decoder or no audio came (the decoder is killed)."""
    if not available():
        raise StreamError(NO_DECODER)
    try:
        proc = subprocess.Popen(  # an argv list, never a shell
            pipeline(source, rate_hz),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,  # GStreamer's errors quote the URL (SPEC 12)
            env=child_env(),
        )
    except OSError:
        raise StreamError(NO_DECODER) from None
    stream = GstStream(proc, timeout_s)
    if not stream.wait_audio():
        stream.close()
        raise StreamError(UNREACHABLE)
    return stream


def open_stream(url: str, rate_hz: int, timeout_s: float) -> PcmStream:
    """The radio stream at ``url`` (a station URL the config validated) as mono PCM at
    ``rate_hz``; :class:`StreamError` when it does not give audio within ``timeout_s``."""
    return open_pipeline(http_source(url), rate_hz, timeout_s)
