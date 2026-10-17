"""A worker that breaks the protocol on purpose (mode in ROGUE_MODE), then idles."""

import json
import os
import struct
import sys
import time

mode = os.environ["ROGUE_MODE"]
out = sys.stdout.buffer
HEADER = struct.Struct(">cII")


def frame(kind: bytes, rid: int, payload: bytes, declared: int | None = None) -> None:
    out.write(HEADER.pack(kind, rid, len(payload) if declared is None else declared) + payload)
    out.flush()


def hello(rate) -> None:
    frame(b"H", 0, json.dumps({"rate_hz": rate, "voice": "v"}).encode())


if mode == "rate_zero":
    hello(0)
elif mode == "rate_text":
    hello("22050")
elif mode == "rate_huge":
    hello(10**9)
else:
    hello(22050)
    sys.stdin.buffer.readline()  # the first request
    if mode == "unknown_kind":
        frame(b"Z", 1, b"")
    elif mode == "oversize":
        frame(b"A", 1, b"", declared=50_000_000)
    elif mode == "odd":
        frame(b"A", 1, b"\x01\x02\x03")
    elif mode == "big_error":
        frame(b"X", 1, b"", declared=10_000_000)
time.sleep(60)
