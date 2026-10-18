"""SPEC 8, 12: a route's request (key and prompt) never follows a redirect to another host.

Real sockets on loopback and the production client defaults (no ``http_client_factory``): the
openai SDK's own client follows redirects, so the router builds its client with them off.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
from roboemot.config.schema import LLMRoute, OpenAISpeechConfig, SafetyConfig
from roboemot.llm import Router
from roboemot.voice.openai_speech import OpenAISpeechBackend
from test_tts import collect, delivery

KEY = "sk-redirect-test-key"
CHAT_OK = (
    b'data: {"id":"x","object":"chat.completion.chunk","created":0,"model":"m","choices":'
    b'[{"index":0,"delta":{"content":"ciao"},"finish_reason":null}]}\n\ndata: [DONE]\n\n'
)


class Recorder(ThreadingHTTPServer):
    """Loopback server that records the request headers it sees."""

    def __init__(self, handler: type[BaseHTTPRequestHandler]) -> None:
        super().__init__(("127.0.0.1", 0), handler)
        self.seen: list[dict[str, str]] = []
        threading.Thread(target=self.serve_forever, daemon=True).start()

    @property
    def port(self) -> int:
        return self.server_address[1]


class Target(BaseHTTPRequestHandler):
    """The hostile destination: answers a valid reply and notes the headers."""

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("content-length", 0)))
        self.server.seen.append({k.lower(): v for k, v in self.headers.items()})  # type: ignore[attr-defined]
        speech = self.path.endswith("/audio/speech")
        body = np.zeros(480, dtype="<i2").tobytes() if speech else CHAT_OK
        self.send_response(200)
        self.send_header("content-type", "audio/pcm" if speech else "text/event-stream")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        pass


@pytest.fixture
def servers() -> Iterator[tuple[Recorder, Recorder]]:
    """(route server that answers 307, destination on another host name)."""
    target = Recorder(Target)

    class Redirect(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get("content-length", 0)))
            self.server.seen.append({k.lower(): v for k, v in self.headers.items()})  # type: ignore[attr-defined]
            self.send_response(307)
            self.send_header("location", f"http://localhost:{target.port}{self.path}")
            self.send_header("content-length", "0")
            self.end_headers()

        def log_message(self, *args) -> None:
            pass

    origin = Recorder(Redirect)
    yield origin, target
    for s in (origin, target):
        s.shutdown()
        s.server_close()


def _router(origin: Recorder) -> Router:
    route = LLMRoute(
        name="cloud",
        base_url=f"http://127.0.0.1:{origin.port}/v1",
        model="m",
        api_key=KEY,
        first_token_timeout_s=5.0,
    )
    return Router([route], SafetyConfig())


async def test_the_llm_router_does_not_follow_a_redirect(servers):
    origin, target = servers
    router = _router(origin)
    try:
        async for _ in router.stream([{"role": "user", "content": "ciao"}]):
            pass
    except Exception:  # the route refuses the redirect; only what the target saw matters
        pass
    assert origin.seen[0]["authorization"] == f"Bearer {KEY}"
    assert target.seen == []  # neither the key nor the prompt body follows a redirect


async def test_the_speech_client_does_not_follow_a_redirect(servers):
    origin, target = servers
    router = _router(origin)
    backend = OpenAISpeechBackend(OpenAISpeechConfig(route="cloud"), lambda: router)
    with pytest.raises(Exception):  # noqa: B017  # a 307 is not audio; refusing it is fine
        await collect(backend, "Ciao!", delivery())
    assert origin.seen[0]["authorization"] == f"Bearer {KEY}"
    assert target.seen == []
