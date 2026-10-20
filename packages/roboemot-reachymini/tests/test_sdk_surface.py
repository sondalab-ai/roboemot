"""The typed SDK surface (``roboemot_reachymini.sdk``) matches the installed SDK (SPEC 4A, 12A).

Runs only where ``reachy_mini`` is installed (the ``sdk`` extra); the members are the ones the
package reads straight from the SDK instead of probing for them.
"""

import inspect

import pytest

pytestmark = pytest.mark.sim  # the CI sim job installs the SDK and runs `pytest -m sim`

pytest.importorskip("reachy_mini")


def test_ws_client_has_the_private_state_the_driver_reads():
    from reachy_mini.io.ws_client import WSClient

    source = inspect.getsource(WSClient.__init__)
    assert "self._is_alive" in source and "self._last_status" in source
    assert "wait" in inspect.signature(WSClient.get_status).parameters


def test_audio_backends_have_clear_player():
    from reachy_mini.media.audio_base import AudioBase

    assert "clear_player" in AudioBase.__abstractmethods__


def test_camera_base_exposes_k():
    from reachy_mini.media.camera_base import CameraBase

    assert isinstance(CameraBase.K, property)
