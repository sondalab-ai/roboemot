"""The face detector runs only the part of its input the picture fills (SPEC 13)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("cv2")

from roboemot.vision import onnx  # noqa: E402
from roboemot.vision.geometry import letterbox_scale  # noqa: E402

CONFIGURED = (320, 240)


class FakeYuNet:
    """Records the input size and the canvas of every run; one fixed face in detector pixels."""

    def __init__(self) -> None:
        self.sizes: list[tuple[int, int]] = []
        self.canvases: list[np.ndarray] = []

    def setInputSize(self, size: tuple[int, int]) -> None:  # noqa: N802 (OpenCV's name)
        self.sizes.append(size)

    def detect(self, canvas: np.ndarray):
        self.canvases.append(canvas)
        row = np.array([[40, 20, 30, 40, 50, 30, 60, 30, 55, 40, 50, 50, 60, 50, 0.9]], np.float32)
        return 1, row


@pytest.fixture
def fake(monkeypatch):
    det = FakeYuNet()
    monkeypatch.setattr(onnx.cv2.FaceDetectorYN, "create", lambda *a: det)
    return det


def frame(width: int, height: int) -> np.ndarray:
    return np.full((height, width, 3), 200, np.uint8)


@pytest.mark.parametrize(
    ("picture", "wanted"),
    [
        ((1280, 720), (320, 192)),  # 16:9 fills 180 rows: 192 is the next whole stride
        ((1536, 864), (320, 192)),
        ((640, 480), (320, 240)),  # 4:3 fills the input: nothing to cut
        ((720, 1280), (160, 240)),  # a tall frame fills 135 columns
        ((320, 100), (320, 128)),
    ],
)
def test_the_input_is_cut_to_the_picture_in_whole_strides(fake, picture, wanted):
    onnx.YuNetDetector("model.onnx", 0.6, CONFIGURED).detect(frame(*picture))
    assert fake.sizes == [wanted]
    assert fake.canvases[0].shape == (wanted[1], wanted[0], 3)


def test_the_cut_never_grows_the_input_or_changes_the_scale(fake):
    for picture in ((1280, 720), (640, 480), (720, 1280), (64, 64)):
        scale = letterbox_scale(*picture, *CONFIGURED)
        (found,) = onnx.YuNetDetector("model.onnx", 0.6, CONFIGURED).detect(frame(*picture))
        w, h = fake.sizes[-1]
        assert w <= CONFIGURED[0] and h <= CONFIGURED[1]
        # boxes and landmarks come back in camera pixels at the configured scale
        assert found.box == pytest.approx((40 / scale, 20 / scale, 70 / scale, 60 / scale))
        assert found.landmarks[0] == pytest.approx((50 / scale, 30 / scale))


def test_the_whole_picture_is_inside_the_cut_input(fake):
    onnx.YuNetDetector("model.onnx", 0.6, CONFIGURED).detect(frame(1280, 720))
    canvas = fake.canvases[0]
    assert (canvas[:180] == 200).all() and (canvas[180:] == 0).all()
