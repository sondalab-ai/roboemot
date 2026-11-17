"""The media channel: a network audio stream decoded to PCM and played through the robot's
speaker, paused while the robot speaks (SPEC 8, Radio playback; SPEC 11, Voice tools)."""

from .decode import PcmStream, StreamError, open_stream
from .player import MediaPlayer

__all__ = ["MediaPlayer", "PcmStream", "StreamError", "open_stream"]
