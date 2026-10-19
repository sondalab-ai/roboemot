"""The configuration a Reachy Mini run sees (core defaults plus the manifest's behaviour)."""

from roboemot.config.schema import RoboEmotConfig


def reachy_config() -> RoboEmotConfig:
    return RoboEmotConfig()
