"""The configuration a Reachy Mini run sees (core defaults plus the manifest's behaviour)."""

from pathlib import Path

from roboemot.affect.clips import load_clip_library
from roboemot.config import ConfigStack
from roboemot.config.schema import RoboEmotConfig
from roboemot.manifest import load_manifest

MANIFEST = (
    Path(__file__).resolve().parents[3] / "configs" / "manifests" / "reachy-mini-wireless.yaml"
)


def reachy_config() -> RoboEmotConfig:
    manifest = load_manifest(MANIFEST, allow_unmeasured_dynamics=True)
    stack = ConfigStack()
    stack.set_manifest_behaviour(
        manifest.model.behaviour, manifest.channels, load_clip_library(manifest)
    )
    return stack.effective()
