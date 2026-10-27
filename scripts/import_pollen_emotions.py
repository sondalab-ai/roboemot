#!/usr/bin/env python3
"""Convert Pollen's Reachy Mini emotions library into RoboEmot expression clips (SPEC 4A, 7).

The 85 recorded moves of ``pollen-robotics/reachy-mini-emotions-library`` (Apache-2.0) are
converted once, at a pinned revision, into ``data/emotions/clips.yaml`` of ``roboemot_reachymini``
together with ``LICENSE`` and ``VENDORED.json`` (revision, licence, attribution, sha256 per source
file and of the outputs). Nothing is downloaded at runtime and the SDK's ``play_move`` is never
used. Downloads are untrusted data: parsed as JSON only, size-capped, hash-checked.

    python scripts/import_pollen_emotions.py              # download the pinned files, convert
    python scripts/import_pollen_emotions.py --from DIR   # use a local copy of the dataset
    python scripts/import_pollen_emotions.py --check      # offline: outputs vs the pins
    python scripts/import_pollen_emotions.py --print-pins # list the upstream tree, print SHA256

Per move: head matrix to ``head.roll/pitch/yaw`` degrees (the driver's Euler convention,
translation dropped), antennas to ``antenna_right`` / ``antenna_left`` (SDK order is
``[right, left]``: ``set_target`` records its arguments verbatim), body yaw to ``body_yaw``. Each
channel is an offset about the manifest home (not about the first frame), eased in from 0 and out
to 0 over ``EASE_S`` or longer (the ramp's peak speed stays under ``EASE_MAX_SPEED_DEG_S``),
dropped when it never leaves ``RDP_TOLERANCE_DEG``, simplified with Ramer-Douglas-Peucker and
rounded. These numbers are constants of this script, not configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import urllib.request
from pathlib import Path

import yaml
from roboemot.config.schema import EXPRESSION_MAX_CYCLE_S
from roboemot_reachymini.driver import ANTENNAS, head_values

REPO = "pollen-robotics/reachy-mini-emotions-library"
REVISION = "873ae49f0b89114b7e535eff0c1f7560d21d9357"
TREE_URL = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}"
FILE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{{name}}"
LICENSE_URL = "https://www.apache.org/licenses/LICENSE-2.0.txt"
LICENSE_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
ATTRIBUTION = (
    "Reachy Mini Emotions Library, Pollen Robotics (https://huggingface.co/datasets/"
    f"{REPO}), Apache-2.0. Converted by RoboEmot (modified). Derivatives keep the "
    "'reachy_mini_community_moves' tag."
)

EASE_S = 0.4  # shortest ramp from 0 to the first frame and from the last frame back to 0
EASE_MAX_SPEED_DEG_S = 90.0  # a longer ramp keeps the min-jerk peak speed under this
MINJERK_PEAK = 1.875  # peak speed of a min-jerk ramp over its mean speed
RDP_TOLERANCE_DEG = 0.5  # simplification tolerance and the "never leaves home" threshold
ROUND_DEG = 0.1  # value resolution of the written keyframes
TIME_RESOLUTION_S = 0.001  # keyframe times are rounded to the millisecond
NAME_RE = r"[\w.-]+\.json"  # an upstream file name: no path separator, no surprises
MAX_FILE_BYTES = 4_000_000  # an upstream move is under 0.2 MB; anything bigger is not one

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_YAML = ROOT / "configs/manifests/reachy-mini-wireless.yaml"
OUT_DIR = ROOT / "packages/roboemot-reachymini/src/roboemot_reachymini/data/emotions"
CLIPS = "clips.yaml"
LICENSE = "LICENSE"
VENDORED = "VENDORED.json"
HEAD_CHANNELS = ("head.roll", "head.pitch", "head.yaw")

# sha256 of every upstream move at REVISION (generate with --print-pins).
SHA256: dict[str, str] = {
    # PINS-BEGIN
    "amazed1.json": "3a80694a7de568c53e72eecb2fed8901d533f9beb0699e336dfa0636944a30e4",
    "anxiety1.json": "e0f783b4cadab1e78c44f53ac3cb05adf32ce6ead9b4d7c52c64fe1a1ffe8140",
    "attentive1.json": "eac086094c388fbe9d6d704ba3705578526de3c0b6343b4774fca965d91a58e4",
    "attentive2.json": "e762ddce57641418c77b99e20dc822027313bc6f637ff5f6ae5208c2f87bb346",
    "boredom1.json": "c607ba78c6858cc5129786b704defab419b940aabb3b86227c5c2e481f4ada01",
    "boredom2.json": "a6153f548c90ba1ce75ed1e2b163dea4d6207c5383fa0b2ea82ec0d1279ecc62",
    "calming1.json": "4d6b4449bdee31507e7657b5daa19e7bdfb2d7625baba3e6267add43a19b3fe2",
    "cheerful1.json": "158a235b49aceaefa4fc245ee34478d6321fcefe16a90846b5a09472ed535186",
    "come1.json": "b890f8eac865c8d78d97afb344717f31d9f1917e243d4e298c088fa2fb25bc71",
    "confused1.json": "0fd7f08416940641befc1f25a7670f8de28f94c13b19fb2fab81f172868474a8",
    "contempt1.json": "9880dc269f436979eb404f786891f1da1db93e0a1e127c7dc940a4ed42738ba0",
    "curious1.json": "16be0c194e33933d13fc38c105be8a47cb6b7de7e220b4567dc3e38bd564f749",
    "dance1.json": "b22453a3c11f96b1764c985409f50176111d56fa002f64d828b3244837ba9579",
    "dance2.json": "6a3f55057e4f39b3ab5fbcad515815b822ca594ccca0fc3692f83c490c06c2c8",
    "dance3.json": "9166e63daa3b2a60b30bf317ba189bbc91c285838bcee27a05359e4f75a1d463",
    "disgusted1.json": "70912b29fa68bbb7bef50edc4a5809ff4b5388aa7653f9ef6581bc4b8c93598f",
    "displeased1.json": "13afc91002a2026889a8233e0b200d36a73c02ba9955afe42badee82f911026a",
    "displeased2.json": "0ae12ab79b91b93c9a8b8b92b31260865feea42c50cbf2e9cb41c002672bafd3",
    "downcast1.json": "0079878a3dbd7be21192764ae9ac7aee5c4a3e472f8cf30995e9227d123bf808",
    "dying1.json": "dd92a79f5771acf01892c6f73a55569657a5a704b9b6465bb9be2f7ae7671a5b",
    "electric1.json": "6184e191c54ac665d58460dfceb5b96847084538259dadd5eb166c074996d2b2",
    "enthusiastic1.json": "49e4c1775cc84b2bd6509863d1efa1ec303f4954f1d807fddd696fbfe6ad3a48",
    "enthusiastic2.json": "3873dd584e93a7686c769255fd2c3aaa23ecccc468662e40d344470147887a72",
    "exhausted1.json": "347f64efe0933a1b8f433ab9240c0e86e5da9c489c9198698296a9f58e47d00b",
    "fear1.json": "3dc118c2f086113397c8004af573042b25590091d24df706b6c2fd12afd29a31",
    "frustrated1.json": "18d67249cc02ba7c1421930f76b50073349bc044fa90f6e9f171df18b8abd1e9",
    "furious1.json": "1121a3a8ee2e458d1171096e49ab0074d1da0f641ad3368149f6053949c94d1b",
    "go_away1.json": "bf57769cda1275418283befe8b7640f167025b7472660678a82fef6582c4f0c6",
    "grateful1.json": "17ca099169b96ab01342ba88b8ca9c9591160e96807773f8d4b557a9d5d78a7f",
    "helpful1.json": "fa895c41dbe8ac9b5b6e9d987bfd8a4235f834dde68d6bcfa66b2b54ae2a206c",
    "helpful2.json": "c246f2ed48e42eb354b76e0a1006c8d065002e4f4d7aa2deb4c2111f447fdf86",
    "impatient1.json": "001b3777fc9cc5b3d1b6105f77df56ea5ece2e37739d343c0c7488185ff62107",
    "impatient2.json": "fdadcf9322df649923bfbf139f2875ff510372bade8b56f64347496f9e60e2df",
    "incomprehensible2.json": "71acba9255088dd7d4e814750a899952b84e140891c06484d7dde5c9451d4e4c",
    "indifferent1.json": "e6bc76d5419e2c358b92b62c0b8e47d21e4ee71d2af0fc54a58f06c1a49970f1",
    "inquiring1.json": "a3f8b057ccebbb48c22dc6017114d6a4a77c6117989d9c93e89b7f4b8320fe0e",
    "inquiring2.json": "835a8e49b0335788ab730d052eb2fc5d08d624c22947e9108acdaa93eb67ed2e",
    "inquiring3.json": "dcaaf83b8ee6ec19cde32df8fea9117103fc9ca55ac52c96168d0f11abaafc86",
    "irritated1.json": "2ebe13a1ed445c0480f4ffdac9b41c86c2bc3486cf0ff259bf0480fb43c24aee",
    "irritated2.json": "bdeba4f4e41188efa838cf389f5085accfc91badd25046a6ecd1dd8bc0d1e493",
    "laughing1.json": "9ece0e9cb9ebb326b08ec0cd1b732f33c921de8b9a248b5ee8753a855370b266",
    "laughing2.json": "3ddd545e7cd903057e3dff872867e34a5d36b1d8f4ada4c653f025eeab5bea52",
    "lonely1.json": "5d76a941576f604a4cc2879e10289c6aafa301dc158875a88139ad9d2e8d3011",
    "lost1.json": "b25b67c6b390175a0c82bece01108751b57ef0fc5c8dd7838fc5b2ed761d21f9",
    "loving1.json": "751a9293c151cce7a46ce07063f7e0df6a8e355dc503a47966f465f7e6428df6",
    "mini-deep-sleep.json": "8885d77819d13da761b84e6ebb524ab3929739896d29aa93b0e67c103064d3ba",
    "no1.json": "15f69b0b359a6fe661d18c15c23453138c566b5293170e473e0188f279b50a88",
    "no_excited1.json": "be6aa88bd63638c7a23ee2bbb9748316334d58dacdbc165638308c423a5f5b85",
    "no_sad1.json": "48453d41defe1c1f0a34c87cd6155068e531e73d69d5b1d79facde81cf7e1baf",
    "oops1.json": "661a5522740ae456add8c05e156bf186a2df6188fd8ee30a5027e46d749edaa6",
    "oops2.json": "cd405a45f9a8284cbf982c8cca6653be00e0becea2d403a65251710af9a6b3cf",
    "proud1.json": "6a3a692adad1b91126da9c5d64876552a4248543c184dca6f961c7a704ed3bc6",
    "proud2.json": "dfe1801d3d4fd876460b11a6ce518d846e360124d15f0b02b671431c75105d7e",
    "proud3.json": "2b4c6c7f99133385383715163f86b49a59adf23cb55b885fd7d53537d7697b82",
    "rage1.json": "87537fbce49b34f34dca5f07671343d515a457acc465632200889301ba32f297",
    "relief1.json": "5c409f15ac4a11e2baf760f4146146b84a2c3ab8a5efc97b2bc5014de20b138f",
    "relief2.json": "35b30d7ab6512fa6f41eda460f3435318cec3799cedad3a8c8d986ae4109f12b",
    "reprimand1.json": "a62bfa67ae35c9816ee7e0538db8c76ce3dcc511ef3d2dea2d43d76e7fa9615a",
    "reprimand2.json": "926c89be7955b756f968d19c48e12fb194bfdb0ed852ab64b77f1c92d83d706a",
    "reprimand3.json": "c4f6d1d12162685bd4e586c846e55230bc4477b17e77bfe95c11aa99a39117d8",
    "resigned1.json": "c67e49957e99bf250642c5a198b8b12963c31bb6437bbf6ff01056e1dfc41b2b",
    "sad1.json": "c34903859fad0efed4ad5564ac2f27ce23da4d87cc43bcfa537dcfe260b604d8",
    "sad2.json": "2ef2420bd1e48f7ff0d174374f58845158460f1a1d9a6251539fd2d3817a733b",
    "scared1.json": "b209ec78040336c5b46a2bb1d14f6d62ce672e6642d7a85fcf9c263f8fcb1cf4",
    "serenity1.json": "759246affdd6c208be7606cb8fc7035906a6806845ba31ea81d3ced08eec8619",
    "shy1.json": "c00f2b66668ab9cb8ec5edecd4c2e6344a42db778677054589533ea4fd3a5b89",
    "sleep1.json": "ba61eb23e7b902da0cb4e2f66ee1df5e7b4bcf2aea42ca5e2a37caeff5746b8a",
    "success1.json": "c319b0c13c867f13b5200621662034c482e3ad079193f4ba5478a793f4c1cf11",
    "success2.json": "11bc930f8eef02db53eedc319372ecaf6e54dcf1a3806ce06526434ce4a68f1b",
    "surprised1.json": "0a85b55d4b2ef686efb42b608556cdba0f7691cce4a66e54f2a79b6fe425e0db",
    "surprised2.json": "fa7ba5318d9797022e94902fa4d8df6d58c278caf589f7fe558ead1c736b5b0c",
    "thoughtful1.json": "327d5253553b9b17afb93f414de95190eb70940abacf3fb35d5cc4edd0c1b115",
    "thoughtful2.json": "fb07c774e2f52d098833e23ee43dece8adf7c85951bcdce5481d8e38fc2f0b0a",
    "tired1.json": "a3cef01a9bf7dbb7d368d123ca1535768dc0eb597606349f35dca744e9f4ce26",
    "toc-toc-toc.json": "2c4d0fc049a9db0e35436615a6b2259c315f16cf96f75cf64eb5761b3a6a3190",
    "uncertain1.json": "897f80471bd2eb3a073dd05dd82bbd428cca2d8051181cf5526bdbbe021dd0e9",
    "uncomfortable1.json": "d1d22d3f2fbf37884343ecd28382f644871b81a989150ea5e4f11bb7670cfdfe",
    "understanding1.json": "75a1341acec63c421bda3b1302aeac8117eecf281d30b0f34a458b3ec973b37a",
    "understanding2.json": "78294e3e9b71a590d488e9578a0c1b123ad4c6236a3a4af8ef4a74707a2cf4e6",
    "waiting.json": "5fc7e14ce0f41e6f61b004a054511651379d35a3bea3f10c48c2b3b368f5b01b",
    "wake-mini-up.json": "e3bfc817fe5fba9abdcd682958f9327e7ed9110f8fc0612871813d6d49d9e1dc",
    "welcoming1.json": "abeb5233b28118c66111875345d441e76ae54346a4e79a25914d07dcc1aedced",
    "welcoming2.json": "625585782534956eaf109fff0b0c4e87f751226ab43dbdd64260d9d9a15417d7",
    "yes1.json": "b355cabc9c7ef844055b2b3bb96ca914118fab1c842429e724f512852aa47568",
    "yes_sad1.json": "f1c80cbb282987ee7707d2850247da9c265e28f53c16b56df69ecafa9869941b",
    # PINS-END
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ------------------------------------------------------------------ conversion


def manifest_homes(path: Path = MANIFEST_YAML) -> dict[str, float]:
    """Home offset in degrees per channel from the robot manifest (head and body yaw: 0)."""
    pose = yaml.safe_load(path.read_text())["pose_groups"]
    homes = {name: 0.0 for name in (*HEAD_CHANNELS, "body_yaw")}
    names, home = pose["antennas"]["names"], pose["antennas"]["home_deg"]
    if tuple(names) != ANTENNAS or len(home) != len(names):
        raise SystemExit(f"manifest antennas {names} do not match the driver order {ANTENNAS}")
    homes.update(zip(names, (float(h) for h in home), strict=True))
    return homes


def parse_move(raw: bytes, name: str) -> tuple[list[float], dict[str, list[float]], str]:
    """Time (s, from 0) and absolute degrees per channel of one upstream move; data, not code."""
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError(f"{name}: {len(raw)} bytes is too large for a move")
    doc = json.loads(raw)
    times, frames = doc["time"], doc["set_target_data"]
    if len(times) != len(frames) or len(times) < 2:
        raise ValueError(f"{name}: time and set_target_data differ in length or are too short")
    times = [float(t) for t in times]
    if not all(math.isfinite(t) for t in times):
        raise ValueError(f"{name}: non-finite time")
    # a few recordings repeat a timestamp (two samples in one tick): keep the first of each
    keep = [0]
    for i in range(1, len(times)):
        if times[i] > times[keep[-1]]:
            keep.append(i)
    times, frames = [times[i] for i in keep], [frames[i] for i in keep]
    if len(times) < 2:
        raise ValueError(f"{name}: fewer than two distinct timestamps")
    series: dict[str, list[float]] = {c: [] for c in (*HEAD_CHANNELS, *ANTENNAS, "body_yaw")}
    for frame in frames:
        head = head_values(frame["head"])
        for c in HEAD_CHANNELS:
            series[c].append(math.degrees(head[c]))
        for c, value in zip(ANTENNAS, frame["antennas"], strict=True):
            series[c].append(math.degrees(float(value)))
        series["body_yaw"].append(math.degrees(float(frame.get("body_yaw", 0.0))))
    if not all(math.isfinite(v) for s in series.values() for v in s):
        raise ValueError(f"{name}: non-finite value")
    return [t - times[0] for t in times], series, str(doc.get("description", ""))


def minjerk(s: float) -> float:
    """Min-jerk ease from 0 to 1 for s in [0, 1]."""
    return s**3 * (10 - 15 * s + 6 * s * s)


def ramp_times(first: float, last: float) -> tuple[float, float]:
    """Ramp-in and ramp-out durations (s) for edge offsets ``first`` and ``last`` (deg).

    ``EASE_S`` at least; longer when the jump from or to 0 would otherwise peak above
    ``EASE_MAX_SPEED_DEG_S`` (a min-jerk ramp peaks at ``MINJERK_PEAK`` x its mean speed).
    """

    def one(jump: float) -> float:
        return max(EASE_S, MINJERK_PEAK * abs(jump) / EASE_MAX_SPEED_DEG_S)

    return one(first), one(last)


def eased_channel(
    times: list[float], offsets: list[float], ramp_in: float, ramp_out: float
) -> tuple[list[float], list[float]]:
    """The offsets with an ease from 0 in front and back to 0 behind; both ends are exactly 0.

    The ramps are sampled at the clip's median step so the simplification sees their curvature.
    """
    diffs = sorted(b - a for a, b in zip(times, times[1:], strict=False))
    step = diffs[min(len(times) // 2, len(diffs) - 1)]
    n_in, n_out = max(2, round(ramp_in / step)), max(2, round(ramp_out / step))
    t_in = [ramp_in * i / n_in for i in range(n_in)]
    v_in = [offsets[0] * minjerk(i / n_in) for i in range(n_in)]
    t_mid = [ramp_in + t for t in times]
    t_out = [ramp_in + times[-1] + ramp_out * i / n_out for i in range(1, n_out + 1)]
    v_out = [offsets[-1] * (1 - minjerk(i / n_out)) for i in range(1, n_out + 1)]
    v_out[-1] = 0.0
    return t_in + t_mid + t_out, v_in + list(offsets) + v_out


def rdp(times: list[float], values: list[float], tolerance: float) -> list[int]:
    """Indices kept by Ramer-Douglas-Peucker on the value error against linear interpolation."""
    keep = {0, len(times) - 1}
    stack = [(0, len(times) - 1)]
    while stack:
        lo, hi = stack.pop()
        worst, at = 0.0, -1
        for i in range(lo + 1, hi):
            frac = (times[i] - times[lo]) / (times[hi] - times[lo])
            err = abs(values[i] - (values[lo] + frac * (values[hi] - values[lo])))
            if err > worst:
                worst, at = err, i
        if worst > tolerance:
            keep.add(at)
            stack += [(lo, at), (at, hi)]
    return sorted(keep)


def convert_move(
    times: list[float], series: dict[str, list[float]], homes: dict[str, float]
) -> dict[str, list[list[float]]]:
    """Keyframes per channel ([t_s, offset_deg]) for one parsed move, all ending together."""
    out: dict[str, list[list[float]]] = {}
    offsets_of = {c: [v - homes[c] for v in vals] for c, vals in series.items()}
    offsets_of = {c: o for c, o in offsets_of.items() if max(abs(x) for x in o) > RDP_TOLERANCE_DEG}
    if not offsets_of:
        return out
    ramp_in, ramp_out = ramp_times(
        max(abs(o[0]) for o in offsets_of.values()), max(abs(o[-1]) for o in offsets_of.values())
    )  # one pair per clip, so every channel ends together
    end = round((times[-1] + ramp_in + ramp_out) / TIME_RESOLUTION_S) * TIME_RESOLUTION_S
    for channel, offsets in offsets_of.items():
        t_all, v_all = eased_channel(times, offsets, ramp_in, ramp_out)
        t_ms = [round(t / TIME_RESOLUTION_S) * TIME_RESOLUTION_S for t in t_all]
        keep = [0]  # millisecond rounding must keep times strictly increasing, ending at `end`
        for i in range(1, len(t_ms) - 1):
            if t_ms[keep[-1]] < t_ms[i] < end:
                keep.append(i)
        t_ms, v_all = [t_ms[i] for i in keep] + [end], [v_all[i] for i in keep] + [0.0]
        picked = rdp(t_ms, v_all, RDP_TOLERANCE_DEG)
        frames = []
        for i in picked:
            value = round(v_all[i] / ROUND_DEG) * ROUND_DEG
            frames.append([round(t_ms[i], 3), round(value, 1) + 0.0])
        frames[0][1] = frames[-1][1] = 0.0
        out[channel] = frames
    if out and max(f[-1][0] for f in out.values()) > EXPRESSION_MAX_CYCLE_S:
        raise ValueError("clip is longer than EXPRESSION_MAX_CYCLE_S")
    return out


def convert(raw: bytes, name: str, homes: dict[str, float]) -> tuple[dict, str]:
    times, series, description = parse_move(raw, name)
    return convert_move(times, series, homes), description


# ------------------------------------------------------------------ output


def num(x: float) -> str:
    text = f"{x:.3f}".rstrip("0")
    return text + "0" if text.endswith(".") else text


def render_yaml(clips: dict[str, tuple[dict, str]]) -> str:
    """The clips file: a header comment, then one entry per clip in name order."""
    lines = [
        f"# Converted from {REPO} revision {REVISION}",
        "# Source licence: Apache-2.0 (see LICENSE). Modified by conversion: head matrix to",
        "# roll/pitch/yaw degrees about home, antennas [right, left], body yaw, eased in and out,",
        "# simplified (scripts/import_pollen_emotions.py). Generated; do not edit by hand.",
        "",
    ]
    for name in sorted(clips):
        channels, description = clips[name]
        text = re.sub(r"\s+", " ", description).strip()
        if text:
            lines.append(f"# {text}")
        lines.append(f"{name}:")
        lines.append("  channels:")
        for channel, frames in channels.items():
            body = ", ".join(f"[{num(t)}, {v:.1f}]" for t, v in frames)
            lines.append(f"    {channel}: [{body}]")
    return "\n".join(lines) + "\n"


def conversion_constants() -> dict[str, float]:
    return {
        "ease_s": EASE_S,
        "ease_max_speed_deg_s": EASE_MAX_SPEED_DEG_S,
        "rdp_tolerance_deg": RDP_TOLERANCE_DEG,
        "round_deg": ROUND_DEG,
        "time_resolution_s": TIME_RESOLUTION_S,
        "homes_deg": manifest_homes(),
    }


def manifest_doc(clips_text: str) -> dict:
    return {
        "source": f"https://huggingface.co/datasets/{REPO}",
        "revision": REVISION,
        "license": "Apache-2.0",
        "attribution": ATTRIBUTION,
        "conversion": conversion_constants(),
        "license_sha256": LICENSE_SHA256,
        "clips_sha256": sha256(clips_text.encode()),
        "files": dict(sorted(SHA256.items())),
    }


# ------------------------------------------------------------------ fetching


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 (fixed https URLs)
        data = resp.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        sys.exit(f"{url}: larger than {MAX_FILE_BYTES} bytes")
    return data


def upstream_moves() -> list[str]:
    tree = json.loads(fetch(TREE_URL))
    names = sorted(e["path"] for e in tree if e["type"] == "file" and e["path"].endswith(".json"))
    for name in names:
        if not re.fullmatch(NAME_RE, name):
            sys.exit(f"unexpected upstream file name {name!r}")
    return names


def read_sources(source: Path | None) -> dict[str, bytes]:
    """Every pinned move, from a local copy or the pinned revision, verified against SHA256."""
    out = {}
    for name, digest in SHA256.items():
        if not re.fullmatch(NAME_RE, name):
            sys.exit(f"unexpected file name in SHA256: {name!r}")
        data = (source / name).read_bytes() if source else fetch(FILE_URL.format(name=name))
        if sha256(data) != digest:
            sys.exit(f"sha256 mismatch for {name}: got {sha256(data)}")
        out[name] = data
    return out


def ensure_license(source: Path | None) -> None:
    path = OUT_DIR / LICENSE
    if path.is_file() and sha256(path.read_bytes()) == LICENSE_SHA256:
        return
    data = (source / LICENSE).read_bytes() if source and (source / LICENSE).is_file() else None
    data = data if data is not None else fetch(LICENSE_URL)
    if sha256(data) != LICENSE_SHA256:
        sys.exit("Apache-2.0 text does not match the pinned sha256")
    path.write_bytes(data)


# ------------------------------------------------------------------ commands


def write(source: Path | None) -> None:
    if not SHA256:
        sys.exit("SHA256 is empty: run --print-pins and paste the result between the PINS markers")
    homes = manifest_homes()
    clips: dict[str, tuple[dict, str]] = {}
    for fname, raw in read_sources(source).items():
        name = fname.removesuffix(".json")
        channels, description = convert(raw, name, homes)
        if not channels:
            sys.exit(f"{name}: every channel stays within {RDP_TOLERANCE_DEG} deg of home")
        clips[name] = (channels, description)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_license(source)
    text = render_yaml(clips)
    (OUT_DIR / CLIPS).write_text(text)
    (OUT_DIR / VENDORED).write_text(json.dumps(manifest_doc(text), indent=2) + "\n")
    frames = sum(len(f) for c, _ in clips.values() for f in c.values())
    print(f"{len(clips)} clips, {frames} keyframes, {len(text)} bytes")


def check() -> int:
    """Offline: VENDORED.json, ``clips.yaml`` and the license agree with the pins in this script.

    This proves the outputs are consistent with the pins (hashes, revision, no stray files). It
    does not re-read the upstream data or re-run the conversion; ``--from DIR`` does that.
    """
    problems = []
    try:
        on_disk = json.loads((OUT_DIR / VENDORED).read_text())
        clips_text = (OUT_DIR / CLIPS).read_bytes()
        license_bytes = (OUT_DIR / LICENSE).read_bytes()
    except (OSError, ValueError) as exc:
        print(f"cannot read the vendored data: {exc}")
        return 1
    expected = manifest_doc(clips_text.decode())
    for key, want in expected.items():
        if key != "clips_sha256" and on_disk.get(key) != want:
            problems.append(f"VENDORED.json {key} differs from the pins in this script")
    if on_disk.get("clips_sha256") != sha256(clips_text):
        problems.append(f"{CLIPS} changed or VENDORED.json is stale")
    if sha256(license_bytes) != LICENSE_SHA256:
        problems.append(f"{LICENSE} is not the pinned Apache-2.0 text")
    if f"revision {REVISION}" not in clips_text.decode().split("\n", 1)[0]:
        problems.append(f"{CLIPS} header does not name the pinned revision")
    extra = {p.name for p in OUT_DIR.iterdir()} - {CLIPS, LICENSE, VENDORED}
    problems += [f"not in the manifest: {name}" for name in sorted(extra)]
    print("\n".join(problems) or "vendored emotions match the manifest")
    return 1 if problems else 0


def print_pins() -> None:
    print("    # PINS-BEGIN")
    for name in upstream_moves():
        print(f'    {name!r}: "{sha256(fetch(FILE_URL.format(name=name)))}",')
    print("    # PINS-END")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from", dest="source", type=Path, help="local copy of the dataset folder")
    ap.add_argument(
        "--check",
        action="store_true",
        help="offline: outputs consistent with the VENDORED.json pins",
    )
    ap.add_argument("--print-pins", action="store_true", help="print SHA256 for REVISION")
    args = ap.parse_args()
    if args.check:
        return check()
    if args.print_pins:
        print_pins()
        return 0
    write(args.source)
    return check()


if __name__ == "__main__":
    sys.exit(main())
