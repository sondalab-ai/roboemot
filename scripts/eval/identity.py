"""Identity accuracy of the shipped face stack on LFW, open set (SPEC 6, 14 Perception eval).

Product code paths: ``YuNetDetector`` and ``SFaceEmbedder`` (``roboemot.vision.onnx``), the
registry and ``Matcher`` with the shipped ``IdentityConfig`` (threshold 0.45, margin 0.1), and
``VisionPipeline`` for the tracks. Protocol per draw: ``k`` identities with at least 11 usable
images are enrolled with 5 images each; every other image is a probe.

* Per evaluation: one probe against the registry, ``Matcher`` with votes_required 1 of 1 (the
  threshold and margin stay the shipped ones). FAR: a probe of a non-enrolled person is accepted
  as anyone. FRR: a probe of an enrolled person is not accepted as that person; split into
  rejected and named as someone else (misidentification).
* Per track: 5 different photos of one person as consecutive frames (0.3 s apart, the first
  one repeated once because the tracker reports a track only after 2 hits) through the
  pipeline, with the shipped 3 of 5 voting. LFW photos of one person are independent shots,
  not video frames, so this is not video evidence.

Embeddings are cached in memory and in the scratch folder only; the result file has counts.

    python -I scripts/eval/identity.py --k 10 --draws 10 --out identity_k10
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

LFW = common.DATASETS / "lfw" / "x" / "lfw"
TRACK_FRAMES = 5
FRAME_GAP_S = 0.3
MIN_IMAGES = 11
ENROLL_IMAGES = 5
STRANGER_TRACKS = 300
TRACKS_PER_PERSON = 6  # random 5-photo subsets of the person's non-enrolment photos
LFW_BENCH = Path(__file__).resolve().parents[1] / "bench" / "results" / "lfw.json"


def image_paths() -> dict[str, list[Path]]:
    """LFW image files per identity (directory name)."""
    out: dict[str, list[Path]] = {}
    for folder in sorted(LFW.iterdir()):
        files = sorted(folder.glob("*.jpg"))
        if files:
            out[folder.name] = files
    return out


def central_face(stages, cfg, image: np.ndarray):
    """The detection nearest the image centre that passes the pipeline's embedding gates."""
    height, width = image.shape[:2]
    faces = [
        d
        for d in stages.face_detector.detect(image)
        if d.score >= cfg.vision.tracker.high_score and d.min_side >= cfg.vision.embed_min_face_px
    ]
    if not faces:
        return None, 0
    centre = np.array([width / 2, height / 2])

    def distance(d) -> float:
        x1, y1, x2, y2 = d.box
        return float(np.linalg.norm(np.array([(x1 + x2) / 2, (y1 + y2) / 2]) - centre))

    return min(faces, key=distance), len(faces)


def extract(stages, cfg, paths: dict[str, list[Path]]):
    """Embedding of the central face of every image (None when no face passes the gates)."""
    import cv2

    cache = common.SCRATCH / "lfw_embeddings.npz"
    names = [(person, i) for person, files in paths.items() for i in range(len(files))]
    if cache.exists():
        data = np.load(cache)
        if len(data["ok"]) == len(names):
            common.log("embeddings from the scratch cache")
            return data["emb"], data["ok"], data["multi"]
    emb = np.zeros((len(names), 128), np.float32)
    ok = np.zeros(len(names), bool)
    multi = np.zeros(len(names), bool)
    for n, (person, i) in enumerate(names):
        image = cv2.imread(str(paths[person][i]))
        face, count = central_face(stages, cfg, image)
        multi[n] = count > 1
        if face is not None:
            emb[n], ok[n] = stages.face_embedder.embed(image, face), True
        if n % 2000 == 0:
            common.log(f"embedded {n}/{len(names)}")
    common.SCRATCH.mkdir(parents=True, exist_ok=True)
    np.savez(cache, emb=emb, ok=ok, multi=multi)
    return emb, ok, multi


class Draw:
    """One random enrolment of ``k`` identities and everything measured on it."""

    def __init__(self, stages, cfg, paths, emb, ok, rng: random.Random, k: int, index: int):
        from roboemot.bus import Bus
        from roboemot.registry import Consent, FileStore, Matcher, Registry
        from roboemot.vision import VisionPipeline

        self.paths, self.emb, self.ok, self.rng, self.cfg = paths, emb, ok, rng, cfg
        self.offset = {}
        n = 0
        for person, files in paths.items():
            self.offset[person] = n
            n += len(files)
        usable = {p: [i for i in range(len(f)) if ok[self.offset[p] + i]] for p, f in paths.items()}
        pool = sorted(p for p, idx in usable.items() if len(idx) >= MIN_IMAGES)
        self.enrolled = rng.sample(pool, k)
        self.usable = usable
        folder = common.SCRATCH / f"registry-{index}"
        shutil.rmtree(folder, ignore_errors=True)
        self.registry = Registry(FileStore(folder, allow_plaintext=True), cfg.identity)
        consent = Consent("eval", datetime(2026, 1, 1, tzinfo=UTC), ("face",))
        self.owner: dict[str, str] = {}
        self.enroll_idx: dict[str, list[int]] = {}
        for person in self.enrolled:
            chosen = rng.sample(usable[person], ENROLL_IMAGES)
            self.enroll_idx[person] = chosen
            record = self.registry.enroll(
                f"p{len(self.owner)}",
                [emb[self.offset[person] + i] for i in chosen],
                stages.face_embedder.model_id,
                consent=consent,
                age_group="adult",
            )
            self.owner[record.person_id] = person
        self.person_of_owner = self.owner
        self.owner_of_person = {v: k for k, v in self.owner.items()}
        self.model_id = stages.face_embedder.model_id
        single = cfg.identity.model_copy(update={"votes_required": 1, "votes_window": 1})
        self.single = Matcher(self.registry, single)
        self.bus = Bus()
        self.percepts: list = []
        self.bus.subscribe("percept.face", lambda m: self.percepts.append(m.payload))
        self.pipeline = VisionPipeline(
            cfg.vision,
            cfg.identity,
            self.bus,
            Matcher(self.registry, cfg.identity),
            face_detector=stages.face_detector,
            face_embedder=stages.face_embedder,
            refresh=False,
            executor=None,
        )

    def per_evaluation(self) -> dict:
        """Single-vote decisions of every probe image against the registry."""
        stats = defaultdict(int)
        stale_margin = 0
        track = 0
        gallery = self.registry.gallery("face", self.model_id)
        for person, files in self.paths.items():
            enrolled = person in self.owner_of_person
            skip = set(self.enroll_idx.get(person, ()))
            for i in range(len(files)):
                n = self.offset[person] + i
                if i in skip:
                    continue
                if not self.ok[n]:
                    stats["enrolled_no_face" if enrolled else "stranger_no_face"] += 1
                    continue
                track += 1
                got = self.single.observe(
                    track, self.emb[n], self.model_id, 0.0, refresh=False
                ).owner_id
                if enrolled:
                    stats["enrolled_probes"] += 1
                    if got == self.owner_of_person[person]:
                        stats["enrolled_correct"] += 1
                    elif got is None:
                        stats["enrolled_rejected"] += 1
                        sims = sorted(
                            (float((rows @ self.emb[n]).max()) for rows in gallery.values()),
                            reverse=True,
                        )
                        if sims and sims[0] >= self.cfg.identity.match_threshold:
                            stale_margin += 1
                    else:
                        stats["enrolled_misidentified"] += 1
                else:
                    stats["stranger_probes"] += 1
                    stats["stranger_accepted"] += got is not None
        stats["enrolled_rejected_by_margin"] = stale_margin
        return dict(stats)

    def run_track(self, frames: list[np.ndarray], base: float) -> dict:
        """Feed one simulated track; the outcome is the central track after the last frame."""
        from roboemot.vision import Frame

        self.pipeline.reset()
        self.percepts.clear()
        first_named = None
        for j, image in enumerate(frames):
            self.percepts.clear()
            self.pipeline.process(Frame(image, base + j * FRAME_GAP_S))
            central = self._central(image)
            if central is not None and central.owner_id and first_named is None:
                first_named = j
        track_ids = {p.track_id for p in self.percepts}
        central = self._central(frames[-1])
        return {
            "owner": None if central is None else central.owner_id,
            "seen": central is not None,
            "first_named_frame": first_named,
            "tracks_in_last_frame": len(track_ids),
        }

    def _central(self, image: np.ndarray):
        if not self.percepts:
            return None
        height, width = image.shape[:2]
        return min(
            self.percepts,
            key=lambda p: (
                ((p.box[0] + p.box[2]) / 2 - 0.5) ** 2 * width**2
                + ((p.box[1] + p.box[3]) / 2 - 0.5) ** 2 * height**2
            ),
        )

    def per_track(self) -> dict:
        """Tracks of 5 photos of an enrolled person, and of 5 photos of a stranger."""
        import cv2

        stats = defaultdict(int)
        clock = 1000.0

        def frames_of(person: str, indices: list[int]) -> list[np.ndarray]:
            images = [cv2.imread(str(self.paths[person][i])) for i in indices]
            return [images[0], *images]

        for person in self.enrolled:
            rest = [i for i in self.usable[person] if i not in set(self.enroll_idx[person])]
            for _ in range(TRACKS_PER_PERSON):
                got = self.run_track(frames_of(person, self.rng.sample(rest, TRACK_FRAMES)), clock)
                clock += 100
                stats["enrolled_tracks"] += 1
                stats["enrolled_unseen"] += not got["seen"]
                if got["owner"] == self.owner_of_person[person]:
                    stats["enrolled_correct"] += 1
                    stats["enrolled_first_named_frame_sum"] += got["first_named_frame"] or 0
                elif got["owner"] is None:
                    stats["enrolled_unknown"] += 1
                else:
                    stats["enrolled_misidentified"] += 1
                stats["enrolled_broken"] += got["tracks_in_last_frame"] > 1
        strangers = sorted(
            p
            for p, idx in self.usable.items()
            if p not in self.owner_of_person and len(idx) >= TRACK_FRAMES
        )
        for person in self.rng.sample(strangers, min(STRANGER_TRACKS, len(strangers))):
            indices = self.rng.sample(self.usable[person], TRACK_FRAMES)
            got = self.run_track(frames_of(person, indices), clock)
            clock += 100
            stats["stranger_tracks"] += 1
            stats["stranger_unseen"] += not got["seen"]
            stats["stranger_accepted"] += got["owner"] is not None
        return dict(stats)


def total(draws: list[dict]) -> dict:
    """Sum the counters of several draws."""
    out: dict[str, int] = defaultdict(int)
    for d in draws:
        for key, value in d.items():
            out[key] += value
    return dict(out)


def summarize(ev: dict, tr: dict) -> dict:
    """Rates with Wilson intervals from pooled counters."""
    r = common.rate
    return {
        "per_evaluation": {
            "enrolled_probes": ev["enrolled_probes"],
            "stranger_probes": ev["stranger_probes"],
            "no_face_enrolled_images": ev.get("enrolled_no_face", 0),
            "no_face_stranger_images": ev.get("stranger_no_face", 0),
            "far": r(ev.get("stranger_accepted", 0), ev["stranger_probes"]),
            "frr": r(ev["enrolled_probes"] - ev.get("enrolled_correct", 0), ev["enrolled_probes"]),
            "frr_rejected": r(ev.get("enrolled_rejected", 0), ev["enrolled_probes"]),
            "frr_rejected_by_margin_only": ev.get("enrolled_rejected_by_margin", 0),
            "misidentified": r(ev.get("enrolled_misidentified", 0), ev["enrolled_probes"]),
        },
        "per_track_3_of_5": {
            "enrolled_tracks": tr["enrolled_tracks"],
            "stranger_tracks": tr["stranger_tracks"],
            "far": r(tr.get("stranger_accepted", 0), tr["stranger_tracks"]),
            "frr": r(tr["enrolled_tracks"] - tr.get("enrolled_correct", 0), tr["enrolled_tracks"]),
            "frr_unknown": r(tr.get("enrolled_unknown", 0), tr["enrolled_tracks"]),
            "misidentified": r(tr.get("enrolled_misidentified", 0), tr["enrolled_tracks"]),
            "enrolled_tracks_split_in_two": tr.get("enrolled_broken", 0),
            "enrolled_tracks_no_face_in_last_frame": tr.get("enrolled_unseen", 0),
            "stranger_tracks_no_face_in_last_frame": tr.get("stranger_unseen", 0),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True, help="identities enrolled per draw")
    ap.add_argument("--draws", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = common.config()
    stages = common.load_stages(cfg, attributes=False, appearance=False, objects=False)
    paths = image_paths()
    emb, ok, multi = extract(stages, cfg, paths)
    common.log(
        f"images {len(ok)}, with a usable face {int(ok.sum())}, several faces {int(multi.sum())}"
    )
    evs, trs, per_draw = [], [], []
    for d in range(args.draws):
        draw = Draw(stages, cfg, paths, emb, ok, random.Random(args.seed * 1000 + d), args.k, d)
        ev, tr = draw.per_evaluation(), draw.per_track()
        evs.append(ev)
        trs.append(tr)
        per_draw.append(summarize(ev, tr))
        far = per_draw[-1]["per_evaluation"]["far"]["rate"]
        common.log(f"draw {d}: eval FAR {far:.5f}, track {per_draw[-1]['per_track_3_of_5']}")
        draw.pipeline.close()
        shutil.rmtree(common.SCRATCH / f"registry-{d}", ignore_errors=True)
    spike = __import__("json").loads(LFW_BENCH.read_text())["pipelines"]["yunet+sface"]
    result = {
        "protocol": {
            "dataset": "LFW (13,233 images, 5,749 identities), open set, enrolled identities have "
            f">= {MIN_IMAGES} usable images",
            "k_enrolled": args.k,
            "draws": args.draws,
            "enroll_images_per_person": ENROLL_IMAGES,
            "track_frames": TRACK_FRAMES,
            "frame_gap_s": FRAME_GAP_S,
            "refresh": False,
            "identity_config": cfg.identity.model_dump(
                include={"match_threshold", "margin", "votes_required", "votes_window"}
            ),
            "detector_gates": {
                "score_min": cfg.vision.tracker.high_score,
                "embed_min_face_px": cfg.vision.embed_min_face_px,
            },
        },
        "images": {
            "total": int(len(ok)),
            "usable_face": int(ok.sum()),
            "several_faces": int(multi.sum()),
        },
        "pooled": summarize(total(evs), total(trs)),
        "per_draw": per_draw,
        "spike_reference_lfw_pairs_yunet_sface": {
            key: spike[key]
            for key in (
                "n_pairs",
                "acc_10fold_mean",
                "std_tar@far0.001",
                "wide_neg_n",
                "wide_tar@far0.001",
                "wide_thr@far0.001",
                "wide_tar@far0.0001",
                "wide_thr@far0.0001",
            )
        },
        "machine": common.cpu_label(),
        "versions": common.versions(),
    }
    print(common.write_result(args.out, result))


if __name__ == "__main__":
    main()
