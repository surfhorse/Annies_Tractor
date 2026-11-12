from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest
from media_synth import (
    arena_frames,
    flash,
    pan_frames,
    remux_with_audio,
    scene_cuts_frames,
    scene_dissolve_frames,
    scene_shot,
    scene_wipe_frames,
    write_video,
)

from afv.media.cuts import Cut, CutAnalysis, CutParams, detect_cuts
from afv.media.frames import iter_frames
from afv.media.normalise import GEMX_FPS, normalise_shot, shot_frame_count
from afv.media.probe import packet_table, probe
from afv.media.shots import DEFAULT_MIN_CONFIDENCE, detect_shots, shots_from_cuts


def _hard(an: CutAnalysis) -> list[Cut]:
    return [c for c in an.cuts if c.kind == "hard"]


def test_hard_cuts_exact(tmp_path: Path) -> None:
    an = detect_cuts(write_video(tmp_path / "c.mp4", scene_cuts_frames([40, 35, 50])))
    assert [c.frame for c in an.cuts] == [40, 75]
    for c in an.cuts:
        assert c.kind == "hard" and c.confidence > 0.6
        assert c.time == pytest.approx((c.frame - 0.5) / 30, abs=1e-6)
    assert an.scores.shape == an.times.shape == (125,)


def test_similar_shot_cut(tmp_path: Path) -> None:
    """Same arena, another camera: histograms barely change (score ~0.11), still a cut."""
    an = detect_cuts(write_video(tmp_path / "arena.mp4", arena_frames(45, 45)))
    assert [(c.frame, c.kind) for c in an.cuts] == [(45, "hard")]
    assert 0.08 < an.cuts[0].score < 0.2 and an.cuts[0].confidence >= DEFAULT_MIN_CONFIDENCE


@pytest.mark.parametrize("fps", [25, 30])
@pytest.mark.parametrize("length", [1, 2, 3, 4])
def test_flashes_are_not_cuts(tmp_path: Path, fps: int, length: int) -> None:
    frames = scene_cuts_frames([45, 45])
    for start in (10, 30, 45, 70):  # inside shots, and on the first frame after the cut
        for i in range(start, start + length):
            frames[i] = flash(frames[i])
    an = detect_cuts(write_video(tmp_path / "flash.mp4", frames, fps=fps))
    assert [c.frame for c in an.cuts] == [45]


def test_exposure_jump_is_not_a_cut(tmp_path: Path) -> None:
    frames = scene_shot(80, 1)
    brighter = [np.clip(f.astype(np.float32) * 1.35, 0, 255).astype(np.uint8) for f in frames[40:]]
    frames = frames[:40] + brighter
    assert detect_cuts(write_video(tmp_path / "exp.mp4", frames)).cuts == []


@pytest.mark.parametrize(("n", "speed"), [(60, 25), (90, 15), (90, 40), (120, 25)])
def test_fast_pans_are_not_cuts(tmp_path: Path, n: int, speed: int) -> None:
    an = detect_cuts(write_video(tmp_path / "pan.mp4", pan_frames(n, speed=speed)))
    assert an.cuts == []


@pytest.mark.parametrize("speed", [15, 40])
def test_cut_between_pans_is_hard(tmp_path: Path, speed: int) -> None:
    a = pan_frames(60, speed=speed, seed=1)
    b = [np.ascontiguousarray(f[:, ::-1]) for f in pan_frames(60, speed=speed, seed=8)]
    an = detect_cuts(write_video(tmp_path / "pp.mp4", a + b))
    assert [(c.frame, c.kind) for c in an.cuts] == [(60, "hard")]


def test_minimum_shot_length(tmp_path: Path) -> None:
    path = write_video(tmp_path / "short.mp4", scene_cuts_frames([40, 5, 40]))
    assert len(detect_cuts(path).cuts) == 1  # 5 frames apart < 0.5 s: keep the stronger
    assert [c.frame for c in detect_cuts(path, CutParams(min_shot=0.1)).cuts] == [40, 45]


@pytest.mark.parametrize("to_black", [False, True])
def test_dissolve_and_fade(tmp_path: Path, to_black: bool) -> None:
    an = detect_cuts(write_video(tmp_path / "d.mp4", scene_dissolve_frames(40, 15, 40, to_black=to_black)))
    (g,) = an.cuts
    assert g.kind == "gradual" and g.confidence > 0.2
    assert 39 <= g.start_frame <= 44 and 50 <= g.frame <= 56 and g.start_time < g.time


@pytest.mark.parametrize("length", [8, 12])
def test_wipe(tmp_path: Path, length: int) -> None:
    an = detect_cuts(write_video(tmp_path / "w.mp4", scene_wipe_frames(40, length, 40)))
    (g,) = an.cuts
    assert g.kind == "gradual" and g.confidence > 0.2
    assert 39 <= g.start_frame <= 42 and 40 + length - 2 <= g.frame <= 40 + length + 1


def test_dip_to_black(tmp_path: Path) -> None:
    a, b = scene_shot(50, 0), scene_shot(50, 2)
    dip = ([(a[40 + i] * (1 - (i + 1) / 9)).astype(np.uint8) for i in range(8)] + [np.zeros_like(a[0])] * 4
           + [(b[i] * ((i + 1) / 9)).astype(np.uint8) for i in range(8)])
    an = detect_cuts(write_video(tmp_path / "dip.mp4", a[:40] + dip + b[8:48]))
    (g,) = an.cuts
    assert g.kind == "gradual" and 39 <= g.start_frame <= 41 and 58 <= g.frame <= 60


def test_gradual_can_be_disabled(tmp_path: Path) -> None:
    path = write_video(tmp_path / "d.mp4", scene_dissolve_frames(40, 15, 40))
    assert all(c.kind == "hard" for c in detect_cuts(path, CutParams(gradual=False)).cuts)


def test_never_emits_zero_confidence(tmp_path: Path) -> None:
    frames = pan_frames(40, speed=30) + scene_cuts_frames([40, 40], seed=3) + pan_frames(40, speed=12, seed=4)
    an = detect_cuts(write_video(tmp_path / "mix.mp4", frames))
    assert an.cuts and all(c.confidence >= an.params.min_confidence > 0 for c in an.cuts)


def _nearest(img: np.ndarray, refs: list[np.ndarray]) -> int:
    m = img.reshape(-1, 3).mean(0)
    return int(np.argmin([np.abs(m - r).sum() for r in refs]))


# (fps, container): mp4 written by PyAV (with and without audio) and remuxes by the pinned ffmpeg
# into MKV/WebM with AAC/Opus, whose seek origin is negative (ffmpeg 7.0.2) while PyAV reports 0
CASES = [(24, "mp4a"), (30, "mp4"), (30000 / 1001, "mkv_aac"), (25, "mkv_opus"), (60, "webm_opus"),
         (50, "mp4"), (24000 / 1001, "mkv_aac"), (60, "mkv_aac")]


@pytest.mark.parametrize(("fps", "container"), CASES)
def test_shots_contain_only_their_frames(tmp_path: Path, ffmpeg: str, fps: float, container: str) -> None:
    """End to end: cuts -> shots -> GEM-X normalisation; every 30 Hz frame of a shot comes from
    that shot (no frame of the neighbour leaks in at either end), and T is predicted exactly."""
    segs = [int(fps * 1.3), int(fps * 0.9) + 1, int(fps * 1.1)]
    frames = scene_cuts_frames(segs)
    refs = [np.mean([f.reshape(-1, 3).mean(0) for f in frames[a:b]], axis=0)
            for a, b in itertools.pairwise([0, *np.cumsum(segs)])]
    path = write_video(tmp_path / "c.mp4", frames, fps=fps, audio=container == "mp4a")
    if container not in ("mp4", "mp4a"):
        path = remux_with_audio(path, tmp_path / f"c.{container.split('_')[0]}", container, ffmpeg)
    shots, an = detect_shots(path)
    assert [c.frame for c in an.cuts] == [segs[0], segs[0] + segs[1]]
    assert len(shots) == 3
    info = probe(path, hash=False)
    for k, shot in enumerate(shots):
        ns = normalise_shot(path, shot.start, shot.duration, tmp_path / f"s{k}.mp4", info=info)
        got = [_nearest(img, refs) for img in iter_frames(ns.path)]
        assert set(got) == {k}, (k, got)
        exact = shot_frame_count(path, shot.start, shot.duration, info=info)
        assert ns.num_frames == shot.num_frames == exact
        assert ns.num_frames >= round(30 * segs[k] / fps) - 1
    assert shots[0].start == 0.0 and shots[-1].end == pytest.approx(info.cv_duration, abs=1e-6)
    assert shots[1].cut_in is an.cuts[0] and shots[0].cut_out is an.cuts[0]
    assert [s.start_frame for s in shots] == [0, segs[0], segs[0] + segs[1]]


def test_long_shots_split_on_ticks(tmp_path: Path) -> None:
    path = write_video(tmp_path / "long.mp4", scene_cuts_frames([70, 65]), audio=True)
    an = detect_cuts(path)
    info = probe(path, hash=False)
    shots = shots_from_cuts(an.cuts, an.times, info, max_duration=0.6, table=packet_table(path))
    assert all(s.duration <= 0.6 + 1e-9 for s in shots) and all(s.split for s in shots)
    for a, b in itertools.pairwise(shots):
        if b.cut_in is None:  # parts of one shot: whole ticks long, timelines join exactly
            ticks = a.duration * GEMX_FPS
            assert abs(ticks - round(ticks)) < 1e-3
            assert b.start == pytest.approx(a.start + round(ticks) / GEMX_FPS, abs=1e-6)
            ns = normalise_shot(path, a.start, a.duration, tmp_path / "part.mp4", info=info)
            assert ns.num_frames == round(ticks)
    # the confidence filter drops the cuts entirely
    assert len(shots_from_cuts(an.cuts, an.times, info, min_confidence=1.01)) == 1


def test_detect_shots_default_min_confidence(tmp_path: Path) -> None:
    path = write_video(tmp_path / "c.mp4", scene_cuts_frames([40, 40]))
    shots, an = detect_shots(path)
    assert len(shots) == 2 and all(c.confidence >= DEFAULT_MIN_CONFIDENCE for c in an.cuts)
    assert len(detect_shots(path, min_confidence=1.01)[0]) == 1
