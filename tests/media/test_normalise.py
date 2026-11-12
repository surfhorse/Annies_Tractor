from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import av
import numpy as np
import pytest
from media_synth import GEMX_VIDEO, colour_cuts_frames, remux_with_audio, write_video

from afv.media.ffmpeg import MediaError, ffmpeg_path
from afv.media.frames import iter_frames
from afv.media.normalise import (
    estimate_frame_count,
    gemx_ffmpeg_args,
    normalise_shot,
    select_interval,
    shot_frame_count,
    shot_timestamps,
)
from afv.media.probe import packet_table, probe

# (start, duration) pairs; None = GEM-X's default (to the end). Chosen to hit the edge cases:
# a first source frame that rounds to tick 1 (cfr fills tick 0), -t just above an integer number
# of ticks, starts between frames, and the end of the source.
INTERVALS = [(0.0, None), (0.0, 1.0), (0.1, 0.5), (0.0137, 1.234), (0.2868, 0.9335),
             (0.0298, 0.239), (0.37, 0.77), (1.0, None), (2.1, None),
             (0.5, 0.01), (0.2, 0.02), (0.31, 1 / 60)]  # sub-tick durations: one frame, not zero


@pytest.fixture(scope="module")
def clips(synth_dir: Path, vfr_video: Path, ffmpeg: str) -> list[Path]:
    """CFR at several rates, with audio (2 streams -> ffmpeg muxes with cfr) and without (vscfr)."""
    out = [vfr_video]
    for fps, audio in [(24, False), (30, False), (60, True), (30000 / 1001, True), (25, True)]:
        name = f"n{fps:.2f}{'a' if audio else ''}.mp4"
        out.append(write_video(synth_dir / name, colour_cuts_frames([37, 41]), fps=fps, audio=audio))
    rng = np.random.default_rng(3)
    t = np.cumsum(np.concatenate([[0.0], rng.choice([1 / 60, 1 / 30, 1 / 24], size=79)]))
    out.append(write_video(synth_dir / "vfr_a.mp4", colour_cuts_frames([40, 40]), times=list(t), audio=True))
    # MKV/WebM with AAC/Opus: negative seek origin in ffmpeg 7.0.2, 0 in PyAV's libavformat
    base = write_video(synth_dir / "n2997.mp4", colour_cuts_frames([37, 41]), fps=30000 / 1001)
    out.append(remux_with_audio(base, synth_dir / "n2997_aac.mkv", "mkv_aac", ffmpeg))
    out.append(remux_with_audio(base, synth_dir / "n2997_opus.webm", "webm_opus", ffmpeg))
    return out


def test_frame_count_follows_gemx_rule(clips: list[Path], tmp_path: Path, ffmpeg: str) -> None:
    """GEM-X's T is the frame count (OpenCV CAP_PROP_FRAME_COUNT) of the mp4 written by its ffmpeg
    command. Our normalised file must have exactly that many decodable frames, and both
    predictions must agree with it."""
    checked = 0
    for clip in clips:
        info = probe(clip, hash=False)
        table = packet_table(clip)
        for start, dur in INTERVALS:
            try:
                select_interval(info, start, dur)
            except MediaError:
                continue
            est = estimate_frame_count(clip, start, dur, info=info, table=table)
            if est == 0:  # no 30 Hz tick in the interval: GEM-X fails, so do we
                assert shot_frame_count(clip, start, dur, info=info) == 0
                with pytest.raises(MediaError, match="no 30 Hz frame"):
                    normalise_shot(clip, start, dur, tmp_path / "s.mp4", info=info)
                continue
            ns = normalise_shot(clip, start, dur, tmp_path / "s.mp4", info=info)
            decoded = sum(1 for _ in iter_frames(ns.path))
            assert ns.num_frames == decoded, (clip.name, start, dur)
            assert shot_frame_count(clip, start, dur, info=info) == decoded, (clip.name, start, dur)
            assert est == decoded, (clip.name, start, dur)
            checked += 1
    assert checked >= 70


def test_cfr_fill_depends_on_stream_count(synth_dir: Path) -> None:
    """Same video with and without an audio track: with audio (2 streams) ffmpeg's mp4 muxing
    uses cfr and duplicates the first frame to t=0 when it rounds to tick 1."""
    frames = colour_cuts_frames([30, 30])
    v = write_video(synth_dir / "fill_v.mp4", frames, fps=24)
    va = write_video(synth_dir / "fill_va.mp4", frames, fps=24, audio=True)
    # -ss 0.1: first kept frame at 0.125 s -> 0.025 s after start -> rounds to tick 1
    assert shot_frame_count(va, 0.1, 0.5) == shot_frame_count(v, 0.1, 0.5) + 1 == 15


def test_command_is_gemx_verbatim(tmp_path: Path) -> None:
    args = gemx_ffmpeg_args("/x/src.mp4", "/y/out.mp4", 1.5, 2.25)
    assert args == [
        "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-ss", "1.500000000", "-i",
        str(Path("/x/src.mp4").resolve()), "-t", "2.250000000", "-map", "0:v:0", "-vf",
        "fps=30,pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-crf", "18", "-pix_fmt",
        "yuv420p", "-an", "-movflags", "+faststart", str(Path("/y/out.mp4").resolve()),
    ]


def test_interval_validation_matches_gemx(cuts_video: Path) -> None:
    info = probe(cuts_video, hash=False)
    assert select_interval(info, 0.0, None) == pytest.approx(125 / 30)
    assert select_interval(info, 1.0, None) == pytest.approx(125 / 30 - 1.0)
    for start, dur in [(-1.0, None), (5.0, None), (0.0, 0.0), (0.0, 61.0), (3.0, 2.0), (float("nan"), 1.0)]:
        with pytest.raises(MediaError):
            select_interval(info, start, dur)


def test_frames_folder_and_timestamps(cuts_video: Path, tmp_path: Path) -> None:
    ns = normalise_shot(cuts_video, 0.5, 1.0, tmp_path / "frames")
    assert ns.num_frames == 30 and (ns.width, ns.height) == (192, 108)
    files = sorted(Path(ns.path).glob("*.jpg"))
    assert len(files) == 30 and files[0].name == "00000.jpg"
    ts = shot_timestamps(0.5, 30)
    assert ts[0] == 0.5 and ts[-1] == pytest.approx(0.5 + 29 / 30)
    np.testing.assert_allclose(ns.timestamps, ts)


def test_frames_folder_safety(cuts_video: Path, tmp_path: Path) -> None:
    target = tmp_path / "frames"
    normalise_shot(cuts_video, 0.0, 0.5, target)
    assert normalise_shot(cuts_video, 0.5, 0.5, target).num_frames == 15  # own folder: replaced
    other = tmp_path / "precious"
    other.mkdir()
    (other / "notes.txt").write_text("keep me")
    with pytest.raises(MediaError, match="refusing"):
        normalise_shot(cuts_video, 0.0, 0.5, other)
    assert (other / "notes.txt").exists()
    with pytest.raises(MediaError, match=r"\.mp4 file or a folder"):
        normalise_shot(cuts_video, 0.0, 0.5, tmp_path / "shot.mov")
    empty = tmp_path / "empty"
    empty.mkdir()
    assert normalise_shot(cuts_video, 0.0, 0.5, empty).num_frames == 15


def test_odd_size_is_padded_even(tmp_path: Path) -> None:
    frames = [f[:107, :191] for f in colour_cuts_frames([12])]
    # libx264 needs even input sizes; encode the odd frame size losslessly via ffv1 in mkv
    src = tmp_path / "odd.mkv"
    with av.open(str(src), "w") as c:
        s = c.add_stream("ffv1", rate=30)
        s.width, s.height, s.pix_fmt = 191, 107, "yuv444p"
        for img in frames:
            for p in s.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    ns = normalise_shot(src, 0.0, None, tmp_path / "odd.mp4")
    assert (ns.width, ns.height) == (192, 108)
    assert ns.num_frames == shot_frame_count(src) == estimate_frame_count(src)


_GEMX_SNIPPET = """
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("gemx_video", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
dur = None if sys.argv[5] == "None" else float(sys.argv[5])
md = m.normalize_video(sys.argv[2], sys.argv[3], start=float(sys.argv[4]), duration=dur)
print(json.dumps({"T": md["normalized"]["frame_count"], "sel": md["source"]["selected_duration"]}))
"""


def _framemd5(path: Path, ffmpeg: str) -> list[str]:
    out = subprocess.run([ffmpeg, "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "framemd5", "-"],
                         capture_output=True, text=True, check=True).stdout
    return [ln.split(",")[-1].strip() for ln in out.splitlines() if ln and not ln.startswith("#")]


def test_identical_to_real_gemx(clips: list[Path], tmp_path: Path, ffmpeg: str, gemx_python: Path) -> None:
    """Run GEM-X's own normalize_video (gem-x env, same ffmpeg on PATH): same T, same decoded frames."""
    env = dict(os.environ, PATH=f"{Path(ffmpeg_path()).parent}{os.pathsep}{os.environ.get('PATH', '')}")
    for clip in [clips[0], clips[3], clips[4], clips[-2], clips[-1]]:  # VFR, 60+a, 29.97+a, MKV, WebM
        for start, dur in [(0.0, None), (0.0137, 1.234), (0.37, 0.77)]:
            ref = tmp_path / "gemx.mp4"
            res = json.loads(subprocess.run(
                [str(gemx_python), "-c", _GEMX_SNIPPET, str(GEMX_VIDEO), str(clip), str(ref),
                 str(start), str(dur)],
                capture_output=True, text=True, check=True, env=env).stdout)
            ns = normalise_shot(clip, start, dur, tmp_path / "ours.mp4")
            assert ns.num_frames == res["T"] == shot_frame_count(clip, start, dur), (clip.name, start, dur)
            assert ns.duration == pytest.approx(res["sel"], abs=1e-12)
            assert _framemd5(Path(ns.path), ffmpeg) == _framemd5(ref, ffmpeg), (clip.name, start, dur)
