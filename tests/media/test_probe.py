from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from media_synth import colour_cuts_frames, remux_with_audio, write_video

from afv.media.ffmpeg import (
    MediaError,
    check_ffmpeg_version,
    ffmpeg_dir,
    ffmpeg_path,
    ffmpeg_version,
    run_ffmpeg,
    seek_origin_us,
)
from afv.media.probe import ClipInfo, opencv_frame_count_fps, packet_times, probe


def test_probe_cfr_with_audio(tmp_path: Path) -> None:
    path = write_video(tmp_path / "a.mp4", colour_cuts_frames([20, 25]), fps=25, audio=True)
    info = probe(path)
    assert info.codec == "h264" and info.container.startswith("mov")
    assert (info.width, info.height) == (192, 108) and info.rotation == 0
    assert info.num_frames == info.nb_frames == info.cv_frame_count == 45
    assert info.cv_fps == pytest.approx(25.0) and info.avg_fps == pytest.approx(25.0)
    assert not info.vfr and info.has_audio and info.num_streams == 2
    assert info.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert info.cv_duration == pytest.approx(45 / 25)
    assert info.notes == []
    assert ClipInfo.from_dict(json.loads(json.dumps(info.to_dict()))) == info


def test_probe_vfr(vfr_video: Path) -> None:
    info = probe(vfr_video, hash=False)
    assert info.vfr and info.num_frames == 90 and info.sha256 == ""
    # OpenCV reports avg_frame_rate, not r_frame_rate, for VFR files
    assert info.cv_fps == pytest.approx(info.avg_fps)
    assert any("variable frame rate" in n for n in info.notes)
    t, _ = packet_times(vfr_video)
    assert t.size == 90 and (t[1:] > t[:-1]).all()


def test_opencv_rule_without_nb_frames() -> None:
    # MKV/WebM: no header count -> floor(duration * fps + 0.5); fps prefers avg_frame_rate
    assert opencv_frame_count_fps(0, 60.0, 29.97, 10.01, 0.0) == (300, 29.97)
    assert opencv_frame_count_fps(0, 30.0, 0.0, 0.0, 2.0) == (60, 30.0)
    assert opencv_frame_count_fps(91, 30.0, 29.97, 3.0, 3.0) == (91, 29.97)


def test_probe_missing(tmp_path: Path) -> None:
    with pytest.raises(MediaError):
        probe(tmp_path / "nope.mp4")


_CV_SNIPPET = """
import cv2, json, sys
out = {}
for p in sys.argv[1:]:
    c = cv2.VideoCapture(p)
    out[p] = [c.get(cv2.CAP_PROP_FRAME_COUNT), c.get(cv2.CAP_PROP_FPS)]
print(json.dumps(out))
"""


def test_opencv_equivalence(tmp_path: Path, vfr_video: Path, gemx_python: Path) -> None:
    """Our OpenCV-equivalent numbers equal real OpenCV's (gem-x env) on CFR, audio and VFR files."""
    files = [
        write_video(tmp_path / "c24.mp4", colour_cuts_frames([10, 13]), fps=24),
        write_video(tmp_path / "c2997a.mp4", colour_cuts_frames([10, 13]), fps=30000 / 1001, audio=True),
        vfr_video,
    ]
    got = json.loads(subprocess.run([str(gemx_python), "-c", _CV_SNIPPET, *map(str, files)],
                                    capture_output=True, text=True, check=True).stdout)
    for f in files:
        info = probe(f, hash=False)
        count, fps = got[str(f)]
        assert info.cv_frame_count == int(count)
        assert info.cv_fps == pytest.approx(fps, rel=1e-9)


def test_seek_origin_from_pinned_ffmpeg(tmp_path: Path, ffmpeg: str) -> None:
    """MKV + AAC: PyAV's libavformat reports start 0, ffmpeg 7.0.2 -0.023 s; -ss follows the binary."""
    src = write_video(tmp_path / "v.mp4", colour_cuts_frames([20]), fps=30)
    mkv = remux_with_audio(src, tmp_path / "v.mkv", "mkv_aac", ffmpeg)
    info = probe(mkv, hash=False)
    assert info.start_time == seek_origin_us(mkv) / 1e6 < 0
    assert info.first_frame_time == pytest.approx(-info.start_time, abs=1e-6)
    assert any("PyAV container start" in n for n in info.notes)
    header = probe(mkv, hash=False, packets=False)
    assert not header.packets_counted and header.start_time == info.start_time
    assert header.cv_frame_count == info.cv_frame_count and header.probe_version == info.probe_version > 0


def test_ffmpeg_dir_and_version(tmp_path: Path, ffmpeg: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AFV_FFMPEG_DIR", str(Path(ffmpeg).parent))
    assert ffmpeg_dir() == Path(ffmpeg).parent and ffmpeg_path() == str(Path(ffmpeg).parent / "ffmpeg")
    assert check_ffmpeg_version(ffmpeg_version()) == ffmpeg_version()
    with pytest.raises(MediaError, match=r"expected 0\.0\.1"):
        check_ffmpeg_version("0.0.1")
    monkeypatch.setenv("AFV_FFMPEG_DIR", str(tmp_path))
    with pytest.raises(MediaError, match="does not contain"):
        ffmpeg_path()


def test_crash_has_readable_message(tmp_path: Path) -> None:
    fake = tmp_path / "ffmpeg"
    fake.write_text("#!/bin/sh\nkill -SEGV $$\n")
    fake.chmod(0o755)
    with pytest.raises(MediaError, match="remux it to mp4"):
        run_ffmpeg(["-version"], str(fake))
    with pytest.raises(MediaError, match="remux it to mp4"):
        seek_origin_us(tmp_path / "ffmpeg", str(fake))
