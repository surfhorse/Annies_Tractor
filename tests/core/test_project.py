from __future__ import annotations

import hashlib

import pytest

from afv.core.project import Project, ProjectError, shot_context, stage_dir


def test_create_and_find(tmp_path, video, project):
    info = project.info
    assert info.source.sha256 == hashlib.sha256(video.read_bytes()).hexdigest()
    assert info.source.size == video.stat().st_size and info.source.title == "Test fight"
    sub = project.root / "shots"
    assert Project.find(sub).root == project.root
    with pytest.raises(ProjectError, match="already has"):
        Project.create(project.root, video)
    with pytest.raises(ProjectError, match="not an afv project"):
        Project(tmp_path)
    with pytest.raises(ProjectError, match="video not found"):
        Project.create(tmp_path / "x", tmp_path / "nope.mp4")


def test_shots(project):
    a = project.add_shot(1.0, 2.0, notes="jab")
    b = project.add_shot(5.0, 1.5, speed_factor=0.25)
    assert (a.id, b.id) == ("001", "002")
    assert project.shot("1").id == "001" and project.shot("002").id == "002"
    assert b.info.speed_factor == 0.25 and b.info.end == pytest.approx(6.5)
    for sub in ("stages", "edits", "exports"):
        assert (a.dir / sub).is_dir()
    with pytest.raises(ProjectError, match="no shot '7'"):
        project.shot("7")
    with pytest.raises(ProjectError, match="already exists"):
        project.add_shot(0, 1, shot_id="001")
    with pytest.raises(ProjectError, match="bad shot interval"):
        project.add_shot(1, 0)
    with pytest.raises(ProjectError, match="bad shot id"):
        project.add_shot(1, 1, shot_id="../evil")
    project.add_shot(9, 1, shot_id="replay_a")
    assert project.shot_ids() == ["001", "002", "replay_a"]


def test_edits(project):
    s = project.add_shot(0, 1)
    assert s.read_edit("fighters") is None
    s.write_edit("fighters", {"fighters": [1, 0]})
    assert s.read_edit("fighters") == {"fighters": [1, 0]}
    s.edit_path("contacts").write_text("{not json")
    with pytest.raises(ProjectError, match=r"contacts\.json is not valid JSON"):
        s.read_edit("contacts")
    with pytest.raises(ProjectError, match="bad edit name"):
        s.edit_path("../x")


def test_source_checks(project, video):
    assert project.source_path() == video.resolve()
    video.write_bytes(b"short")
    with pytest.raises(ProjectError, match="changed since it was hashed"):
        project.source_path()
    video.unlink()
    with pytest.raises(ProjectError, match="source video missing"):
        project.source_path()


def test_bad_json_is_readable(project):
    s = project.add_shot(0, 1)
    (s.dir / "shot.json").write_text('{"id": "001"}')
    with pytest.raises(ProjectError, match=r"shot\.json is invalid"):
        _ = s.info


def test_shot_context_for_stages(project, video):
    s = project.add_shot(3.0, 2.0, speed_factor=0.5)
    ctx = shot_context(s.dir)
    assert ctx.source_path == str(video.resolve()) and ctx.start == 3.0 and ctx.speed_factor == 0.5
    assert ctx.source_url == "https://example.com/v"
    assert stage_dir(s.dir, "ingest") == s.stage_dir("ingest")
