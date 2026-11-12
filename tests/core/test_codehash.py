"""Stage source in the cache: normalised hashing, dependency discovery, CPU/model-stage policy."""

from __future__ import annotations

import ast
import importlib
import json
import sys
import textwrap
from pathlib import Path

import pytest

from afv.core import codehash
from afv.core import runner as runner_mod
from afv.core.runner import RunError, Runner


def run(shot, fakes, **kw):
    to = kw.pop("to", "smooth")
    from_stage = kw.pop("from_stage", None)
    fakes.calls.clear()
    events: list[dict] = []
    rep = Runner(shot, report=events.append, **kw).run(to=to, from_stage=from_stage)
    rep.events = events
    return rep


@pytest.fixture(autouse=True)
def _fresh_codehash_caches():
    codehash.clear_cache()
    yield
    codehash.clear_cache()


# ---------------------------------------------------------------------------
# Normalised source hash
# ---------------------------------------------------------------------------

SRC = '''
"""Module doc."""
import numpy as np


def speed(x, fps=30.0):
    """Doc."""
    # a comment
    return np.diff(x) * fps
'''


@pytest.mark.parametrize(
    "variant",
    [
        SRC.replace("# a comment", "# another comment\n    # and more"),
        SRC.replace('"""Doc."""', '"""A much longer docstring\n\n    over lines."""'),
        SRC.replace('"""Module doc."""', ""),
        SRC.replace("return np.diff(x) * fps", "return (\n        np.diff(x)\n        * fps\n    )"),
        SRC.replace("fps=30.0", "fps = 30.0").replace("\n\n\n", "\n\n\n\n"),
        SRC.replace('"""Doc."""', "'''Doc.'''"),
    ],
)
def test_comment_docstring_and_format_edits_keep_the_hash(variant):
    assert codehash.normalised_source_hash(variant) == codehash.normalised_source_hash(SRC)


@pytest.mark.parametrize(
    "variant",
    [SRC.replace("fps=30.0", "fps=60.0"), SRC.replace("np.diff(x)", "np.gradient(x)"), SRC + "\nK = 1\n"],
)
def test_code_edits_change_the_hash(variant):
    assert codehash.normalised_source_hash(variant) != codehash.normalised_source_hash(SRC)


def test_empty_and_docstring_only_modules_are_equal():
    h = codehash.normalised_source_hash
    assert h("") == h('"""Only a docstring."""\n') == h("# a comment\n")
    assert h("def f():\n    pass\n") == h('def f():\n    """Doc."""\n')


def test_json_data_is_hashed_after_parsing():
    a = codehash.json_hash(b'{"a": 1, "b": [1, 2]}')
    assert a == codehash.json_hash(b'{\n  "b": [1,2],\n  "a": 1\n}\n')
    assert a != codehash.json_hash(b'{"a": 2, "b": [1, 2]}')


@pytest.mark.parametrize(
    "src, module, filename, want",
    [
        ("from . import x", "afv.models.camera", "camera.py", "afv.models"),
        ("from .workers import x", "afv.models.camera", "camera.py", "afv.models.workers"),
        ("from ..motion.fk import x", "afv.models.camera", "camera.py", "afv.motion.fk"),
        ("from .bvh import x", "afv.export", "__init__.py", "afv.export.bvh"),
        ("from .. import x", "afv.export", "__init__.py", "afv"),
        ("from ... import x", "afv.export", "__init__.py", None),
        ("from afv.core import types", "x", "x.py", "afv.core"),
    ],
)
def test_resolve_from(src, module, filename, want):
    node = ast.parse(src).body[0]
    pkg = codehash._package_of(module, Path(filename))
    assert codehash._resolve_from(node, pkg) == want


def test_real_stage_dependencies():
    from afv.models.camera import CameraVipe
    from afv.models.gemx import Pose3dGemx
    from afv.models.track import DetectTrackStage

    rec = codehash.stage_code(DetectTrackStage)
    assert rec is not None and rec["python"] == codehash.PYTHON
    files = set(rec["files"])
    assert {"afv.models.track", "afv.core.types", "src/afv/models/workers/track_worker.py"} <= files
    assert "afv.media.normalise" in files  # imported inside a function
    assert "afv.models" in files and "afv.media" in files  # parent packages' __init__.py
    assert not files & codehash.PLUMBING  # runner, project, envs ... are not part of any stage's code

    pose = set(codehash.stage_code(Pose3dGemx)["files"])
    assert {"src/afv/models/workers/gemx_worker.py", "src/afv/models/workers/soma_worker.py"} <= pose
    assert "src/afv/models/workers/vipe_worker.py" not in pose  # named in camera.py, a stage module
    cam = set(codehash.stage_code(CameraVipe)["files"])
    assert {"src/afv/models/workers/vipe_worker.py", "src/afv/models/workers/gemx_worker.py"} <= cam
    assert "src/afv/models/workers/soma_worker.py" not in cam  # imported inside gemx_worker's pose code

    from afv.export.stage import ExportStage
    from afv.motion.stages import PairStage

    assert "blender/afv_export.py" in codehash.stage_code(ExportStage)["files"]  # via a helper module
    files = set(codehash.stage_code(PairStage)["files"])
    assert {"afv.motion.pair", "afv.motion.stages", "src/afv/core/data/skeletons/soma77.json"} <= files


def _write_pkg(tmp_path, monkeypatch, stage_src: str, name: str):
    pkg = tmp_path / "pkg"
    (pkg / "workers").mkdir(parents=True, exist_ok=True)
    worker = "import helper_w\nimport numpy\n\ndef f():\n    import lazy_w\n"
    (pkg / "workers" / "my_worker.py").write_text(worker)
    (pkg / "workers" / "helper_w.py").write_text("X = 1\n")
    (pkg / "workers" / "lazy_w.py").write_text("Y = 1\n")
    (pkg / "workers" / "other_worker.py").write_text("Z = 1\n")
    (pkg / f"{name}.py").write_text(stage_src)
    monkeypatch.syspath_prepend(str(pkg))
    sys.modules.pop(name, None)
    return pkg, importlib.import_module(name)


def test_worker_scripts_and_their_sibling_imports(tmp_path, monkeypatch):
    src = "from afv.core.types import Tracks\n\nWORKER = 'my_worker.py'\n\nclass S:\n    pass\n"
    pkg, mod = _write_pkg(tmp_path, monkeypatch, src, "mystage_cw")
    labels = sorted(codehash.stage_code(mod.S)["files"])
    assert "mystage_cw" in labels and "afv.core.types" in labels
    assert any(lbl.endswith("my_worker.py") for lbl in labels)
    assert any(lbl.endswith("helper_w.py") for lbl in labels)  # module-level import
    assert not any(lbl.endswith("lazy_w.py") for lbl in labels)  # function-level import
    before = codehash.stage_code(mod.S)["hash"]
    (pkg / "workers" / "helper_w.py").write_text("X = 2\n")
    assert codehash.stage_code(mod.S)["hash"] != before
    sys.modules.pop("mystage_cw", None)


def test_declared_worker_scripts_and_code_deps(tmp_path, monkeypatch):
    src = (
        "WORKER = 'my_worker.py'\n\nclass S:\n    worker_scripts = ('other_worker.py',)\n"
        "    code_deps = ('afv.motion.retime', 'pkg_data.json')\n"
    )
    _, mod = _write_pkg(tmp_path, monkeypatch, src, "mystage_dw")
    monkeypatch.setattr(codehash, "REPO_ROOT", tmp_path)
    (tmp_path / "pkg_data.json").write_text('{"a": 1}')
    labels = codehash.stage_code(mod.S)["files"]
    assert any(k.endswith("other_worker.py") for k in labels)
    assert not any(k.endswith("my_worker.py") for k in labels)  # the declaration replaces the strings
    assert "afv.motion.retime" in labels and "pkg_data.json" in labels
    h = codehash.stage_code(mod.S)["hash"]
    (tmp_path / "pkg_data.json").write_text('{\n  "a": 1\n}\n')
    assert codehash.stage_code(mod.S)["hash"] == h  # JSON reformatting does not count
    (tmp_path / "pkg_data.json").write_text('{"a": 2}')
    assert codehash.stage_code(mod.S)["hash"] != h
    sys.modules.pop("mystage_dw", None)


def test_scan_cache_is_keyed_on_the_module(tmp_path):
    p = tmp_path / "m.py"
    p.write_text("from . import sib\n")
    assert "a.b" in codehash._scan(p, "a.b.m").imports
    assert "c" in codehash._scan(p, "c.m").imports


# ---------------------------------------------------------------------------
# Runner policy (code records controlled by the test)
# ---------------------------------------------------------------------------


@pytest.fixture
def code(monkeypatch):
    """{class name: version string} -> the code record the runner sees (on disk and as snapshot)."""
    versions: dict[str, str] = {}

    def fake(cls):
        v = versions.get(cls.__name__, "1")
        if v == "fail":
            return None
        files = {f"afv.fake.{cls.__name__.lower()}": v}
        return {"hash": f"{cls.__name__}:{v}", "files": files, "python": "3.12"}

    monkeypatch.setattr(runner_mod, "stage_code", fake)
    monkeypatch.setattr(runner_mod, "snapshot_code", fake)
    return versions


@pytest.fixture
def done(shot, fakes, do_pick, code):
    do_pick(shot)
    rep = run(shot, fakes)
    assert rep.outcome == "ok", rep.message
    return shot


def _drop_code_records(shot, names):
    for name in names:
        p = shot.stage_dir(name) / "meta.json"
        meta = json.loads(p.read_text())
        meta.pop("code")
        p.write_text(json.dumps(meta))


def test_meta_records_code(done):
    meta = done.meta("smooth")
    assert meta["code"] == {"hash": "Smooth:1", "files": {"afv.fake.smooth": "1"}, "python": "3.12"}
    assert meta["code_policy"] == "auto"
    assert "code" not in meta["key_parts"]  # compared apart from the key


def test_cpu_stage_reruns_on_code_change(done, fakes, code):
    code["Smooth"] = "2"
    plan = {s.name: (s.state, s.reason) for s in Runner(done).plan().stages}
    assert plan["smooth"] == ("stale", "code changed: afv.fake.smooth")
    rep = run(done, fakes)
    assert fakes.calls == ["smooth"]
    assert next(s.reason for s in rep.stages if s.name == "smooth") == "code changed: afv.fake.smooth"
    assert done.meta("smooth")["code"]["hash"] == "Smooth:2"
    run(done, fakes)
    assert fakes.calls == []


def test_cpu_rerun_with_same_output_keeps_downstream(done, fakes, code):
    code["Pose"] = "2"  # pose3d is a CPU fake here; its output does not change
    run(done, fakes)
    assert fakes.calls == ["pose3d"]


def test_model_stage_code_change_is_reported_not_rerun(done, fakes, code, monkeypatch):
    monkeypatch.setattr(fakes.Pose, "env", "gemx")
    code["Pose"] = "2"
    plan = {s.name: (s.state, s.reason) for s in Runner(done).plan().stages}
    state, reason = plan["pose3d"]
    assert state == "stale_code"
    assert reason == (
        "code changed: afv.fake.pose; result kept: --code-changes rerun (reuses worker caches) "
        "or --from pose3d"
    )
    assert plan["smooth"][0] == "fresh"  # downstream uses the kept result
    rep = run(done, fakes)
    assert fakes.calls == []
    assert next(s.state for s in rep.stages if s.name == "pose3d") == "stale_code"
    skip = next(e for e in rep.events if e["event"] == "stage_skip" and e["stage"] == "pose3d")
    assert skip["code_changed"] == "code changed: afv.fake.pose"

    run(done, fakes, code_changes="rerun")
    assert fakes.calls == ["pose3d"]  # same output: smooth stays fresh
    assert done.meta("pose3d")["code"]["hash"] == "Pose:2"


def test_from_reruns_a_model_stage_with_changed_code(done, fakes, code, monkeypatch):
    monkeypatch.setattr(fakes.Pose, "env", "gemx")
    code["Pose"] = "2"
    run(done, fakes, from_stage="pose3d")
    assert fakes.calls == ["pose3d", "smooth"]
    assert done.meta("pose3d")["code"]["hash"] == "Pose:2"


def test_accept_applies_to_model_stages_only(done, fakes, code, monkeypatch):
    monkeypatch.setattr(fakes.Pose, "env", "gemx")
    code["Pose"] = "2"
    code["Smooth"] = "2"
    out = done.meta("pose3d")["output_hash"]
    rep = run(done, fakes, code_changes="accept")
    assert fakes.calls == ["smooth"]  # a CPU stage with changed code always re-runs
    assert [e["stage"] for e in rep.events if e.get("code_accepted")] == ["pose3d"]
    meta = done.meta("pose3d")
    assert meta["code"]["hash"] == "Pose:2" and meta["output_hash"] == out
    assert meta["code_accepted"]["previous"] == "Pose:1"
    plan = Runner(done).plan(to="smooth").stages
    assert {(s.state, s.reason) for s in plan} == {("fresh", "")}


def test_accept_on_results_from_before_code_hashing(done, fakes, code, monkeypatch):
    monkeypatch.setattr(fakes.Pose, "env", "gemx")
    _drop_code_records(done, ("pose3d", "smooth"))
    rep = run(done, fakes, code_changes="accept")
    assert fakes.calls == ["smooth"]  # CPU: re-runs; model stage: the current code is recorded
    assert done.meta("pose3d")["code"]["hash"] == "Pose:1"
    assert done.meta("pose3d")["code_accepted"]["previous"] is None
    note = "code not recorded (result predates code hashing)"
    assert any(e.get("code_accepted") == note for e in rep.events)
    assert {(s.state, s.reason) for s in Runner(done).plan(to="smooth").stages} == {("fresh", "")}


def test_results_without_code_record(done, fakes, code, monkeypatch):
    """Results from before code hashing: CPU stages re-run, model stages stay fresh with a note."""
    monkeypatch.setattr(fakes.Pose, "env", "gemx")
    _drop_code_records(done, ("ingest", "detect_track", "camera", "pose3d", "smooth"))
    note = "code not recorded (result predates code hashing)"
    plan = {s.name: (s.state, s.reason) for s in Runner(done).plan(to="smooth").stages}
    assert plan["detect_track"] == ("stale", note)  # a CPU fake here
    assert plan["camera"][0] == "pending"  # until detect_track has re-run (same output, it turns out)
    run(done, fakes, code_changes="rerun")  # rerun does not apply to unrecorded model results
    assert fakes.calls == ["ingest", "detect_track", "camera:vipe", "smooth"]  # the CPU fakes
    plan = {s.name: (s.state, s.reason) for s in Runner(done).plan(to="smooth").stages}
    assert plan["pose3d"] == ("fresh", note)
    assert plan["smooth"] == ("fresh", "")


def test_python_version_change_is_named(done, fakes, code, monkeypatch):
    monkeypatch.setattr(fakes.Pose, "env", "gemx")
    p = done.stage_dir("pose3d") / "meta.json"
    meta = json.loads(p.read_text())
    meta["code"] = {**meta["code"], "python": "3.11", "hash": "other", "files": {"afv.x": "a", "afv.y": "b"}}
    p.write_text(json.dumps(meta))
    st = next(s for s in Runner(done).plan(to="smooth").stages if s.name == "pose3d")
    assert st.state == "stale_code" and st.reason.startswith("Python version changed (3.11 -> 3.12)")
    assert "afv.x" not in st.reason


def test_failed_hash_counts_as_changed(done, fakes, code, monkeypatch, caplog):
    code["Smooth"] = "fail"
    monkeypatch.setattr(fakes.Pose, "env", "gemx")
    code["Pose"] = "fail"
    plan = {s.name: (s.state, s.reason) for s in Runner(done).plan(to="smooth").stages}
    assert plan["smooth"] == ("stale", "code could not be hashed (counted as changed)")
    assert plan["pose3d"][0] == "stale_code" and plan["pose3d"][1].startswith("code could not be hashed")
    run(done, fakes, code_changes="accept")  # nothing to accept: the model result stays flagged
    assert fakes.calls == ["smooth"]
    assert next(s for s in Runner(done).plan(to="smooth").stages if s.name == "pose3d").state == "stale_code"


def test_code_policy_attribute_overrides_env(done, fakes, code, monkeypatch):
    monkeypatch.setattr(fakes.Smooth, "code_policy", "report", raising=False)
    code["Smooth"] = "2"
    assert next(s for s in Runner(done).plan().stages if s.name == "smooth").state == "stale_code"
    monkeypatch.setattr(fakes.Pose, "env", "gemx")
    monkeypatch.setattr(fakes.Pose, "code_policy", "auto", raising=False)
    code["Pose"] = "2"
    run(done, fakes)
    assert fakes.calls == ["pose3d"]


def test_cached_policy_reruns_for_module_changes_only(done, fakes, code, monkeypatch):
    """camera's default: its worker outputs are cached, so an afv-code change re-runs it (cheaply);
    a worker script change is only reported."""
    monkeypatch.setattr(fakes.CameraVipe, "env", "vipe")
    assert Runner.code_policy("camera", fakes.CameraVipe) == "cached"
    code["CameraVipe"] = "2"  # an afv module changed
    run(done, fakes)
    assert fakes.calls == ["camera:vipe"]

    plain = runner_mod.stage_code

    def worker_change(cls):
        rec = plain(cls)
        if cls is fakes.CameraVipe:
            files = {**rec["files"], "src/afv/models/workers/vipe_worker.py": "new"}
            rec = {**rec, "hash": "w", "files": files}
        return rec

    monkeypatch.setattr(runner_mod, "stage_code", worker_change)
    st = next(s for s in Runner(done).plan(to="smooth").stages if s.name == "camera")
    assert st.state == "stale_code" and "vipe_worker.py" in st.reason


def test_stale_reason_names_every_cause(done, fakes, code):
    code["Smooth"] = "2"
    plan = Runner(done, params={"smooth": {"strength": 0.7}}).plan()
    why = next(s.reason for s in plan.stages if s.name == "smooth")
    assert why == "params: strength, code changed: afv.fake.smooth"


def test_version_and_upstream_wording(done, fakes, monkeypatch):
    monkeypatch.setattr(fakes.Smooth, "version", "2")
    plan = {s.name: s.reason for s in Runner(done, params={"camera": {"focal": 5.0}}).plan().stages}
    assert plan["pose3d"] == "after camera re-runs"
    run(done, fakes, params={"camera": {"focal": 5.0}})
    rep = run(done, fakes, params={"camera": {"focal": 6.0}})
    reasons = {s.name: s.reason for s in rep.stages}
    assert reasons["pose3d"] == "upstream: camera"
    monkeypatch.setattr(fakes.Smooth, "version", "3")
    plan = {s.name: s.reason for s in Runner(done, params={"camera": {"focal": 6.0}}).plan().stages}
    assert plan["smooth"] == "version 2 -> 3"


def test_bad_code_changes_value(shot):
    with pytest.raises(RunError, match="code_changes"):
        Runner(shot, code_changes="sometimes")


# ---------------------------------------------------------------------------
# End to end with a real stage module on disk
# ---------------------------------------------------------------------------

STAGE_SRC = textwrap.dedent(
    '''
    """A CPU smooth stage written to disk by the test."""
    import pathlib

    from pydantic import BaseModel

    from afv.core.stage import register
    from afv.core.types import BodyMotion

    CALLS = []
    EDIT = []


    @register("disk")
    class DiskSmooth:
        name, version, env, inputs, licences, weight_hash = "smooth", "1", "cpu", ("pose3d",), (), ""

        class Params(BaseModel):
            strength: float = 0.5

        def run(self, ctx, inputs, params):
            CALLS.append("smooth")
            if EDIT:  # someone saves an edit to this file while the stage runs
                p = pathlib.Path(__file__)
                p.write_text(p.read_text().replace("fps=30.0", "fps=29.97"))
            m = inputs["pose3d"]
            return BodyMotion(frames=m.frames, timestamps=m.timestamps, skeleton=m.skeleton, fps=30.0,
                              actors=m.actors)
    '''
)


def _disk_stage(tmp_path, monkeypatch, registry, name):
    mod_dir = tmp_path / "stagemods"
    mod_dir.mkdir()
    path = mod_dir / f"{name}.py"
    path.write_text(STAGE_SRC)
    monkeypatch.syspath_prepend(str(mod_dir))
    sys.modules.pop(name, None)
    mod = importlib.import_module(name)
    registry["smooth"].pop("default")
    return path, mod


def _new_process(mod):
    """What a later afv process sees: fresh caches and the module as it is on disk now."""
    codehash.clear_cache()
    return importlib.reload(mod)


def test_editing_a_stage_file(tmp_path, monkeypatch, shot, fakes, do_pick, registry):
    path, mod = _disk_stage(tmp_path, monkeypatch, registry, "disk_smooth_cw")
    do_pick(shot)
    assert run(shot, fakes).outcome == "ok" and mod.CALLS == ["smooth"]

    path.write_text(STAGE_SRC.replace('"""A CPU', '"""Edited docs. A CPU') + "\n# trailing comment\n")
    mod = _new_process(mod)
    run(shot, fakes)
    assert mod.CALLS == []  # docstring/comment edits: still fresh

    path.write_text(STAGE_SRC.replace("fps=30.0", "fps=29.97"))
    mod = _new_process(mod)
    plan = {s.name: (s.state, s.reason) for s in Runner(shot).plan().stages}
    assert plan["smooth"] == ("stale", "code changed: disk_smooth_cw")
    run(shot, fakes)
    assert mod.CALLS == ["smooth"]
    assert next(s for s in Runner(shot).plan(to="smooth").stages if s.name == "smooth").state == "fresh"
    sys.modules.pop("disk_smooth_cw", None)


@pytest.mark.parametrize("forced", [False, True])
def test_code_edited_during_a_run_is_stale_afterwards(
    tmp_path, monkeypatch, shot, fakes, do_pick, registry, forced
):
    """The record stored is the code that ran (snapshotted at run start), not what is on disk when
    the result is written."""
    path, mod = _disk_stage(tmp_path, monkeypatch, registry, f"disk_edit_{int(forced)}")
    do_pick(shot)
    if forced:
        run(shot, fakes)
    mod.EDIT.append(True)
    run(shot, fakes, from_stage="smooth" if forced else None)
    mod.EDIT.clear()
    assert "fps=29.97" in path.read_text()
    st = next(s for s in Runner(shot).plan(to="smooth").stages if s.name == "smooth")
    assert (st.state, st.reason) == ("stale", f"code changed: disk_edit_{int(forced)}")
    sys.modules.pop(f"disk_edit_{int(forced)}", None)


def test_code_saved_since_afv_started_is_not_run(tmp_path, monkeypatch, shot, fakes, do_pick, registry):
    """afv batch: a file saved between two shots in one process. The stage is not run with the old
    code still loaded (its result would be stale at once); the shot stops, to be run again."""
    path, mod = _disk_stage(tmp_path, monkeypatch, registry, "disk_batch")
    do_pick(shot)
    run(shot, fakes)
    first = shot.meta("smooth")
    path.write_text(STAGE_SRC.replace("fps=30.0", "fps=29.97"))
    rep = run(shot, fakes, params={"smooth": {"strength": 0.9}})
    assert rep.outcome == "code_changed" and mod.CALLS == ["smooth"]  # not run again
    st = rep.stages[-1]
    assert (st.name, st.state) == ("smooth", "skipped") and "code changed since afv started" in st.reason
    assert any(e["event"] == "code_changed" for e in rep.events)
    assert shot.meta("smooth") == first  # the stored result is untouched
    mod = _new_process(mod)  # the next afv process runs it with the new code
    rep = run(shot, fakes, params={"smooth": {"strength": 0.9}})
    assert rep.outcome == "ok" and mod.CALLS == ["smooth"]
    sys.modules.pop("disk_batch", None)


def test_parent_packages_are_hashed_not_followed():
    from afv.models.gemx import Pose3dGemx

    files = set(codehash.stage_code(Pose3dGemx)["files"])
    assert {"afv.media", "afv.models", "afv.media.ffmpeg"} <= files  # parents hashed
    assert not {"afv.media.cuts", "afv.media.shots"} & files  # what afv/media/__init__.py imports


def test_default_worker_scripts_for_pose3d(tmp_path, monkeypatch):
    """pose3d's workers count even if its module stops naming them (soma_worker is imported only
    inside gemx_worker's functions)."""
    src = "class P:\n    name = 'pose3d'\n"
    _, mod = _write_pkg(tmp_path, monkeypatch, src, "mystage_pose")
    labels = codehash.stage_code(mod.P)["files"]
    assert {"src/afv/models/workers/gemx_worker.py", "src/afv/models/workers/soma_worker.py"} <= set(labels)
    sys.modules.pop("mystage_pose", None)


# ---------------------------------------------------------------------------
# Cache-only re-runs (code_policy "cached")
# ---------------------------------------------------------------------------


@pytest.fixture
def cam_cache(fakes, monkeypatch):
    """Fake camera as a model stage that honours Ctx.cache_only: {"hit": bool, "seen": [cache_only...]}."""
    from afv.core.stage import CacheOnlyMiss

    state = {"hit": True, "seen": []}
    orig = fakes.CameraVipe.run

    def run_(self, ctx, inputs, params):
        state["seen"].append(ctx.cache_only)
        if ctx.cache_only and not state["hit"]:
            raise CacheOnlyMiss("no cached vipe_worker.py output for this request")
        return orig(self, ctx, inputs, params)

    monkeypatch.setattr(fakes.CameraVipe, "run", run_)
    monkeypatch.setattr(fakes.CameraVipe, "env", "vipe")
    return state


def test_cached_rerun_is_cache_only(done, fakes, code, cam_cache):
    code["CameraVipe"] = "2"
    st = next(s for s in Runner(done).plan(to="smooth").stages if s.name == "camera")
    assert st.state == "stale" and st.reason.endswith("re-runs from its caches only")
    run(done, fakes)
    assert fakes.calls == ["camera:vipe"] and cam_cache["seen"] == [True]
    assert done.meta("camera")["code"]["hash"] == "CameraVipe:2"


def test_cached_rerun_miss_keeps_the_result(done, fakes, code, cam_cache):
    cam_cache["hit"] = False
    code["CameraVipe"] = "2"
    before = done.meta("camera")
    rep = run(done, fakes)
    assert rep.outcome == "ok" and fakes.calls == []  # nothing downstream re-ran
    st = next(s for s in rep.stages if s.name == "camera")
    assert st.state == "stale_code" and "needs a model re-run" in st.reason
    assert "--code-changes rerun or --from camera" in st.reason
    assert done.meta("camera") == before
    assert not list(done.work_root.glob("failed-*"))  # a miss is not a failure
    run(done, fakes, code_changes="rerun")  # an explicit re-run is a full one
    assert cam_cache["seen"][-1] is False and fakes.calls == ["camera:vipe"]


def test_camera_worker_cache_only_never_launches(tmp_path, monkeypatch):
    """The camera.py side of Ctx.cache_only: a WorkerCache miss raises instead of running a worker."""
    from afv.core.stage import CacheOnlyMiss
    from afv.models import camera

    def boom(*a, **k):
        raise AssertionError("a worker was launched in a cache-only run")

    monkeypatch.setattr(camera, "run_env_worker", boom)
    monkeypatch.setattr(camera, "env_fingerprint", lambda env: "fake-env")
    wc = camera.WorkerCache(tmp_path / "cache", {"frame_range": None}, "src", cache_only=True)
    with pytest.raises(CacheOnlyMiss, match=r"no cached vipe_worker\.py output"):
        wc.call("vipe", "vipe_worker.py", {"command": "vipe"}, tmp_path)


def test_stat_scope_and_ttl_reuse_stats(tmp_path, monkeypatch):
    f = tmp_path / "m.py"
    f.write_text("x = 1\n")
    calls = []
    real = Path.stat

    def counting(self, *a, **kw):
        if self == f:
            calls.append(1)
        return real(self, *a, **kw)

    monkeypatch.setattr(Path, "stat", counting)
    with codehash.stat_scope():
        h = codehash._scan(f, None).hash
        assert codehash._scan(f, None).hash == h
    assert len(calls) == 1  # one stat per scope
    codehash._scan(f, None)
    assert len(calls) == 2  # no scope, no TTL: stat again
    try:
        codehash.set_stat_ttl(60)
        codehash._scan(f, None)
        f.write_text("x = 2\n")
        assert codehash._scan(f, None).hash == h  # within the window the old stat is reused
        assert len(calls) == 3
        codehash.set_stat_ttl(0)
        assert codehash._scan(f, None).hash != h
    finally:
        codehash.set_stat_ttl(0)
