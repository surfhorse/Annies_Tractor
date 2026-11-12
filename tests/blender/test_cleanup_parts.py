"""The cleanup tools' bpy-free parts: edit files, the ``afv serve`` client (against a fake worker),
the one-shot fallback command lines, the take's tracking boxes and the overlay geometry."""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import pytest
from take_helpers import TESTS_DIR, addon_module, synthetic_shot

from afv import take as tk
from afv.core.types import Tracks

edits = addon_module("edits")
serve = addon_module("serve")
takeio = addon_module("takeio")
FAKE = TESTS_DIR / "fake_serve.py"


# --- edits --------------------------------------------------------------------------------------
def test_paint_contact_cuts_overlaps_and_keeps_unknown_keys():
    data = {"version": 3, "spans": [
        {"role": "A", "start": 10, "end": 30, "state": "plant", "foot": "left", "note": "mine"},
        {"role": "A", "start": 0, "end": 50, "state": "free", "foot": "right"},
        {"role": "B", "start": 0, "end": 50, "state": "plant", "foot": "left"},
        {"role": "A", "start": 0, "end": 50, "state": "plant", "channels": ["LeftFoot"]},
    ]}
    before = json.dumps(data)
    out = edits.paint_contact(data, "A", "left", 15, 20, "free")
    assert json.dumps(data) == before  # the input is not changed
    assert out["version"] == 3
    left_a = [s for s in out["spans"] if s.get("role") == "A" and s.get("foot") == "left"]
    assert [(s["start"], s["end"], s["state"]) for s in left_a] == [(10, 15, "plant"), (20, 30, "plant"),
                                                                    (15, 20, "free")]
    assert all(s.get("note") == "mine" for s in left_a[:2])
    assert {json.dumps(s) for s in data["spans"][1:]} <= {json.dumps(s) for s in out["spans"]}
    cleared = edits.paint_contact(out, "A", "left", 0, 100, "clear")
    assert not [s for s in cleared["spans"] if s.get("role") == "A" and s.get("foot") == "left"]
    assert edits.paint_contact({}, "B", "right", 3, 4, "plant") == {
        "spans": [{"role": "B", "start": 3, "end": 4, "state": "plant", "foot": "right"}]}
    with pytest.raises(edits.EditError):
        edits.paint_contact({}, "A", "left", 5, 5, "plant")
    with pytest.raises(edits.EditError):
        edits.paint_contact({}, "A", "middle", 1, 5, "plant")


def test_hands_spans_split_both():
    d = edits.mark_hands({}, "A", "both", 10, 30)
    d = edits.mark_hands(d, "A", "left", 15, 20, off=False)  # left hand back on inside
    got = sorted((s["start"], s["end"], s["side"]) for s in d["spans"])
    assert got == [(10, 15, "both"), (15, 20, "right"), (20, 30, "both")]
    d2 = edits.mark_hands(d, "A", "right", 0, 100, off=False)
    assert sorted((s["start"], s["end"], s["side"]) for s in d2["spans"]) == [(10, 15, "left"),
                                                                                (20, 30, "left")]
    assert edits.mark_hands(d, "B", "both", 0, 5)["spans"][-1]["role"] in ("A", "B")


def test_pins_and_swaps():
    pin = {"role": "A", "frame": 40, "radius": 6, "preset": "unity_humanoid",
           "bones": {"LeftArm": [0.0, 0.0, 0.0, 1.0]}, "root": None}
    d = edits.add_pin({"other": 1}, pin)
    d = edits.add_pin(d, {**pin, "frame": 10})
    right = {"RightArm": [0.0, 0.0, 1.0, 0.0]}
    d = edits.add_pin(d, {**pin, "radius": 9, "bones": right, "root": [1, 2, 3]})
    assert [(p["frame"], p["radius"]) for p in d["pins"]] == [(10, 6), (40, 9)] and d["other"] == 1
    assert d["pins"][1]["bones"] == {"LeftArm": [0.0, 0.0, 0.0, 1.0], "RightArm": [0.0, 0.0, 1.0, 0.0]}
    d = edits.add_pin(d, {**pin, "bones": {"LeftArm": [1.0, 0.0, 0.0, 0.0]}})  # same frame: merged
    assert d["pins"][1]["bones"]["LeftArm"] == [1.0, 0.0, 0.0, 0.0] and "RightArm" in d["pins"][1]["bones"]
    assert d["pins"][1]["root"] == [1, 2, 3]  # kept: the new pin has none
    d, gone = edits.delete_pin(d, 0)
    assert gone["frame"] == 10 and len(d["pins"]) == 1
    assert edits.pin_range({"frame": 3, "radius": 8}, 0, 100) == (0, 12)
    assert edits.pin_range({"frame": 98, "radius": 8}, 0, 100) == (90, 101)
    with pytest.raises(edits.EditError):
        edits.add_pin({}, {**pin, "bones": {}})
    s = edits.add_swap({}, 40, (2, 1))
    assert s == {"swaps": [{"start": 40, "end": None, "objects": [2, 1]}]}
    with pytest.raises(edits.EditError):
        edits.add_swap({}, 40, (1, 1))
    assert edits.delete_swap(s, 0)[0] == {"swaps": []}


def test_validate_mirrors_the_workers_checks(tmp_path):
    ok = {"spans": [{"role": "A", "start": 1, "end": 3, "state": "plant", "foot": "left"}], "x": 1}
    edits.validate("contacts", ok)
    bad = [
        ("contacts", {"spans": [{"role": "C", "start": 1, "end": 3, "state": "plant", "foot": "left"}]}),
        ("contacts", {"spans": [{"role": "A", "start": 3, "end": 3, "state": "plant", "foot": "left"}]}),
        ("contacts", {"spans": [{"role": "A", "start": 1, "end": 3, "state": "plant"}]}),
        ("hands", {"spans": [{"role": "A", "start": 1, "end": 3, "side": "up"}]}),
        ("pins", {"pins": [{"role": "A", "frame": 1, "preset": "p", "bones": {"X": [0, 0, 0, 3]}}]}),
        ("pins", {"pins": [{"role": "A", "frame": -1, "preset": "p", "bones": {"X": [0, 0, 0, 1]}}]}),
        ("pins", {"pins": [{"role": "A", "frame": 1, "bones": {"X": [0, 0, 0, 1]}}]}),
        ("ids", {"swaps": [{"start": 4, "end": None, "objects": [2, 2]}]}),
        ("ids", {"swaps": [{"start": 4, "end": 4, "objects": [1, 2]}]}),
    ]
    for name, data in bad:
        with pytest.raises(edits.EditError):
            edits.validate(name, data)
    with pytest.raises(edits.EditError):
        edits.write_local(tmp_path, "hands", bad[3][1])
    assert not edits.edit_path(tmp_path, "hands").exists()
    edits.write_local(tmp_path, "contacts", ok)
    edits.write_local(tmp_path, "contacts", {"spans": []})  # top-level keys left out are kept
    assert edits.read(tmp_path, "contacts") == {"spans": [], "x": 1}
    edits.write_local(tmp_path, "contacts", None)
    assert not edits.edit_path(tmp_path, "contacts").exists()


def test_history(tmp_path):
    assert edits.history(tmp_path) == [] and edits.drop_history(tmp_path, "x") is None
    pins = {"pins": []}
    for i in range(edits.HISTORY_MAX + 3):
        edits.push_history(tmp_path, {"id": f"e{i}", "label": f"fix {i}", "name": "pins", "before": None,
                                      "after_hash": edits.state_hash(pins)})
    h = edits.history(tmp_path)
    assert len(h) == edits.HISTORY_MAX and h[-1]["label"] == f"fix {edits.HISTORY_MAX + 2}"
    assert edits.history_path(tmp_path) == tmp_path / ".addon" / "history.json"
    assert not (tmp_path / "edits" / edits.OLD_HISTORY).exists()
    edits.write_local(tmp_path, "pins", pins)
    edits.revert_check(tmp_path, h[-1])
    assert edits.entry_matches(tmp_path, h[-1])
    pin = {"role": "A", "frame": 1, "preset": "p", "root": [0, 0, 0]}
    edits.write_local(tmp_path, "pins", {"pins": [pin]})
    with pytest.raises(edits.EditError, match="changed since"):
        edits.revert_check(tmp_path, h[-1])
    assert edits.drop_history(tmp_path, h[-1]["id"])["label"] == h[-1]["label"]
    assert len(edits.history(tmp_path)) == len(h) - 1
    size = edits.history_path(tmp_path).stat().st_size
    assert size < 200 * edits.HISTORY_MAX  # a before-snapshot and a hash, not two full files
    assert edits.canonical("mixamorig:Hips", "mixamorig:") == "Hips"
    assert edits.canonical("Hips", "x:") == "Hips"


def test_history_migrates_the_old_file(tmp_path):
    old = tmp_path / "edits" / edits.OLD_HISTORY
    old.parent.mkdir()
    after = {"spans": []}
    fix = {"label": "paint", "name": "contacts", "before": None, "after": after}
    old.write_text(json.dumps({"fixes": [fix]}))
    (h,) = edits.history(tmp_path)
    assert not old.exists() and edits.history_path(tmp_path).is_file()
    assert "after" not in h and h["after_hash"] == edits.state_hash(after)


def test_settings_revert(tmp_path):
    shot = {"settings": {"params": {"smooth": {"strength": 0.75, "region_strength": {"legs": 2}}}}}
    (tmp_path / "shot.json").write_text(json.dumps(shot))
    before = edits.shot_smoothing(tmp_path)
    assert before["smooth.strength"] == 0.75 and before["smooth.region_strength.legs"] == 2
    assert before["smooth.region_strength.arms"] is None  # not set at shot level
    sets = edits.revert_sets(before)
    assert "smooth.strength=0.75" in sets and "smooth.region_strength.arms=null" in sets
    assert "smooth.region_strength.legs=2" in sets and len(sets) == 7
    entry = {"label": "smoothing applied", "settings_before": before, "after_hash": edits.state_hash(before)}
    assert edits.entry_matches(tmp_path, entry)
    shot["settings"]["params"]["smooth"]["strength"] = 0.9  # changed elsewhere since
    (tmp_path / "shot.json").write_text(json.dumps(shot))
    with pytest.raises(edits.EditError, match="smoothing settings changed since"):
        edits.revert_check(tmp_path, entry)


SPAN = {"role": "A", "start": 1, "end": 3, "state": "plant", "foot": "left"}
PIN = {"role": "A", "frame": 1, "preset": "p", "bones": {"X": [0, 0, 0, 1]}}
DIFF_CASES = [  # the review's 28 cases and a few more
    ("contacts", {"spans": [{**SPAN, "start": 1.0}]}),
    ("contacts", {"spans": [{**SPAN, "start": "1"}]}),
    ("contacts", {"spans": [{**SPAN, "end": 3.5}]}),
    ("contacts", {"spans": [{**SPAN, "start": True}]}),
    ("contacts", {"spans": [{**SPAN, "foot": None, "channels": [1, 2]}]}),
    ("contacts", {"spans": [{**SPAN, "foot": None, "channels": ["LeftFoot"]}]}),
    ("contacts", {"spans": [{**SPAN, "foot": None}]}),
    ("contacts", {"spans": [SPAN], "note": float("nan")}),
    ("contacts", {"spans": None}),
    ("contacts", {}),
    ("contacts", {"spans": [{**SPAN, "start": -1}]}),
    ("contacts", {"spans": [{**SPAN, "state": "hover"}]}),
    ("contacts", {"spans": [{**SPAN, "foot": "middle"}]}),
    ("contacts", {"spans": [{**SPAN, "role": "C"}]}),
    ("hands", {"spans": [{"role": "A", "start": 1, "end": 3}]}),
    ("hands", {"spans": [{"role": "B", "start": 0, "end": 1, "side": "both", "x": [1]}]}),
    ("pins", {"pins": [{**PIN, "radius": 2.0}]}),
    ("pins", {"pins": [{**PIN, "radius": -1}]}),
    ("pins", {"pins": [{**PIN, "bones": {"X": [0, 0, 0, "1"]}}]}),
    ("pins", {"pins": [{**PIN, "bones": {"X": [0, 0, 0, True]}}]}),
    ("pins", {"pins": [{**PIN, "bones": {}, "root": [0, 1, 2]}]}),
    ("pins", {"pins": [{**PIN, "bones": {}}]}),
    ("pins", {"pins": [{**PIN, "root": [0, 1]}]}),
    ("pins", {"pins": [{**PIN, "preset": None}]}),
    ("pins", {"pins": [{**PIN, "frame": "5"}]}),
    ("pins", {"pins": [{**PIN, "bones": None}]}),
    ("pins", {"pins": [{**PIN, "bones": {"X": [0, 0, 0, 3]}}]}),
    ("pins", {"pins": [{**PIN, "root_motion": "world", "prefix": "", "root_rotation": [0, 0, 0, 1]}]}),
    ("ids", {"swaps": [{"start": 1, "objects": [1, 2]}]}),
    ("ids", {"swaps": [{"start": 1, "end": None, "objects": [0, 2]}]}),
    ("ids", {"swaps": [{"start": 1, "end": None, "objects": [1.0, 2]}]}),
    ("ids", {"swaps": [{"start": 1, "end": None, "objects": [1, 2, 3]}]}),
    ("ids", {"swaps": [{"start": 4, "end": 4, "objects": [1, 2]}]}),
    ("fighters", {"nonsense": 1}),
    ("bogus", {}),
]


@pytest.mark.parametrize(("name", "data"), DIFF_CASES)
def test_validation_agrees_with_the_worker(name, data):
    """The add-on's fallback validation and afv.core.edits (what afv serve uses) accept and refuse
    the same payloads."""
    from afv.core import edits as core

    def verdict(fn):
        try:
            fn(name, data)
            return "ok"
        except Exception as e:  # both raise their own EditError
            return f"refused ({type(e).__name__})"

    addon, worker = verdict(edits.validate), verdict(core.validate_edit)
    assert (addon == "ok") == (worker == "ok"), f"add-on {addon}, worker {worker}"


def test_bone_sides():
    assert edits.foot_side("LeftToeBase") == "left" and edits.foot_side("mixamorig:RightUpLeg") == "right"
    assert edits.foot_side("afv_RightFoot", "afv_") == "right"
    assert edits.foot_side("LeftHand") is None and edits.foot_side("Hips") is None
    assert edits.hand_side("RightForeArm") == "right" and edits.hand_side("LeftHandIndex2") == "left"
    assert edits.hand_side("LeftLeg") is None


def test_write_local_is_atomic_and_readable(tmp_path):
    p = edits.write_local(tmp_path, "contacts", {"spans": []})
    assert p == tmp_path / "edits" / "contacts.json" and edits.read(tmp_path, "contacts") == {"spans": []}
    assert [q.name for q in p.parent.iterdir()] == ["contacts.json"]  # no temp file left
    assert edits.read(tmp_path, "pins") == {}
    p.write_text("{nope")
    with pytest.raises(edits.EditError, match="not valid JSON"):
        edits.read(tmp_path, "contacts")


def test_basis_to_local_inverts_the_builders_basis():
    rng = np.random.default_rng(1)
    rest = rng.normal(size=(20, 4))
    rest /= np.linalg.norm(rest, axis=-1, keepdims=True)
    local = rng.normal(size=(20, 4))
    local /= np.linalg.norm(local, axis=-1, keepdims=True)
    conj = rest * np.array([1, -1, -1, -1])
    basis = edits._qmul(edits._qmul(conj, local), rest)  # the builder: basis = R^-1 local R
    got = edits.basis_to_local(rest, basis)  # xyzw
    want = np.concatenate([local[:, 1:], local[:, :1]], axis=-1)
    err = np.minimum(np.abs(got - want).max(-1), np.abs(got + want).max(-1))
    assert err.max() < 1e-12 and np.all(got[:, 3] >= 0)


def test_smooth_settings_and_sets(tmp_path):
    (tmp_path / "shot.json").write_text(json.dumps(
        {"settings": {"params": {"smooth": {"strength": 0.5, "region_strength": {"arms": 0.25}}}}}))
    s = edits.smooth_settings(tmp_path)
    assert s["strength"] == 0.5
    assert s["region_strength"]["arms"] == 0.25 and s["region_strength"]["legs"] == 1.0
    sets = edits.smooth_sets(0.5, {"arms": 0.25})
    assert sets[0] == "smooth.strength=0.5" and "smooth.region_strength.arms=0.25" in sets
    assert len(sets) == 1 + len(edits.REGIONS)


# --- the one-shot fallback -----------------------------------------------------------------------
def test_cli_args():
    base = {"project": "/mnt/d/p", "shot": "001"}
    assert serve.cli_args("run", {**base, "from": "contacts", "to": "hands_face", "frames": "10:70",
                                  "sets": ["smooth.strength=0.5"], "yes": True}) == [
        "run", "001", "--project", "/mnt/d/p", "--from", "contacts", "--to", "hands_face",
        "--frames", "10:70",
        "--set", "smooth.strength=0.5", "--yes"]
    assert serve.cli_args("take", {**base, "preset": "godot"}) == ["take", "001", "--project", "/mnt/d/p",
                                                                   "--preset", "godot"]
    assert serve.cli_args("export", {**base, "preset": "unreal", "yes": False})[-2:] == ["--preset", "unreal"]
    with pytest.raises(ValueError):
        serve.cli_args("edit", base)
    assert serve.frames_arg(10, 70) == "10:70"
    argv = serve.serve_argv("Ubuntu", "~/afv/afv-env/bin/afv")
    assert argv[:5] == ["wsl.exe", "-d", "Ubuntu", "-e", "bash"]
    assert argv[-1].endswith("~/afv/afv-env/bin/afv serve")
    with pytest.raises(ValueError):
        serve.serve_argv("Ubuntu", "$HOME/afv")


# --- the serve client against the fake worker ---------------------------------------------------
def _client(tmp_path, **cfg):
    log = tmp_path / "serve_log.jsonl"
    cfgp = tmp_path / f"cfg_{len(list(tmp_path.glob('cfg_*')))}.json"
    cfgp.write_text(json.dumps({"log": str(log), **cfg}))
    return serve.ServeClient(lambda: [sys.executable, str(FAKE), str(cfgp)], ping_timeout=20), log


def _wait(req, timeout=20.0):
    req.join(timeout)
    assert req.finished, "no reply"
    req.drain()
    assert req.done
    return req


def _log(log):
    return [json.loads(x) for x in log.read_text().splitlines()]


def test_serve_requests_progress_and_edit(tmp_path):
    client, log = _client(tmp_path)
    try:
        assert client.status() == "not started"
        proj = tmp_path / "proj"
        req = client.submit("edit", {"project": str(proj), "shot": "001", "name": "contacts",
                                     "data": {"spans": []}}, "edit")
        assert client.wait_ready() and client.confirmed
        _wait(req)
        written = proj / "shots" / "001" / "edits" / "contacts.json"
        assert req.ok and json.loads(written.read_text()) == {"spans": []}
        run = _wait(client.submit("run", {"project": str(proj), "shot": "001", "from": "smooth",
                                          "to": "hands_face", "frames": "3:9"}, "run"))
        assert run.ok and run.result()["outcome"] == "ok"
        kinds = [e["event"] for e in run.events]
        assert kinds[-1] == "result" and "progress" in kinds
        assert run.result()["stages"][0]["range"] == "3:9"
        cmds = [r["cmd"] for r in _log(log)]
        assert cmds == ["ping", "edit", "run"] and len({r["id"] for r in _log(log)}) == 3
        assert "running" in client.status()
    finally:
        client.shutdown(wait=5)
    assert not client.running


def test_fake_worker_validates_edits(tmp_path):
    client, _log_path = _client(tmp_path)
    try:
        spans = [{"role": "A", "start": 5, "end": 2, "side": "left"}]
        base = {"project": str(tmp_path), "shot": "1", "name": "hands"}
        bad = _wait(client.submit("edit", {**base, "data": {"spans": spans}}, "edit"))
        assert not bad.ok and "end is exclusive" in bad.failure()
        gone = _wait(client.submit("edit", {**base, "data": None}, "edit"))
        assert gone.ok and gone.result()["removed"]
    finally:
        client.shutdown(wait=5)


def test_serve_error_review_and_cancel(tmp_path):
    client, log = _client(tmp_path, fail_run="contacts: boom")
    try:
        r = _wait(client.submit("run", {"project": "/x", "shot": "1"}, "run"))
        assert not r.ok and r.failure() == "contacts: boom" and not r.unavailable
    finally:
        client.shutdown(wait=5)
    client, log = _client(tmp_path, review_once=True, run_s=30)
    try:
        r = _wait(client.submit("run", {"project": "/x", "shot": "1"}, "run"))
        assert not r.ok and r.review()["hints"] == [{"kind": "swap"}] and "id swap" in r.review()["message"]
        slow = client.submit("run", {"project": "/x", "shot": "1", "yes": True}, "run")
        time.sleep(0.5)
        slow.cancel()
        _wait(slow)
        assert slow.cancelled and not slow.ok and slow.failure() == "cancelled"
        sent = _log(log)
        assert sent[-1]["cmd"] == "cancel" and sent[-1]["args"] == {"target_id": slow.id}
        assert slow.elapsed < 10
    finally:
        client.shutdown(wait=5)


def test_serve_restarts_after_it_died(tmp_path):
    client, _log_path = _client(tmp_path, die_after=3)  # ping, run, then dies on the next request
    try:
        assert _wait(client.submit("run", {"project": "/x", "shot": "1"}, "a")).ok
        dead = _wait(client.submit("run", {"project": "/x", "shot": "1"}, "b"))
        assert not dead.ok and not dead.unavailable and "stopped (exit 9)" in dead.failure()
        for _ in range(100):
            if not client.running:
                break
            time.sleep(0.05)
        again = _wait(client.submit("run", {"project": "/x", "shot": "1"}, "c"))  # a new worker
        assert again.ok and client.restarts == 1
    finally:
        client.shutdown(wait=5)


def test_serve_unavailable_and_restart_limit(tmp_path):
    no_serve = "import sys; print('no serve here'); sys.exit(2)"
    client = serve.ServeClient(lambda: [sys.executable, "-c", no_serve])
    r = _wait(client.submit("run", {"project": "/x", "shot": "1"}, "run"))
    assert r.unavailable and "could not start (exit 2)" in r.failure() and "no serve here" in r.failure()
    with pytest.raises(serve.ServeError, match="cannot start"):
        serve.ServeClient(lambda: [str(tmp_path / "missing.exe")]).submit("ping", {}, "p")
    starts = 1  # the request above started the first worker
    for _ in range(serve.MAX_RESTARTS + 3):
        try:
            _wait(client.submit("run", {"project": "/x", "shot": "1"}, "run"))
            starts += 1
        except serve.ServeError as e:
            assert "stopped 4 times" in str(e)
            break
    assert starts == serve.MAX_RESTARTS + 1 == 4  # a start and 3 restarts a minute, then it gives up


def test_serve_exits_when_the_client_goes(tmp_path):
    client, _log_path = _client(tmp_path)
    _wait(client.submit("ping", {}, "p"))
    proc = client._proc
    proc.stdin.close()  # Blender quit: the pipe closes
    assert proc.wait(timeout=10) == 0


# --- tracking boxes in the take ------------------------------------------------------------------
def test_take_tracks_and_overlay_geometry(tmp_path):
    bm, cam = synthetic_shot(n_frames=12, width=640, height=360)
    n = 14  # detect_track has two frames more than the motion (and starts one earlier)
    boxes = np.full((n, 3, 4), np.nan, np.float32)
    for f in range(n):
        boxes[f, 0] = (100 + f, 50, 300 + f, 650)  # source pixels (1280 x 720)
        boxes[f, 1] = (700, 40, 900, 700)
        if f % 2:
            boxes[f, 2] = (500, 10, 600, 300)
    tracks = Tracks(frames=np.arange(-1, n - 1), timestamps=np.zeros(n), track_ids=np.array([1, 2, 3]),
                    boxes=boxes, scores=np.ones((n, 3), np.float32), fighters={"A": 1, "B": 2})
    tb = tk.tracks_on_frames(tracks, bm.frames, (0.5, 0.5), (640, 360), {3: "referee"})
    inp = tk.TakeInputs(motion=bm, camera=cam, shot_id="007", project_dir="/x/proj", tracks=tb)
    tj = tk.write_take(tmp_path / "take", inp)
    meta = json.loads(tj.read_text())
    assert meta["tracks"]["ids"] == [1, 2, 3] and meta["tracks"]["fighters"] == {"A": 1, "B": 2}
    assert meta["tracks"]["labels"] == {"3": "referee"} and meta["tracks"]["image_size"] == [640, 360]
    assert meta["tracks"]["npz"] in tk.package_files(meta)
    td = takeio.load_tracks(tj)
    assert td.ids == [1, 2, 3] and td.role_of(2) == "B" and td.role_of(3) is None
    assert td.labels[3] == "referee"
    at0 = dict(td.boxes_at(0))  # take frame 0 = detect_track frame index 1 (odd: the referee is there)
    assert at0[1] == pytest.approx((50.5, 25.0, 150.5, 325.0)) and 3 in at0
    assert 3 not in dict(td.boxes_at(1))
    assert td.boxes_at(999) == []
    old = tk.write_take(tmp_path / "take_old", tk.TakeInputs(motion=bm, camera=cam, shot_id="007",
                                                             project_dir="/x/proj"))
    assert takeio.load_tracks(old) is None and json.loads(old.read_text())["tracks"] is None

    corners = ((10.0, 410.0), (650.0, 410.0), (10.0, 50.0))  # region: y up, the image 640 x 360
    assert takeio.image_to_region(corners, (640, 360), 0, 0) == (10.0, 410.0)
    assert takeio.image_to_region(corners, (640, 360), 640, 360) == (650.0, 50.0)
    assert takeio.image_to_region(corners, (640, 360), 320, 90) == (330.0, 320.0)
    region = [(1, (0, 100, 50, 0)), (2, (20, 60, 30, 40)), (3, (200, 0, 300, 100))]
    assert takeio.box_at(region, 25, 50) == 2  # the smaller of two overlapping boxes
    assert takeio.box_at(region, 5, 5) == 1 and takeio.box_at(region, 150, 50) is None
    assert takeio.inside_frame(corners, 330, 320) and not takeio.inside_frame(corners, 5, 320)
    assert not takeio.inside_frame(corners, 330, 420)


def test_strip_bars():
    spans = [{"role": "B", "start": 5, "end": 9, "severity": "bad"},
             {"role": "A", "start": 0, "end": 0, "severity": "warn"},
             {"role": "A", "start": 7, "end": 12, "severity": "warn"}]
    bars = takeio.strip_bars(spans, ["A", "B"])
    assert bars == [(0, 0.0, 1.0, "warn"), (0, 7.0, 13.0, "warn"), (1, 5.0, 10.0, "bad")]
