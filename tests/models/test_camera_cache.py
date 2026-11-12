"""The camera stage's worker-output cache (<shot>/.cache/camera) and the gravity re-level helper (CPU;
fake workers). The runner's re-level pass re-runs camera with ``up_cam0``: only the CPU part may run."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import pytest
from test_camera_flow import F, N, Q, _flow_handler, _r_w_cv, _standing_fighter, _vipe_handler
from test_camera_stages import FakeWorkers, _pitch_yaw, _rx, _tracks

from afv.core.stage import Ctx, StageError
from afv.models import camera as camera_mod
from afv.models.camera import (
    CameraParams,
    CameraStatic,
    CameraVipe,
    WorkerCache,
    camera_to_world_cv,
    gravity_disagrees,
    make_camera_track,
    pitch_only_up,
    relevel_override,
    validate_up_cam0,
)
from afv.models.gemx import body_motion_from_fk, camera_from_gemx, pose3d_summary


class Shot:
    """One shot folder with a real (dummy) source file and fake workers counting their calls."""

    def __init__(self, tmp: Path, monkeypatch, handlers: dict):
        self.tmp, self.n = tmp, 0
        self.video = tmp / "source.mp4"
        self.video.write_bytes(b"not really a video")
        self.fake = FakeWorkers(tmp, handlers)

        def call(env, script, request, work_dir, **kw):
            out = self.fake(env, script, request, work_dir, **kw)
            if request["command"] != "normalize":
                return out
            # like ffmpeg: the normalised video's bytes depend on the source
            Path(request["out_mp4"]).write_bytes(b"30Hz:" + Path(request["video"]).read_bytes())
            return {**out, "frame_count": N}

        monkeypatch.setattr(camera_mod, "run_env_worker", call)

    def run(self, boxes=None, deterministic=False, force=False, stage=CameraVipe, **params):
        self.n += 1
        work = self.tmp / f"work{self.n}"
        work.mkdir()
        ctx = Ctx(shot_dir=self.tmp / "shot", work_dir=work, deterministic=deterministic, force=force)
        before = len(self.fake.calls)
        p = CameraParams(video=str(self.video), **params)
        cam = stage().run(ctx, {"detect_track": _tracks(n=N, boxes=boxes)}, p)
        qc = json.loads((work / "qc.json").read_text())
        return cam, qc, [r["command"] for _, _, r in self.fake.calls[before:]]


@pytest.fixture
def shot(tmp_path, monkeypatch):
    mirrored = np.einsum("ab,nbc->nac", Q, _r_w_cv(-0.1))  # ViPE wrong: the flow camera is used
    return Shot(
        tmp_path, monkeypatch, {"vipe": _vipe_handler(mirrored), "flow": _flow_handler(np.full(N, F))}
    )


def test_relevel_pass_reruns_only_the_cpu_part(shot):
    cam1, qc1, calls1 = shot.run()
    assert calls1 == ["normalize", "vipe", "flow"]
    assert set(qc1["cache"].values()) == {"miss"}
    tilt = np.radians(3.0)  # the re-level: up 3 deg away from GeoCalib's, in camera 0's axes
    r0 = camera_mod.camera_to_world_cv(cam1)[0][0]
    up_cam0 = r0.T @ np.array([0.0, np.cos(tilt), np.sin(tilt)])
    cam2, qc2, calls2 = shot.run(up_cam0=list(up_cam0))
    assert calls2 == []  # ViPE, flow and the normalised video all come from the cache
    assert qc2["cache"] == {"normalize": "hit", "vipe": "hit", "flow": "hit"}
    assert qc2["gravity"] == "override" and qc2["rotation_source"] == "flow"
    # only the world was re-levelled: the same camera motion, pitched by the 3 degrees
    (r1, _), (r2, _) = camera_mod.camera_to_world_cv(cam1), camera_mod.camera_to_world_cv(cam2)
    rel1 = np.einsum("ba,tbc->tac", r1[0], r1)
    rel2 = np.einsum("ba,tbc->tac", r2[0], r2)
    assert np.allclose(rel1, rel2, atol=1e-9)
    p1, p2 = _pitch_yaw(cam1)[0], _pitch_yaw(cam2)[0]
    assert abs(p2[0] - p1[0]) == pytest.approx(3.0, abs=0.05)
    assert (shot.tmp / "work2" / "shot30.mp4").exists()  # outputs restored into the new work folder


def test_cache_is_never_stale(shot):
    shot.run()
    boxes = np.tile(np.array([[[600, 300, 650, 400]]], np.float32), (N, 1, 1))
    _, qc, calls = shot.run(boxes=boxes)  # new mask input: flow recomputed, ViPE (no boxes) reused
    assert calls == ["flow"] and qc["cache"]["vipe"] == "hit"
    _, _, calls = shot.run(pipeline="pose_only")  # a ViPE parameter
    assert calls == ["vipe"]
    _, _, calls = shot.run(deterministic=True)  # golden runs never reuse a non-golden GPU solve
    assert calls == ["vipe"]  # (normalising and the CPU flow tracks do not depend on it)
    # a replaced source (new size/mtime) misses everything
    shot.video.write_bytes(b"a different source video")
    os.utime(shot.video, ns=(1, 1))
    _, _, calls = shot.run()
    assert calls == ["normalize", "vipe", "flow"]
    # a corrupted entry is detected (sha256 per file), deleted and recomputed
    entries = [m.parent for m in (shot.tmp / "shot/.cache/camera").glob("*/manifest.json")]
    for e in entries:
        for f in e.iterdir():
            if f.suffix == ".npz":
                f.write_bytes(b"garbage")
    _, qc, calls = shot.run()
    assert calls == ["vipe", "flow"] and qc["cache"]["normalize"] == "hit"
    _, qc, calls = shot.run()
    assert calls == [] and set(qc["cache"].values()) == {"hit"}
    assert not list((shot.tmp / "shot/.cache/camera").glob(".tmp-*"))  # no half-written entries


def test_cache_can_be_switched_off(shot):
    shot.run(cache=False)
    _, qc, calls = shot.run(cache=False)
    assert calls == ["normalize", "vipe", "flow"] and qc["cache"] == "off"
    assert not (shot.tmp / "shot/.cache").exists()


def _qc(diff, height, up=(0.01, -0.99, 0.08)):
    return {
        "gravity": {"pitch_difference_deg": diff, "height_change_if_gemx_m": height, "gemx_up_cam0": list(up)}
    }


def test_relevel_override_decision():
    # Who Am I in the projects: +4.8 deg, +1.78 m -> re-level; UFC: -0.9 deg -> no
    assert relevel_override(_qc(4.8, 1.78)) == {"up_cam0": [0.01, -0.99, 0.08]}
    assert relevel_override(_qc(-4.0, -1.2)) is not None
    assert relevel_override(_qc(-0.88, -0.06)) is None
    assert relevel_override(_qc(4.0, 0.2)) is None  # big angle, close camera: height barely moves
    assert relevel_override(_qc(0.4, 1.0)) is None
    assert relevel_override({}) is None and relevel_override(None) is None
    assert relevel_override(_qc(5.0, 2.0, up=(0.0, 0.0, 0.0))) is None
    assert relevel_override(_qc(5.0, 2.0, up=(float("nan"), 1.0, 0.0))) is None
    assert gravity_disagrees(_qc(4.8, 1.78)["gravity"]) and not gravity_disagrees({"pitch_difference_deg": 9})


def test_force_recomputes_and_refreshes(shot):
    shot.run()
    _, qc, calls = shot.run(force=True)  # `afv run --from camera`: the runner sets Ctx.force
    assert calls == ["normalize", "vipe", "flow"] and set(qc["cache"].values()) == {"forced"}
    _, qc, calls = shot.run()  # the refreshed entries serve the next normal run
    assert calls == [] and set(qc["cache"].values()) == {"hit"}


def test_ffmpeg_and_environment_changes_miss(shot, monkeypatch):
    shot.run()
    monkeypatch.setattr(camera_mod, "ffmpeg_identity", lambda: "/other/ffmpeg:1:2")
    _, _, calls = shot.run()  # same normalised bytes from the (fake) new ffmpeg: ViPE/flow reused
    assert calls == ["normalize"]
    real = camera_mod.env_fingerprint
    monkeypatch.setattr(
        camera_mod, "env_fingerprint", lambda env: real(env) + (";upgraded" if env == "vipe" else "")
    )
    _, _, calls = shot.run()  # the vipe env changed: both of its workers re-run
    assert calls == ["vipe", "flow"]


def test_each_command_is_keyed_on_what_reaches_it(shot):
    shot.run()
    _, qc, calls = shot.run(stage=CameraStatic, gravity="level", focal_px=6000.0)  # vipe -> static
    assert calls == [] and qc["cache"] == {"normalize": "hit"}  # no re-normalising


def test_sweep_and_prune(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    old, new = root / ".tmp-old", root / ".tmp-new"
    old.mkdir(parents=True)
    new.mkdir()
    os.utime(old, (1, 1))
    WorkerCache(root, {}, "src")
    assert not old.exists() and new.exists()  # a killed run's folder is swept, a running one kept
    script_sha = camera_mod.file_sha256(camera_mod.WORKERS / "vipe_worker.py")
    for i in range(12):
        e = root / f"e{i:02d}"
        e.mkdir()
        material = {"command": "flow", "script_name": "vipe_worker.py", "script": script_sha if i else "old"}
        (e / "manifest.json").write_text(json.dumps({"material": material, "files": {}, "response": {}}))
        os.utime(e / "manifest.json", (1000 + i, 1000 + i))
    other = root / "gravity"
    other.mkdir()
    (other / "manifest.json").write_text(json.dumps({"material": {"command": "gravity"}, "files": {}}))
    cache = WorkerCache(root, {}, "src")
    material = {"command": "flow", "script_name": "vipe_worker.py", "script": script_sha}
    cache.prune(material, keep=root / "e01")
    left = sorted(p.name for p in root.iterdir() if not p.name.startswith("."))
    assert "e00" not in left  # made by another version of the worker script
    assert "e01" in left and "gravity" in left  # the entry just used, and other commands, stay
    assert sum(n.startswith("e") for n in left) == camera_mod.CACHE_KEEP_PER_COMMAND
    assert {"e11", "e10"} <= set(left)  # newest kept


def test_a_valid_entry_survives_a_failed_copy(tmp_path, monkeypatch):
    def worker(env, script, request, work_dir, **kw):
        Path(request["out_npz"]).write_bytes(b"result")
        return {"out_npz": request["out_npz"]}

    monkeypatch.setattr(camera_mod, "run_env_worker", worker)
    cache = WorkerCache(tmp_path / "cache", {}, "src")
    req = {"command": "gravity", "out_npz": str(tmp_path / "a.npz")}
    cache.call("vipe", "vipe_worker.py", req, tmp_path)
    key, material = cache.key("vipe", "vipe_worker.py", req)
    blocked = tmp_path / "blocked.npz"
    blocked.mkdir()  # the destination is a folder: copying out fails
    assert cache._load(tmp_path / "cache" / key, {**req, "out_npz": str(blocked)}, material) is None
    assert (tmp_path / "cache" / key / "manifest.json").exists()
    assert cache._load(tmp_path / "cache" / key, {**req, "out_npz": str(tmp_path / "b.npz")}, material)
    assert (tmp_path / "b.npz").read_bytes() == b"result"


def test_up_override_is_validated_and_pitch_only(shot):
    for bad in ([0.0, 0.0, 0.0], [1.0, float("nan"), 0.0], [0.0, 1.0, 0.0], [1.0, 2.0]):
        with pytest.raises(StageError):
            validate_up_cam0(bad)
    ref = validate_up_cam0([np.sin(np.radians(3.0)), -1.0, -0.05])  # GeoCalib: 3 deg of roll
    ovr = validate_up_cam0([np.sin(np.radians(-2.0)), -1.0, 0.09])  # GEM-X: other pitch and roll
    out = pitch_only_up(ovr, ref)
    pr = camera_mod._pitch_roll_deg
    assert pr(out)[0] == pytest.approx(pr(ovr)[0], abs=1e-6)  # GEM-X's pitch
    assert pr(out)[1] == pytest.approx(pr(ref)[1], abs=0.3)  # GeoCalib's roll
    cam1, _, _ = shot.run()
    r0 = camera_mod.camera_to_world_cv(cam1)[0][0]
    far = r0.T @ np.array([0.0, np.cos(np.radians(15.0)), np.sin(np.radians(15.0))])
    _, qc, _ = shot.run(up_cam0=list(far))
    assert "up_override_far_from_geocalib" in qc["summary"]["flags"]
    assert qc["up_override"]["angle_from_geocalib_deg"] == pytest.approx(15.0, abs=0.1)
    with pytest.raises(StageError):
        shot.run(up_cam0=[0.0, 1.0, 0.0])  # upside down


# ---------------------------------------------------------------------------------------------
# pose3d's gravity check and the full re-level round trip
# ---------------------------------------------------------------------------------------------


def _align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation taking unit vector a to unit vector b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def _with_gemx_up(fk, up_cv: np.ndarray):
    """``fk`` whose GEM-X gravity, in each camera frame (OpenCV), is ``up_cv`` [T,3]."""
    v = np.einsum("tba,tb->ta", fk.incam_rot[:, 0], up_cv)  # incam^T u
    g = fk.global_rot.copy()
    for t in range(len(g)):
        g[t, 0] = _align(np.array([0.0, 1.0, 0.0]), v[t]).T
    return dataclasses.replace(fk, global_rot=g)


def _true_up_cv(r_c2w: np.ndarray) -> np.ndarray:
    return np.einsum("tba,b->ta", r_c2w, [0.0, 1.0, 0.0])


def test_gravity_check_uses_the_same_frames_for_camera_and_gemx():
    fk = _standing_fighter(n=N)
    base = camera_from_gemx(fk)
    r0, c = camera_to_world_cv(base)
    tilt = np.stack([_rx(-16.0 * t / (N - 1)) for t in range(N)])  # the camera tilts 16 deg down
    r = np.einsum("tab,tbc->tac", tilt, r0)
    cam = make_camera_track(r, c, base.focal_px, width=base.width, height=base.height, cx=base.cx, cy=base.cy)
    observed = np.arange(N) >= N // 2  # the fighter is seen in the second half only
    fk = dataclasses.replace(_with_gemx_up(fk, _true_up_cv(r)), observed=observed)
    _, qc = body_motion_from_fk({"A": fk}, cam)
    g = qc["gravity"]
    assert abs(g["pitch_difference_deg"]) < 0.05  # whole-shot camera median vs observed GEM-X: -4 deg
    down = np.degrees(np.arcsin(-r[:, 1, 2]))
    assert g["camera_down_pitch_deg"] == pytest.approx(np.median(down[observed]), abs=0.01)
    assert g["frames_used"] == observed.sum() and relevel_override(qc) is None


def test_gravity_check_without_usable_frames_says_so():
    fk = dataclasses.replace(_standing_fighter(n=N), observed=np.zeros(N, bool))
    motion, qc = body_motion_from_fk({"A": fk}, camera_from_gemx(fk))
    g = qc["gravity"]
    assert g["pitch_difference_deg"] is None and "error" in g and "NaN" not in json.dumps(g)
    assert "gravity_unchecked" in pose3d_summary(motion, {**qc, "fighters": _qc_fighters(qc)})["flags"]
    assert relevel_override(qc) is None


def _qc_fighters(qc):
    return {
        r: {
            "reprojection_px_median": 1.0,
            "gravity_disagreement_deg_median": 1.0,
            "observed_fraction": 1.0,
            "unobserved_spans": [],
            **q,
        }
        for r, q in qc["fighters"].items()
    }


@pytest.mark.parametrize("err_deg", [5.0, -5.0])
def test_relevel_round_trip(tmp_path, monkeypatch, err_deg):
    """GeoCalib wrong in pitch; ViPE's world not camera 0 (Q); pose3d measures the signed pitch
    difference, relevel_override's up_cam0 brings it to ~0, and the camera moves the predicted way."""
    r_w = _r_w_cv(0.1)  # truth: camera pitched 8 deg down, panning
    r_v = np.einsum("ab,nbc->nac", Q, r_w)  # ViPE's world: rotated, so camera 0 is not identity
    frames = np.array([0, 13, 27, 39])
    wrong = _rx(err_deg)  # GeoCalib's up off in pitch (about camera x)

    def vipe(req):
        c2w = np.tile(np.eye(4), (N, 1, 1))
        c2w[:, :3, :3] = r_v
        up_cam = np.einsum("ab,nb->na", wrong, _true_up_cv(r_w)[frames])
        np.savez(
            req["out_npz"],
            c2w=c2w,
            intrinsics=np.tile([F, F, 640.0, 360.0], (N, 1)),
            up_cam=up_cam,
            up_frames=frames,
            up_weight=np.ones(4),
            up_focal=np.full(4, F),
        )
        return {"out_npz": req["out_npz"], "frames": N, "gravity": "geocalib"}

    shot = Shot(tmp_path, monkeypatch, {"vipe": vipe})
    fk = _with_gemx_up(_standing_fighter(n=N), _true_up_cv(r_w))  # GEM-X: the true gravity
    cam1, _, _ = shot.run(rotation_source="vipe")
    _, qc1 = body_motion_from_fk({"A": fk}, cam1)
    g1 = qc1["gravity"]
    assert abs(g1["pitch_difference_deg"]) == pytest.approx(5.0, abs=0.3)
    override = relevel_override(qc1)
    assert override is not None
    u = np.asarray(override["up_cam0"])
    r0 = r_v[0]
    assert np.degrees(np.arccos(np.clip((r0 @ u) @ (r0.T @ u), -1, 1))) > 10  # a frame mix-up would show
    cam2, qc_cam2, calls = shot.run(rotation_source="vipe", **override)
    assert calls == [] and qc_cam2["gravity"] == "override"
    _, qc2 = body_motion_from_fk({"A": fk}, cam2)
    g2 = qc2["gravity"]
    assert abs(g2["pitch_difference_deg"]) < 0.3 and relevel_override(qc2) is None
    dh = g2["camera_height_m"] - g1["camera_height_m"]
    assert np.sign(dh) == np.sign(g1["height_change_if_gemx_m"])
    assert abs(dh) == pytest.approx(abs(g1["height_change_if_gemx_m"]), rel=0.3)
    pitch2, _ = _pitch_yaw(cam2)
    assert np.allclose(pitch2, -8.0, atol=0.3)  # back to the true camera pitch


# ---------------------------------------------------------------------------------------------
# pose3d: fighters scaled about the camera centre onto the shared floor
# ---------------------------------------------------------------------------------------------


def _smaller(fk, k: float):
    """The same fighter seen as a k-times smaller person k-times closer (identical image)."""
    return dataclasses.replace(
        fk,
        incam_pos=fk.incam_pos * k,
        global_pos=fk.global_pos * k,
        tpose_pos=fk.tpose_pos * k,
        sole_y=fk.sole_y * k,
    )


@pytest.mark.parametrize(("k", "cap", "capped"), [(0.96, 0.05, False), (0.85, 0.05, True)])
def test_floor_scale_puts_both_fighters_on_the_floor_without_moving_the_image(k, cap, capped):
    a = _standing_fighter(n=N)
    fks = {"A": a, "B": _smaller(a, k)}
    cam = camera_from_gemx(a)
    plain, qc0 = body_motion_from_fk(fks, cam, floor_scale_max=0.0)
    scaled, qc1 = body_motion_from_fk(fks, cam, floor_scale_max=cap)
    gap0 = qc0["floor"]["fighter_floor_spread_m"]
    sc = qc1["floor"]["scale"]
    assert qc1["fighters"]["B"]["root_prior_alignment"]["before_scale"] is True  # stale: labelled
    assert gap0 > 0.05 and sc["B"]["capped"] == capped
    if not capped:
        assert qc1["floor"]["fighter_floor_spread_m"] < 0.01  # soles on one floor
        ratio = sc["B"]["scale"] / sc["A"]["scale"]
        assert ratio == pytest.approx(1 / k, rel=0.01)  # B's size restored relative to A
        assert sc["B"]["stature_after_m"] / sc["A"]["stature_after_m"] == pytest.approx(1.0, abs=0.01)
    else:
        assert abs(sc["B"]["scale"] - 1.0) <= cap + 1e-9 and sc["B"]["wanted"] > 1 + cap
        assert (
            "B_floor_scale_capped" in pose3d_summary(scaled, {**qc1, "fighters": _qc_fighters(qc1)})["flags"]
        )
    # every joint still projects where it did
    from afv.core.skeletons import get_skeleton
    from afv.models.camera import project, shifted_camera
    from afv.motion.fk import forward_kinematics

    sk = get_skeleton("soma77")
    for role in "AB":
        uv = []
        for motion in (plain, scaled):
            act = motion.actors[role]
            pos, _ = forward_kinematics(sk.with_offsets(act.rest_offsets), act.rot, act.root)
            uv.append(project(shifted_camera(cam, motion.world_shift), pos)[0])
        assert np.abs(uv[0] - uv[1]).max() < 0.05  # px


def test_floor_scale_is_off_for_a_fallback_world():
    a = _standing_fighter(n=N)
    _, qc = body_motion_from_fk(
        {"A": a, "B": _smaller(a, 0.96)}, camera_from_gemx(a), root_mode="gemx_world", floor_scale_max=0.05
    )
    assert "scale" not in qc["floor"]


def test_floor_scale_needs_real_contact_support():
    a = _standing_fighter(n=N)
    logits = np.full_like(a.contact_logits, -20.0)
    logits[:3, :4] = 20.0  # three planted frames x 4 channels = contact weight 12 ...
    few = dataclasses.replace(_smaller(a, 0.96), contact_logits=logits)
    logits2 = logits.copy()
    logits2[2, :4] = -20.0  # ... two frames = 8 < FLOOR_SCALE_MIN_CONTACT
    fewer = dataclasses.replace(few, contact_logits=logits2)
    _, qc = body_motion_from_fk({"A": a, "B": fewer}, camera_from_gemx(a))
    b = qc["floor"].get("scale", {}).get("B") or {}
    assert b.get("scale", 1.0) == 1.0 and "too little contact" in b.get("reason", "")
    _, qc = body_motion_from_fk({"A": a, "B": few}, camera_from_gemx(a))
    assert qc["floor"]["scale"]["B"]["scale"] != 1.0 and qc["floor"]["scale"]["B"]["contact_weight"] >= 10


def test_floor_scale_is_exact_when_the_camera_height_moves():
    a = _standing_fighter(n=N)
    base = camera_from_gemx(a)
    r, c = camera_to_world_cv(base)
    c = np.zeros((N, 3))
    c[:, 1] = np.linspace(0.0, 1.0, N)  # the camera rises 1 m over the shot (use_translation)
    cam = make_camera_track(r, c, base.focal_px, width=base.width, height=base.height, cx=base.cx, cy=base.cy)
    logits = np.full_like(a.contact_logits, -20.0)
    logits[N // 2 :, :4] = 20.0  # planted only while the camera is high: not at its median height
    b = dataclasses.replace(_smaller(a, 0.96), contact_logits=logits)
    a = dataclasses.replace(a, contact_logits=logits)
    _, qc = body_motion_from_fk({"A": a, "B": b}, cam, floor_scale_max=0.1)
    assert qc["floor"]["scale"]["B"]["scale"] != 1.0
    assert qc["floor"]["fighter_floor_spread_m"] < 0.002  # soles on one floor after scaling


def test_floor_scale_defaults_agree_and_are_bounded():
    import inspect

    from pydantic import ValidationError

    from afv.models import gemx

    default = inspect.signature(gemx.body_motion_from_fk).parameters["floor_scale_max"].default
    assert default == gemx.Pose3dParams().floor_scale_max == gemx.FLOOR_SCALE_MAX
    for bad in (-0.01, 0.2):
        with pytest.raises(ValidationError):
            gemx.Pose3dParams(floor_scale_max=bad)
