"""Blender-side checks for the retarget engine tests (``test_retarget_engine*.py``), run headless:

    blender -b --factory-startup --disable-autoexec -noaudio --python retarget_checks.py -- job.json

job.json: {"result_path", "addon_parent", "mode", ...}

- synthetic: {"take", "take_prefixed"}: synthetic target rigs built in the take's scene (Rigify-style
  ``.L/.R`` names in a 60 deg A-pose with a branching head, moved and turned 30 deg; copies turned 10
  and 90 deg and one parented under a turned empty; an anatomical centimetre rig with a forward
  ``Sacrum``, twist bones mid-chain, a Copy Rotation knee like Girush's, unnamed fingers and an
  existing action; Unity names under a 0.01-scaled object facing +Y with a root bone and no Spine1;
  a 3ds Max Biped), all with odd rolls, mixed rotation modes and a skinned mesh; Blender's real Rigify
  metarig and a generated Rigify rig; plus the refusals (non-uniform scale, lying rig, excluded source,
  contradicting map, generated Rigify), a swapped map, an error while writing, the map files, a tweak
  key, an unreachable lock, a source without plant masks, a prefixed source with a root bone and a
  frame subset.
- character: {"take", "character", "facing", "runs": [{"role", "lock_feet"}]}: imports the
  character (FBX, or appends the objects of a .blend), guess_map, one retarget per run, then a tweak
  key on fighter A and one more free run.

Every retarget is measured here from the evaluated scene, independently of the module's maths and
report (``measure``: facing and hips scale recomputed, penetration on the evaluated foot mesh, toe
tip and heel), and the target's state is compared before/after (``snapshot``).
"""

from __future__ import annotations

import itertools
import json
import math
import os
import random
import sys
import tempfile
import time
import traceback

import bpy
from mathutils import Matrix, Vector

# --- the canonical chain, written out again here (independent of retarget.py) -----------------------
TAIL = {"Hips": "Spine", "Spine": "Spine1", "Spine1": "Spine2", "Spine2": "Neck", "Neck": "Head",
        "Head": "HeadTop_End"}
PARENT = {"Spine": "Hips", "Spine1": "Spine", "Spine2": "Spine1", "Neck": "Spine2", "Head": "Neck",
          "HeadTop_End": "Head"}
for _s in ("Left", "Right"):
    _chain = [f"{_s}Shoulder", f"{_s}Arm", f"{_s}ForeArm", f"{_s}Hand", f"{_s}HandMiddle1"]
    PARENT[f"{_s}Shoulder"] = "Spine2"
    for _a, _b in itertools.pairwise(_chain):
        TAIL[_a] = _b
        if _b != f"{_s}HandMiddle1":
            PARENT[_b] = _a
    for _f in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
        PARENT[f"{_s}Hand{_f}1"] = f"{_s}Hand"
        for _k in (1, 2, 3):
            TAIL[f"{_s}Hand{_f}{_k}"] = f"{_s}Hand{_f}{_k + 1}"
            PARENT[f"{_s}Hand{_f}{_k + 1}"] = f"{_s}Hand{_f}{_k}"
    _leg = [f"{_s}UpLeg", f"{_s}Leg", f"{_s}Foot", f"{_s}ToeBase", f"{_s}Toe_End"]
    PARENT[f"{_s}UpLeg"] = "Hips"
    for _a, _b in itertools.pairwise(_leg):
        TAIL[_a] = _b
        PARENT[_b] = _a
FEET = {f"{s}{b}" for s in ("Left", "Right") for b in ("Foot", "ToeBase", "Toe_End")}
AXES = {"+X": Vector((1, 0, 0)), "-X": Vector((-1, 0, 0)), "+Y": Vector((0, 1, 0)), "-Y": Vector((0, -1, 0)),
        "+Z": Vector((0, 0, 1)), "-Z": Vector((0, 0, -1))}
FPS = 30.0


def canon_order(names):
    def depth(c):
        d = 0
        while c in PARENT:
            c, d = PARENT[c], d + 1
        return d
    return sorted(names, key=lambda c: (depth(c), c))


def mapped_child(c, mapped):
    n = TAIL.get(c)
    while n is not None and n not in mapped:
        n = TAIL.get(n)
    return n


def source_canonical(src):
    prefix = src["afv_hips"][:-4]
    return {b.name[len(prefix):]: b.name for b in src.data.bones if b.name.startswith(prefix)}


def world_forward(obj, axis):
    """``obj``'s forward in world (horizontal unit) for a forward ``axis`` in its armature axes."""
    v = obj.matrix_world.to_3x3().normalized() @ AXES[axis]
    v.z = 0.0
    return v.normalized()


def descendants(obj, name):
    root = obj.data.bones[name]
    return [root.name, *[b.name for b in root.children_recursive]]


def skinned(obj):
    """Meshes of the view layer with an enabled Armature modifier on ``obj`` (the module's rule)."""
    bpy.context.view_layer.update()
    return [o for o in bpy.context.view_layer.objects if o.type == "MESH" and any(
        m.type == "ARMATURE" and m.object == obj and m.show_viewport for m in o.modifiers)]


def foot_vertices(obj, fam):
    """{mesh object: (vertex indices of its evaluated mesh weighted >= 25 % to the bones of ``fam``,
    their rest world z)}; rest = evaluated with the armature in its rest position (modifiers before
    the Armature one included)."""
    meshes = skinned(obj)
    out = {}
    old = obj.data.pose_position
    obj.data.pose_position = "REST"
    bpy.context.view_layer.update()
    try:
        dg = bpy.context.evaluated_depsgraph_get()
        for m in meshes:
            names = {g.index: g.name for g in m.vertex_groups}
            ev = m.evaluated_get(dg)
            ids, zs = [], []
            for v in ev.data.vertices:
                ws = [(names.get(g.group), g.weight) for g in v.groups]
                tot = sum(w for _, w in ws)
                if tot > 0 and sum(w for n, w in ws if n in fam) / tot >= 0.25:
                    ids.append(v.index)
                    zs.append((ev.matrix_world @ v.co).z)
            if ids:
                out[m] = (ids, zs)
    finally:
        obj.data.pose_position = old
        bpy.context.view_layer.update()
    return out


def _min_z(mobj):
    """Lowest world z of ``mobj``'s evaluated mesh."""
    import numpy as np

    ob = mobj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    co = np.empty(3 * len(ob.data.vertices))
    ob.data.vertices.foreach_get("co", co)
    mw = np.array(ob.matrix_world)
    return float((co.reshape(-1, 3) @ mw[2, :3]).min() + mw[2, 3])


def _zs(mobj, ids):
    ob = mobj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    co = [0.0] * (3 * len(ob.data.vertices))
    ob.data.vertices.foreach_get("co", co)
    mw = ob.matrix_world
    return [(mw @ Vector(co[3 * i:3 * i + 3])).z for i in ids]


# --- measurement ----------------------------------------------------------------------------------
def _angle_deg(q):
    return math.degrees(2.0 * math.atan2(Vector(q[1:]).length, abs(q[0])))


def _twist_deg(q, axis):
    tw = 2.0 * math.atan2(Vector(q[1:]).dot(axis), q[0])
    tw = (tw + math.pi) % (2 * math.pi) - math.pi
    return abs(math.degrees(tw))


def measure(scene, src, tgt, bone_map, rep, facing_axis, frames=None):
    """Errors of the evaluated target against the evaluated source. The facing yaw and the hips scale
    are recomputed here (the target's forward is ``facing_axis`` in its armature axes; the take
    armature's is +Z), penetration is measured on the evaluated foot mesh, toe tip and heel."""
    bpy.context.view_layer.update()
    f0, f1 = rep["frames"] if frames is None else frames
    sb = source_canonical(src)
    pairs = canon_order([c for c in bone_map if c in sb and bone_map[c] in tgt.data.bones])
    mapped = set(pairs)
    ik = set(rep.get("ik_bones", []))  # the foot lock's IK'd legs: not measured against the source
    fs, ft = world_forward(src, "+Z"), world_forward(tgt, facing_axis)
    yaw_rad = math.atan2(ft.cross(fs).z, ft.dot(fs))
    yaw = Matrix.Rotation(yaw_rad, 3, "Z")
    s_rest = {c: src.matrix_world @ src.data.bones[sb[c]].matrix_local for c in sb}
    t_rest = {c: tgt.matrix_world @ tgt.data.bones[bone_map[c]].matrix_local for c in pairs}
    child, align, d_s = {}, {}, {}
    for c in pairs:
        k = mapped_child(c, mapped)
        child[c] = k
        if c == "Hips" or c in FEET:
            align[c] = yaw
        elif k is not None:
            dt = t_rest[k].translation - t_rest[c].translation
            ds = s_rest[k].translation - s_rest[c].translation
            d_s[c] = ds.normalized()
            align[c] = (yaw @ dt).rotation_difference(ds).to_matrix() @ yaw
        else:
            p = PARENT.get(c)
            while p is not None and p not in align:
                p = PARENT.get(p)
            align[c] = align[p] if p else yaw
    # the feet: every bone below each mapped foot, its skinned vertices, toe tips, heel
    fams = {s: descendants(tgt, bone_map[f"{s}Foot"]) for s in ("Left", "Right") if f"{s}Foot" in mapped}
    verts = {s: foot_vertices(tgt, fam) for s, fam in fams.items()}
    mw = tgt.matrix_world
    rest_pts = [mw @ tgt.data.bones[n].head_local for fam in fams.values() for n in fam]
    rest_pts += [mw @ tgt.data.bones[n].tail_local for fam in fams.values() for n in fam]
    floor = min([p.z for p in rest_pts] + [z for vs in verts.values() for _ids, zs in vs.values()
                                           for z in zs])
    all_meshes = skinned(tgt)
    tgt.data.pose_position = "REST"
    bpy.context.view_layer.update()
    whole_floor = min((_min_z(m) for m in all_meshes), default=None)
    tgt.data.pose_position = "POSE"
    bpy.context.view_layer.update()
    whole = []
    heel = {}
    for s in fams:
        ank = t_rest[f"{s}Foot"]
        h = ank.translation.copy()
        h.z = floor
        heel[s] = ank.inverted() @ h  # in the foot bone's frame (world rest)
    legs = ("LeftUpLeg", "RightUpLeg")
    anchor = legs if all(c in mapped for c in legs) else ("Hips",)

    def mid(d):
        return sum((d[c].translation for c in anchor), Vector()) / len(anchor)

    a_s_rest, a_t_rest = mid(s_rest), mid(t_rest)
    k_auto = (a_t_rest.z - floor) / a_s_rest.z
    k = rep["hips_scale"] if rep.get("hips_scale_given") else k_auto
    size = 100.0 / k
    s_heel = {}
    for s in ("Left", "Right"):
        if f"{s}Foot" in sb:
            h = s_rest[f"{s}Foot"].translation.copy()
            h.z = 0.0
            s_heel[s] = s_rest[f"{s}Foot"].inverted() @ h
    plant_props = {kk[len("afv_plant_"):]: kk for kk in src.keys() if kk.startswith("afv_plant_")}
    worst = {}
    dir_err = twist_err = rot_err = hips_rot = feet_rot = anchor_err = 0.0
    out_sides = {s: {"ankle": [], "toe_pos": [], "pen": [], "src_depth": [], "foot": [], "toe": []}
                 for s in fams}
    for f in range(f0, f1 + 1):
        scene.frame_set(f)
        dg = bpy.context.evaluated_depsgraph_get()
        es, et = src.evaluated_get(dg), tgt.evaluated_get(dg)
        ws = {c: es.matrix_world @ es.pose.bones[sb[c]].matrix for c in sb}
        wt = {c: et.matrix_world @ et.pose.bones[bone_map[c]].matrix for c in pairs}
        plants = {ch: float(es.get(prop, 0.0)) > 0.5 for ch, prop in plant_props.items()}
        a_goal = a_t_rest + k * (mid(ws) - a_s_rest)
        anchor_err = max(anchor_err, (mid(wt) - a_goal).length * size)
        for c in pairs:
            if bone_map[c] in ik:
                continue
            d_src = ws[c].to_quaternion() @ s_rest[c].to_quaternion().inverted()
            rel = wt[c].to_quaternion() @ (align[c].to_quaternion() @ t_rest[c].to_quaternion()).inverted()
            e = d_src.inverted() @ rel
            q = (e.w, e.x, e.y, e.z)
            kc = child[c]
            if c == "Hips":
                hips_rot = max(hips_rot, _angle_deg(q))
            elif c in FEET:
                feet_rot = max(feet_rot, _angle_deg(q))
                worst[c] = max(worst.get(c, 0.0), _angle_deg(q))
            elif kc is None:
                rot_err = max(rot_err, _angle_deg(q))
                worst[c] = max(worst.get(c, 0.0), _angle_deg(q))
            else:
                v_t = wt[kc].translation - wt[c].translation
                v_s = ws[kc].translation - ws[c].translation
                a = math.degrees(v_t.angle(v_s, 0.0))
                t = _twist_deg(q, d_s[c])
                dir_err = max(dir_err, a)
                twist_err = max(twist_err, t)
                worst[c] = max(worst.get(c, 0.0), a, t)
        if whole_floor is not None:
            low = min(_min_z(m) for m in all_meshes)
            whole.append(round((whole_floor - low) * size, 4))
        for s, fam in fams.items():
            pts = [et.matrix_world @ et.pose.bones[n].head for n in fam]
            pts += [et.matrix_world @ et.pose.bones[n].tail for n in fam]
            pts.append(wt[f"{s}Foot"] @ heel[s])
            z = min([p.z for p in pts] + [zz for m, (ids, _zr) in verts[s].items() for zz in _zs(m, ids)])
            o = out_sides[s]
            o["pen"].append(round((floor - z) * size, 4))
            o["ankle"].append([round(x * size, 4) for x in wt[f"{s}Foot"].translation])
            if f"{s}ToeBase" in mapped:
                o["toe_pos"].append([round(x * size, 4) for x in wt[f"{s}ToeBase"].translation])
            sp = []
            for b in ("Foot", "ToeBase", "Toe_End"):
                if s + b in sb:
                    pb = es.pose.bones[sb[s + b]]
                    sp += [es.matrix_world @ pb.head, es.matrix_world @ pb.tail]
            if s in s_heel:
                sp.append(ws[f"{s}Foot"] @ s_heel[s])
            o["src_depth"].append(round(max(0.0, -min(p.z for p in sp)) * 100.0, 4))
            o["foot"].append(plants.get(f"{s}Foot", False))
            o["toe"].append(plants.get(f"{s}ToeBase", False))
    slide = {}
    for s in ("Left", "Right"):
        if s not in out_sides:
            slide[s.lower()] = None
            continue
        o = out_sides[s]
        pf, pt = o["foot"], o["toe"]
        if not o["toe_pos"]:  # no toe bone: a planted toe holds the ankle
            pf, pt = [a or b for a, b in zip(pf, pt, strict=True)], [False] * len(pf)
        speeds = []
        for i in range(len(pf) - 1):
            if pf[i] and pf[i + 1]:
                key = "ankle"
            elif pt[i] and pt[i + 1] and not pf[i] and not pf[i + 1]:
                key = "toe_pos"
            else:
                continue
            speeds.append((Vector(o[key][i + 1]) - Vector(o[key][i])).length * FPS)
        speeds.sort()
        h = len(speeds) // 2
        med = (speeds[h] if len(speeds) % 2 else 0.5 * (speeds[h - 1] + speeds[h])) if speeds else None
        slide[s.lower()] = med
        slide[s.lower() + "_frames"] = len(speeds)
    top = sorted(worst.items(), key=lambda kv: -kv[1])[:4]
    return {"dir_err_deg": dir_err, "twist_err_deg": twist_err, "rot_err_deg": rot_err,
            "hips_rot_deg": hips_rot, "feet_rot_deg": feet_rot, "anchor_err_cm": anchor_err,
            "slide_cm_s": slide, "pen_cm": max((max(o["pen"]) for o in out_sides.values()), default=0.0),
            "sides": out_sides, "whole_mesh_pen": whole, "yaw_deg": math.degrees(yaw_rad), "k_auto": k_auto,
            "floor": floor,
            "mesh_feet": any(verts.values()), "worst": [[c, round(v, 4)] for c, v in top],
            "mapped": len(pairs)}


# --- state snapshots (acceptance 5) ------------------------------------------------------------------
def _r(v, nd=6):
    return [round(float(x), nd) for x in v]


def action_hash(a):
    out = []
    for layer in a.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                for fc in bag.fcurves:
                    co = [0.0] * (2 * len(fc.keyframe_points))
                    fc.keyframe_points.foreach_get("co", co)
                    out.append([fc.data_path, fc.array_index, len(fc.keyframe_points),
                                round(sum(co), 6), round(sum(x * x for x in co), 4)])
    return sorted(out)


def snapshot(scene, tgt):
    """The target's persistent state, the scene's selection, active object and modes (an appended
    .blend's matrix_world is stale until the first depsgraph update, hence the update)."""
    vl = bpy.context.view_layer
    vl.update()
    meshes = []
    for o in scene.objects:
        if o.type != "MESH" or not (o.parent == tgt or any(m.type == "ARMATURE" and m.object == tgt
                                                           for m in o.modifiers)):
            continue
        n = len(o.data.vertices)
        co = [0.0] * (3 * n)
        o.data.vertices.foreach_get("co", co)
        meshes.append([o.name, [_r(r) for r in o.matrix_basis], [_r(r) for r in o.matrix_parent_inverse],
                       o.parent_type, o.parent_bone, n, round(sum(co), 4),
                       round(sum(x * x for x in co), 3), [(m.type, m.object.name if getattr(m, "object", None)
                                                          else None, m.show_viewport) for m in o.modifiers],
                       sorted(g.name for g in o.vertex_groups), o.parent.name if o.parent else None])
    ad = tgt.animation_data
    anim = ("REPLACE", 1.0, "HOLD", True, False) if ad is None else (
        ad.action_blend_type, round(ad.action_influence, 6), ad.action_extrapolation, ad.use_nla,
        ad.use_tweak_mode)
    active = vl.objects.active
    return {
        "object": [[_r(r) for r in tgt.matrix_world], _r(tgt.location), _r(tgt.rotation_euler),
                   _r(tgt.rotation_quaternion), _r(tgt.scale), tgt.rotation_mode,
                   tgt.parent.name if tgt.parent else None],
        "rest": {b.name: [_r(b.head_local), _r(b.tail_local), [_r(r) for r in b.matrix_local],
                          b.parent.name if b.parent else None, b.use_connect] for b in tgt.data.bones},
        "pose_modes": {pb.name: pb.rotation_mode for pb in tgt.pose.bones},
        "constraints": {pb.name: [c.type for c in pb.constraints] for pb in tgt.pose.bones if pb.constraints},
        "meshes": sorted(meshes),
        "actions": {a.name: action_hash(a) for a in bpy.data.actions},
        "fake_users": {a.name: a.use_fake_user for a in bpy.data.actions},
        "nla": [[t.name, t.mute, [s.name for s in t.strips]] for t in ad.nla_tracks] if ad else [],
        "anim": list(anim),
        "frame": [scene.frame_current, round(scene.frame_subframe, 6)],
        "objects": sorted(o.name for o in scene.objects),
        "selected": sorted(o.name for o in scene.objects if o.select_get()),
        "active": active.name if active else None,
        "modes": [tgt.mode, active.mode if active else None],
    }


def compare(before, after, new_action):
    """What changed apart from the new action (and its assignment): a list of strings, empty = ok."""
    bad = [key for key in ("object", "rest", "pose_modes", "constraints", "meshes", "nla", "anim", "frame",
                           "objects", "selected", "active", "modes") if before[key] != after[key]]
    for name, h in before["actions"].items():
        if after["actions"].get(name) != h:
            bad.append(f"action {name}")
    extra = set(after["actions"]) - set(before["actions"])
    if extra != {new_action}:
        bad.append(f"new actions {sorted(extra)}")
    for name, fake in before["fake_users"].items():
        if after["fake_users"].get(name) != fake and not after["fake_users"].get(name):
            bad.append(f"fake user dropped on {name}")
    return bad


def run(scene, src, tgt, bone_map, facing_axis, **kw):
    before = snapshot(scene, tgt)
    from afv_addon import retarget as rt

    t = time.perf_counter()
    rep = rt.retarget(scene, src, tgt, bone_map, **kw)
    wall = time.perf_counter() - t
    after = snapshot(scene, tgt)
    act = bpy.data.actions[rep["action"]]
    ad = tgt.animation_data
    bags = act.layers[0].strips[0].channelbags
    keys = sorted({len(fc.keyframe_points) for bag in bags for fc in bag.fcurves})
    info = {"report": rep, "wall_s": wall, "changed": compare(before, after, rep["action"]),
            "assigned": ad.action == act, "fake_user": act.use_fake_user, "key_counts": keys,
            "frame_after": after["frame"],
            "fake_users_set": sorted(n for n, f in after["fake_users"].items()
                                     if f and not before["fake_users"].get(n, True))}
    if kw.get("hips_scale") is not None:
        rep = {**rep, "hips_scale_given": True}
    info["measure"] = measure(scene, src, tgt, bone_map, rep, facing_axis)
    return info


def brief(info):
    """A run without its per-frame arrays (for results that do not need them)."""
    m = {k: v for k, v in info["measure"].items() if k != "sides"}
    return {**info, "measure": m}


# --- synthetic rigs -----------------------------------------------------------------------------------
def V(*a):
    return Vector(a)


def joints(arm_deg):
    j = {"hips": V(0, 0, 0.95), "spine": V(0, 0.005, 1.03), "spine1": V(0, 0.01, 1.15),
         "spine2": V(0, 0.01, 1.28), "neck": V(0, 0.0, 1.50), "neck_mid": V(0, -0.005, 1.55),
         "head": V(0, -0.01, 1.60), "headtop": V(0, -0.01, 1.80)}
    a = math.radians(arm_deg)
    for side, s in (("L", 1.0), ("R", -1.0)):
        d = V(s * math.cos(a), 0, -math.sin(a))
        j["clav" + side] = V(s * 0.03, 0, 1.45)
        j["arm" + side] = V(s * 0.17, 0, 1.45)
        j["elbow" + side] = j["arm" + side] + 0.28 * d
        j["wrist" + side] = j["elbow" + side] + 0.25 * d
        j["handend" + side] = j["wrist" + side] + 0.08 * d
        fingers = (("thumb", -0.035), ("index", -0.025), ("middle", -0.008), ("ring", 0.008),
                   ("pinky", 0.024))
        for name, y in fingers:
            base = j["wrist" + side] + (0.02 if name == "thumb" else 0.085) * d + V(0, y, 0)
            seg = (d + V(0, -0.7, 0)).normalized() if name == "thumb" else d
            for k in range(4):
                j[f"{name}{k}{side}"] = base + 0.03 * k * seg
        j["thigh" + side] = V(s * 0.10, 0, 0.92)
        j["knee" + side] = V(s * 0.105, -0.015, 0.50)
        j["knee2" + side] = j["knee" + side] + V(0, 0.002, -0.05)
        j["ankle" + side] = V(s * 0.11, 0.02, 0.08)
        j["toe" + side] = V(s * 0.11, -0.10, 0.02)
        j["toeend" + side] = V(s * 0.11, -0.17, 0.02)
    return j


def _mid(a, b):
    return (a + b) * 0.5


def style_rigify(j):
    b = [("spine", None, j["hips"], j["spine"]), ("spine.001", "spine", j["spine"], j["spine1"]),
         ("spine.002", "spine.001", j["spine1"], j["spine2"]),
         ("spine.003", "spine.002", j["spine2"], j["neck"]),
         ("spine.004", "spine.003", j["neck"], j["neck_mid"]),
         ("spine.005", "spine.004", j["neck_mid"], j["head"]),
         ("spine.006", "spine.005", j["head"], j["headtop"]),
         # a branching head like the real metarig: face -> eyes, jaw
         ("face", "spine.006", j["head"] + V(0, -0.02, 0.02), j["head"] + V(0, -0.08, 0.02)),
         ("jaw", "face", j["head"] + V(0, -0.03, 0.0), j["head"] + V(0, -0.09, -0.03))]
    exp = {"Hips": "spine", "Spine": "spine.001", "Spine1": "spine.002", "Spine2": "spine.003",
           "Neck": "spine.004", "Head": "spine.006"}
    for side, s, S in (("L", 1, "Left"), ("R", -1, "Right")):
        x = "." + side
        b += [("eye" + x, "face", j["head"] + V(s * 0.03, -0.08, 0.07), j["head"] + V(s * 0.03, -0.1, 0.07)),
              ("pelvis" + x, "spine", j["hips"], j["hips"] + V(s * 0.1, -0.05, 0.1)),
              ("breast" + x, "spine.003", j["spine2"] + V(s * 0.1, -0.05, 0),
               j["spine2"] + V(s * 0.1, -0.15, 0)),
              ("shoulder" + x, "spine.003", j["clav" + side], j["arm" + side]),
              ("upper_arm" + x, "shoulder" + x, j["arm" + side], j["elbow" + side]),
              ("forearm" + x, "upper_arm" + x, j["elbow" + side], j["wrist" + side]),
              ("hand" + x, "forearm" + x, j["wrist" + side], j["handend" + side]),
              ("thigh" + x, "spine", j["thigh" + side], j["knee" + side]),
              ("shin" + x, "thigh" + x, j["knee" + side], j["ankle" + side]),
              ("foot" + x, "shin" + x, j["ankle" + side], j["toe" + side]),
              ("toe" + x, "foot" + x, j["toe" + side], j["toeend" + side]),
              ("heel.02" + x, "foot" + x, j["ankle" + side] + V(-0.03, 0.05, -0.06),
               j["ankle" + side] + V(0.03, 0.05, -0.06))]
        exp.update({f"{S}Shoulder": "shoulder" + x, f"{S}Arm": "upper_arm" + x,
                    f"{S}ForeArm": "forearm" + x, f"{S}Hand": "hand" + x, f"{S}UpLeg": "thigh" + x,
                    f"{S}Leg": "shin" + x, f"{S}Foot": "foot" + x, f"{S}ToeBase": "toe" + x})
        for k in (1, 2, 3):
            parent = "hand" + x if k == 1 else f"thumb.0{k - 1}{x}"
            b.append((f"thumb.0{k}{x}", parent, j[f"thumb{k - 1}{side}"], j[f"thumb{k}{side}"]))
            exp[f"{S}HandThumb{k}"] = f"thumb.0{k}{x}"
        for i, name in enumerate(("index", "middle", "ring", "pinky"), start=1):
            b.append((f"palm.0{i}{x}", "hand" + x, j["wrist" + side], j[f"{name}0{side}"]))
            for k in (1, 2, 3):
                b.append((f"f_{name}.0{k}{x}", f"palm.0{i}{x}" if k == 1 else f"f_{name}.0{k - 1}{x}",
                          j[f"{name}{k - 1}{side}"], j[f"{name}{k}{side}"]))
                exp[f"{S}Hand{name.capitalize()}{k}"] = f"f_{name}.0{k}{x}"
    return b, exp


def style_anatomical(j):
    sacrum_head = j["spine"] + V(0, 0.07, 0)  # behind the spine: the bone points forward
    a = _mid(j["spine"], j["spine1"])
    b = [("Sacrum", None, sacrum_head, j["spine"]), ("Sacrum.001", "Sacrum", j["spine"], j["hips"]),
         ("Spine1", "Sacrum", j["spine"], a), ("Spine1.001", "Spine1", a, j["spine1"]),
         ("Spine2", "Spine1.001", j["spine1"], j["spine2"]), ("Spine3", "Spine2", j["spine2"], j["neck"]),
         ("Spine3.002", "Spine3", j["neck"], j["head"]), ("head", "Spine3.002", j["head"], j["headtop"]),
         ("genitalia", "Sacrum.001", j["hips"] + V(0, -0.05, -0.03), j["hips"] + V(0, -0.1, -0.03))]
    exp = {"Hips": "Sacrum", "Spine": "Spine1", "Spine1": "Spine2", "Spine2": "Spine3", "Neck": "Spine3.002",
           "Head": "head"}
    for side, s, S in (("L", 1, "Left"), ("R", -1, "Right")):
        x = "." + side
        ua, fa = _mid(j["arm" + side], j["elbow" + side]), _mid(j["elbow" + side], j["wrist" + side])
        b += [("eye" + x, "head", j["head"] + V(s * 0.03, -0.08, 0.07), j["head"] + V(s * 0.03, -0.1, 0.07)),
              ("Clavicle" + x, "Spine3", j["clav" + side], j["arm" + side]),
              ("Humerus" + x, "Clavicle" + x, j["arm" + side], ua),
              ("HumerusRoll" + x, "Humerus" + x, ua, j["elbow" + side]),
              ("LowerArm" + x, "HumerusRoll" + x, j["elbow" + side], fa),
              ("LowerArm" + x + ".001", "LowerArm" + x, fa, j["wrist" + side]),
              ("hand001" + x, "LowerArm" + x + ".001", j["wrist" + side], j["handend" + side]),
              ("Femur" + x, "Sacrum.001", j["thigh" + side], j["knee" + side]),
              ("Tibia" + x, "Femur" + x, j["knee" + side], j["knee2" + side]),
              ("Tibia" + x + ".001", "Tibia" + x, j["knee2" + side], j["ankle" + side]),
              ("Foot" + x, "Tibia" + x + ".001", j["ankle" + side], j["toe" + side]),
              ("Foot" + x + ".001", "Foot" + x, j["toe" + side], j["toeend" + side])]
        exp.update({f"{S}Shoulder": "Clavicle" + x, f"{S}Arm": "Humerus" + x, f"{S}ForeArm": "LowerArm" + x,
                    f"{S}Hand": "hand001" + x, f"{S}UpLeg": "Femur" + x, f"{S}Leg": "Tibia" + x,
                    f"{S}Foot": "Foot" + x, f"{S}ToeBase": "Foot" + x + ".001"})
        nums = {"thumb": (2, 3, 4), "index": (5, 7, 8), "middle": (9, 10, 11), "ring": (12, 13, 14),
                "pinky": (15, 16, 17)}
        for name, ids in nums.items():
            for k, i in enumerate(ids, start=1):
                parent = "hand001" + x if k == 1 else f"hand{ids[k - 2]:03d}{x}"
                b.append((f"hand{i:03d}{x}", parent, j[f"{name}{k - 1}{side}"], j[f"{name}{k}{side}"]))
                exp[f"{S}Hand{name.capitalize()}{k}"] = f"hand{i:03d}{x}"
    b.append(("right_sword_mount", "hand001.R", j["wristR"] + V(0, 0, -0.02), j["wristR"] + V(0, 0, -0.05)))
    return b, exp


def style_unity(j):
    b = [("Root", None, V(0, 0, 0), V(0, -0.2, 0)), ("Hips", "Root", j["hips"], j["spine"]),
         ("Spine", "Hips", j["spine"], j["spine2"]), ("UpperChest", "Spine", j["spine2"], j["neck"]),
         ("Neck", "UpperChest", j["neck"], j["head"]),
         ("Neck_Twist_A", "Neck", j["neck_mid"], j["neck_mid"] + V(0, 0, 0.03)),
         ("Head", "Neck", j["head"], j["headtop"])]
    exp = {"Hips": "Hips", "Spine": "Spine", "Spine2": "UpperChest", "Neck": "Neck", "Head": "Head"}
    seg = {1: "Proximal", 2: "Intermediate", 3: "Distal"}
    for side, s, S in (("L", 1, "Left"), ("R", -1, "Right")):
        p = S + "_"
        b += [(p + "Eye", "Head", j["head"] + V(s * 0.03, -0.08, 0.07), j["head"] + V(s * 0.03, -0.1, 0.07)),
              (p + "Shoulder", "UpperChest", j["clav" + side], j["arm" + side]),
              (p + "UpperArm", p + "Shoulder", j["arm" + side], j["elbow" + side]),
              (p + "LowerArm", p + "UpperArm", j["elbow" + side], j["wrist" + side]),
              (p + "Hand", p + "LowerArm", j["wrist" + side], j["handend" + side]),
              (p + "UpperLeg", "Hips", j["thigh" + side], j["knee" + side]),
              (p + "LowerLeg", p + "UpperLeg", j["knee" + side], j["ankle" + side]),
              (p + "Foot", p + "LowerLeg", j["ankle" + side], j["toe" + side]),
              (p + "Toes", p + "Foot", j["toe" + side], j["toeend" + side]),
              (p + "ToesEnd", p + "Toes", j["toeend" + side], j["toeend" + side] + V(0, -0.05, 0))]
        exp.update({f"{S}Shoulder": p + "Shoulder", f"{S}Arm": p + "UpperArm", f"{S}ForeArm": p + "LowerArm",
                    f"{S}Hand": p + "Hand", f"{S}UpLeg": p + "UpperLeg", f"{S}Leg": p + "LowerLeg",
                    f"{S}Foot": p + "Foot", f"{S}ToeBase": p + "Toes", f"{S}Toe_End": p + "ToesEnd"})
        for name in ("thumb", "index", "middle", "ring", "pinky"):
            N = name.capitalize()
            for k in (1, 2, 3):
                parent = p + "Hand" if k == 1 else f"{p}{N}{seg[k - 1]}"
                b.append((f"{p}{N}{seg[k]}", parent, j[f"{name}{k - 1}{side}"], j[f"{name}{k}{side}"]))
                exp[f"{S}Hand{N}{k}"] = f"{p}{N}{seg[k]}"
    return b, exp


def style_biped(j):
    """3ds Max Biped: thighs under ``Bip01 Pelvis``, the clavicles under ``Bip01 Neck``."""
    b = [("Bip01", None, V(0, 0, 0), V(0, -0.1, 0)), ("Bip01 Pelvis", "Bip01", j["hips"], j["spine"]),
         ("Bip01 Spine", "Bip01 Pelvis", j["spine"], j["spine1"]),
         ("Bip01 Spine1", "Bip01 Spine", j["spine1"], j["spine2"]),
         ("Bip01 Spine2", "Bip01 Spine1", j["spine2"], j["neck"]),
         ("Bip01 Neck", "Bip01 Spine2", j["neck"], j["head"]),
         ("Bip01 Head", "Bip01 Neck", j["head"], j["headtop"])]
    exp = {"Hips": "Bip01 Pelvis", "Spine": "Bip01 Spine", "Spine1": "Bip01 Spine1", "Spine2": "Bip01 Spine2",
           "Neck": "Bip01 Neck", "Head": "Bip01 Head"}
    for side, S in (("L", "Left"), ("R", "Right")):
        p = f"Bip01 {side} "
        b += [(p + "Clavicle", "Bip01 Neck", j["clav" + side], j["arm" + side]),
              (p + "UpperArm", p + "Clavicle", j["arm" + side], j["elbow" + side]),
              (p + "Forearm", p + "UpperArm", j["elbow" + side], j["wrist" + side]),
              (p + "Hand", p + "Forearm", j["wrist" + side], j["handend" + side]),
              (p + "Thigh", "Bip01 Pelvis", j["thigh" + side], j["knee" + side]),
              (p + "Calf", p + "Thigh", j["knee" + side], j["ankle" + side]),
              (p + "Foot", p + "Calf", j["ankle" + side], j["toe" + side]),
              (p + "Toe0", p + "Foot", j["toe" + side], j["toeend" + side])]
        exp.update({f"{S}Shoulder": p + "Clavicle", f"{S}Arm": p + "UpperArm", f"{S}ForeArm": p + "Forearm",
                    f"{S}Hand": p + "Hand", f"{S}UpLeg": p + "Thigh", f"{S}Leg": p + "Calf",
                    f"{S}Foot": p + "Foot", f"{S}ToeBase": p + "Toe0"})
    return b, exp


RIGS = {
    # name: (style, arm angle below horizontal, data scale, object location, rotation (deg), object scale)
    "syn_rigify": (style_rigify, 60.0, 1.0, (2.0, 1.0, 0.0), (0.0, 0.0, 30.0), 1.0),
    "syn_anatomical": (style_anatomical, 25.0, 100.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.0),
    "syn_unity_cm": (style_unity, 0.0, 100.0, (-1.5, 0.5, 0.0), (0.0, 0.0, 180.0), 0.01),
}
TURNED = {"syn_rot10": 10.0, "syn_rot90": 90.0}  # Rigify-style copies turned about Z (free runs)


def build_rig(scene, name, style, arm_deg, unit=1.0, loc=(0, 0, 0), rot=(0, 0, 0), scale=1.0, seed=0,
              parent=None):
    bones, expected = style(joints(arm_deg))
    rng = random.Random(seed)
    arm = bpy.data.armatures.new(name)
    obj = bpy.data.objects.new(name, arm)
    scene.collection.objects.link(obj)
    obj.parent = parent
    obj.location = loc
    obj.rotation_euler = [math.radians(a) for a in rot]
    obj.scale = (scale, scale, scale) if isinstance(scale, float) else scale
    vl = bpy.context.view_layer
    vl.objects.active = obj
    with bpy.context.temp_override(active_object=obj, object=obj):
        bpy.ops.object.mode_set(mode="EDIT")
        made = {}
        for n, par, head, tail in bones:
            eb = arm.edit_bones.new(n)
            eb.head = head * unit
            eb.tail = tail * unit
            eb.roll = rng.uniform(-math.pi, math.pi)  # odd rolls everywhere
            if par is not None:
                eb.parent = made[par]
                eb.use_connect = (made[par].tail - eb.head).length < 1e-6 and rng.random() < 0.5
            made[n] = eb
        bpy.ops.object.mode_set(mode="OBJECT")
    for pb in obj.pose.bones:  # mixed rotation modes (keys must follow the bone's own mode)
        pb.rotation_mode = rng.choice(["QUATERNION", "QUATERNION", "XYZ", "ZXY"])
    _skin(scene, obj)
    return obj, expected


def _skin(scene, arm_obj):
    """A mesh with one triangle per bone (rigid to it) plus a sole under each foot bone."""
    verts, faces, groups = [], [], []
    for b in arm_obj.data.bones:
        h, t = b.head_local, b.tail_local
        y = t - h
        x = y.orthogonal().normalized() * max(y.length * 0.1, 1e-3)
        base = len(verts)
        verts += [h + x, h - x, t]
        faces.append((base, base + 1, base + 2))
        groups.append((b.name, [base, base + 1, base + 2]))
    mesh = bpy.data.meshes.new(arm_obj.name + "_mesh")
    mesh.from_pydata([tuple(v) for v in verts], [], faces)
    mobj = bpy.data.objects.new(arm_obj.name + "_mesh", mesh)
    scene.collection.objects.link(mobj)
    mobj.parent = arm_obj
    for n, idx in groups:
        mobj.vertex_groups.new(name=n).add(idx, 1.0, "REPLACE")
    mobj.modifiers.new("Armature", "ARMATURE").object = arm_obj


def add_old_action(obj, bone_names):
    """An existing active action on the target (keys on some bones), like Girush's own actions."""
    from bpy_extras import anim_utils

    act = bpy.data.actions.new(obj.name + "_old")
    slot = act.slots.new(id_type="OBJECT", name=obj.name)
    bag = anim_utils.action_ensure_channelbag_for_slot(act, slot)
    for n in bone_names:
        pb = obj.pose.bones[n]
        prop = "rotation_quaternion" if pb.rotation_mode == "QUATERNION" else "rotation_euler"
        rest = [1.0, 0.0, 0.0, 0.0] if prop == "rotation_quaternion" else [0.0, 0.0, 0.0]
        for i in range(len(rest)):
            fc = bag.fcurves.new(f'pose.bones["{n}"].{prop}', index=i)
            fc.keyframe_points.add(2)
            off = 0.3 if i == 1 else 0.0
            fc.keyframe_points.foreach_set("co", [0.0, rest[i] + off, 20.0, rest[i] - off])
            fc.update()
    ad = obj.animation_data_create()
    ad.action = act
    ad.action_slot = slot
    return act


def add_copy_rotation_knees(obj):
    """Girush's knee: the shin copies its knee piece's local rotation."""
    for x in ("L", "R"):
        c = obj.pose.bones[f"Tibia.{x}.001"].constraints.new("COPY_ROTATION")
        c.target = obj
        c.subtarget = f"Tibia.{x}"
        c.owner_space = c.target_space = "LOCAL"


def rest_offset_deg(scene, obj, names, frames):
    """Worst angle (deg) of these pose bones away from their rest relative to the parent (evaluated)."""
    worst = 0.0
    for f in frames:
        scene.frame_set(f)
        ev = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        for n in names:
            pb = ev.pose.bones[n]
            rel = pb.parent.matrix.inverted() @ pb.matrix if pb.parent else pb.matrix
            rest = obj.data.bones[n]
            rest_rel = rest.matrix_local
            if rest.parent:
                rest_rel = rest.parent.matrix_local.inverted() @ rest_rel
            d = rel.to_quaternion().rotation_difference(rest_rel.to_quaternion())
            worst = max(worst, math.degrees(d.angle), (rel.translation - rest_rel.translation).length)
    return worst


# --- modes ----------------------------------------------------------------------------------------
def _import_take(path):
    from afv_addon import builder

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    builder.import_take(path, scene)
    arms = {o["afv_role"]: o for o in scene.objects if o.get("afv_kind") == "armature"}
    return scene, arms


def _sides_ok(bone_map):
    """Every Left*/Right* canonical bone maps to a bone whose name says the same side (or no side)."""
    from afv_addon import retarget as rt

    bad = []
    for c, n in bone_map.items():
        side = "L" if c.startswith("Left") else "R" if c.startswith("Right") else None
        got = rt._parse_name(n)[0]
        if (side is not None and got not in (side, None)) or (side is None and got is not None):
            bad.append([c, n])
    return bad


def _guess(rt, obj, expected=None):
    g = rt.guess_map(obj)
    out = {"guess": g, "missing": rt.missing_required(g), "sides_bad": _sides_ok(g),
           "duplicates": len(g) - len(set(g.values())), "bones": len(obj.data.bones)}
    if expected is not None:
        out["expected"] = expected
    return out


def _raises(fn):
    try:
        fn()
    except ValueError as e:
        return str(e)
    return None


def mapio(tmp):
    from afv_addon import retarget as rt

    out = {}
    p = os.path.join(tmp, "m.afvmap.json")
    data = rt.new_map({"Hips": "a", "Spine": "b"}, facing="+Y", hips_scale=2.0, target="T")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump({**data, "extra": {"keep": 1}, "bones": {}}, fh)
    rt.write_map(p, data)
    raw = json.load(open(p, encoding="utf-8"))
    out["kept_unknown"] = raw.get("extra") == {"keep": 1} and raw["bones"] == {"Hips": "a", "Spine": "b"}
    out["roundtrip"] = rt.read_map(p) == {**data, "extra": {"keep": 1}}
    out["no_tmp_left"] = sorted(os.listdir(tmp)) == ["m.afvmap.json"]
    bad_cases = {
        "format": {**data, "format": "x"}, "version_bool": {**data, "version": True},
        "version_2": {**data, "version": 2}, "bones_list": {**data, "bones": []},
        "unknown_bone": {**data, "bones": {"Tail": "x"}}, "empty_name": {**data, "bones": {"Hips": ""}},
        "twice": {**data, "bones": {"Hips": "a", "Spine": "a"}}, "facing": {**data, "facing": "up"},
        "hips_bool": {**data, "hips_scale": True}, "hips_str": {**data, "hips_scale": "1"},
        "hips_neg": {**data, "hips_scale": -1}, "hips_nan": {**data, "hips_scale": float("nan")},
        "target_num": {**data, "target": 3},
    }
    rejected = {}
    for k, d in bad_cases.items():
        q = os.path.join(tmp, f"bad_{k}.json")
        with open(q, "w", encoding="utf-8") as fh:
            json.dump(d, fh)
        rejected[k] = _raises(lambda q=q: rt.read_map(q))
        os.remove(q)
    out["rejected"] = rejected
    try:
        rt.write_map(p, bad_cases["twice"])
        out["write_rejects"] = False
    except ValueError:
        out["write_rejects"] = json.load(open(p, encoding="utf-8")) == raw  # file untouched
    obj = bpy.data.objects.new("map_holder", None)
    out["from_empty"] = rt.map_from_object(obj)
    obj[rt.MAP_PROP] = json.dumps({**data, "note": "mine"})
    rt.map_to_object(obj, {**data, "facing": None})
    back = rt.map_from_object(obj)
    out["object_roundtrip"] = (back.get("note") == "mine" and back["facing"] is None
                               and back["bones"] == data["bones"])
    obj[rt.MAP_PROP] = "{broken"
    out["object_broken"] = _raises(lambda: rt.map_from_object(obj))
    z = rt.new_map({"Hips": "a"}, facing="+Z")
    rt.write_map(p, z)
    out["z_roundtrip"] = rt.read_map(p)["facing"] == "+Z"
    out["missing"] = rt.missing_required({"Hips": "a"})
    out["missing_full"] = rt.missing_required(data)
    bpy.data.objects.remove(obj)
    return out


def tweak_forearm(scene, src):
    """+40 deg about X on LeftForeArm, a constant key on fighter ``src``'s "AFV tweaks" strip."""
    from bpy_extras import anim_utils

    strip = next(t for t in src.animation_data.nla_tracks if t.name == "AFV tweaks").strips[0]
    bag = anim_utils.action_ensure_channelbag_for_slot(strip.action, strip.action_slot)
    prefix = src["afv_hips"][:-4]
    q = Matrix.Rotation(math.radians(40.0), 4, "X").to_quaternion()
    for i in range(4):
        fc = bag.fcurves.new(f'pose.bones["{prefix}LeftForeArm"].rotation_quaternion', index=i)
        fc.keyframe_points.add(2)
        fc.keyframe_points.foreach_set("co", [float(scene.frame_start), q[i], float(scene.frame_end), q[i]])
        fc.update()


def measure_dir(scene, obj, a, b, frames):
    out = []
    for f in frames:
        scene.frame_set(f)
        ev = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        mw = ev.matrix_world
        out.append(mw @ ev.pose.bones[b].head - mw @ ev.pose.bones[a].head)
    return out


def synthetic(job):
    from afv_addon import retarget as rt

    res = {}
    with tempfile.TemporaryDirectory() as tmp:
        res["mapio"] = mapio(tmp)
    scene, arms = _import_take(job["take"])
    src = arms["A"]
    res["source_canonical"] = len(rt.canonical_bones(src))
    res["frames"] = [scene.frame_start, scene.frame_end]
    rigs = {}
    for i, (name, (style, arm_deg, unit, loc, rot, scale)) in enumerate(RIGS.items()):
        rigs[name] = build_rig(scene, name, style, arm_deg, unit, loc, rot, scale, seed=i)
    add_copy_rotation_knees(rigs["syn_anatomical"][0])
    old = add_old_action(rigs["syn_anatomical"][0], ["eye.L", "Femur.L", "genitalia"])
    res["rigs"] = {}
    for name, (obj, expected) in rigs.items():
        r = _guess(rt, obj, expected)
        r["free"] = run(scene, src, obj, r["guess"], "-Y", lock_feet=False)
        r["locked"] = run(scene, src, obj, r["guess"], "-Y", lock_feet=True)
        res["rigs"][name] = r
    # turned copies and one under a turned, moved empty: facing in the armature's own axes
    empty = bpy.data.objects.new("turned_parent", None)
    scene.collection.objects.link(empty)
    empty.location = (1.0, -2.0, 0.0)
    empty.rotation_euler = (0.0, 0.0, math.radians(45.0))
    turned = {n: build_rig(scene, n, style_rigify, 60.0, rot=(0, 0, deg), loc=(3.0, 0, 0), seed=7)
              for n, deg in TURNED.items()}
    turned["syn_parented"] = build_rig(scene, "syn_parented", style_rigify, 60.0, rot=(0, 0, 20.0), seed=8,
                                       parent=empty)
    res["turned"] = {}
    for name, (obj, _exp) in turned.items():
        g = rt.guess_map(obj)
        res["turned"][name] = brief(run(scene, src, obj, g, "-Y", lock_feet=False))
    anat = rigs["syn_anatomical"][0]
    res["old_action"] = {"exists": old.name in bpy.data.actions, "fake_user": old.use_fake_user,
                         "unmapped_at_rest_deg": rest_offset_deg(scene, anat, ["eye.L", "genitalia"],
                                                                 [0, 10, 20, 40])}
    rig, _ = rigs["syn_rigify"]
    guess = rt.guess_map(rig)
    scene.frame_set(7)  # a frame subset, and the scene frame restored to a non-start frame
    sub = run(scene, src, rig, guess, "-Y", frames=(20, 80), lock_feet=True)
    res["subset"] = {"frames": sub["report"]["frames"], "changed": sub["changed"],
                     "key_counts": sub["key_counts"], "frame_after": sub["frame_after"],
                     "measure": brief(sub)["measure"]}
    k = res["rigs"]["syn_rigify"]["free"]["report"]["hips_scale"]
    hi = run(scene, src, rig, guess, "-Y", lock_feet=True, hips_scale=1.6 * k)  # hips far too high
    res["unreachable"] = {"notes": hi["report"]["notes"], "hips_scale": hi["report"]["hips_scale"]}
    full = rt.new_map(guess, facing="+X", hips_scale=0.9 * k, target=rig.name)
    fm = rt.retarget(scene, src, rig, full, lock_feet=False)
    res["full_map"] = {"facing": fm["facing"], "hips_scale": fm["hips_scale"], "yaw": fm["facing_yaw_deg"]}
    errors = {}
    for key, kw in {"hips_missing": {"bone_map": {"Spine": "spine.001"}},
                    "twice": {"bone_map": {**guess, "Spine": guess["Hips"]}},
                    "frames": {"bone_map": guess, "frames": (30, 10)},
                    "facing": {"bone_map": guess, "facing": "up"},
                    "scale": {"bone_map": guess, "hips_scale": 0},
                    "hierarchy": {"bone_map": {**guess, "LeftForeArm": "shin.L", "LeftLeg": "forearm.L"}},
                    }.items():
        bm = kw.pop("bone_map")
        errors[key] = _raises(lambda bm=bm, kw=kw: rt.retarget(scene, src, rig, bm, **kw))
    res["errors"] = errors
    res["refusals"] = refusals(scene, src, rt)
    def flip(c):
        return ("Right" + c[4:]) if c.startswith("Left") else ("Left" + c[5:]) if c.startswith("Right") else c

    swapped = {flip(c): n for c, n in guess.items()}
    res["swapped_notes"] = rt.retarget(scene, src, rig, swapped, lock_feet=False)["notes"]
    res["write_error"] = write_error(scene, src, rig, guess, rt)
    res["mute_error"] = mute_error(scene, src, rig, guess, rt)
    res["write_error_tweak"] = write_error_tweak(scene, src, rig, guess, rt)
    res["yup_target"] = yup_target(scene, src, arms["B"], rt)
    res["close_locks"] = close_locks(scene, src, rig, guess, rt)
    bare = src.copy()  # a source without plant masks
    scene.collection.objects.link(bare)
    for kk in [kk for kk in bare.keys() if kk.startswith("afv_plant_")]:
        del bare[kk]
    res["no_plants"] = {"notes": rt.retarget(scene, bare, rig, guess, lock_feet=True)["notes"]}
    # a tweak key on fighter A's "AFV tweaks" track: the result follows it
    before = measure_dir(scene, rig, guess["LeftForeArm"], guess["LeftHand"], [30, 120, 250])
    tweak_forearm(scene, src)
    tw = run(scene, src, rig, guess, "-Y", lock_feet=False)
    after = measure_dir(scene, rig, guess["LeftForeArm"], guess["LeftHand"], [30, 120, 250])
    res["tweak"] = {"measure": brief(tw)["measure"],
                    "changed_deg": [math.degrees(a.angle(b)) for a, b in zip(before, after, strict=True)]}
    res["guess_extra"] = guess_extra(scene, src, rt)
    from afv_addon import builder  # a prefixed source rig with a root bone (unity_generic, mixamorig:)

    builder.import_take(job["take_prefixed"], scene)
    psrc = next(o for o in scene.objects if o.get("afv_kind") == "armature" and o.get("afv_role") == "A"
                and o.data.bones.get("root") is not None)
    res["prefixed"] = {"canonical": rt.canonical_bones(psrc),
                       "run": brief(run(scene, psrc, rig, guess, "-Y", lock_feet=True))}
    return res


def refusals(scene, src, rt):
    """Each must raise ValueError (message returned)."""
    out = {}
    squash, _ = build_rig(scene, "syn_squash", style_rigify, 60.0, scale=(1.25, 1.0, 0.8), seed=3)
    g = rt.guess_map(squash)
    out["nonuniform_target"] = _raises(lambda: rt.retarget(scene, src, squash, g))
    lying, _ = build_rig(scene, "syn_lying", style_rigify, 60.0, rot=(-90.0, 0, 0), seed=4)
    out["lying"] = _raises(lambda: rt.retarget(scene, src, lying, rt.guess_map(lying)))
    rig = bpy.data.objects["syn_rigify"]
    g = rt.guess_map(rig)
    coll = src.users_collection[0]
    lc = next(c for c in bpy.context.view_layer.layer_collection.children if c.collection == coll)
    lc.exclude = True
    try:
        out["excluded_source"] = _raises(lambda: rt.retarget(scene, src, rig, g))
    finally:
        lc.exclude = False
    old = tuple(src.scale)
    src.scale = (1.0, 1.0, 1.3)
    bpy.context.view_layer.update()
    try:
        out["nonuniform_source"] = _raises(lambda: rt.retarget(scene, src, rig, g))
    finally:
        src.scale = old
        bpy.context.view_layer.update()
    return out


def write_error_tweak(scene, src, rig, guess, rt):
    """The target in NLA tweak mode, the assignment fails after tweak mode was left: tweak mode, the
    tweaked strip and the actions are back as they were, no partial action is left."""
    ad = rig.animation_data
    keep = ad.action
    track = ad.nla_tracks.new()
    track.name = "user track"
    strip = track.strips.new("user strip", 0, keep)
    ad.action = None
    for t in ad.nla_tracks:
        t.select = t == track
    strip.select = True
    ad.nla_tracks.active = track
    ad.use_tweak_mode = True
    names = set(bpy.data.actions.keys())
    was = {"tweak": ad.use_tweak_mode, "action": ad.action.name if ad.action else None}
    real = rt._assign

    def failing(*a, **kw):
        raise RuntimeError("simulated failure")

    rt._assign = failing
    try:
        rt.retarget(scene, src, rig, guess, lock_feet=False, action_name="should_vanish")
        raised = False
    except RuntimeError:
        raised = True
    finally:
        rt._assign = real
    now = {"tweak": ad.use_tweak_mode, "action": ad.action.name if ad.action else None}
    out = {"raised": raised, "was": was, "now": now,
           "new_actions": sorted(set(bpy.data.actions.keys()) - names)}
    ad.use_tweak_mode = False  # back to the test's own state
    ad.nla_tracks.remove(track)
    ad.action = keep
    return out


def mute_error(scene, src, rig, guess, rt):
    """Switching off the second of the target's Armature modifiers fails (as on a non-editable linked
    mesh): retarget raises, and every modifier it had switched off is on again, no action is left."""
    extra = bpy.data.objects[rig.name + "_mesh"].copy()  # a second skinned mesh (its modifier copied)
    scene.collection.objects.link(extra)
    bpy.context.view_layer.update()
    mods = [md for o in skinned(rig) for md in o.modifiers if md.type == "ARMATURE"]
    before = [md.show_viewport for md in mods]
    names = set(bpy.data.actions.keys())
    real, calls = rt._set_viewport, [0]

    def failing(md, value):
        calls[0] += 1
        if calls[0] == 2:
            raise RuntimeError("simulated: modifier not editable")
        real(md, value)

    rt._set_viewport = failing
    try:
        rt.retarget(scene, src, rig, guess, lock_feet=False, action_name="should_vanish")
        raised = False
    except RuntimeError:
        raised = True
    finally:
        rt._set_viewport = real
    out = {"raised": raised, "calls": calls[0], "modifiers": len(mods), "before": before,
           "after": [md.show_viewport for md in mods],
           "new_actions": sorted(set(bpy.data.actions.keys()) - names)}
    bpy.data.objects.remove(extra)
    return out


def write_error(scene, src, rig, guess, rt):
    """An error while writing keys: no partial action is left, the target's action is unchanged."""
    names = set(bpy.data.actions.keys())
    before = rig.animation_data.action.name if rig.animation_data and rig.animation_data.action else None
    real, calls = rt._new_fcurve, [0]

    def failing(*a, **kw):
        calls[0] += 1
        if calls[0] > 30:
            raise RuntimeError("simulated failure")
        return real(*a, **kw)

    rt._new_fcurve = failing
    try:
        rt.retarget(scene, src, rig, guess, lock_feet=False, action_name="should_vanish")
        raised = False
    except RuntimeError:
        raised = True
    finally:
        rt._new_fcurve = real
    after = rig.animation_data.action.name if rig.animation_data and rig.animation_data.action else None
    return {"raised": raised, "new_actions": sorted(set(bpy.data.actions.keys()) - names),
            "action_kept": before == after}


def yup_target(scene, src, tgt, rt):
    """Fighter A onto fighter B's take armature (Y-up armature data): the reported facing (+Z) goes
    through a map file and back into retarget, with the same result."""
    g = rt.guess_map(tgt)
    first = rt.retarget(scene, src, tgt, g, lock_feet=False)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "b.afvmap.json")
        rt.write_map(path, rt.new_map(g, facing=first["facing"], target=tgt.name))
        back = rt.read_map(path)
    second = rt.retarget(scene, src, tgt, back, lock_feet=False)
    same = action_hash(bpy.data.actions[first["action"]]) == action_hash(bpy.data.actions[second["action"]])
    return {"facing": first["facing"], "read_facing": back["facing"], "second_facing": second["facing"],
            "yaw": [first["facing_yaw_deg"], second["facing_yaw_deg"]], "same_keys": same,
            "dir": [first["max_dir_err_deg"], second["max_dir_err_deg"]]}


def bigmesh(job):
    """A ~1M-vertex skinned mesh on a synthetic rig: retarget time (free and locked), and that the
    feet points still come from it."""
    from afv_addon import retarget as rt

    scene, arms = _import_take(job["take"])
    rig, _ = build_rig(scene, "syn_big", style_rigify, 60.0, seed=11)
    import numpy as np

    bones = list(rig.data.bones)
    feet = {"foot.L", "toe.L", "heel.02.L", "foot.R", "toe.R", "heel.02.R"}
    n_feet = int(job["verts"] * job.get("feet_share", 0.3))  # dense feet: a big share below the ankles
    per_foot = n_feet // len(feet)
    per = (job["verts"] - n_feet) // (len(bones) - len(feet))
    rng = np.random.default_rng(0)
    pts, groups, start = [], [], 0
    for b in bones:
        cnt = per_foot if b.name in feet else per
        h, t = np.array(b.head_local), np.array(b.tail_local)
        u = rng.random((cnt, 1))
        q = h + u * (t - h) + rng.normal(0.0, 0.02, (cnt, 3))
        if b.name in feet:  # a sole-like slab under the foot: from the floor up to the ankle
            q[:, 2] = rng.uniform(0.0, max(h[2], t[2]), cnt)
        groups.append((b.name, start, cnt))
        pts.append(q)
        start += cnt
    co = np.concatenate(pts)
    mesh = bpy.data.meshes.new("big")
    mesh.vertices.add(len(co))
    mesh.vertices.foreach_set("co", co.ravel())
    mobj = bpy.data.objects.new("big", mesh)
    scene.collection.objects.link(mobj)
    mobj.parent = rig
    for name, start, count in groups:
        mobj.vertex_groups.new(name=name).add(list(range(start, start + count)), 1.0, "REPLACE")
    mobj.modifiers.new("Armature", "ARMATURE").object = rig
    g = rt.guess_map(rig)
    ankle = max(rig.data.bones["foot.L"].head_local.z, rig.data.bones["foot.R"].head_local.z)
    out = {"verts": len(co), "below_ankle": int((co[:, 2] <= ankle).sum())}
    for lock in (False, True):
        t = time.perf_counter()
        rep = rt.retarget(scene, arms["A"], rig, g, lock_feet=lock)
        out["locked" if lock else "free"] = {"seconds": rep["seconds"], "wall": time.perf_counter() - t,
                                             "penetration_cm": rep["penetration_cm"], "notes": rep["notes"]}
    return out


def close_locks(scene, src, rig, guess, rt):
    """Two plants of the left foot three frames apart (frames 100-129 and 133-160 of a copy of fighter
    A): their locks sit at opposite sides of the moving retargeted foot, so the fade across the short
    gap must be harder than the design limit and the notes must say so, with the right numbers."""
    from bpy_extras import anim_utils

    dup = src.copy()
    dup.animation_data.action = None
    track = next(t for t in dup.animation_data.nla_tracks if t.name == "AFV solved")
    strip = track.strips[0]
    act = strip.action.copy()
    strip.action = act
    strip.action_slot = act.slots[0]
    scene.collection.objects.link(dup)
    bag = anim_utils.action_get_channelbag_for_slot(act, strip.action_slot)
    for fc in bag.fcurves:
        if not fc.data_path.startswith('["afv_plant_'):
            continue
        on = fc.data_path == '["afv_plant_LeftFoot"]'
        co = [0.0] * (2 * len(fc.keyframe_points))
        fc.keyframe_points.foreach_get("co", co)
        for i in range(0, len(co), 2):
            f = round(co[i])
            co[i + 1] = 1.0 if on and (100 <= f <= 129 or 133 <= f <= 160) else 0.0
        fc.keyframe_points.foreach_set("co", co)
        fc.update()
    bpy.context.view_layer.update()
    free = run(scene, dup, rig, guess, "-Y", lock_feet=False)
    locked = run(scene, dup, rig, guess, "-Y", lock_feet=True)
    return {"free": free["measure"], "locked": locked["measure"], "notes": locked["report"]["notes"]}


def guess_extra(scene, src, rt):
    """3ds Max Biped, Blender's real Rigify metarig and a rig generated from it."""
    out = {}
    biped, exp = build_rig(scene, "syn_biped", style_biped, 0.0, loc=(-3.0, 0, 0), seed=5)
    out["biped"] = _guess(rt, biped, exp)
    import addon_utils

    addon_utils.enable("rigify", default_set=True)
    for o in list(bpy.context.selected_objects):
        o.select_set(False)
    bpy.ops.object.armature_human_metarig_add()
    meta = bpy.context.active_object
    meta.name = "metarig"
    meta.location = (0.0, 3.0, 0.0)
    r = _guess(rt, meta)
    r["free"] = brief(run(scene, src, meta, r["guess"], "-Y", lock_feet=False))
    out["metarig"] = r
    bpy.ops.pose.rigify_generate()
    gen = bpy.context.active_object
    out["generated"] = {"bones": len(gen.data.bones), "guess": _raises(lambda: rt.guess_map(gen)),
                        "retarget": _raises(lambda: rt.retarget(scene, src, gen, {"Hips": "torso"}))}
    return out


def load_character(scene, path):
    """Import the character; the take's fps and frame range are put back (the FBX importer sets the
    scene to the file's 24 fps)."""
    keep = (scene.render.fps, scene.render.fps_base, scene.frame_start, scene.frame_end)
    before = set(scene.objects)
    if path.lower().endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=path)
    elif path.lower().endswith((".glb", ".gltf")):
        bpy.ops.import_scene.gltf(filepath=path)
    else:  # append the objects of a .blend (its scripts are not run)
        with bpy.data.libraries.load(path, link=False) as (src, dst):
            dst.objects = list(src.objects)
        for o in dst.objects:
            if o is not None and o.type in ("ARMATURE", "MESH", "EMPTY"):
                scene.collection.objects.link(o)
    scene.render.fps, scene.render.fps_base, scene.frame_start, scene.frame_end = keep
    new = [o for o in scene.objects if o not in before and o.type == "ARMATURE"]
    return max(new, key=lambda o: len(o.data.bones))


def character(job):
    from afv_addon import retarget as rt

    scene, arms = _import_take(job["take"])
    t = time.perf_counter()
    tgt = load_character(scene, job["character"])
    res = {"load_s": time.perf_counter() - t, "target": tgt.name}
    res.update(_guess(rt, tgt))
    guess = res["guess"]
    res["runs"] = []
    for spec in job["runs"]:
        r = run(scene, arms[spec["role"]], tgt, guess, job["facing"], lock_feet=spec["lock_feet"])
        res["runs"].append({**spec, **r})
    rt.retarget(scene, arms["A"], tgt, guess, lock_feet=False)  # fighter A without the tweak
    before = measure_dir(scene, tgt, guess["LeftForeArm"], guess["LeftHand"], [30, 60, 90])
    tweak_forearm(scene, arms["A"])
    tw = run(scene, arms["A"], tgt, guess, job["facing"], lock_feet=False)
    after = measure_dir(scene, tgt, guess["LeftForeArm"], guess["LeftHand"], [30, 60, 90])
    res["tweak"] = {"measure": brief(tw)["measure"],
                    "changed_deg": [math.degrees(a.angle(b)) for a, b in zip(before, after, strict=True)]}
    return res


def main():
    job_path = sys.argv[sys.argv.index("--") + 1]
    with open(job_path, encoding="utf-8") as fh:
        job = json.load(fh)
    sys.path.insert(0, job["addon_parent"])
    result = {"ok": False}
    try:
        result["out"] = {"synthetic": synthetic, "character": character, "bigmesh": bigmesh}[job["mode"]](job)
        result["ok"] = True
    except Exception:
        result["error"] = traceback.format_exc()
        print(result["error"])
    with open(job["result_path"], "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, default=str)
    if not result["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
