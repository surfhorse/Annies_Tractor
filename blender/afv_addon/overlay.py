"""Drawing with Blender's ``gpu`` module: the confidence strip and the tracking boxes.

- **Confidence strip**: one thin row per fighter at the top of the Timeline / Dope Sheet / NLA /
  Graph editors (just under the frame numbers), a bar per QC span coloured by severity (amber =
  warn, red = bad). The timeline markers stay as the fallback (and carry the labels).
- **Tracking boxes**: detect_track's boxes (from the take package) over the camera view in the 3D
  viewport, with the object id, the fighter role and label; fighter A red, B blue, others grey.
  ``afv.pick_boxes`` highlights the boxes being picked.

Nothing here runs in background mode (no GPU there); the geometry is in ``takeio`` (pure, tested).
"""

from __future__ import annotations

import os

import bpy

from . import builder, takeio

_HANDLES: list[tuple[type, object, str]] = []
ROLE_COLOR = {"A": (0.95, 0.3, 0.25, 1.0), "B": (0.3, 0.55, 1.0, 1.0)}
OTHER_COLOR = (0.75, 0.75, 0.75, 0.9)
PICK_COLOR = (1.0, 0.9, 0.1, 1.0)


class _Pick:
    active = False
    selected: list[int] = []  # noqa: RUF012 (module state)
    rec: dict | None = None  # the take being picked on (afv.pick_boxes)


PICK = _Pick()
_SPAN_CACHE: dict = {"raw": None, "spans": []}
_TRACK_CACHE: dict = {}  # take.json path -> (mtime, tracks)


def _show_strip(context) -> bool:
    addon = context.preferences.addons.get(__package__)
    return getattr(addon.preferences, "draw_strip", True) if addon is not None else True


def _spans(scene):
    raw = scene.get(builder.PROP_SPANS, "[]")
    if raw != _SPAN_CACHE["raw"]:
        _SPAN_CACHE["raw"] = raw
        _SPAN_CACHE["spans"] = builder.scene_spans(scene)
    return _SPAN_CACHE["spans"]


def tracks_for(scene, rec=None):
    """The tracking boxes of take ``rec`` (default: the scene's active take), cached by take.json
    path and mtime; None without boxes."""
    rec = rec if rec is not None else builder.scene_take(scene)
    if rec is None:
        return None
    try:
        tj = builder.resolve_take_json(rec)
        mtime = os.stat(tj).st_mtime_ns
    except (builder.BuildError, OSError):
        return None
    hit = _TRACK_CACHE.get(tj)
    if hit is None or hit[0] != mtime:
        try:
            tracks = takeio.load_tracks(tj)
        except takeio.TakeError:
            tracks = None
        if len(_TRACK_CACHE) > 8:
            _TRACK_CACHE.clear()
        hit = _TRACK_CACHE[tj] = (mtime, tracks)
    return hit[1]


def take_camera(scene, rec):
    """The camera object of take ``rec`` (each imported copy has its own)."""
    return builder._take_objects(scene, rec)["camera"] if rec is not None else None


def camera_corners(scene, region, rv3d, cam=None):
    """Region pixels of the camera frame's (top-left, top-right, bottom-left), or None. Only when
    the viewport looks through ``cam`` (default: the scene camera)."""
    from bpy_extras.view3d_utils import location_3d_to_region_2d

    cam = cam if cam is not None else scene.camera
    if cam is None or rv3d is None or rv3d.view_perspective != "CAMERA" or scene.camera != cam:
        return None
    dg = bpy.context.evaluated_depsgraph_get()
    ev = cam.evaluated_get(dg)
    tr, _br, bl, tl = (ev.matrix_world @ v for v in ev.data.view_frame(scene=scene))
    pts = [location_3d_to_region_2d(region, rv3d, p) for p in (tl, tr, bl)]
    if any(p is None for p in pts):
        return None
    return tuple((p.x, p.y) for p in pts)


def region_boxes(scene, region, rv3d, rec=None):
    """[(object id, (x0, y0, x1, y1) region pixels)] of take ``rec`` (default: the scene's active
    take) at the current frame, its tracks and the camera frame's corners; ([], None, None) when
    nothing can be drawn (no boxes, or the viewport does not look through that take's camera)."""
    rec = rec if rec is not None else builder.scene_take(scene)
    tracks = tracks_for(scene, rec)
    if tracks is None or not tracks.image_size[0]:
        return [], None, None
    corners = camera_corners(scene, region, rv3d, take_camera(scene, rec))
    if corners is None:
        return [], tracks, None
    out = []
    for obj, (x0, y0, x1, y1) in tracks.boxes_at(scene.frame_current):
        a = takeio.image_to_region(corners, tracks.image_size, x0, y0)
        b = takeio.image_to_region(corners, tracks.image_size, x1, y1)
        out.append((obj, (a[0], a[1], b[0], b[1])))
    return out, tracks, corners


# --- draw callbacks --------------------------------------------------------------------------------
def draw_strip():
    import gpu
    from gpu_extras.batch import batch_for_shader

    context = bpy.context
    scene, region = context.scene, context.region
    if scene is None or region is None or not _show_strip(context):
        return
    spans = _spans(scene)
    if not spans:
        return
    rec = builder.scene_take(scene) or {}
    roles = list(rec.get("roles") or sorted({s.get("role", "") for s in spans}))
    ui = context.preferences.system.ui_scale
    h, gap = 4.0 * ui, 1.0 * ui
    top = region.height - 24.0 * ui  # under the frame numbers (scrubbing area)
    v2d = region.view2d
    pos, col = [], []
    for row, f0, f1, sev in takeio.strip_bars(spans, roles):
        x0 = v2d.view_to_region(f0, 0.0, clip=False)[0]
        x1 = v2d.view_to_region(f1, 0.0, clip=False)[0]
        if x1 < 0 or x0 > region.width:
            continue
        x0, x1 = max(x0, -1.0), min(x1, region.width + 1.0)
        x1 = max(x1, x0 + 2.0)  # a one-frame span stays visible when zoomed out
        y1 = top - row * (h + gap)
        y0 = y1 - h
        c = takeio.SEVERITY_COLOR.get(sev, takeio.SEVERITY_COLOR["warn"])
        pos += [(x0, y0), (x1, y0), (x1, y1), (x0, y0), (x1, y1), (x0, y1)]
        col += [c] * 6
    if not pos:
        return
    shader = gpu.shader.from_builtin("SMOOTH_COLOR")
    batch = batch_for_shader(shader, "TRIS", {"pos": pos, "color": col})
    gpu.state.blend_set("ALPHA")
    shader.bind()
    batch.draw(shader)
    gpu.state.blend_set("NONE")


def draw_boxes():
    import blf
    import gpu
    from gpu_extras.batch import batch_for_shader

    context = bpy.context
    scene, region = context.scene, context.region
    space = context.space_data
    if scene is None or region is None or space is None or space.type != "VIEW_3D":
        return
    props = getattr(scene, "afv", None)
    if not PICK.active and not (props is not None and props.show_boxes):
        return
    rec = PICK.rec if PICK.active and PICK.rec is not None else builder.record_for(
        scene, context.view_layer.objects.active if context.view_layer else None)
    boxes, tracks, _ = region_boxes(scene, region, context.region_data, rec)
    if not boxes:
        return
    ui = context.preferences.system.ui_scale
    shader = gpu.shader.from_builtin("POLYLINE_UNIFORM_COLOR")
    gpu.state.blend_set("ALPHA")
    font = 0
    blf.size(font, 13.0 * ui)
    for obj, (x0, y0, x1, y1) in boxes:
        role = tracks.role_of(obj)
        picked = PICK.active and obj in PICK.selected
        color = PICK_COLOR if picked else ROLE_COLOR.get(role, OTHER_COLOR)
        pts = [(x0, y0), (x1, y0), (x1, y0), (x1, y1), (x1, y1), (x0, y1), (x0, y1), (x0, y0)]
        batch = batch_for_shader(shader, "LINES", {"pos": pts})
        shader.bind()
        shader.uniform_float("viewportSize", (region.width, region.height))
        shader.uniform_float("lineWidth", (3.0 if picked else 1.5) * ui)
        shader.uniform_float("color", color)
        batch.draw(shader)
        label = tracks.labels.get(obj)
        text = f"#{obj}" + (f" {role}" if role else "") + (f" {label}" if label else "")
        blf.color(font, *color)
        blf.position(font, min(x0, x1) + 3 * ui, max(y0, y1) - 15 * ui, 0)
        blf.draw(font, text)
    if PICK.active:
        msg = (f"Swap IDs from frame {scene.frame_current}: click two boxes "
               f"({len(PICK.selected)}/2 picked), Esc or right-click to cancel")
        blf.color(font, *PICK_COLOR)
        blf.position(font, 12 * ui, 12 * ui, 0)
        blf.draw(font, msg)
    gpu.state.blend_set("NONE")


_SPACES_STRIP = ("SpaceDopeSheetEditor", "SpaceNLA", "SpaceGraphEditor")


def register():
    if bpy.app.background:
        return  # no drawing without a window
    for name in _SPACES_STRIP:
        space = getattr(bpy.types, name)
        _HANDLES.append((space, space.draw_handler_add(draw_strip, (), "WINDOW", "POST_PIXEL"), "WINDOW"))
    v3d = bpy.types.SpaceView3D
    _HANDLES.append((v3d, v3d.draw_handler_add(draw_boxes, (), "WINDOW", "POST_PIXEL"), "WINDOW"))


def unregister():
    while _HANDLES:
        space, handle, region = _HANDLES.pop()
        try:
            space.draw_handler_remove(handle, region)
        except (ValueError, RuntimeError):
            pass
    PICK.active, PICK.selected, PICK.rec = False, [], None


def redraw_all():
    wm = bpy.context.window_manager
    if wm is None:
        return
    for win in wm.windows:
        for area in win.screen.areas:
            if area.type in ("VIEW_3D", "DOPESHEET_EDITOR", "NLA_EDITOR", "GRAPH_EDITOR"):
                area.tag_redraw()
