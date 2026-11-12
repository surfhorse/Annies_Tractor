"""The shot browser's drawing (``gpu``) and its pick state.

- **Pick fighters**: the prompt frame's numbered boxes over the browser camera's view (the 3D viewport
  looks through it; the frame image fills the camera frame), fighter A red, B blue, extras yellow, the
  others white; a help line at the bottom. The geometry is the Swap IDs tool's (``overlay.camera_corners``,
  ``takeio.image_to_region`` / ``box_at``), so a click hits what is drawn.
- **Shot strip**: in the Timeline / Dope Sheet of the browser scene, one bar per shot under the frame
  numbers: grey candidate, green queued, dark red skipped; a yellow tick on shots where fighters are
  likely; the selected shot outlined.

Nothing draws in background mode.
"""

from __future__ import annotations

import bpy

from . import browser_core as core
from . import overlay, takeio

ROLE_COLORS = [(0.95, 0.3, 0.25, 1.0), (0.3, 0.55, 1.0, 1.0)]  # fighter A, fighter B (as overlay.py)
EXTRA_COLOR = (1.0, 0.85, 0.1, 1.0)
FREE_COLOR = (0.92, 0.92, 0.92, 0.9)
STATE_COLORS = {"candidate": (0.55, 0.55, 0.55, 0.8), "queued": (0.25, 0.75, 0.35, 0.95),
                "skipped": (0.45, 0.15, 0.15, 0.8)}
LIKELY_COLOR = (1.0, 0.85, 0.1, 0.95)
SCENE_PROP = "afv_browser"  # the browser scene's marker (JSON: project, root_win)
CAMERA_NAME = "AFV Shots camera"

SESSION = core.Session()  # the open project's shot list


class _Pick:
    """The pick in progress (one at a time): the shot, its prompt, and the numbers chosen so far."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.shot = ""
        self.waiting = False  # the prompt is being fetched (scan or detection in afv)
        self.wait_text = ""  # what it waits for, with an estimate
        self.wait_until: float | None = None  # monotonic time the wait should end (estimate)
        self.saving = False  # the pick was sent to afv: the clicks stay until it is saved
        self.prompt: dict | None = None
        self.frame: int | None = None  # prompt frame (30 Hz shot timeline)
        self.source = ""  # scan | stage | detected
        self.image = ""  # the frame image shown behind the boxes (a path Blender reads)
        self.size = (0, 0)  # pixel size the boxes are in (the image's)
        self.selected: list[int] = []  # detection numbers: [A, B, extras...]
        self.current: dict | None = None  # the shot's fighters edit before this pick

    def boxes(self) -> list[tuple[int, tuple[float, float, float, float]]]:
        """[(detection number, (x0, y0, x1, y1) image pixels)] of the prompt."""
        out = []
        for k, b in enumerate((self.prompt or {}).get("boxes") or []):
            try:
                out.append((k, tuple(float(v) for v in b[:4])))
            except (TypeError, ValueError):
                continue
        return out

    def toggle(self, k: int) -> None:
        """Click on detection ``k``: the first two chosen are fighters A and B, then extras; clicking a
        chosen one takes it out (the ones after it move up)."""
        if k in self.selected:
            self.selected.remove(k)
        elif any(n == k for n, _ in self.boxes()):
            self.selected.append(k)

    def role(self, k: int) -> str:
        if k not in self.selected:
            return ""
        i = self.selected.index(k)
        return "A" if i == 0 else "B" if i == 1 else "extra"

    def text(self) -> str:
        sel = self.selected
        parts = [f"A = #{sel[0]}" if sel else "A = ?", f"B = #{sel[1]}" if len(sel) > 1 else "B = ?"]
        if len(sel) > 2:
            parts.append("extra " + ", ".join(f"#{k}" for k in sel[2:]))
        return ", ".join(parts)


PICK = _Pick()


def browser_camera(scene):
    cam = bpy.data.objects.get(CAMERA_NAME)
    return cam if cam is not None and scene is not None and cam.name in scene.objects else None


def region_boxes(scene, region, rv3d):
    """[(detection number, region box)] of the pick and the camera frame's corners; ([], None) when the
    viewport does not look through the browser camera."""
    cam = browser_camera(scene)
    if cam is None or not PICK.active or PICK.prompt is None or not PICK.size[0]:
        return [], None
    corners = overlay.camera_corners(scene, region, rv3d, cam)
    if corners is None:
        return [], None
    out = []
    for k, (x0, y0, x1, y1) in PICK.boxes():
        a = takeio.image_to_region(corners, PICK.size, x0, y0)
        b = takeio.image_to_region(corners, PICK.size, x1, y1)
        out.append((k, (a[0], a[1], b[0], b[1])))
    return out, corners


def box_color(k: int):
    role = PICK.role(k)
    if role == "A":
        return ROLE_COLORS[0]
    if role == "B":
        return ROLE_COLORS[1]
    return EXTRA_COLOR if role else FREE_COLOR


def wait_line() -> str:
    """What the pick waits for, with the time left (estimate)."""
    import time

    text = PICK.wait_text or "finding the people"
    if PICK.wait_until is not None:
        left = PICK.wait_until - time.monotonic()
        text += f", about {left:.0f} s left" if left > 1 else ", any moment now"
    return text


def help_line() -> str:
    if PICK.waiting:
        return f"Shot {PICK.shot}: {wait_line()}... Esc cancels"
    if PICK.saving:
        return f"Shot {PICK.shot}: saving {PICK.text()}..."
    n = len(PICK.boxes())
    if not n:
        return f"Shot {PICK.shot}: nobody was found on this frame. Esc to leave (trim the shot, scan again)"
    step = ("click fighter A" if not PICK.selected else "click fighter B" if len(PICK.selected) == 1
            else "click extras (referee...) or press Enter to save")
    return (f"Pick fighters, shot {PICK.shot}: {step}  |  {PICK.text()}  |  Enter save, Backspace undo, "
            "Esc cancel, click a chosen box to drop it")


# --- draw callbacks --------------------------------------------------------------------------------
def draw_pick():
    import blf
    import gpu
    from gpu_extras.batch import batch_for_shader

    context = bpy.context
    scene, region = context.scene, context.region
    if not PICK.active or scene is None or region is None or context.space_data is None:
        return
    if context.space_data.type != "VIEW_3D" or browser_camera(scene) is None:
        return
    boxes, _corners = region_boxes(scene, region, context.region_data)
    ui = context.preferences.system.ui_scale
    font = 0
    gpu.state.blend_set("ALPHA")
    if boxes:
        shader = gpu.shader.from_builtin("POLYLINE_UNIFORM_COLOR")
        blf.size(font, 16.0 * ui)
        for k, (x0, y0, x1, y1) in boxes:
            color = box_color(k)
            chosen = k in PICK.selected
            pts = [(x0, y0), (x1, y0), (x1, y0), (x1, y1), (x1, y1), (x0, y1), (x0, y1), (x0, y0)]
            batch = batch_for_shader(shader, "LINES", {"pos": pts})
            shader.bind()
            shader.uniform_float("viewportSize", (region.width, region.height))
            shader.uniform_float("lineWidth", (3.5 if chosen else 1.5) * ui)
            shader.uniform_float("color", color)
            batch.draw(shader)
            role = PICK.role(k)
            text = f"#{k}" + (f" {role}" if role else "")
            blf.color(font, *color)
            blf.position(font, min(x0, x1) + 4 * ui, max(y0, y1) - 20 * ui, 0)
            blf.draw(font, text)
    blf.size(font, 14.0 * ui)
    blf.color(font, *EXTRA_COLOR)
    blf.position(font, 12 * ui, 12 * ui, 0)
    blf.draw(font, help_line())
    gpu.state.blend_set("NONE")


def _strip_rects(scene, region, ui):
    """[(x0, y0, x1, y1, color)] of the shot strip in a timeline region."""
    if not SESSION.open or scene.get(SCENE_PROP) is None:
        return []
    fps = SESSION.fps
    v2d = region.view2d
    h = 5.0 * ui
    top = region.height - 24.0 * ui
    sel = getattr(getattr(scene, "afv_shots", None), "shot", "")
    out, outline = [], None
    for r in SESSION.all_rows():
        if not core.has_range(r):
            continue
        a, b = core.shot_frames(r, fps)
        x0 = v2d.view_to_region(a, 0.0, clip=False)[0]
        x1 = v2d.view_to_region(b + 1, 0.0, clip=False)[0]
        if x1 < 0 or x0 > region.width:
            continue
        x0, x1 = max(x0, -1.0), min(x1, region.width + 1.0)
        x1 = max(x1 - 1.0, x0 + 1.0)  # a gap between neighbours
        out.append((x0, top - h, x1, top, STATE_COLORS.get(r.get("state", "queued"), STATE_COLORS["queued"])))
        if core.likely(r):
            out.append((x0, top + 1.0 * ui, x1, top + 3.0 * ui, LIKELY_COLOR))
        if str(r.get("id")) == sel:
            outline = (x0, top - h - 2 * ui, x1, top + 4 * ui)
    if outline is not None:
        x0, y0, x1, y1 = outline
        w = 1.5 * ui
        white = (1.0, 1.0, 1.0, 1.0)
        out += [(x0, y0, x1, y0 + w, white), (x0, y1 - w, x1, y1, white), (x0, y0, x0 + w, y1, white),
                (x1 - w, y0, x1, y1, white)]
    return out


def draw_strip():
    import gpu
    from gpu_extras.batch import batch_for_shader

    context = bpy.context
    scene, region = context.scene, context.region
    if scene is None or region is None:
        return
    rects = _strip_rects(scene, region, context.preferences.system.ui_scale)
    if not rects:
        return
    pos, col = [], []
    for x0, y0, x1, y1, c in rects:
        pos += [(x0, y0), (x1, y0), (x1, y1), (x0, y0), (x1, y1), (x0, y1)]
        col += [c] * 6
    shader = gpu.shader.from_builtin("SMOOTH_COLOR")
    batch = batch_for_shader(shader, "TRIS", {"pos": pos, "color": col})
    gpu.state.blend_set("ALPHA")
    shader.bind()
    batch.draw(shader)
    gpu.state.blend_set("NONE")


_HANDLES: list = []


def register():
    if bpy.app.background:
        return
    v3d = bpy.types.SpaceView3D
    _HANDLES.append((v3d, v3d.draw_handler_add(draw_pick, (), "WINDOW", "POST_PIXEL")))
    dope = bpy.types.SpaceDopeSheetEditor
    _HANDLES.append((dope, dope.draw_handler_add(draw_strip, (), "WINDOW", "POST_PIXEL")))


def unregister():
    while _HANDLES:
        space, handle = _HANDLES.pop()
        try:
            space.draw_handler_remove(handle, "WINDOW")
        except (ValueError, RuntimeError):
            pass
    PICK.reset()
    SESSION.clear()
