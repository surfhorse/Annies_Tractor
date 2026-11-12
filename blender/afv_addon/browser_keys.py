"""Keyboard shortcuts of the shot browser, while the mouse is over the 3D viewport or the timeline of the
AFV Shots scene (elsewhere, and in other scenes, the keys keep their usual meaning: the operator's poll
fails and Blender goes on to its own keymap):

    I / O                 Set in / Set out at the playhead
    Q / S                 Queue / Skip, then the next shot in the list's order
    Shift+Q               Queue & pick (queue the shot, then pick its fighters)
    Page Down / Page Up   next / previous shot in the list's order
    L                     next shot where fighters are likely
    Enter                 Pick fighters
    Space                 play (Blender's own)

The items are registered in the keymaps that would otherwise take these keys first: "Object Mode" and
"3D View" over the 3D viewport (Blender's own I inserts a keyframe, S scales), "Dopesheet" and "Dopesheet
Generic" over the timeline. N stays Blender's (the sidebar, where the AFV panel is). While picking, the
pick's own keys win (Enter saves, Backspace, digits, Esc).
"""

from __future__ import annotations

import bpy
from bpy.props import EnumProperty, StringProperty

from . import browser_jobs as jobs
from .browser import PICK, SESSION, _Op, browser_scene, props, set_state

KEYS = (  # (action, key, shift, text for the panel)
    ("in", "I", False, "I / O: set in / out"),
    ("out", "O", False, ""),
    ("queue", "Q", False, "Q / S: queue / skip, then next"),
    ("skip", "S", False, ""),
    ("queue_pick", "Q", True, "Shift+Q: queue & pick"),
    ("next", "PAGE_DOWN", False, "Page Down / Up: next / previous shot"),
    ("prev", "PAGE_UP", False, ""),
    ("next_likely", "L", False, "L: next likely shot"),
    ("pick", "RET", False, "Enter: pick fighters"),
)
# keymap -> the actions registered there (None: all): "Object Mode" only takes the keys it would otherwise
# use itself (I keys, S scales); the rest go through "3D View"
KEYMAPS = (("Object Mode", "EMPTY", ("in", "out", "skip")), ("3D View", "VIEW_3D", None),
           ("Dopesheet", "DOPESHEET_EDITOR", None), ("Dopesheet Generic", "DOPESHEET_EDITOR", None))
HELP = [text for _a, _k, _s, text in KEYS if text] + ["Space: play"]
_KEYMAPS: list = []


def in_browser(context) -> bool:
    return SESSION.open and context.scene is not None and context.scene == browser_scene()


def queue_and_pick(context, shot: str, invoke: bool):
    """Queue the shot (if it is not), then start picking its fighters."""
    if SESSION.rows[shot].get("state") != "queued":
        set_state(context, shot, "queued", advance=False)
    mode = "INVOKE_DEFAULT" if invoke else "EXEC_DEFAULT"
    return bpy.ops.afv.browser_pick(mode, shot=shot)


class AFV_OT_browser_queue_pick(_Op, bpy.types.Operator):
    """Queue the shot (Run queue runs it), then pick its fighters"""

    bl_idname = "afv.browser_queue_pick"
    bl_label = "Queue & pick"

    shot: StringProperty(description="Shot id (default: the selected shot)")

    @classmethod
    def poll(cls, context):
        return SESSION.open and browser_scene() is not None and not PICK.active

    def _shot(self, context):
        shot = self.shot or props(context).shot
        return shot if shot in SESSION.rows else None

    def execute(self, context):
        shot = self._shot(context)
        if shot is None:
            return self.fail("select a shot first")
        queue_and_pick(context, shot, invoke=False)
        return {"FINISHED"}

    def invoke(self, context, event):
        shot = self._shot(context)
        if shot is None:
            return self.fail("select a shot first")
        queue_and_pick(context, shot, invoke=True)
        return {"FINISHED"}


class AFV_OT_browser_key(_Op, bpy.types.Operator):
    """A shot browser shortcut (I / O, Q / S, Shift+Q, Page Down / Up, L, Enter)"""

    bl_idname = "afv.browser_key"
    bl_label = "Shot browser key"
    bl_options = {"INTERNAL"}  # noqa: RUF012 (Blender's API)

    action: EnumProperty(items=[(a, a, "") for a, *_ in KEYS])

    @classmethod
    def poll(cls, context):
        return in_browser(context) and not PICK.active

    def invoke(self, context, event):
        return self.run(context, invoke=True)

    def execute(self, context):
        return self.run(context, invoke=False)

    def run(self, context, invoke):
        a = self.action
        shot = props(context).shot
        try:
            if a in ("in", "out"):
                return bpy.ops.afv.browser_set_edge(edge=a)
            if a in ("next", "prev"):
                return bpy.ops.afv.browser_nav(step=1 if a == "next" else -1)
            if a == "next_likely":
                return bpy.ops.afv.browser_nav(step=1, likely=True)
            if shot not in SESSION.rows:
                return self.fail("select a shot first (N: the next one)")
            if a in ("queue", "skip"):
                set_state(context, shot, "queued" if a == "queue" else "skipped")
                return {"FINISHED"}
            if a == "pick":
                return bpy.ops.afv.browser_pick("INVOKE_DEFAULT" if invoke else "EXEC_DEFAULT", shot=shot)
            if a == "queue_pick":
                queue_and_pick(context, shot, invoke)
                return {"FINISHED"}
        except RuntimeError as e:  # an operator's poll failed (e.g. no selected shot)
            jobs.set_status(str(e).splitlines()[0], "warning")
            return {"CANCELLED"}
        return {"CANCELLED"}


def register_keymaps():
    kc = bpy.context.window_manager.keyconfigs.addon if bpy.context.window_manager else None
    if kc is None:
        return  # background: no keymaps
    for name, space, only in KEYMAPS:
        km = kc.keymaps.new(name=name, space_type=space)
        for action, key, shift, _text in KEYS:
            if only is not None and action not in only:
                continue
            kmi = km.keymap_items.new(AFV_OT_browser_key.bl_idname, key, "PRESS", shift=shift)
            kmi.properties.action = action
            _KEYMAPS.append((km, kmi))


def unregister_keymaps():
    while _KEYMAPS:
        km, kmi = _KEYMAPS.pop()
        try:
            km.keymap_items.remove(kmi)
        except (ReferenceError, RuntimeError):
            pass


CLASSES = (AFV_OT_browser_queue_pick, AFV_OT_browser_key)
