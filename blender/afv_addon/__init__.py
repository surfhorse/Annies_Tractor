"""AFV: review and clean up afv takes (fight footage to 3D animation) in Blender.

Sidebar panels (3D viewport, N, tab AFV): "Shots", the shot browser (``browser.py``: a long video's
shots, trimmed, queued, picked and run in the background), and "AFV": Import take, Re-import,
Next/Previous issue, Export, and the cleanup tools (contacts, pins, smoothing, range re-runs, hands,
ID swaps) through one ``afv serve`` worker; the Retarget sub-panel puts a fighter's motion onto your own
character and exports it. See README.md in this folder for install and use. The scene builder
(``builder.py``) is shared with headless ``afv blender``.
"""

from __future__ import annotations

import bpy

from . import browser, cleanup, ops, overlay, retarget_ui, ui

_CLASSES = (*ui.CLASSES, *ops.CLASSES, *cleanup.CLASSES)
_HANDLERS = (
    (bpy.app.handlers.load_post, ops.on_load_post),
    (bpy.app.handlers.save_pre, ops.on_save_pre),
    (bpy.app.handlers.save_post, ops.on_save_post),
)


def register():
    browser.register()  # first: the Shots panel comes before the take panel
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.afv = bpy.props.PointerProperty(type=ui.AFVSceneProps)
    retarget_ui.register()  # after the AFV panel (its parent)
    for handlers, fn in _HANDLERS:
        if fn not in handlers:
            handlers.append(fn)
    overlay.register()


def unregister():
    ops.cancel_job()
    ops.shutdown_serve()
    ops.STATE.review = None
    overlay.unregister()
    for handlers, fn in _HANDLERS:
        if fn in handlers:
            handlers.remove(fn)
    retarget_ui.unregister()
    del bpy.types.Scene.afv
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
    browser.unregister()
