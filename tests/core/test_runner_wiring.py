"""Round 2 wiring of the runner: inherited params (ContactParams from contacts into smooth/export),
the default target, and the real stage graph (ingest -> ... -> hands_face -> export)."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
from pydantic import BaseModel, Field

import afv.export.stage
import afv.media.stage
import afv.models.camera
import afv.models.gemx
import afv.models.track
import afv.motion.hands_face
import afv.motion.stages  # noqa: F401  (register the real stages at collection time)
from afv.core import runner as runner_mod
from afv.core.config import ConfigError
from afv.core.runner import DEFAULT_TARGET, RunError, Runner
from afv.core.stage import STAGE_ORDER, register
from afv.core.types import ActorMotion, BodyMotion


class Contact(BaseModel):
    reach: float = 0.96
    min_plant_frames: int = 3


def _motion(n=4):
    rot = np.zeros((n, 2, 4), np.float32)
    rot[..., 3] = 1
    return BodyMotion(
        frames=np.arange(n), timestamps=np.arange(n) / 30.0, skeleton="soma_provisional", fps=30.0,
        actors={"A": ActorMotion(rot=rot, root=np.zeros((n, 3), np.float32))},
    )  # fmt: skip


@pytest.fixture
def chain(fakes):
    """contacts -> smooth with smooth.contact inherited from contacts' params."""
    seen: dict[str, dict] = {}

    @register("default")
    class Contacts:
        name, version, env, inputs, licences, weight_hash = "contacts", "1", "cpu", (), (), ""
        Params = Contact

        def run(self, ctx, inputs, params):
            seen["contacts"] = params.model_dump()
            return _motion()

    @register("default")
    class Smooth:
        name, version, env, inputs, licences, weight_hash = "smooth", "1", "cpu", ("contacts",), (), ""
        inherit_params: ClassVar[dict[str, str]] = {"contact": "contacts"}

        class Params(BaseModel):
            strength: float = 1.0
            contact: Contact = Field(default_factory=Contact)

        def run(self, ctx, inputs, params):
            seen["smooth"] = params.model_dump()
            return _motion()

    return seen


def test_contact_params_carry_from_contacts_to_smooth(chain, shot):
    # ROUND2_NOTES (E): smooth/retime re-pinned with default ContactParams, not those contacts ran with
    info = shot.info
    info.settings.params = {"contacts": {"reach": 0.9}}
    shot.save_info(info)
    r = Runner(shot)
    assert r.run(to="smooth").outcome == "ok"
    assert chain["contacts"]["reach"] == 0.9
    assert chain["smooth"]["contact"] == {"reach": 0.9, "min_plant_frames": 3}
    # an explicit smooth.contact setting still wins over the inherited value, key by key
    r = Runner(shot, params={"smooth": {"contact": {"min_plant_frames": 5}}})
    assert r.run(to="smooth").outcome == "ok"
    assert chain["smooth"]["contact"] == {"reach": 0.9, "min_plant_frames": 5}


def test_inherited_params_are_part_of_the_key(chain, shot):
    r = Runner(shot)
    r.run(to="smooth")
    key = shot.meta("smooth")["key"]
    info = shot.info
    info.settings.params = {"contacts": {"min_plant_frames": 4}}
    shot.save_info(info)
    # contacts re-runs with the new params; its output (fake) is identical, so the ONLY thing that can
    # make smooth stale is its inherited params
    Runner(shot).run(to="contacts")
    plan = Runner(shot).plan(to="smooth")
    states = {s.name: (s.state, s.reason) for s in plan.stages}
    assert states["contacts"][0] == "fresh"
    assert states["smooth"] == ("stale", "params: contact")
    Runner(shot).run(to="smooth")
    assert shot.meta("smooth")["key"] != key
    assert shot.meta("smooth")["params"]["contact"]["min_plant_frames"] == 4


def test_null_on_an_inherited_field_re_inherits(chain, shot):
    info = shot.info
    info.settings.params = {"contacts": {"reach": 0.9}, "smooth": {"contact": {"min_plant_frames": 5}}}
    shot.save_info(info)
    r = Runner(shot)
    assert r.params_for("smooth", r.resolve("smooth")[0]).contact.min_plant_frames == 5
    # --set smooth.contact=null: back to "as contacts" (not to the Params defaults)
    r = Runner(shot, params={"smooth": {"contact": None}})
    assert r.params_for("smooth", r.resolve("smooth")[0]).contact.model_dump() == {
        "reach": 0.9, "min_plant_frames": 3}
    # a nested null re-inherits that key only
    r = Runner(shot, params={"smooth": {"contact": {"min_plant_frames": None, "reach": 0.5}}})
    assert r.params_for("smooth", r.resolve("smooth")[0]).contact.model_dump() == {
        "reach": 0.5, "min_plant_frames": 3}


def test_inherit_params_cycle_is_an_error(fakes, shot):
    @register("default")
    class Contacts:
        name, version, env, inputs, licences, weight_hash = "contacts", "1", "cpu", (), (), ""
        inherit_params: ClassVar[dict[str, str]] = {"smoothing": "smooth"}

        class Params(BaseModel):
            smoothing: dict = Field(default_factory=dict)

        def run(self, ctx, inputs, params):
            return None

    @register("default")
    class Smooth:
        name, version, env, inputs, licences, weight_hash = "smooth", "1", "cpu", ("contacts",), (), ""
        inherit_params: ClassVar[dict[str, str]] = {"contact": "contacts"}

        class Params(BaseModel):
            contact: dict = Field(default_factory=dict)

        def run(self, ctx, inputs, params):
            return None

    with pytest.raises(RunError, match="circular inherit_params"):
        Runner(shot).run(to="smooth")


def test_stale_reason_names_a_new_param_whose_value_is_none():
    from afv.core.cache import diff_parts

    old = {"params": {"a": 1}, "stage": "x"}
    assert diff_parts(old, {"params": {"a": 1, "up_cam0": None}, "stage": "x"}) == ["params: up_cam0"]


def test_inherit_into_unknown_field_is_an_error(fakes, shot):
    @register("default")
    class Smooth:
        name, version, env, inputs, licences, weight_hash = "smooth", "1", "cpu", (), (), ""
        inherit_params: ClassVar[dict[str, str]] = {"nope": "contacts"}

        class Params(BaseModel):
            pass

        def run(self, ctx, inputs, params):
            return None

    with pytest.raises(RunError, match="unknown field 'nope'"):
        Runner(shot).run(to="smooth")


def test_bad_inherited_upstream_settings_fail_fast(chain, shot):
    info = shot.info
    info.settings.params = {"contacts": {"reach": "far"}}
    shot.save_info(info)
    with pytest.raises(ConfigError):
        Runner(shot).run(to="smooth")


# ---------------------------------------------------------------------------
# The real stage graph (no model is run: only resolution and params)
# ---------------------------------------------------------------------------


def test_default_target_is_export_and_every_stage_has_an_implementation(shot):
    assert DEFAULT_TARGET == "export"
    r = Runner(shot)
    assert r.needed("export") == list(STAGE_ORDER)  # hands_face exists; nothing is skipped
    assert r.resolve("detect_track")[0].inputs == ()  # ingest always runs first; not part of the key
    assert r.resolve("hands_face")[0].inputs == ("smooth",)
    assert r.resolve("export")[0].inputs == ("hands_face",)
    assert "afv.motion.hands_face" in runner_mod.STAGE_MODULES
    assert r.module_status["afv.motion.hands_face"] == "ok"


def test_real_smooth_and_export_inherit_contact_params(shot):
    info = shot.info
    info.settings.params = {"contacts": {"reach": 0.9, "min_plant_frames": 5}}
    info.speed_factor = 0.5
    shot.save_info(info)
    r = Runner(shot)
    for name in ("smooth", "export"):
        p = r.params_for(name, r.resolve(name)[0])
        assert p.contact.reach == 0.9 and p.contact.min_plant_frames == 5, name
    exp = r.params_for("export", r.resolve("export")[0])
    assert exp.speed_factor == 0.5  # shot field injected: retime applies the shot's speed


def test_smooth_key_does_not_depend_on_speed_factor(shot):
    r = Runner(shot)
    cls, impl = r.resolve("smooth")
    parts = r.key_parts("smooth", cls, impl, r.params_for("smooth", cls), {"contacts": "x"}, {})
    assert "speed_factor" not in parts["shot"] and "speed_factor" not in parts["params"]
    cls, impl = r.resolve("export")
    parts = r.key_parts("export", cls, impl, r.params_for("export", cls), {"hands_face": "x"}, {})
    assert parts["shot"]["speed_factor"] == 1.0  # export still depends on it
