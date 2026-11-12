"""``afv serve``'s ``presets`` request (round 5, track R2): the engine presets with their exporter
arguments, for the add-on's Export character. Answered at once, needs no project."""

from __future__ import annotations

from afv.export.presets import PRESETS


def test_presets_request(client):
    res, evs = client.request("presets")
    assert res["event"] == "result" and len(evs) == 1
    assert [p["name"] for p in res["presets"]] == list(PRESETS)
    for p in res["presets"]:
        assert p == PRESETS[p["name"]].to_dict()  # the FBX arguments as afv.export.presets has them
        assert p["fbx"]["use_selection"] is False and p["fbx"]["object_types"] == ["ARMATURE", "MESH"]
    res, _ = client.request("ping")
    assert res["pong"]  # the worker carries on


def test_unknown_cmd_lists_presets(client):
    res, _ = client.request("frobnicate")
    assert res["event"] == "error" and "presets" in res["message"]
