from __future__ import annotations

import pytest
from pydantic import BaseModel

from afv.core.config import ConfigError, parse_set, resolve_params


class Region(BaseModel):
    arms: float = 1.0
    legs: float = 1.0


class Params(BaseModel):
    strength: float = 0.5
    region: Region = Region()
    maybe: Region | None = None
    free: dict[str, float] = {}


def test_nested_unknown_keys_are_rejected():
    p = resolve_params("smooth", Params, parse_set(["smooth.region.arms=0.3"])["smooth"])
    assert p.region.arms == 0.3
    with pytest.raises(ConfigError, match=r"region\.armz"):
        resolve_params("smooth", Params, {"region": {"armz": 0.3}})
    with pytest.raises(ConfigError, match=r"maybe\.x"):
        resolve_params("smooth", Params, {"maybe": {"x": 1}})
    assert resolve_params("smooth", Params, {"free": {"anything": 2.0}}).free == {"anything": 2.0}


def test_parse_set_errors():
    with pytest.raises(ConfigError, match="unknown stage"):
        parse_set(["smoth.strength=1"])
    with pytest.raises(ConfigError, match="expected"):
        parse_set(["smooth"])
    assert parse_set(["smooth.strength=null"]) == {"smooth": {"strength": None}}
