"""VolSurface.save / load: versioned JSON schema round-trips."""

import json

import numpy as np
import pytest

from src.pysvi import VolSurface, calibrate_surface
from src.pysvi.models import ArbitrageFreedom

from tests.conftest import SURFACE_RATE as R


@pytest.mark.parametrize("model", ["svi", "natural", "ssvi", "essvi", "jw", "dsvi", "sabr"])
def test_round_trip_all_models(surface_df, tmp_path, model):
    surface = VolSurface.fit(surface_df, model=model, r=R)
    path = tmp_path / f"{model}.json"
    surface.save(path)
    loaded = VolSurface.load(path)

    K = np.array([92.0, 100.0, 108.0])
    for T in surface.maturities:
        np.testing.assert_array_equal(loaded.iv(K, T), surface.iv(K, T))
    np.testing.assert_array_equal(loaded.price(K, 0.7), surface.price(K, 0.7))
    assert type(loaded.model) is type(surface.model)
    assert loaded.r == surface.r
    assert loaded.interp_method == surface.interp_method


def test_report_and_condition_survive(surface_df, tmp_path):
    surface = calibrate_surface(
        surface_df, model="ssvi",
        arbitrage_condition=ArbitrageFreedom.NO_BUTTERFLY, r=R,
    )
    path = tmp_path / "s.json"
    surface.save(path)
    loaded = VolSurface.load(path)
    assert ArbitrageFreedom.NO_BUTTERFLY in loaded.model.arbitrage_condition
    assert ArbitrageFreedom.NO_CALENDAR in loaded.model.arbitrage_condition
    assert loaded.fit_report is not None
    assert loaded.fit_report.calendar_enforced
    assert loaded.fit_report.n_ok == surface.fit_report.n_ok
    assert str(loaded.fit_report) == str(surface.fit_report)


def test_schema_is_documented_json(surface_df, tmp_path):
    surface = VolSurface.fit(surface_df, model="svi", r=R)
    path = tmp_path / "s.json"
    surface.save(path)
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == 1
    assert payload["model"] == "SVI"
    assert {"pysvi_version", "r", "interp_method", "slices", "fit_report"} <= set(payload)
    assert all({"maturity", "params"} <= set(item) for item in payload["slices"])


def test_unknown_schema_and_model_rejected(surface_df, tmp_path):
    surface = VolSurface.fit(surface_df, model="svi")
    path = tmp_path / "s.json"
    surface.save(path)
    payload = json.loads(path.read_text())

    bad = dict(payload, schema_version=99)
    p1 = tmp_path / "bad_version.json"
    p1.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="schema_version"):
        VolSurface.load(p1)

    bad = dict(payload, model="HestonQuadratic")
    p2 = tmp_path / "bad_model.json"
    p2.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="unknown model"):
        VolSurface.load(p2)
