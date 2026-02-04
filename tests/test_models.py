import pytest
from hypothesis import given, strategies as st
from src.pysvi.models import *
from src.pysvi.calibration import get_model


def test_svi_total_variance_formula():
    """Verify implementation matches published formula."""
    k = np.array([0.0, 0.1, -0.1])
    # a + b{ρ(k-m) + sqrt[(k-m)² + σ²]}
    expected = np.array([0.03, 0.0274, 0.0374])
    result = svi_total_variance(k, a=0.01, b=0.1, rho=-0.5, m=0.0, sigma=0.2)
    np.testing.assert_allclose(result, expected, rtol=1e-2)


@given(k=st.lists(st.floats(-1.0, 1.0), min_size=1, max_size=100))
def test_svi_positivity(k):
    """SVI total variance always non-negative."""
    k = np.array(k)
    w = svi_total_variance(k, 0.01, 0.1, -0.5, 0.0, 0.2)
    assert np.all(w >= -1e-8)  # Numerical tolerance


def test_models_factory():
    """Factory returns correct instances."""
    assert isinstance(get_model("svi"), SVI)
    assert isinstance(get_model("SSVI"), SSVI)
    assert isinstance(get_model("EsSvI"), ESSVI)

    with pytest.raises(KeyError, match="invalid"):
        get_model("invalid")


def test_parametrization_abstract():
    """ABC cannot be instantiated."""
    from src.pysvi.models import Parametrization
    with pytest.raises(TypeError):
        Parametrization()
