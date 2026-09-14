"""Calibration controls: objectives, robust losses, initialisation."""

import numpy as np
import pytest

from src.pysvi.models import (
    SVI, NaturalSVI, SSVI, SABR, svi_total_variance,
)
from src.pysvi import _kernels as K

TRUE = {"a": 0.01, "b": 0.12, "rho": -0.6, "m": 0.01, "sigma": 0.25}
_K_GRID = np.linspace(-0.4, 0.4, 41)
_W_TRUE = svi_total_variance(_K_GRID, **TRUE)


def _fit_rmse(params, w_ref=_W_TRUE, k=_K_GRID):
    w_fit = svi_total_variance(k, **{p: params[p] for p in ("a", "b", "rho", "m", "sigma")})
    return float(np.sqrt(np.mean((w_ref - w_fit) ** 2)))


# ── Objectives ───────────────────────────────────────────────────────

@pytest.mark.parametrize("objective", ["total_variance", "implied_vol", "price", "vega_weighted"])
def test_objectives_recover_clean_smile(objective):
    """Every residual space recovers a clean smile to tiny w-RMSE.

    multi_start decouples objective-space correctness from the raw-SVI
    landscape's local minima (the default start can land in a poor basin).
    """
    params = SVI().calibrate(
        _K_GRID, _W_TRUE, objective=objective, initialization="multi_start"
    )
    assert params is not None
    assert _fit_rmse(params) < 1e-5, objective


def test_black_call_matches_py_vollib():
    """The price-space kernel matches py_vollib's Black formula."""
    from py_vollib.black import black
    T = 0.25
    for k_i, w in [(-0.2, 0.012), (0.0, 0.01), (0.15, 0.011), (0.4, 0.02)]:
        expected = black("c", 1.0, float(np.exp(k_i)), T, 0.0, float(np.sqrt(w / T)))
        got = K._PLAIN["black_call"](k_i, w)
        np.testing.assert_allclose(got, expected, rtol=1e-10)


def test_bid_ask_objective_stays_inside_band(atm_slice):
    """bid_ask calibration lands inside the quoted band where feasible."""
    noisy = _W_TRUE * (1.0 + 0.02 * np.sin(37.0 * _K_GRID))  # noisy mids
    w_bid = _W_TRUE - 5e-4
    w_ask = _W_TRUE + 5e-4
    params = SVI().calibrate(
        _K_GRID, noisy, objective="bid_ask", w_bid=w_bid, w_ask=w_ask
    )
    assert params is not None
    w_fit = svi_total_variance(_K_GRID, **params)
    violation = np.maximum(w_bid - w_fit, 0.0) + np.maximum(w_fit - w_ask, 0.0)
    assert float(violation.max()) < 1e-6, f"band violation {violation.max():.2e}"


def test_unknown_objective_and_loss_raise():
    with pytest.raises(ValueError, match="unknown objective"):
        SVI().calibrate(_K_GRID, _W_TRUE, objective="prices")
    with pytest.raises(ValueError, match="unknown loss"):
        SVI().calibrate(_K_GRID, _W_TRUE, loss="l1")
    with pytest.raises(ValueError, match="w_bid"):
        SVI().calibrate(_K_GRID, _W_TRUE, objective="bid_ask")


# ── Robust losses ────────────────────────────────────────────────────

def test_robust_loss_resists_corrupted_wing():
    """One corrupted far-wing quote distorts l2 far more than soft_l1."""
    w_bad = _W_TRUE.copy()
    w_bad[-1] *= 1.5  # single bad call-wing quote
    clean = slice(0, -1)

    p_l2 = SVI().calibrate(_K_GRID, w_bad)
    p_robust = SVI().calibrate(_K_GRID, w_bad, loss="soft_l1")
    assert p_l2 is not None and p_robust is not None

    rmse_l2 = _fit_rmse(p_l2, _W_TRUE[clean], _K_GRID[clean])
    rmse_robust = _fit_rmse(p_robust, _W_TRUE[clean], _K_GRID[clean])
    assert rmse_robust < rmse_l2, (rmse_robust, rmse_l2)
    assert rmse_robust < 1e-4


@pytest.mark.parametrize("loss", ["huber", "soft_l1", "cauchy"])
def test_robust_losses_recover_clean_smile(loss):
    """Robust losses reproduce the l2 fit on clean data (multi_start
    decouples the check from raw SVI's local-minimum landscape)."""
    params = SVI().calibrate(
        _K_GRID, _W_TRUE, loss=loss, initialization="multi_start"
    )
    assert params is not None
    assert _fit_rmse(params) < 1e-4, loss


def test_explicit_f_scale_accepted():
    params = SVI().calibrate(
        _K_GRID, _W_TRUE, loss="huber", f_scale=1e-4,
        initialization="multi_start",
    )
    assert params is not None
    assert _fit_rmse(params) < 1e-4


# ── Initialisation ───────────────────────────────────────────────────

def test_multi_start_deterministic():
    """Two multi_start runs give bitwise-identical parameters."""
    p1 = SVI().calibrate(_K_GRID, _W_TRUE, initialization="multi_start")
    p2 = SVI().calibrate(_K_GRID, _W_TRUE, initialization="multi_start")
    assert p1 is not None
    assert p1 == p2


def test_multi_start_escapes_bad_basin():
    """Regression: the default start converges to a poor local optimum on
    this clean smile under tight tolerances; multi_start recovers the
    true fit (review-issue acceptance case for pysvi#9)."""
    p_default = SVI().calibrate(_K_GRID, _W_TRUE)
    p_multi = SVI().calibrate(_K_GRID, _W_TRUE, initialization="multi_start")
    assert p_default is not None and p_multi is not None
    assert _fit_rmse(p_multi) < 1e-5
    assert _fit_rmse(p_multi) < _fit_rmse(p_default) / 10.0


def test_multi_start_never_worse_than_default():
    """multi_start's best-of includes a run of the default start under
    the default path's own options, so it can never lose to it."""
    rng = np.random.default_rng(7)
    w_noisy = _W_TRUE + 2e-5 * rng.standard_normal(_K_GRID.size)
    p_default = SVI().calibrate(_K_GRID, w_noisy)
    p_multi = SVI().calibrate(_K_GRID, w_noisy, initialization="multi_start")
    assert p_default is not None and p_multi is not None
    mse_default = float(np.mean((w_noisy - svi_total_variance(_K_GRID, **p_default)) ** 2))
    mse_multi = float(np.mean((w_noisy - svi_total_variance(_K_GRID, **p_multi)) ** 2))
    assert mse_multi <= mse_default + 1e-14


def test_multi_start_never_worse_all_models():
    """The multi_start >= default invariant, across the SVI family.

    Regression for a real-data failure: on a steep short-dated equity
    put skew, every tight-tolerance multi-start run of jump-wings ended
    with L-BFGS-B's ABNORMAL status (line search exhausted below the
    achievable precision) at good points, all were discarded by the
    converged-only selection, and a bound-pinned early-"convergence"
    fit won -- 7x worse than the plain default path. Selection now
    compares objective values over all finite results and always
    includes the default start under the default path's own options.
    """
    from src.pysvi.models import ESSVI, JumpWings

    # Steep left wing, tiny ATM total variance -- the shape (taken from
    # a fitted real SPY slice, T ~ 0.11) that triggered the failure.
    steep = {"a": -0.0065, "b": 0.053, "rho": -0.70, "m": -0.06, "sigma": 0.20}
    k = np.linspace(-0.41, 0.10, 61)
    rng = np.random.default_rng(11)
    w = svi_total_variance(k, **steep) * (1.0 + 0.02 * rng.standard_normal(k.size))
    T = 0.108
    theta = float(np.interp(0.0, k, w))

    cases = [
        (SVI(), {}),
        (NaturalSVI(), {}),
        (SSVI(), {"theta": theta}),
        (ESSVI(), {"theta": theta, "theta_ref": theta}),
        (JumpWings(), {"T": T}),
    ]
    for model, kwargs in cases:
        name = type(model).__name__
        p_default = model.calibrate(k, w, **kwargs)
        p_multi = model.calibrate(k, w, initialization="multi_start", **kwargs)
        assert p_default is not None, name
        assert p_multi is not None, name
        mse_default = float(np.mean((w - model.total_variance(k, p_default)) ** 2))
        mse_multi = float(np.mean((w - model.total_variance(k, p_multi)) ** 2))
        assert mse_multi <= mse_default * (1.0 + 1e-9) + 1e-16, (
            f"{name}: multi_start MSE {mse_multi:.3e} worse than "
            f"default {mse_default:.3e}"
        )


def test_multi_start_other_models(atm_slice):
    """multi_start runs for SSVI and SABR."""
    from src.pysvi.calibration import prepare_slice
    k, w_target, F = prepare_slice(atm_slice)
    T = float(atm_slice["maturity"].iloc[0])
    theta = float(np.nanmin(atm_slice["iv"] ** 2 * atm_slice["maturity"]))
    p_ssvi = SSVI().calibrate(k, w_target, theta=theta, initialization="multi_start")
    assert p_ssvi is not None
    p_sabr = SABR().calibrate(k, w_target, T=T, F=F, beta=1.0,
                              initialization="multi_start")
    assert p_sabr is not None


def test_jump_wings_initialization():
    """Data-driven wing readoff works for SVI and NaturalSVI."""
    p_svi = SVI().calibrate(_K_GRID, _W_TRUE, initialization="jump_wings")
    assert p_svi is not None
    assert _fit_rmse(p_svi) < 1e-5

    model = NaturalSVI()
    p_nat = model.calibrate(_K_GRID, _W_TRUE, initialization="jump_wings")
    assert p_nat is not None
    w_fit = model.total_variance(_K_GRID, p_nat)
    assert float(np.sqrt(np.mean((_W_TRUE - w_fit) ** 2))) < 1e-5


def test_jump_wings_rejected_elsewhere():
    with pytest.raises(ValueError, match="jump_wings"):
        SSVI().calibrate(_K_GRID, _W_TRUE, theta=0.01, initialization="jump_wings")


def test_unknown_initialization_raises():
    with pytest.raises(ValueError, match="unknown initialization"):
        SVI().calibrate(_K_GRID, _W_TRUE, initialization="random")


# ── Controls compose with the numba backend ──────────────────────────

def test_controls_backend_parity(backend_mode):
    """Objectives and losses run identically under both backends."""
    params = SVI().calibrate(
        _K_GRID, _W_TRUE, objective="vega_weighted", loss="soft_l1",
        initialization="multi_start",
    )
    assert params is not None
    assert _fit_rmse(params) < 1e-4


# Real SPY slice (2026-09-14 snapshot, T ~ 0.108): see the regression
# test below for provenance.
_JW_REAL_K = np.array([
    -0.28774615474558535, -0.2790880920024708, -0.2619936586431707,
    -0.25355478999730613, -0.2451865403267894, -0.23688773751209438,
    -0.22865723837557897, -0.22049392773641793, -0.21239671750379863,
    -0.20436454580653443, -0.19639637615735747, -0.18849119665024425,
    -0.18064801918921827, -0.17286587874716336, -0.1651438326532531,
    -0.1574809599076839, -0.14987636052246464, -0.14232915488708167,
    -0.1348384831579241, -0.12740350467040606, -0.12002339737278345,
    -0.11269735728071058, -0.10542459795163085, -0.09820434997814372,
    -0.09103586049953122, -0.08391839273066716, -0.07685122550757476,
    -0.06983365284892826, -0.06286498353283484, -0.06147705825798679,
    -0.06009105665010967, -0.05870697338417113, -0.05732480315721949,
    -0.05594454068826111, -0.054566180718139824, -0.053189718009416465,
    -0.051815147346249896, -0.05044246353427775, -0.049071661400499,
    -0.04770273579315734, -0.046335681581624315, -0.044970493656284416,
    -0.043607166928420246, -0.04224569633009918, -0.04088607681405965,
    -0.0395283033535992, -0.03817237094246334, -0.03681827459473409,
    -0.03546600934472046, -0.03411557024684893, -0.0327669523755554,
    -0.03142015082517672, -0.030075160709844048, -0.028731977163376315,
    -0.02739059533917501, -0.026051010410118588, -0.024713217568458568,
    -0.023377212025716364, -0.022042989012579752, -0.020710543778801164,
    -0.019379871593096, -0.018050967743042278, -0.016723827534979814,
    -0.015398446293911103, -0.01407481936340284, -0.012752942105487093,
    -0.011432809900564243, -0.010114418147305893, -0.008797762262559086,
    -0.007482837681250044, -0.006169639856289638, -0.004858164258478826,
    -0.0035484063764153703, -0.0022403617164000553, -0.0009340258023445132,
    0.00037060582432023217, 0.0016735376047364032, 0.00297477396270818,
    0.004274319304793708, 0.005572178020393634, 0.006868354481840284,
    0.008162853044486044, 0.009455678046791169, 0.010746833810411237,
    0.012036324640282961, 0.013324154824711621, 0.014610328635455753,
    0.015894850327812383, 0.01717772414070151, 0.01973854500237533,
    0.02101650044786732, 0.022292824807472033, 0.023567522239472588,
    0.02484059688627067, 0.026112052874467325, 0.027381894314943193,
    0.02865012530293828, 0.0299167499181314, 0.031181772224718067,
    0.03244519627149002, 0.03370702609191212, 0.03496726570419983,
    0.03622591911139597, 0.037482990301447056, 0.03873848324727927,
    0.03999240190687296, 0.04124475022333884, 0.042495532124991436,
    0.04374475152542333, 0.044992412323578576, 0.04623851840382572,
    0.047483073636030604, 0.048726081875627575, 0.04996754696369247,
    0.05120747272701294, 0.052445862978159494, 0.053682721515555774,
    0.05491805212354851, 0.05615185857247716, 0.05738414461874216,
    0.05984417045960445, 0.06107191769792683, 0.062298159421171104,
    0.06352289931706695, 0.0647461410598111, 0.06596788831013364,
    0.06718814471536298, 0.07326719079174503, 0.07930950524770772,
    0.09128569629442353, 0.10312015394142629,
])
_JW_REAL_W = np.array([
    0.01718305257442991, 0.01670311680969616, 0.015404595792007444,
    0.01479789045634891, 0.014188702190308973, 0.008710943076332198,
    0.013035033955731896, 0.012542458207308597, 0.012141926547800943,
    0.01127312713968798, 0.010770934776717988, 0.010262791870026372,
    0.009941832828032047, 0.009482575887580716, 0.009076263790856787,
    0.0085388579947952, 0.008163982763742528, 0.007722171839847394,
    0.007294936179077053, 0.006840656026427722, 0.006559105329282249,
    0.006197864895205132, 0.005833847984454328, 0.005504818627596722,
    0.005199451265732981, 0.004868752741985235, 0.004545384471048568,
    0.004297633360375387, 0.00400683966232944, 0.0039675425506684545,
    0.0038716956911894005, 0.0038429681642972845, 0.00379221054975721,
    0.0037526694410430026, 0.0036986516698322243, 0.0036249886032452003,
    0.003544882201059321, 0.0035401405439629977, 0.003509113527083209,
    0.0033201530107899906, 0.003439930987444507, 0.0033113857579762144,
    0.003304419260360834, 0.0032532494303591356, 0.003250436688335602,
    0.003170638808767879, 0.0030905265410275003, 0.0031129158311271918,
    0.0030251870739802047, 0.0030587226233202603, 0.002810795060987052,
    0.002856053830243662, 0.002854267554308997, 0.0028158428567721058,
    0.0026665153869565123, 0.0027006229575161794, 0.0026176307831608527,
    0.00261755182295173, 0.0026289950641568277, 0.0026032353699500087,
    0.002466302801152619, 0.002561373594765329, 0.002481313212416116,
    0.002479905463064765, 0.0022889820929831474, 0.00239234382937435,
    0.0022210850968784348, 0.002289009742178456, 0.0022950322682224046,
    0.002281530047199782, 0.0021366774730645077, 0.002173172169823524,
    0.0022311088095454948, 0.0021893651077367917, 0.0021387647046471154,
    0.0016760877104580357, 0.0019465020651832094, 0.001729164673492172,
    0.0016042717149885647, 0.0017009143087597836, 0.0016871538506769668,
    0.001624172171426116, 0.0016820827658525779, 0.0015635483642014235,
    0.0017246596289018131, 0.0015585193087812833, 0.0015237659061443259,
    0.0014289554656299747, 0.0014403023246737834, 0.0014427822622588829,
    0.0014277498055241965, 0.0013889489294259532, 0.0013630390269626389,
    0.0013564176540997591, 0.0015018557235067635, 0.0013661113806505672,
    0.001339472138182823, 0.001279338023142242, 0.0012688805130336952,
    0.0012851014281363296, 0.0012990690112581339, 0.0013688059611985366,
    0.001256042135787219, 0.0012330816995117842, 0.0012413767598961826,
    0.0012135270432728775, 0.001275954168623889, 0.0012270495414661376,
    0.0012178697415041822, 0.001194605222906801, 0.00119716665389774,
    0.001202194287483029, 0.0012101937467400286, 0.0013099271977534662,
    0.0012990454160363818, 0.001218587977160532, 0.0013174971294242265,
    0.0012304822920405276, 0.0013189857497339045, 0.0013110750276441603,
    0.0013257276098437615, 0.001260094838280581, 0.0012417063731862098,
    0.001333243484643037, 0.0012676360898536633, 0.001284004150168087,
    0.0012740751988546551, 0.0013345840032333866, 0.001387751593561039,
    0.0016240297406281204, 0.0019693690084878306,
])


def test_jw_multi_start_real_slice_regression():
    """JW multi_start on the exact real SPY slice that exposed the defect.

    Full-precision quotes from a 2026-09-14 SPY snapshot (T ~ 0.108,
    steep put skew, tiny ATM total variance). On this data every tight
    multi-start L-BFGS-B run terminated ABNORMAL at good points and was
    discarded by the old converged-only selection; a bound-pinned fit
    that "converged" early won with a 41x worse objective than the
    plain default path. The values are landscape-sensitive: rounding
    them to even 7 decimals no longer reproduces the cascade.
    """
    from src.pysvi.models import ArbitrageFreedom, JumpWings

    T = 0.10792284588863521
    model = JumpWings(ArbitrageFreedom.NO_BUTTERFLY)
    p_default = model.calibrate(_JW_REAL_K, _JW_REAL_W, T=T)
    p_multi = model.calibrate(_JW_REAL_K, _JW_REAL_W, T=T,
                              initialization="multi_start")
    assert p_default is not None and p_multi is not None
    mse_default = float(np.mean(
        (_JW_REAL_W - model.total_variance(_JW_REAL_K, p_default)) ** 2))
    mse_multi = float(np.mean(
        (_JW_REAL_W - model.total_variance(_JW_REAL_K, p_multi)) ** 2))
    assert mse_multi <= mse_default * (1.0 + 1e-9) + 1e-16
