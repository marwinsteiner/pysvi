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


def test_crossed_bid_ask_band_rejected():
    """A crossed band (w_bid > w_ask) raises instead of silently fitting
    toward an impossible band -- the usual cause is swapped kwargs."""
    with pytest.raises(ValueError, match="crossed band"):
        SVI().calibrate(
            _K_GRID, _W_TRUE, objective="bid_ask",
            w_bid=_W_TRUE + 5e-4, w_ask=_W_TRUE - 5e-4,  # swapped
        )
    # a single crossed quote is enough
    w_bid = _W_TRUE - 5e-4
    w_ask = _W_TRUE + 5e-4
    w_bid_bad = w_bid.copy(); w_bid_bad[3] = w_ask[3] + 1e-6
    with pytest.raises(ValueError, match="1 quote"):
        SVI().calibrate(_K_GRID, _W_TRUE, objective="bid_ask",
                        w_bid=w_bid_bad, w_ask=w_ask)


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


# Real SPY slice (2026-09-17 live-book snapshot, T ~ 0.119): see the
# validity-aware-selection regression test below for provenance.
_IV_REAL_K = np.array([
    -0.2839548281071514, -0.2752967653640367, -0.2667130216726453,
    -0.2582023320047366, -0.24976346335887203, -0.24139521368835534,
    -0.23309641087366031, -0.2248659117371449, -0.2167026010979839,
    -0.2086053908653646, -0.20057321916810025, -0.19260504951892343,
    -0.18469987001181007, -0.17685669255078423, -0.17529541198383167,
    -0.17373656521454062, -0.17062614280014815, -0.1690745521087292,
    -0.16752536512189992, -0.16597857440360128, -0.1644341725522271,
    -0.16289215220041184, -0.16135250601481899, -0.1582803069778488,
    -0.15674773962807076, -0.15521751744730306, -0.15368963326924992,
    -0.15216407996041273, -0.1506408504198913, -0.14911993757918457,
    -0.1476013344019946, -0.14608503388403052, -0.14457102905281552,
    -0.14305931296749355, -0.1415498787186393, -0.1400427194280679,
    -0.1385378282486477, -0.1370351983641126, -0.1355348229888781,
    -0.134036695367856, -0.13104715651949, -0.12955573193282002,
    -0.12806652838135218, -0.126579539259774, -0.12509475799219472,
    -0.12361217803197197, -0.12213179286153765, -0.1206535959922268,
    -0.11917758096410616, -0.11770374134580566, -0.11623207073434935,
    -0.11476256275498943, -0.11329521106103962, -0.11183000933371172,
    -0.11036695128195126, -0.1089060306422765, -0.1074472411786165,
    -0.10599057668215245, -0.10453603097115806, -0.1030835978908432,
    -0.10163327131319665, -0.10018504513683188, -0.09873891328683215,
    -0.09729486971459854, -0.09585290839769733, -0.09441302333970963,
    -0.09297520857008218, -0.09153945814397789, -0.09010576614212935,
    -0.08867412667069151, -0.08724453386109715, -0.08581698186991167,
    -0.0843914648786907, -0.08296797709383685, -0.08154651274645942,
    -0.0801270660922331, -0.07870963141125974, -0.077294203007929,
    -0.07588077521078206, -0.07446934237237425, -0.0730598988691407,
    -0.07165243910126087, -0.07024695749252605, -0.06884344849020581,
    -0.06744190656491741, -0.06604232621049409, -0.06464470194385619,
    -0.06324902830488138, -0.06185529985627762, -0.060463511183455054,
    -0.059073656894400806, -0.05768573161955266, -0.05629973001167557,
    -0.05491564674573705, -0.05353347651878543, -0.05215321404982696,
    -0.0507748540797057, -0.04939839137098236, -0.04802382070781582,
    -0.04665113689584358, -0.04528033476206497, -0.043911409154723226,
    -0.04254435494319023, -0.04117916701785023, -0.0398158402899862,
    -0.03845436969166505, -0.03709475017562554, -0.03573697671516512,
    -0.03438104430402928, -0.03302694795630006, -0.03167468270628634,
    -0.03032424360841494, -0.028975625737121324, -0.027628824186742663,
    -0.026283834071409905, -0.02494065052494231, -0.023599268700740916,
    -0.022259683771684517, -0.02092189093002452, -0.019585885387282342,
    -0.018251662374145636, -0.016919217140367072, -0.015588544954661934,
    -0.014259641104608237, -0.012932500896545684, -0.011607119655477107,
    -0.010283492724968757, -0.008961615467053033, -0.007641483262130095,
    -0.006323091508871881, -0.005006435624124987, -0.0036915110428159675,
    -0.002378313217855473, -0.001066837620044796, 0.0002429202620187468,
    0.0015509649220340385, 0.002857300836089446, 0.004161932462754501,
    0.005464864243170426, 0.00676610060114218, 0.008065645943227685,
    0.009363504658827808, 0.010659681120274435, 0.011954179682920172,
    0.013247004685225272, 0.014538160448845318, 0.01582765127871702,
    0.017115481463145656, 0.018401655273889764, 0.01968617696624659,
    0.020969050779135693, 0.022250280935184184, 0.023529871640809247,
    0.024807827086301432, 0.026084151445906124, 0.027358848877906655,
    0.028631923524704717, 0.029903379512901347, 0.031173220953377406,
    0.03244145194137247, 0.03370807655656535, 0.034973098863151995,
    0.03623652290992414, 0.03749835273034622, 0.0387585923426339,
    0.04001724574983003, 0.0412743169398813, 0.042529809885713275,
    0.04378372854530695, 0.0450360768617728, 0.046286858763425594,
    0.047536078163857465, 0.048783738962012685, 0.05002984504225981,
    0.05127440027446467, 0.05251740851406162, 0.053758873602126486,
    0.054998799365446945, 0.056237189616593686, 0.05747404815398994,
    0.05870937876198265, 0.05994318521091107, 0.06117547125717626,
    0.062406240643308845, 0.06363549709803851, 0.06486324433636087,
    0.06608948605960532, 0.06731422595550114, 0.06853746769824527,
    0.06975921494856759, 0.07097947135379712, 0.07219824054792649,
    0.07341552615167814, 0.07463133177256798, 0.07584566100497002,
    0.07705851743017926, 0.07826990461647629, 0.07947982611918948,
    0.08068828548075802, 0.08189528623079324, 0.08310083188614183,
    0.08430492595094637, 0.08670877326234097, 0.08790853345424499,
    0.08910685594635366, 0.09507702293285743, 0.10101175845267214,
    0.10691148057986039, 0.11860752034305157, 0.13589901745311264,
])
_IV_REAL_W = np.array([
    0.016969329813203305, 0.01627559961622058, 0.015680676290949363,
    0.015078718799531252, 0.014472505042860212, 0.013937071030334125,
    0.013390399495960877, 0.012835848776175043, 0.012332247947785143,
    0.011765767227082127, 0.011246532138447675, 0.010764672795803634,
    0.010272471111510187, 0.009737180185128341, 0.005396960161221751,
    0.009592901409355937, 0.00937723679508806, 0.00933513139203576,
    0.009226565742104259, 0.009149686698931327, 0.009040976808908259,
    0.008962835840430932, 0.008883808312511832, 0.008666025319345143,
    0.008585727352061772, 0.008532228263769993, 0.00844993591891958,
    0.008366986790275763, 0.008283416951792777, 0.008173807738839677,
    0.008114553558504595, 0.00800485038786986, 0.007943611637131521,
    0.007833900026541047, 0.007747754575595806, 0.007661201566706697,
    0.0076075603333244455, 0.007508765602340785, 0.007442054209730379,
    0.007353328713383711, 0.007215236940606863, 0.007125044671962433,
    0.007053920408935836, 0.0069630378066013405, 0.006890513364831063,
    0.00681715084645839, 0.0067429931857003215, 0.006685411079283109,
    0.0065924568394316796, 0.006532781789137736, 0.006471764954021125,
    0.006393561099726281, 0.006330388747314778, 0.006235531375392425,
    0.006178197782548019, 0.00611945848137977, 0.006037928904314403,
    0.005963032121776397, 0.005914914088913697, 0.005824797417848757,
    0.005760960086042939, 0.0057088822581241195, 0.005642657356166701,
    0.005575459426110056, 0.005501322882401663, 0.005438333965666918,
    0.005374253400731616, 0.005309133140530741, 0.005248530544014495,
    0.005181360144266152, 0.005118567798756752, 0.005059852785070188,
    0.0050099627764599065, 0.004938727151486504, 0.004881248666502409,
    0.004827184841642215, 0.0047578490241240575, 0.004710273833586934,
    0.004652157245046343, 0.004592775712718946, 0.00453639985670348,
    0.004478685003010074, 0.004411631851201823, 0.004367353276599491,
    0.004317352357651687, 0.0042581958978208075, 0.004201503249974867,
    0.004139879801057883, 0.004101805659787851, 0.00402713007162295,
    0.0039889706883336545, 0.003938779494414873, 0.003883737587365076,
    0.0038335809768501572, 0.0037725374609376603, 0.003728423280813629,
    0.0036765021171233803, 0.0036287790722345967, 0.0035764684048191888,
    0.003544701748889809, 0.003483415495525283, 0.0034447506366098823,
    0.0034013156415695473, 0.003355890665351793, 0.003308547224586226,
    0.0032496180168704258, 0.003210760728084931, 0.003169689721053592,
    0.0031241897341034415, 0.003087934546045842, 0.0030471231216138896,
    0.003004117005729857, 0.0029547523838432464, 0.002915944478139364,
    0.0028850146312217194, 0.002837446501551968, 0.0027957026371876284,
    0.0027613399461349107, 0.002716839197384028, 0.0026831674162291217,
    0.002652392385676759, 0.0026152775785546565, 0.0025791714826274717,
    0.002542211173105027, 0.002525115864663904, 0.002475830120141958,
    0.0024460759104293042, 0.0024217392640989532, 0.002381221814737663,
    0.0023509886185776553, 0.0023242104052921342, 0.0023023209121996262,
    0.00226936505031759, 0.002255296772573422, 0.0019585309031240282,
    0.0019351272756437219, 0.001898418931686824, 0.0018857511184389945,
    0.0018554290483716687, 0.0018317465026326475, 0.0018032637179804805,
    0.0017812557659216935, 0.0017543640047390284, 0.0017310149446275308,
    0.001716844398323378, 0.0016948583525246998, 0.0016749430505302429,
    0.0016585509323215897, 0.0016355960414650927, 0.0016205349872059837,
    0.0015988090639269298, 0.00158366109213937, 0.0015662503778093937,
    0.0015541974618035, 0.001538392858596713, 0.0015202703170276318,
    0.0015143666690911656, 0.0015032232106934565, 0.0014967627386540494,
    0.0014849314013008013, 0.0014709735124924326, 0.0014621465546511042,
    0.0014589798598715238, 0.0014503801964675096, 0.0014439912358458712,
    0.0014359788150211585, 0.0014284432258366054, 0.0014260243117057438,
    0.001417839685792505, 0.00141538699726943, 0.0014094784729108921,
    0.0014076528336566733, 0.0014078619021118254, 0.0014020045292148751,
    0.001404244400660721, 0.001406553320996352, 0.0014058482428198707,
    0.001408641382476603, 0.0014013605838214143, 0.0014125673039787389,
    0.001406008002565503, 0.0014202839717221188, 0.001415549183614062,
    0.0014253111920264536, 0.0014233681388791202, 0.0014381579805974808,
    0.0014513193756668535, 0.001451691796255919, 0.0014606086147009516,
    0.0014673697909839106, 0.0014717581602484795, 0.0014870998719541395,
    0.0015009896412721338, 0.0015133000947751278, 0.0015238871072897203,
    0.0015325860902628643, 0.0015392071471376475, 0.0015617072317195935,
    0.0015645657917473551, 0.0015847838090150388, 0.0016041182442511597,
    0.0016224987092844252, 0.00165608055451514, 0.0016711018076451523,
    0.001684805665077881, 0.001761952565688823, 0.0018829964152490186,
    0.001990353234919115, 0.010426534669864223, 0.01263764994083014,
])

def test_validity_aware_selection_real_slice_regression():
    """Bound-pinned winners no longer veto the whole fit; unreliable
    success flags no longer discard good basins.

    Full-precision quotes from a 2026-09-17 live-book SPY snapshot
    (T ~ 0.119, 210 quotes). Under objective='implied_vol' with
    NO_BUTTERFLY this slice exposed two stacked defects:

    1. multi_start's best-of winner sat exactly AT |rho| = 0.999, and
       the post-fit validity check then returned None for the whole
       fit while valid runners-up were discarded -- selection now
       filters invalid candidates via a per-model validity predicate.
    2. Every good candidate (20x lower objective than the surviving
       trivial flat fits) terminated ABNORMAL even after a
       default-tolerance polish -- L-BFGS-B stalls below achievable
       precision at perfectly good minima on noisy objectives -- so
       the converged-only rule kept only degenerate b -> 0 fits.
       Settled endpoints are now accepted regardless of the status
       flag; the objective value is the oracle, not scipy's status.
    """
    from src.pysvi.models import ArbitrageFreedom

    model = SVI(ArbitrageFreedom.NO_BUTTERFLY)
    p_default = model.calibrate(_IV_REAL_K, _IV_REAL_W,
                                objective="implied_vol")
    p_multi = model.calibrate(_IV_REAL_K, _IV_REAL_W,
                              objective="implied_vol",
                              initialization="multi_start")
    assert p_default is not None and p_multi is not None
    assert abs(p_multi["rho"]) < 0.999 and p_multi["b"] > 1e-4

    def vol_rmse(p):
        w_fit = svi_total_variance(
            _IV_REAL_K, **{n: p[n] for n in ("a", "b", "rho", "m", "sigma")})
        return float(np.sqrt(np.mean(
            (np.sqrt(np.maximum(w_fit, 0.0)) - np.sqrt(_IV_REAL_W)) ** 2)))

    # the degenerate flat-fit family scores ~2.4e-2 here; the good
    # basin ~5.3e-3 -- both paths must land in the good basin, and
    # multi_start must not lose to default
    assert vol_rmse(p_default) < 1e-2
    assert vol_rmse(p_multi) < 1e-2
    assert vol_rmse(p_multi) <= vol_rmse(p_default) * (1.0 + 1e-6)
