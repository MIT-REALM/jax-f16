import inspect
from typing import Callable

import ipdb
import jax
import numpy as np
from aerobench.lowlevel.adc import adc as adc_ref
from aerobench.lowlevel.dampp import dampp as dampp_ref
from aerobench.lowlevel.low_level_controller import LowLevelController as LowLevelControllerRef
from aerobench.lowlevel.morellif16 import Morellif16 as morelli_f16_ref
from aerobench.lowlevel.pdot import pdot as pdot_ref
from aerobench.lowlevel.rtau import rtau as rtau_ref
from aerobench.lowlevel.subf16_model import subf16_model as subf16_model_ref
from aerobench.lowlevel.tgear import tgear as tgear_ref
from aerobench.lowlevel.thrust import thrust as thrust_ref

from jax_f16.f16_types import INNER_CONTROL_NU, OUTER_CONTROL_NU, PLANE_STATE_NX
from jax_f16.lowlevel.adc import adc
from jax_f16.lowlevel.dampp import dampp
from jax_f16.lowlevel.morellif16 import morelli_f16
from jax_f16.lowlevel.pdot import pdot
from jax_f16.lowlevel.rtau import rtau
from jax_f16.lowlevel.subf16_model import subf16_model
from jax_f16.lowlevel.tgear import tgear
from jax_f16.lowlevel.thrust import thrust

low_level_fns = [
    (adc, adc_ref),
    (dampp, dampp_ref),
    (morelli_f16, morelli_f16_ref),
    (rtau, rtau_ref),
    (pdot, pdot_ref),
    (tgear, tgear_ref),
    (thrust, thrust_ref),
]

# Check that the results match at least on double precision.
jax.config.update("jax_enable_x64", True)


def check_same_output(my_fn: Callable, ref_fn: Callable, n_params: int, check_iters: int = 384):
    rng = np.random.default_rng(seed=5123)

    for ii in range(check_iters):
        args = rng.uniform(1e-3, 200.0, size=(n_params,))

        my_out = np.array(my_fn(*args))
        ref_out = np.array(ref_fn(*args))

        assert my_out.shape == ref_out.shape
        assert np.allclose(my_out, ref_out)


def test_lowlevel_fns():
    print("[test_f16_ll] test_lowlevel_fns")
    for my_impl, ref_impl in low_level_fns:
        # 1: Number of arguments match.
        my_sig, ref_sig = inspect.signature(my_impl), inspect.signature(ref_impl)
        assert len(my_sig.parameters) == len(ref_sig.parameters)

        n_params = len(my_sig.parameters)

        # 2: Check that they give the same output for random inputs.
        check_same_output(my_impl, ref_impl, n_params)


def test_subf16_model():
    print("[test_f16_ll] test_subf16_model")
    rng = np.random.default_rng(seed=2123844)

    n_iters = 256
    for ii in range(n_iters):
        x = rng.uniform(-80.0, 80.0, size=(PLANE_STATE_NX,))
        u = rng.uniform(-60.0, 60.0, size=(INNER_CONTROL_NU,))

        model = "morelli"
        my_out = subf16_model(x, u, model).to_array()

        a, b, c, d, e = subf16_model_ref(x, u, model)
        ref_out = np.array([*a, b, c, d, e])

        assert np.allclose(my_out, ref_out)


def test_subf16_model2():
    print("[test_f16_ll] test_subf16_model2")
    llc = LowLevelControllerRef()

    x0 = np.array(
        [540, 0.037027160081059704, 0, -0.39269908169872414, -0.47123889803846897, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 1000, 9]
    )
    x0 = np.concatenate([x0, np.zeros(3)], axis=0)
    u_ref = np.zeros(OUTER_CONTROL_NU)

    x_ctrl, u_deg = llc.get_u_deg(u_ref, x0)

    model = "morelli"
    my_xd, _, _, _, _ = subf16_model(x0[0:13], u_deg, model)
    ref_xd, _, _, _, _ = subf16_model_ref(x0[0:13], u_deg, model)

    assert np.allclose(my_xd, ref_xd)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        test_lowlevel_fns()
        test_subf16_model()
        test_subf16_model2()
