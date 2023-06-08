from typing import NamedTuple

import jax.numpy as jnp

from jax_f16.f16_types import (
    FULL_STATE_NX,
    OUTER_CONTROL_NU,
    PLANE_STATE_NX,
    F16ModelType,
    FullState,
    InnerOuterControl,
    OuterControl,
    PlaneState,
)
from jax_f16.lowlevel.low_level_controller import LowLevelController
from jax_f16.lowlevel.subf16_model import subf16_model


class ControlledF16Out(NamedTuple):
    xd: FullState
    u_rad: InnerOuterControl
    Nz: float
    ps: float
    Ny_r: float


def full_to_plane_state(full_state: FullState) -> PlaneState:
    assert full_state.shape == (FULL_STATE_NX,)
    return full_state[0:PLANE_STATE_NX]


def controlled_f16(
    x_f16: FullState, u_ref: OuterControl, f16_model: F16ModelType = "morelli", v2_integrators=False
) -> ControlledF16Out:
    """Returns the LQR-controlled F-16 state derivatives and more.

    Removed t because dynamics are not time-varying.
    Also removed llc as a parameter because only one such llc is ever used.
    """
    assert not v2_integrators
    assert x_f16.shape == (FULL_STATE_NX,)
    assert u_ref.shape == (OUTER_CONTROL_NU,)
    assert f16_model == "morelli"

    llc = LowLevelController()
    x_ctrl, u_deg = llc.get_u_deg(u_ref, x_f16)

    # Note: Control vector (u) for subF16 is in units of degrees
    plane_state = full_to_plane_state(x_f16)
    xd_model, Nz, Ny, _, _ = subf16_model(plane_state, u_deg, f16_model)

    # Nonlinear (Actual): ps = p * cos(alpha) + r * sin(alpha)
    ps = x_ctrl[4] * jnp.cos(x_ctrl[0]) + x_ctrl[5] * jnp.sin(x_ctrl[0])

    # Calculate (side force + yaw rate) term
    Ny_r = Ny + x_ctrl[5]

    # integrators from low-level controller
    start = len(xd_model)
    end = start + llc.get_num_integrators()
    int_der = llc.get_integrator_derivatives(u_ref, Nz, ps, Ny_r)

    xd = jnp.concatenate([xd_model, int_der], axis=0)
    assert xd.shape == (FULL_STATE_NX,)

    u_rad: InnerOuterControl = jnp.concatenate([jnp.deg2rad(u_deg), u_ref[0:3]])
    return ControlledF16Out(xd, u_rad, Nz, ps, Ny_r)
