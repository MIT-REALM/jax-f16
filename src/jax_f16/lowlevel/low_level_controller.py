import jax.numpy as jnp
import numpy as np

from jax_f16.f16_types import (
    FULL_STATE_NX,
    FullState,
    InnerControl,
    LLCIntState,
    LQRState,
    OuterControl,
)
from jax_f16.utils.jax_types import FloatScalar


class CtrlLimits:
    "Control Limits"
    ThrottleMax = 1  # Afterburner on for throttle > 0.7
    ThrottleMin = 0
    ElevatorMaxDeg = 25
    ElevatorMinDeg = -25
    AileronMaxDeg = 21.5
    AileronMinDeg = -21.5
    RudderMaxDeg = 30
    RudderMinDeg = -30

    NzMax = 6
    NzMin = -1

    min_control = np.array([ThrottleMin, ElevatorMinDeg, AileronMinDeg, RudderMinDeg])
    max_control = np.array([ThrottleMax, ElevatorMaxDeg, AileronMaxDeg, RudderMaxDeg])


class LowLevelController:
    """low level flight controller"""

    old_k_long = np.array([[-156.8801506723475, -31.037008068526642, -38.72983346216317]], dtype=float)
    old_k_lat = np.array(
        [[37.84483, -25.40956, -6.82876, -332.88343, -17.15997], [-23.91233, 5.69968, -21.63431, 64.49490, -88.36203]],
        dtype=float,
    )

    old_xequil = np.array(
        [502.0, 0.0389, 0.0, 0.0, 0.0389, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 9.0567], dtype=float
    ).transpose()
    old_uequil = np.array([0.1395, -0.7496, 0.0, 0.0], dtype=float).transpose()

    def __init__(self, gain_str="old"):
        # Hard coded LQR gain matrix from matlab version
        assert gain_str == "old"

        # Longitudinal Gains
        K_long = LowLevelController.old_k_long
        K_lat = LowLevelController.old_k_lat

        self.K_lqr = np.zeros((3, 8))
        self.K_lqr[:1, :3] = K_long
        self.K_lqr[1:, 3:] = K_lat

        # equilibrium points from BuildLqrControllers.py
        self.xequil = LowLevelController.old_xequil
        self.uequil = LowLevelController.old_uequil

        # Oswin's changes.
        self.xequil_full = np.zeros(FULL_STATE_NX)
        self.xequil_full[: len(self.xequil)] = self.xequil

    def get_u_deg(self, u_ref: OuterControl, f16_state: FullState) -> tuple[LQRState, InnerControl]:
        """Get the reference commands for the control surfaces"""

        # Calculate perturbation from trim state
        # x_delta = f16_state.copy()
        # x_delta[: len(self.xequil)] -= self.xequil
        x_delta = f16_state - self.xequil_full

        ## Implement LQR Feedback Control
        # Reorder states to match controller:
        # [alpha, q, int_e_Nz, beta, p, r, int_e_ps, int_e_Ny_r]
        x_ctrl = jnp.array([x_delta[i] for i in [1, 7, 13, 2, 6, 8, 14, 15]])

        # Initialize control vectors
        # u_deg = np.zeros((4,))  # throt, ele, ail, rud
        #
        # # Calculate control using LQR gains
        # u_deg[1:4] = np.dot(-self.K_lqr, x_ctrl)  # Full Control
        #
        # # Set throttle as directed from output of getOuterLoopCtrl(...)
        # u_deg[0] = u_ref[3]
        #
        # # Add in equilibrium control
        # u_deg[0:4] += self.uequil

        lqr_control = jnp.dot(-self.K_lqr, x_ctrl)
        u_deg = jnp.concatenate([u_ref[3, None], lqr_control], axis=0)
        u_deg = u_deg + self.uequil

        u_deg = jnp.clip(u_deg, CtrlLimits.min_control, CtrlLimits.max_control)

        # ## Limit controls to saturation limits
        # ctrlLimits = CtrlLimits
        # # Limit throttle from 0 to 1
        # u_deg[0] = max(min(u_deg[0], ctrlLimits.ThrottleMax), ctrlLimits.ThrottleMin)
        # # Limit elevator from -25 to 25 deg
        # u_deg[1] = max(min(u_deg[1], ctrlLimits.ElevatorMaxDeg), ctrlLimits.ElevatorMinDeg)
        # # Limit aileron from -21.5 to 21.5 deg
        # u_deg[2] = max(min(u_deg[2], ctrlLimits.AileronMaxDeg), ctrlLimits.AileronMinDeg)
        # # Limit rudder from -30 to 30 deg
        # u_deg[3] = max(min(u_deg[3], ctrlLimits.RudderMaxDeg), ctrlLimits.RudderMinDeg)

        return x_ctrl, u_deg

    @staticmethod
    def get_num_integrators() -> int:
        """Get the number of integrators in the low-level controller"""
        return 3

    @staticmethod
    def get_integrator_derivatives(
        u_ref: OuterControl, Nz: FloatScalar, ps: FloatScalar, Ny_r: FloatScalar
    ) -> LLCIntState:
        """get the derivatives of the integrators in the low-level controller"""
        return jnp.array([Nz - u_ref[0], ps - u_ref[1], Ny_r - u_ref[2]])
