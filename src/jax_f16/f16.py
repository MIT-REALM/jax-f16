import numpy as np

from jax_f16.f16_types import FULL_STATE_NX, OUTER_CONTROL_NU, FullState, OuterControl
from jax_f16.highlevel.controlled_f16 import controlled_f16


class F16:
    NX = FULL_STATE_NX
    NU = OUTER_CONTROL_NU

    VT, ALPHA, BETA, PHI, THETA, PSI, P, Q, R, PN, PE, H, POW, NZINT, PSINT, NYRINT = range(NX)
    NZ, PS, NYR, THRTL = range(NU)

    # ENU
    POS = np.array([PE, PN, H])
    POS2D = np.array([PE, PN])
    ANGLES = np.array([PHI, THETA, PSI])
    ANGLE_DIMS = np.array([ALPHA, BETA, PHI, THETA, PSI])
    PQR = np.array([P, Q, R])

    def xdot(self, state: FullState, control: OuterControl) -> FullState:
        controlled_f16_out = controlled_f16(state, control)
        return controlled_f16_out.xd

    @staticmethod
    def trim_state() -> FullState:
        vt = 5.02087232e02
        alpha, beta, phi, theta, psi = 3.17104739e-02, 3.81559052e-10, 1.20344518e-08, 3.17104739e-02, 0.0
        P, Q, R = -8.98301794e-12, 1.01512007e-11, 3.78297644e-10
        alt = 3.60002011e03
        power = 7.64291231e00
        nz, ps, pypr = 2.51838260e-03, -1.06352405e-11, 6.53983409e-10
        #               vₜ    α    β      ϕ      θ    ψ  P  Q  R pn pe    h  pow    nz   ps  ny+r
        x = np.array([vt, alpha, beta, phi, theta, psi, P, Q, R, 0, 0, alt, power, nz, ps, pypr])
        return x

    @staticmethod
    def trim_control() -> OuterControl:
        nz, ps, nyr = -5.02734948e-04, -6.16608611e-20, -1.10228341e-08
        thrtl = -2.18080950e-02
        return np.array([nz, ps, nyr, thrtl])
