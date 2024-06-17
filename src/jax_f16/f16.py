import jax.numpy as jnp
import numpy as np

from jax_f16.f16_types import FULL_STATE_NX, OUTER_CONTROL_NU, FullState, OuterControl
from jax_f16.f16_utils import Vec2, Vec3, rotx, roty, rotz
from jax_f16.highlevel.controlled_f16 import controlled_f16
from jax_f16.utils.jax_types import FloatScalar


class F16:
    NX = FULL_STATE_NX
    NU = OUTER_CONTROL_NU

    VT, ALPHA, BETA, PHI, THETA, PSI, P, Q, R, PN, PE, H, POW, NZINT, PSINT, NYRINT = range(NX)
    NZ, PS, NYR, THRTL = range(NU)

    # ENU
    POS = np.array([PE, PN, H])
    POS_NEU = np.array([PN, PE, H])
    POS2D = np.array([PE, PN])
    POS2D_NED = np.array([PN, PE])
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

    @staticmethod
    def state(vt: FloatScalar, alphabeta: Vec2, rpy: Vec3, pqr: Vec3, pos: Vec3, pow: FloatScalar, ints: Vec3):
        vt = np.array([vt])
        pow = np.array([pow])
        arrs = [alphabeta, rpy, pqr, pos, ints]
        alphabeta, rpy, pqr, pos, ints = [np.array(a) for a in arrs]
        assert vt.shape == (1,)
        assert alphabeta.shape == (2,)
        assert rpy.shape == (3,)
        assert pqr.shape == (3,)
        assert pos.shape == (3,)
        assert pow.shape == (1,)
        assert ints.shape == (3,)
        state = np.concatenate([vt, alphabeta, rpy, pqr, pos, pow, ints], axis=0)
        assert state.shape == (F16.NX,)
        return state


def compute_f16_vel_angles(state: FullState) -> Vec3:
    """Compute cos / sin of [gamma, sigma], the pitch & yaw of the velocity vector."""
    assert state.shape == (F16.NX,)
    # 1: Compute {}^{W}R^{F16}.
    R_W_F16 = rotz(state[F16.PSI]) @ roty(state[F16.THETA]) @ rotx(state[F16.PHI])
    assert R_W_F16.shape == (3, 3)

    # 2: Compute v_{F16}
    ca, sa = jnp.cos(state[F16.ALPHA]), jnp.sin(state[F16.ALPHA])
    cb, sb = jnp.cos(state[F16.BETA]), jnp.sin(state[F16.BETA])
    v_F16 = jnp.array([ca * cb, sb, sa * cb])

    # 3: Compute v_{W}
    v_W = R_W_F16 @ v_F16
    assert v_W.shape == (3,)

    # 4: Back out cos and sin of gamma and sigma.
    cos_sigma = v_W[0]
    sin_sigma = v_W[1]
    sin_gamma = v_W[2]

    out = jnp.array([cos_sigma, sin_sigma, sin_gamma])
    assert out.shape == (3,)
    return out
