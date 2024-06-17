import jax.numpy as jnp
import numpy as np

from jax_f16.f16 import F16, compute_f16_vel_angles
from jax_f16.f16_types import FullState, OuterControl, S


def wrap_to_pi(x):
    """Wrap angle to [-pi, pi]."""
    return (x + np.pi) % (2 * np.pi) - np.pi


def get_sin_gamma(x_f16):
    "get the path angle gamma"

    alpha = x_f16[F16.ALPHA]  # AoA           (rad)
    beta = x_f16[F16.BETA]  # Sideslip      (rad)
    phi = x_f16[F16.PHI]  # Roll anle     (rad)
    theta = x_f16[F16.THETA]  # Pitch angle   (rad)

    sin_gamma = (jnp.cos(alpha) * jnp.sin(theta) - jnp.sin(alpha) * jnp.cos(theta) * jnp.cos(phi)) * jnp.cos(beta) - (
        jnp.cos(theta) * jnp.sin(phi)
    ) * jnp.sin(beta)
    return sin_gamma


class F16PIDController:
    def __init__(self, alt_setpoint: float):
        self._alt_setpoint = alt_setpoint
        self._vt_setpoint = 502.0

        self._kp_alt = 0.025
        self._ki_alt = 5e-3
        self._kgamma_alt = 25

        self._kp_vt = 0.25

    def get_control(self, state: FullState) -> OuterControl:
        vt, alpha = state[S.VT], state[S.ALPHA]
        theta = state[S.THETA]
        h = state[S.ALT]

        # Path angle (rad)
        gamma = theta - alpha

        # PI control on altitude, using Nz.
        h_err = self._alt_setpoint - h
        h_err_integral = 0.0
        Nz = self._kp_alt * h_err + self._ki_alt * h_err_integral
        Nz = jnp.clip(Nz, -1.0, 6.0)

        # (Psuedo) Derivative control using path angle
        Nz = Nz - self._kgamma_alt * gamma
        Nz = jnp.clip(Nz, -1.0, 6.0)

        # try to maintain a fixed airspeed near trim point
        throttle = -self._kp_vt * (vt - self._vt_setpoint)

        # Zero Ps and Ny+R
        return jnp.array([Nz, 0.0, 0.0, throttle])


class F16N0PIDController:
    """Cascaded PID controller that tries to stabilize to PN=0, Psi=pi/2."""

    def __init__(self, alt_setpoint: float):
        self._alt_setpoint = alt_setpoint
        self._vt_setpoint = 502.0

        # Gains for speed control
        self.k_vt = 0.25

        # Gains for altitude tracking
        self.k_alt = 0.005
        self.k_h_dot = 0.02

        # Gains for heading tracking
        self.k_prop_psi = 5
        self.k_der_psi = 0.5

        # Gains for roll tracking
        self.k_prop_phi = 0.75
        self.k_der_phi = 0.5
        self.max_bank_deg = 65  # maximum bank angle setpoint

        self.kp_psi_pn = 1e-3
        self.kd_psi_pn = 8e-3

    def get_control(self, state: FullState) -> OuterControl:
        # Compute a desired heading angle based on the error in pN and the error to pi/2.
        return self.get_control_all(state)[0]

    def get_control_all(self, state: FullState) -> tuple[OuterControl, dict[str, float]]:
        psi_cmd_orig = np.pi / 2

        vel_angles = compute_f16_vel_angles(state)[0]
        vn = state[S.VT] * vel_angles

        pn_err = 0.0 - state[F16.PN]
        # Exponential decay.
        vn_err_alpha = 0.05
        vn_cmd = vn_err_alpha * pn_err
        vn_err = vn_cmd - vn

        # Set psi_cmd using this pseudo PD.
        psi_offset_pn_p = -self.kp_psi_pn * pn_err
        psi_offset_pn_d = -self.kd_psi_pn * vn_err
        psi_offset_pn = psi_offset_pn_p + psi_offset_pn_d
        psi_cmd = psi_cmd_orig + psi_offset_pn.clip(-0.9 * np.pi / 2, 0.9 * np.pi / 2)

        # Track the commanded psi by setting the roll angle.
        psi_err = wrap_to_pi(psi_cmd - wrap_to_pi(state[S.PSI]))
        phi_cmd = psi_err * self.k_prop_psi - state[S.R] * self.k_der_psi

        # Bound to acceptable bank angles:
        max_bank_rad = np.deg2rad(self.max_bank_deg)
        phi_cmd = phi_cmd.clip(-max_bank_rad, max_bank_rad)

        # Track the roll angle using the Ps control.
        ps_cmd = (phi_cmd - wrap_to_pi(state[S.PHI])) * self.k_prop_phi - state[S.P] * self.k_der_phi

        # Track altitude for nz cmd.
        h_err = self._alt_setpoint - state[F16.H]
        # If this is positive, then we are falling (because NED)??
        # h_dot = vn[2]
        h_dot = state[F16.VT] * get_sin_gamma(state)

        nz_alt = self.k_alt * h_err - self.k_h_dot * h_dot

        # Get nz to do a level turn. Pull g's to maintain altitude during bank based on trig
        # i.e., if cos(phi) ~ 0, then nz_turn = 1/cos(phi) - 1
        nz_roll = 1 / jnp.cos(state[F16.PHI]) - 1
        nz_roll = jnp.where(jnp.abs(state[F16.PHI]) < 0.1, 0.0, nz_roll)

        # if below target alt (h_err > 0), or we are near level, then allow negative nz.
        # Otherwise, descend in bank (no negative Gs).
        nz = nz_alt + nz_roll
        nz = jnp.where((h_err > 0) | (jnp.abs(state[F16.PHI]) < np.deg2rad(15)), nz, jnp.maximum(0.0, nz))

        # Throttle PID.
        throttle = self.k_vt * (self._vt_setpoint - state[F16.VT])
        throttle = throttle.clip(0.0, 1.0)

        control = jnp.array([nz, ps_cmd, 0.0, throttle])
        return control, {
            "pn_err": pn_err,
            "vn": vn,
            "vn_cmd": vn_cmd,
            "vn_err": vn_err,
            "psi_offset_pn_p": psi_offset_pn_p,
            "psi_offset_pn_d": psi_offset_pn_d,
            "psi_offset_pn": psi_offset_pn,
            "psi_cmd": psi_cmd,
            "phi_cmd": phi_cmd,
        }
