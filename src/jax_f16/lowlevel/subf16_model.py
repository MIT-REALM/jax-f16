from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from jax_f16.f16_types import INNER_CONTROL_NU, PLANE_STATE_NX, C, F16ModelType, InnerControl, PlaneState, S
from jax_f16.lowlevel.adc import adc
from jax_f16.lowlevel.dampp import dampp
from jax_f16.lowlevel.morellif16 import morelli_f16
from jax_f16.lowlevel.pdot import pdot
from jax_f16.lowlevel.tgear import tgear
from jax_f16.lowlevel.thrust import thrust


class SubF16ModelOut(NamedTuple):
    xd: PlaneState
    Nz: float
    Ny: float
    az: float
    ay: float

    def to_array(self):
        return np.array([*self.xd, self.Nz, self.Ny, self.az, self.ay])


class SubF16ModelConstants:
    xcg = 0.35

    s = 300
    b = 30
    cbar = 11.32
    rm = 1.57e-3
    xcgr = 0.35
    he = 160.0
    c1 = -0.770
    c2 = 0.02755
    c3 = 1.055e-4
    c4 = 1.642e-6
    c5 = 0.9604
    c6 = 1.759e-2
    c7 = 1.792e-5
    c8 = -0.7336
    c9 = 1.587e-5
    # Radians to degrees
    rtod = 57.29578
    g = 32.17


def subf16_model(x: PlaneState, u: InnerControl, model: F16ModelType, adjust_cy: bool = True) -> SubF16ModelOut:
    assert x.shape == (PLANE_STATE_NX,)
    assert u.shape == (INNER_CONTROL_NU,)

    k = SubF16ModelConstants()

    # thtlc, el, ail, rdr = u

    # air data computer and engine model
    amach, qbar = adc(x[S.VT], x[S.ALT])
    cpow = tgear(u[C.THROTTLE])

    xd_12 = pdot(x[S.POWER], cpow)

    t = thrust(x[S.POWER], x[S.ALT], amach)
    # dail = ail / 20
    # drdr = rdr / 30

    # component build up.
    assert model == "morelli"
    u_rad = jnp.deg2rad(u[1:])
    cxt, cyt, czt, clt, cmt, cnt = morelli_f16(
        x[S.ALPHA], x[S.BETA], *u_rad, *x[S.PQR], k.cbar, k.b, x[S.VT], k.xcg, k.xcgr
    )

    # add damping derivatives
    tvt = 0.5 / x[S.VT]
    b2v = k.b * tvt
    cq = k.cbar * x[S.Q] * tvt

    alpha_deg = jnp.rad2deg(x[S.ALPHA])
    p, q, r = x[S.PQR]
    phi, theta, psi = x[S.PHI], x[S.THETA], x[S.PSI]

    # get ready for state equations
    d = dampp(alpha_deg)
    cxt = cxt + cq * d[0]
    cyt = cyt + b2v * (d[1] * r + d[2] * p)
    czt = czt + cq * d[3]
    clt = clt + b2v * (d[4] * r + d[5] * p)
    cmt = cmt + cq * d[6] + czt * (k.xcgr - k.xcg)
    cnt = cnt + b2v * (d[7] * r + d[8] * p) - cyt * (k.xcgr - k.xcg) * k.cbar / k.b
    cbta = jnp.cos(x[2])
    u = x[S.VT] * jnp.cos(x[1]) * cbta
    v = x[S.VT] * jnp.sin(x[2])
    w = x[S.VT] * jnp.sin(x[1]) * cbta
    sth = jnp.sin(theta)
    cth = jnp.cos(theta)
    sph = jnp.sin(phi)
    cph = jnp.cos(phi)
    spsi = jnp.sin(psi)
    cpsi = jnp.cos(psi)
    qs = qbar * k.s
    qsb = qs * k.b
    rmqs = k.rm * qs
    gcth = k.g * cth
    qsph = q * sph
    ay = rmqs * cyt
    az = rmqs * czt

    # force equations
    udot = r * v - q * w - k.g * sth + k.rm * (qs * cxt + t)
    vdot = p * w - r * u + gcth * sph + ay
    wdot = q * u - p * v + gcth * cph + az
    dum = u * u + w * w

    xd_0 = (u * udot + v * vdot + w * wdot) / x[S.VT]
    xd_1 = (u * wdot - w * udot) / dum
    xd_2 = (x[S.VT] * vdot - v * xd_0) * cbta / dum

    # kinematics
    xd_3 = p + (sth / cth) * (qsph + r * cph)
    xd_4 = q * cph - r * sph
    xd_5 = (qsph + r * cph) / cth

    # moments
    xd_6 = (k.c2 * p + k.c1 * r + k.c4 * k.he) * q + qsb * (k.c3 * clt + k.c4 * cnt)
    xd_7 = (k.c5 * p - k.c7 * k.he) * r + k.c6 * (r * r - p * p) + qs * k.cbar * k.c7 * cmt
    xd_8 = (k.c8 * p - k.c2 * r + k.c9 * k.he) * q + qsb * (k.c4 * clt + k.c9 * cnt)

    # navigation
    t1 = sph * cpsi
    t2 = cph * sth
    t3 = sph * spsi
    s1 = cth * cpsi
    s2 = cth * spsi
    s3 = t1 * sth - cph * spsi
    s4 = t3 * sth + cph * cpsi
    s5 = sph * cth
    s6 = t2 * cpsi + t3
    s7 = t2 * spsi - t1
    s8 = cph * cth

    xd_9 = u * s1 + v * s3 + w * s6  # north speed
    xd_10 = u * s2 + v * s4 + w * s7  # east speed
    xd_11 = u * sth - v * s5 - w * s8  # vertical speed

    # outputs
    xa = 15.0  # sets distance normal accel is in front of the c.g. (xa = 15.0 at pilot)
    az = az - xa * xd_7  # moves normal accel in front of c.g.

    ####################################
    ###### peter additions below ######
    if adjust_cy:
        ay = ay + xa * xd_8  # moves side accel in front of c.g.

    # For extraction of Nz
    Nz = (-az / k.g) - 1  # zeroed at 1 g, positive g = pulling up
    Ny = ay / k.g

    #              [  vt, alph, beta,  phi, thet,  psi,    P,    Q,    R,   PN,    PE,     H,   POW]
    xd = jnp.array([xd_0, xd_1, xd_2, xd_3, xd_4, xd_5, xd_6, xd_7, xd_8, xd_9, xd_10, xd_11, xd_12])
    assert xd.shape == x.shape

    return SubF16ModelOut(xd, Nz, Ny, az, ay)
