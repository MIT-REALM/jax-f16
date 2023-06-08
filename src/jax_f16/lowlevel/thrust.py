"""
Stanle Bak
Python F-16
Thrust function
"""

import jax.numpy as jnp

from jax_f16.utils.jax_types import FloatScalar


class ThrustConstants:
    a = jnp.array(
        [
            [1060, 670, 880, 1140, 1500, 1860],
            [635, 425, 690, 1010, 1330, 1700],
            [60, 25, 345, 755, 1130, 1525],
            [-1020, -170, -300, 350, 910, 1360],
            [-2700, -1900, -1300, -247, 600, 1100],
            [-3600, -1400, -595, -342, -200, 700],
        ],
        dtype=float,
    ).T
    b = jnp.array(
        [
            [12680, 9150, 6200, 3950, 2450, 1400],
            [12680, 9150, 6313, 4040, 2470, 1400],
            [12610, 9312, 6610, 4290, 2600, 1560],
            [12640, 9839, 7090, 4660, 2840, 1660],
            [12390, 10176, 7750, 5320, 3250, 1930],
            [11680, 9848, 8050, 6100, 3800, 2310],
        ],
        dtype=float,
    ).T
    c = jnp.array(
        [
            [20000, 15000, 10800, 7000, 4000, 2500],
            [21420, 15700, 11225, 7323, 4435, 2600],
            [22700, 16860, 12250, 8154, 5000, 2835],
            [24240, 18910, 13760, 9285, 5700, 3215],
            [26070, 21075, 15975, 11115, 6860, 3950],
            [28886, 23319, 18300, 13484, 8642, 5057],
        ],
        dtype=float,
    ).T


def thrust(power: FloatScalar, alt: FloatScalar, rmach: FloatScalar) -> FloatScalar:
    """thrust lookup-table version"""
    k = ThrustConstants

    # if alt < 0:
    #     alt = 0.01  # uh, why not 0?
    alt = jnp.where(alt < 0, 0.01, alt)

    h = 0.0001 * alt
    i = jnp.fix(h).astype(jnp.int32)

    # if i >= 5:
    #     i = 4
    i = jnp.clip(i, a_max=4)

    dh = h - i
    rm = 5 * rmach
    m = jnp.fix(rm).astype(jnp.int32)

    # if m >= 5:
    #     m = 4
    # elif m <= 0:
    #     m = 0
    m = jnp.clip(m, a_min=0, a_max=4)

    dm = rm - m
    cdh = 1 - dh

    # do not increment these, since python is 0-indexed while matlab is 1-indexed
    # i = i + 1
    # m = m + 1

    s = k.b[i, m] * cdh + k.b[i + 1, m] * dh
    t = k.b[i, m + 1] * cdh + k.b[i + 1, m + 1] * dh
    tmil = s + (t - s) * dm

    s0 = k.a[i, m] * cdh + k.a[i + 1, m] * dh
    t0 = k.a[i, m + 1] * cdh + k.a[i + 1, m + 1] * dh
    tidl = s0 + (t0 - s0) * dm
    thrst0 = tidl + (tmil - tidl) * power * 0.02

    s1 = k.c[i, m] * cdh + k.c[i + 1, m] * dh
    t1 = k.c[i, m + 1] * cdh + k.c[i + 1, m + 1] * dh
    tmax = s1 + (t1 - s1) * dm
    thrst1 = tmil + (tmax - tmil) * (power - 50) * 0.02

    thrst = jnp.where(power < 50, thrst0, thrst1)

    return thrst
