"""
Stanley Bak
Python F-16
power derivative (pdot)
"""
import jax.numpy as jnp

from jax_f16.lowlevel.rtau import rtau
from jax_f16.utils.jax_types import FloatScalar


def pdot(power: FloatScalar, cpow: FloatScalar) -> FloatScalar:
    """Rate of change of power.
    :param power: Actual power.
    :param cpow: Power command.
    :return:
    """

    #                 [t,    p2]
    case1 = jnp.array([5.0, cpow])
    case2 = jnp.array([rtau(60 - power), 60])
    case3 = jnp.array([5, 40])
    case4 = jnp.array([rtau(cpow - power), cpow])

    t, p2 = jnp.where(cpow >= 50, jnp.where(power >= 50, case1, case2), jnp.where(power >= 50, case3, case4))
    pd = t * (p2 - power)

    return pd
