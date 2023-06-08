"""
Stanley Bak
Python F-16

Rtau function
"""
import jax.numpy as jnp

from jax_f16.utils.jax_types import FloatScalar


def rtau(dp: FloatScalar) -> FloatScalar:
    """Reciprocal time constant.
    :param dp:
    :return:
    """

    # if dp <= 25:
    #     rt = 1.0
    # elif dp >= 50:
    #     rt = .1
    # else:
    #     rt = 1.9 - .036 * dp
    rt = jnp.where(dp <= 25, 1.0, jnp.where(dp >= 50, 0.1, 1.9 - 0.036 * dp))
    return rt
