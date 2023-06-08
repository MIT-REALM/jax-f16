import jax.numpy as jnp

from jax_f16.utils.jax_types import FloatScalar


def tgear(thtl: FloatScalar) -> FloatScalar:
    """Power command vs throttle relationship.
    :param thtl:
    :return:
    """

    # if thtl <= 0.77:
    #     tg = 64.94 * thtl
    # else:
    #     tg = 217.38 * thtl - 117.38
    tg = jnp.where(thtl <= 0.77, 64.94 * thtl, 217.38 * thtl - 117.38)

    return tg
