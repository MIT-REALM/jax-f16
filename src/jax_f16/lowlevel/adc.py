from typing import NamedTuple

import jax.numpy as jnp

from jax_f16.utils.jax_types import FloatScalar


class ADCOut(NamedTuple):
    amac: FloatScalar
    """Mach number"""
    qbar: FloatScalar
    """Dynamic pressure"""


def adc(vt: FloatScalar, alt: FloatScalar) -> ADCOut:
    """Converts velocity (vt) and altitude (alt) to mach number (amach) and dynamic pressure (qbar)

    See pages 63-65 of Stevens & Lewis, "Aircraft Control and Simulation", 2nd edition
    """

    # vt = freestream air speed

    ro = 2.377e-3
    tfac = 1 - 0.703e-5 * alt

    # if alt >= 35000:  # in stratosphere
    #     t = 390
    # else:
    #     t = 519 * tfac  # 3 rankine per atmosphere (3 rankine per 1000 ft)
    t = jnp.where(alt >= 35_000, 390, 519 * tfac)

    # rho = freestream mass density
    rho = ro * tfac**4.14

    # a = speed of sound at the ambient conditions
    # speed of sound in a fluid is the sqrt of the quotient of the modulus of elasticity over the mass density
    a = jnp.sqrt(1.4 * 1716.3 * t)

    # amach = mach number
    amach = vt / a

    # qbar = dynamic pressure
    qbar = 0.5 * rho * vt * vt

    return ADCOut(amach, qbar)
