import jax.numpy as jnp
import numpy as np

from jax_f16.utils.jax_types import Arr, Float

Scalar = float | tuple[float, float] | Float[Arr, "*b"]
Vec2 = list[Scalar] | tuple[Scalar] | Float[Arr, "*b 2"]
Vec3 = list[Scalar] | tuple[Scalar, Scalar, Scalar] | Float[Arr, "*b 3"]

RotMat3D = Float[Arr, "3 3"]


def rotz(psi: Scalar) -> RotMat3D:
    c, s = jnp.cos(psi), jnp.sin(psi)
    return jnp.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def roty(theta: Scalar) -> RotMat3D:
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotx(phi: Scalar) -> RotMat3D:
    c, s = jnp.cos(phi), jnp.sin(phi)
    return jnp.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def f16state(vt: Scalar, ab: Vec2, rpy: Vec3, pqr: Vec3, pos3d: Vec3, power: Scalar, ints: Vec3) -> Float[Arr, "*b nx"]:
    """Convenience function for specify F16 state / F16 bounds.
        vt is Scalar -> (nx, )
        vt is (2,) -> (nx, 2)
    :param vt:
    :param ab:
    :param rpy:
    :param pqr:
    :param pos3d:
    :param power:
    :param ints:
    :return:
    """
    if isinstance(vt, (float, int)):
        return np.array([vt, *ab, *rpy, *pqr, *pos3d, power, *ints])
    if isinstance(vt, tuple):
        # Turn everything into an array.
        arrs = [vt, ab, rpy, pqr, pos3d, power, ints]
        for ii, tup in enumerate(arrs):
            arrs[ii] = np.array(tup)
        vt, ab, rpy, pqr, pos3d, power, ints = arrs

    vt, power = np.expand_dims(vt, -2), np.expand_dims(power, -2)
    return np.concatenate([vt, ab, rpy, pqr, pos3d, power, ints], axis=-2)
