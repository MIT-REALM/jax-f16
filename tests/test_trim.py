import jax
import numpy as np

from jax_f16.f16 import F16

jax.config.update("jax_enable_x64", True)


def test_trim():
    f16 = F16()

    x = F16.trim_state()
    u = F16.trim_control()
    xdot = np.array(f16.xdot(x, u))
    assert xdot.shape == (F16.NX,)

    # Everything except PN should be small.
    xdot[F16.PN] = 0
    assert np.allclose(xdot, 0.0, atol=5e-8)


if __name__ == "__main__":
    test_trim()
