import ipdb
import jax
import numpy as np
from aerobench.highlevel.controlled_f16 import controlled_f16 as controlled_f16_ref
from aerobench.lowlevel.low_level_controller import LowLevelController as LowLevelControllerRef

from jax_f16.f16 import F16
from jax_f16.f16_types import OUTER_CONTROL_NU

llc = LowLevelControllerRef()

# Check that the results match at least on double precision, since there are errors when using f32.
jax.config.update("jax_enable_x64", True)


def integrate_f16_ref_euler(x0: np.ndarray, u_traj: np.ndarray, h: float):
    xtraj = [x0]
    x = x0
    for u in u_traj:
        xd = controlled_f16_ref(0.0, x, u, llc)[0]
        x = x + xd * h
        xtraj.append(x)
    return np.stack(xtraj, axis=0)


def integrate_f16_euler(x0: np.ndarray, u_traj: np.ndarray, h: float):
    dyn = F16()

    xtraj = [x0]
    x = x0
    for u in u_traj:
        xd = dyn.xdot(x, u)
        x = x + xd * h
        xtraj.append(x)
    return np.stack(xtraj, axis=0)


def test_highlevel_fns():
    h = 5e-4
    T = 256

    dyn = F16()

    x0 = np.array(
        [540, 0.037027160081059704, 0, -0.39269908169872414, -0.47123889803846897, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 1000, 9]
    )
    x0 = np.concatenate([x0, np.zeros(3)], axis=0)

    u_traj = np.zeros((T, OUTER_CONTROL_NU))
    u_traj[:100, 0] = 5
    u_traj[100:200, 1] = -5
    u_traj[200:250, 2] = 10
    u_traj[:, 3] = 0.5

    my_dx = dyn.xdot(x0, u_traj[0])
    ref_dx = controlled_f16_ref(0.0, x0, u_traj[0], llc)[0]
    assert np.allclose(my_dx, ref_dx)

    x_traj = integrate_f16_euler(x0, u_traj, h)
    ref_xtraj = integrate_f16_ref_euler(x0, u_traj, h)

    assert np.allclose(x_traj, ref_xtraj)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        test_highlevel_fns()
