from typing import Literal, Union

import numpy as np
from jaxtyping import Array, Float

arr = Union[np.ndarray, Array]

PLANE_STATE_NX = 13
FULL_STATE_NX = 16
OUTER_CONTROL_NU = 4
INNER_CONTROL_NU = 4

PlaneState = Float[arr, "13"]
FullState = Float[arr, "16"]

# [alpha, q, int_e_Nz, beta, p, r, int_e_ps, int_e_Ny_r]
LQRState = Float[arr, "8"]

LLCIntState = Float[arr, "3"]

# [Nz, ps, Ny+R, throttle]
OuterControl = Float[arr, "4"]

# [throt, ele, ail, rud]
InnerControl = Float[arr, "4"]

# [throt, ele, ail, rud, Nz_ref, ps_ref, Ny_r_ref]
InnerOuterControl = Float[arr, "7"]

F16ModelType = Literal["morelli"]


class S:
    VT = 0
    ALPHA = 1
    BETA = 2
    #
    PHI = 3
    THETA = 4
    PSI = 5
    #
    P = 6
    Q = 7
    R = 8
    #
    ALT = 11
    POWER = 12

    PQR = slice(6, 9)


class C:
    THROTTLE = 0
    EL = 1
    AIL = 2
    RDR = 3
