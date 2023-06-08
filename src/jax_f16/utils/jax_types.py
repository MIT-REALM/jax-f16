from typing import Union

import numpy as np
from jaxtyping import Array, Float

Arr = Union[np.ndarray, Array]

AnyFloat = Float[Arr, "*"]
FloatScalar = Float[Arr, ""]

State = Float[Arr, "nx"]
