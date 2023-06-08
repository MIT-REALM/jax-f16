<div align="center">

# Jax F16 Dynamics

[Installation](#installation) •
[Quickstart](#quickstart) •
[Citations](#citations)

</div>

## Installation
Make sure [jax](https://github.com/google/jax#installation) and [jaxtyping](https://github.com/google/jaxtyping) are installed.
Then, either install from PyPI
```shell
pip install jax-f16
```
or install from github 
```shell
pip install --upgrade git+https://github.com/mit-realm/jax-f16.git
```

## Quickstart
```python
from jax_f16.f16 import F16

f16 = F16()
x, u = f16.trim_state(), f16.trim_control()
assert x.shape == (F16.NX,) and u.shape == (F16.NU,)
xdot = f16.xdot(x, u)
```

## Citations
If you would like to use `jax-f16` in a publication, please cite our paper for this implementation
```bibtex
@inproceedings{So-RSS-23, 
    AUTHOR    = {Oswin So AND Chuchu Fan}, 
    TITLE     = {{Solving Stabilize-Avoid Optimal Control via Epigraph Form and Deep Reinforcement Learning}}, 
    BOOKTITLE = {Proceedings of Robotics: Science and Systems}, 
    YEAR      = {2023}, 
} 
```
as well as the paper for the original implementation
```bibtex
@inproceedings{heidlauf2018verification,
  title={Verification Challenges in F-16 Ground Collision Avoidance and Other Automated Maneuvers.},
  author={Heidlauf, Peter and Collins, Alexander and Bolender, Michael and Bak, Stanley},
  booktitle={ARCH@ ADHS},
  pages={208--217},
  year={2018}
}
```
