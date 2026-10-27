"""ML r1: complexidade adaptativa sem stacking: peso dos modelos de ML (XGB/MLP no resíduo) = clip((n-200)/2000,0,0.8);
alpha do ridge = max(10, 30000/n)."""
import sys; sys.path.insert(0, "harness/models"); import numpy as np
from r1_ml_lib import StackModel
def _w(n):
    a = float(np.clip((n - 200) / 2000, 0, 0.8)); return {"m1": 1 - a, "m2": a / 2, "m3": a / 2}
class Model(StackModel):
    name = "r1_ml_adapt"; use18 = True; alpha_n = 30000.; fixed_w = staticmethod(_w)
