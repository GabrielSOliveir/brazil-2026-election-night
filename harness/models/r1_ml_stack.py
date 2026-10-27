"""ML r1: stacking NNLS (soma 1) de m0 swing, m1 ridge hierárquico, m2 ridge+XGB resíduo, m3 ridge+MLP resíduo, m4 XGB direto.
Pesos estimados out-of-fold (GroupKFold por município, 3 folds) só nas zonas apuradas; fallback m1 se n<300."""
import sys; sys.path.insert(0, "harness/models")
from r1_ml_lib import StackModel
class Model(StackModel):
    name = "r1_ml_stack"; use18 = True
