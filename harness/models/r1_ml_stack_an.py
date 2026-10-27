"""ML r1: como r1_ml_stack, mas alpha do ridge = max(10, 10000/n) (shrinkage extra com poucas zonas)."""
import sys; sys.path.insert(0, "harness/models")
from r1_ml_lib import StackModel
class Model(StackModel):
    name = "r1_ml_stack_an"; use18 = True; alpha_n = 10000.
