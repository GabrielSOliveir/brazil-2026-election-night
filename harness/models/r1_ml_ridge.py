"""ML r1: ridge de transferência hierárquico (alvo = virada y-b; UF/mun/zona com shrinkage; feats 2018 + p18) + modelo de v2. Sem stacking."""
import sys; sys.path.insert(0, "harness/models")
from r1_ml_lib import StackModel
class Model(StackModel):
    name = "r1_ml_ridge"; use18 = True; stack = False
