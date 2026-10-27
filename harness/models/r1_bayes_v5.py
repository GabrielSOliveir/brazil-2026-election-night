"""r1_bayes_v5 — igual ao v4, com priori do viés de seleção tau0 = 0,2pp (era 0,1pp): cobre o começo do cenário real
(amostra inicial não representativa: exterior + zonas pequenas, viés de comparecimento no NE)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r1_bayes_v4 as B4
B4.TAU0 = 0.002
class Model(B4.Model):
    name = "r1_bayes_v5"
