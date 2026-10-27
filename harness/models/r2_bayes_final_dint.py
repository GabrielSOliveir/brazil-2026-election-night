"""r2_bayes_final_dint — ablação congelada do r2_bayes_final: SEM a camada probit (só o corretor sequencial
interno do bayes, como no r1_bayes_v5). Todo o resto idêntico (DVA=0,3, log tamanho, kappa=0,08)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2_bayes_final as F
class Model(F.Model):
    name = "r2_bayes_final_dint"
    cfg = dict(F.Model.cfg, SHIFT="dint")
