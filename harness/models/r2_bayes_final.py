"""r2_bayes_final — candidato principal (CONGELADO; hiperparâmetros escolhidos só em 2018).
Virada s = y - b com hierarquia reg > UF > zona eleitoral > município (partial pooling, gaussiano exato, MAP dos
hiperparâmetros) + forma de transferência (Tebet/Ciro/outros x b) + curva em b + log(tamanho da zona).
Comparecimento log(v2/validos1) hierárquico, ENCOLHIDO para a razão nacional no começo: a = min(1, q/0,30).
Viés de ordem de chegada:
  - probit (agente ts) sobre a virada residualizada; spike-and-slab pi=1%, slab 1pp -> dispara só com tendência clara;
  - validação sequencial interna (bayes): erro ao prever o fim do já apurado a partir do começo -> d_int e tau^2;
  - combinação: shift = probit + (1 - P(tendência)) * (-lam * d_int).
Intervalo 90% (t6): posterior conjunta (beta + efeitos vistos) + grupos NÃO vistos (correlacionado) + ruído
+ comparecimento + W^2 (tau^2 + var do probit), com tau0 = 0,08 * sqrt(sd_reg^2 + sd_uf^2) (escala com a
heterogeneidade geográfica da virada estimada na própria apuração: largo em 2018, estreito em 2022)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import r2_bayes_dev as D
import r1_bayes_v2 as B, r1_bayes_v3 as B3

def feats_s(d):
    X, b = B3.feats_s(d)
    return np.column_stack([X, np.log(d.a1.values) - 9.0]), b

D.feats_s = feats_s; D.feats_v = B.feats_v
D.S_B0 = B3.S_B0 + [0.0]; D.S_SD = B3.S_SD + [0.02]; D.S_PRIOR = B.S_PRIOR
D.V_B0, D.V_SD, D.V_PRIOR = B.V_B0, B.V_SD, B.V_PRIOR

class Model(D.Model):
    name = "r2_bayes_final"
    cfg = dict(TAU0=0.002, KAPPA=0.08, NU0=2.0, NU=6.0, SHIFT="switch", DVA=0.3, GEO=1.0,
               PI=0.01, PTAU=0.01, PRES=True)
