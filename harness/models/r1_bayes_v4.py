"""r1_bayes_v4 — v3 com:
 (a) correção de média pelo viés sequencial interno ENCOLHIDO: shift = lam * d_int, lam = tau^2/(tau^2+se_int^2)
     (d_int = erro médio de virada ao prever o "fim" do já apurado a partir do "começo");
 (b) tau^2 calibrado com o erro interno da FATIA TOTAL (virada + comparecimento), por unidade de peso restante.
Ideia: se o que chega depois é sistematicamente diferente do que chegou antes (cidades grandes chegam tarde,
ou o cenário 'pior'), isso já aparece DENTRO do apurado; o modelo aprende o tamanho (tau) e a direção (d_int)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.stats import t as tdist
from r1_bayes_common import HG, keys
import r1_bayes_v2 as B, r1_bayes_v3 as B3

feats_v = B.feats_v   # (testado: + log tamanho -> pior no início do cenário real; removido)
V_B0 = B.V_B0; V_SD = B.V_SD

TAU0 = float(os.environ.get("BTAU0", "0.001")); NU0 = 2.0; CUTS = (0.3, 0.5, 0.7)

def sub(kk, ii): return {L: kk[L][ii] for L in kk}

class Model(B3.Model):
    name = "r1_bayes_v4"
    nu = 6
    shift = True

    def fit(self, R):
        k = keys(R); v = R.v2.values.astype(float); V1 = R.validos1.values.astype(float); L2 = R.L2.values
        X, b = B3.feats_s(R); s = R.y.values - b; Xv = feats_v(R); dv = np.log(v / V1)
        self.ms = HG(B3.S_B0, B3.S_SD, B.S_PRIOR).fit(s, X, k, v)
        self.mv = HG(V_B0, V_SD, B.V_PRIOR).fit(dv, Xv, k, v)
        o = np.argsort(R.ordem.values); cf = np.cumsum(v[o]) / v.sum(); ex = []; d = []; sl = []
        for c in CUTS:
            i1 = o[cf <= c]; i2 = o[cf > c]
            if len(i1) < 20 or len(i2) < 20: continue
            hs = HG(B3.S_B0, B3.S_SD, B.S_PRIOR).fit(s[i1], X[i1], sub(k, i1), v[i1], th=self.ms.th)
            hv = HG(V_B0, V_SD, B.V_PRIOR).fit(dv[i1], Xv[i1], sub(k, i1), v[i1], th=self.mv.th)
            Ms, us = hs.design(X[i2], sub(k, i2)); Mv, uv = hv.design(Xv[i2], sub(k, i2))
            yh = b[i2] + Ms @ hs.post; vh = V1[i2] * np.exp(Mv @ hv.post)
            W2 = vh.sum() / (v[i1].sum() + vh.sum())
            nat_p = (L2[i1].sum() + (yh * vh).sum()) / (v[i1].sum() + vh.sum()); nat_t = L2.sum() / v.sum()
            Vt = v[i1].sum() + vh.sum()
            se2 = hs.var_lin(Ms, us, vh / Vt, vh) + hv.var_lin(Mv, uv, (yh - nat_p) * vh / Vt, vh)
            e = (nat_p - nat_t) / W2                       # viés por unidade de peso restante
            ex.append(e * e - se2 / W2 ** 2)
            w = v[i2] / v[i2].sum(); d.append(w @ (yh - R.y.values[i2])); sl.append(se2 / W2 ** 2)
        self.tau2 = (NU0 * TAU0 ** 2 + (np.sum(np.maximum(ex, 0)) if ex else 0)) / (NU0 + len(ex))
        self.dint = float(np.mean(d)) if d else 0.0
        self.sint2 = float(np.mean(sl)) if sl else 1.0
        self.lam = self.tau2 / (self.tau2 + self.sint2)
        return self

    def predict(self, U):
        k = keys(U); X, b = B3.feats_s(U); Xv = feats_v(U)
        Ms, us = self.ms.design(X, k); Mv, uv = self.mv.design(Xv, k)
        self._c = (Ms, us, Mv, uv)
        y = b + Ms @ self.ms.post - (self.lam * self.dint if self.shift else 0.0)
        v2 = U.validos1.values * np.exp(Mv @ self.mv.post)
        return pd.DataFrame({"y": y, "v2": v2}, index=U.index)
