"""r1_bayes_v2 — virada hierárquica (como r1_bayes_hier) + modelo hierárquico de COMPARECIMENTO VÁLIDO
dv_z = log(v2_z / validos1_z) com os mesmos níveis (reg, uf, zona, mun). v2 previsto = validos1*exp(dv).
Intervalo nacional: delta-method somando (i) incerteza da virada e (ii) do comparecimento,
cada uma com posterior conjunta dos efeitos vistos + variância a priori dos grupos não vistos (correlacionada)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.stats import t as tdist
from r1_bayes_common import HG, keys

def feats_s(d):
    V = d.validos1.values.astype(float); b = d.c13.values / (d.c13.values + d.c22.values)
    r15 = d.c15.values / V; r12 = d.c12.values / V; ro = d.c_out.values / V
    bn = (d.brancos1.values + d.nulos1.values) / d.comp1.values; ab = d.abst1.values / d.a1.values
    X = np.column_stack([np.ones(len(d)), r15, r12, ro, r15 * b, r12 * b, ro * b, bn - 0.06, ab - 0.2])
    return X, b

def feats_v(d):
    V = d.validos1.values.astype(float); b = d.c13.values / (d.c13.values + d.c22.values)
    r = (d.c15.values + d.c12.values + d.c_out.values) / V
    bn = (d.brancos1.values + d.nulos1.values) / d.comp1.values; ab = d.abst1.values / d.a1.values
    da = np.log(d.aptos2.values / d.a1.values)
    return np.column_stack([np.ones(len(d)), b - 0.5, r - 0.1, bn - 0.06, ab - 0.2, da])

S_B0 = [0.0, 0.5, 0.5, 0.5, -1.0, -1.0, -1.0, 0.0, 0.0]
S_SD = [0.05, 0.5, 0.5, 0.5, 0.7, 0.7, 0.7, 0.2, 0.2]
S_PRIOR = dict(reg=0.008, uf=0.008, zon=0.008, mun=0.006, e=0.006)
V_B0 = [0.0, 0, 0, 0, 0, 1.0]
V_SD = [0.05, 0.2, 0.2, 0.3, 0.3, 0.5]
V_PRIOR = dict(reg=0.02, uf=0.02, zon=0.015, mun=0.015, e=0.015)

class Model:
    name = "r1_bayes_v2"
    nu = 6
    def fit(self, R):
        k = keys(R); v = R.v2.values.astype(float)
        X, b = feats_s(R); self.ms = HG(S_B0, S_SD, S_PRIOR).fit(R.y.values - b, X, k, v)
        Xv = feats_v(R); self.mv = HG(V_B0, V_SD, V_PRIOR).fit(np.log(v / R.validos1.values), Xv, k, v)
        return self

    def predict(self, U):
        k = keys(U); X, b = feats_s(U); Xv = feats_v(U)
        Ms, us = self.ms.design(X, k); Mv, uv = self.mv.design(Xv, k)
        self._c = (Ms, us, Mv, uv)
        y = b + Ms @ self.ms.post; v2 = U.validos1.values * np.exp(Mv @ self.mv.post)
        return pd.DataFrame({"y": y, "v2": v2}, index=U.index)

    def nat_se(self, R, U, pred):
        Ms, us, Mv, uv = self._c
        vh = np.asarray(pred["v2"], float); yh = np.clip(np.asarray(pred["y"], float), 0, 1)
        Vt = R.v2.sum() + vh.sum(); nat = (R.L2.sum() + (yh * vh).sum()) / Vt
        var = self.ms.var_lin(Ms, us, vh / Vt, vh)
        var += self.mv.var_lin(Mv, uv, (yh - nat) * vh / Vt, vh)
        return nat, np.sqrt(var)

    def national_interval(self, R, U, pred):
        nat, se = self.nat_se(R, U, pred); zq = tdist.ppf(0.95, self.nu)
        return nat - zq * se, nat + zq * se
