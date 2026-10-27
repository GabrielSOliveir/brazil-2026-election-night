"""r1_bayes_v3 — v2 + (a) termos (b-0,5)^2, (b-0,5)^3 na virada; (b) termo de DISCREPÂNCIA/seleção calibrado
dentro da própria apuração: com os hiperparâmetros já estimados, refaz o posterior usando só o prefixo
(30/50/70% dos votos já chegados, por ordem de chegada) e prevê o resto do que já chegou. O excesso do erro
interno sobre o erro previsto (e^2 - se^2) estima tau^2 = variância do viés médio de virada entre
"o que chegou antes" e "o que chega depois" (priori inv-gama: tau0=0,1pp, nu0=2).
Intervalo nacional: var_modelo + (peso restante)^2 * tau^2, quantil t(nu)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.stats import t as tdist
from r1_bayes_common import HG, keys
import r1_bayes_v2 as B

def feats_s(d):
    X, b = B.feats_s(d)
    return np.column_stack([X, (b - 0.5) ** 2, (b - 0.5) ** 3]), b

S_B0 = B.S_B0 + [0.0, 0.0]; S_SD = B.S_SD + [0.3, 0.5]
TAU0 = 0.001; NU0 = 2.0; CUTS = (0.3, 0.5, 0.7)

class Model(B.Model):
    name = "r1_bayes_v3"
    nu = 6
    shift = os.environ.get("BSHIFT", "0") == "1"   # True: também corrige a média pelo viés interno (não usado por padrão)

    def fit(self, R):
        k = keys(R); v = R.v2.values.astype(float)
        X, b = feats_s(R); s = R.y.values - b
        self.ms = HG(S_B0, S_SD, B.S_PRIOR).fit(s, X, k, v)
        Xv = B.feats_v(R); self.mv = HG(B.V_B0, B.V_SD, B.V_PRIOR).fit(np.log(v / R.validos1.values), Xv, k, v)
        # calibração interna (sequencial) do viés de seleção
        o = np.argsort(R.ordem.values); cf = np.cumsum(v[o]) / v.sum(); ex = []; d = []; sl = []
        for c in CUTS:
            i1 = o[cf <= c]; i2 = o[cf > c]
            if len(i1) < 20 or len(i2) < 20: continue
            sub = lambda kk, ii: {L: kk[L][ii] for L in kk}
            h = HG(S_B0, S_SD, B.S_PRIOR).fit(s[i1], X[i1], sub(k, i1), v[i1], th=self.ms.th)
            M2, u2 = h.design(X[i2], sub(k, i2)); w = v[i2] / v[i2].sum()
            e = w @ (M2 @ h.post - s[i2]); se2 = h.var_lin(M2, u2, w, v[i2])
            ex.append(e * e - se2); d.append(e); sl.append(se2)
        self.tau2 = (NU0 * TAU0 ** 2 + np.sum(np.maximum(ex, 0) if ex else 0)) / (NU0 + len(ex))
        self.dint = float(np.mean(d)) if d else 0.0
        self.sint2 = float(np.mean(sl)) if sl else 1.0
        self.lam = self.tau2 / (self.tau2 + self.sint2)   # encolhimento bayesiano do viés interno
        return self

    def predict(self, U):
        k = keys(U); X, b = feats_s(U); Xv = B.feats_v(U)
        Ms, us = self.ms.design(X, k); Mv, uv = self.mv.design(Xv, k)
        self._c = (Ms, us, Mv, uv)
        y = b + Ms @ self.ms.post - (self.lam * self.dint if self.shift else 0.0)
        v2 = U.validos1.values * np.exp(Mv @ self.mv.post)
        return pd.DataFrame({"y": y, "v2": v2}, index=U.index)

    def national_interval(self, R, U, pred):
        nat, se = self.nat_se(R, U, pred)
        vh = np.asarray(pred["v2"], float); W = vh.sum() / (R.v2.sum() + vh.sum())
        se = np.sqrt(se ** 2 + W ** 2 * self.tau2); zq = tdist.ppf(0.95, self.nu)
        self.last = dict(se=se, W=W, tau=np.sqrt(self.tau2), dint=self.dint, sint=np.sqrt(self.sint2), lam=self.lam / 100)
        return nat - zq * se, nat + zq * se
