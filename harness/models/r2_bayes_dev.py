"""r2_bayes_dev — versão de DESENVOLVIMENTO (tunada só em 2018), configurável por variáveis de ambiente.
Base = r1_bayes_v5 (virada hierárquica reg>UF>zona>mun + comparecimento hierárquico + termo de seleção tau).
Novidades:
  BSHIFT  = dint | probit | both | none   -> correção de viés de ordem (dint = validação sequencial interna do bayes;
            probit = corretor do agente ts, r1_ts_common.probit_shift, spike-and-slab sobre a tendência em probit(u))
  BDVA    = c  -> comparecimento: dv = a*hier + (1-a)*log(razão nacional), a = min(1, q/c)  (encolhe p/ razão no começo)
  BTAU0, BNU0, BNU, BGEO  -> priori de tau, gl do t, inflação da variância de UFs/regiões não vistas.
Intervalo: var(modelo) + W^2 tau^2 + var(deslocamento probit)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.stats import t as tdist, norm
from r1_bayes_common import HG, keys
import r1_bayes_v2 as B, r1_bayes_v3 as B3
from r1_ts_common import probit_shift

E = os.environ.get
CFG = dict(TAU0=float(E("BTAU0", "0.002")), NU0=float(E("BNU0", "2")), NU=float(E("BNU", "6")),
           SHIFT=E("BSHIFT", "dint"), DVA=float(E("BDVA", "0")), GEO=float(E("BGEO", "1")),
           PI=float(E("BPI", "0.01")), PTAU=float(E("BPTAU", "0.01")), PRES=E("BPRES", "0") == "1", KAPPA=float(E("BKAPPA", "0")))
CUTS = (0.3, 0.5, 0.7)
LA_S = E("BLA_S", "0") == "1"; LA_V = E("BLA_V", "0") == "1"
def feats_s(d):
    X, b = B3.feats_s(d)
    return (np.column_stack([X, np.log(d.a1.values) - 9.0]) if LA_S else X), b
def feats_v(d):
    X = B.feats_v(d)
    return np.column_stack([X, np.log(d.a1.values) - 9.0]) if LA_V else X
S_B0, S_SD, S_PRIOR = B3.S_B0 + [0.0] * LA_S, B3.S_SD + [0.02] * LA_S, B.S_PRIOR
V_B0, V_SD, V_PRIOR = B.V_B0 + [0.0] * LA_V, B.V_SD + [0.05] * LA_V, B.V_PRIOR

def sub(kk, ii): return {L: kk[L][ii] for L in kk}

def probit_post(R, U, pi, ts, resid):
    """deslocamento probit (média) e sua variância a posteriori (mistura spike-and-slab)."""
    sh, gam, se = probit_shift(R, U, slab=(pi, ts), resid=resid, return_all=True)
    if gam == 0 or se == 0 or not np.isfinite(se): return 0.0, 0.0, 0.0
    se2 = se ** 2; t2 = gam ** 2 / se2; r = se2 / (se2 + ts ** 2)
    lbf = 0.5 * np.log(r) + 0.5 * t2 * (1 - r); ps = 1 / (1 + (1 - pi) / pi * np.exp(-min(lbf, 50)))
    m = gam * ts ** 2 / (ts ** 2 + se2); vs = ts ** 2 * se2 / (ts ** 2 + se2)
    vg = ps * (vs + m * m) - (ps * m) ** 2
    k = sh / (ps * m) if ps * m != 0 else 0.0          # (gU - gR)
    return sh, k * k * vg, ps

class Model:
    name = "r2_bayes_dev"
    cfg = CFG

    def fit(self, R):
        c = self.cfg
        k = keys(R); v = R.v2.values.astype(float); V1 = R.validos1.values.astype(float); L2 = R.L2.values
        X, b = feats_s(R); s = R.y.values - b; Xv = feats_v(R); dv = np.log(v / V1)
        self.ms = HG(S_B0, S_SD, S_PRIOR).fit(s, X, k, v)
        self.mv = HG(V_B0, V_SD, V_PRIOR).fit(dv, Xv, k, v)
        self.ratio = np.log(v.sum() / V1.sum())
        o = np.argsort(R.ordem.values); cf = np.cumsum(v[o]) / v.sum(); ex = []; d = []; sl = []
        for cut in CUTS:
            i1 = o[cf <= cut]; i2 = o[cf > cut]
            if len(i1) < 20 or len(i2) < 20: continue
            hs = HG(S_B0, S_SD, S_PRIOR).fit(s[i1], X[i1], sub(k, i1), v[i1], th=self.ms.th)
            hv = HG(V_B0, V_SD, V_PRIOR).fit(dv[i1], Xv[i1], sub(k, i1), v[i1], th=self.mv.th)
            Ms, us = hs.design(X[i2], sub(k, i2)); Mv, uv = hv.design(Xv[i2], sub(k, i2))
            yh = b[i2] + Ms @ hs.post; vh = V1[i2] * np.exp(Mv @ hv.post)
            Vt = v[i1].sum() + vh.sum(); W2 = vh.sum() / Vt
            nat_p = (L2[i1].sum() + (yh * vh).sum()) / Vt; nat_t = L2.sum() / v.sum()
            se2 = hs.var_lin(Ms, us, vh / Vt, vh) + hv.var_lin(Mv, uv, (yh - nat_p) * vh / Vt, vh)
            e = (nat_p - nat_t) / W2
            ex.append(e * e - se2 / W2 ** 2)
            w = v[i2] / v[i2].sum(); d.append(w @ (yh - R.y.values[i2])); sl.append(se2 / W2 ** 2)
        tau0 = c["TAU0"] if c["KAPPA"] <= 0 else c["KAPPA"] * np.hypot(self.ms.sd["reg"], self.ms.sd["uf"])
        self.tau0 = tau0
        self.tau2 = (c["NU0"] * tau0 ** 2 + (np.sum(np.maximum(ex, 0)) if ex else 0)) / (c["NU0"] + len(ex))
        self.dint = float(np.mean(d)) if d else 0.0
        self.sint2 = float(np.mean(sl)) if sl else 1.0
        self.lam = self.tau2 / (self.tau2 + self.sint2)
        self.R = R
        return self

    def predict(self, U):
        c = self.cfg; R = self.R
        k = keys(U); X, b = feats_s(U); Xv = feats_v(U)
        Ms, us = self.ms.design(X, k); Mv, uv = self.mv.design(Xv, k)
        sh = 0.0; self.pvar = 0.0
        if c["SHIFT"] in ("dint", "both"): sh -= self.lam * self.dint
        if c["SHIFT"] in ("probit", "both"):
            p, pv, ps = probit_post(R, U, c["PI"], c["PTAU"], c["PRES"]); sh += p; self.pvar = pv
        if c["SHIFT"] == "switch":   # probit quando a tendência de ordem é clara (ps alto); senão o dint do bayes
            p, pv, ps = probit_post(R, U, c["PI"], c["PTAU"], c["PRES"])
            sh = p + (1 - ps) * (-self.lam * self.dint); self.pvar = pv; self.ps = ps
        self.pshift = sh
        y = b + Ms @ self.ms.post + sh
        dvh = Mv @ self.mv.post
        if c["DVA"] > 0:
            q = R.v2.sum() / (R.v2.sum() + U.validos1.sum() * np.exp(self.ratio))
            a = min(1.0, q / c["DVA"]); dvh = a * dvh + (1 - a) * self.ratio
        self._c = (Ms, us, Mv, uv)
        return pd.DataFrame({"y": y, "v2": U.validos1.values * np.exp(dvh)}, index=U.index)

    def national_interval(self, R, U, pred):
        c = self.cfg; Ms, us, Mv, uv = self._c
        vh = np.asarray(pred["v2"], float); yh = np.clip(np.asarray(pred["y"], float), 0, 1)
        Vt = R.v2.sum() + vh.sum(); nat = (R.L2.sum() + (yh * vh).sum()) / Vt; W = vh.sum() / Vt
        if c["GEO"] != 1:
            sds, sdv = dict(self.ms.sd), dict(self.mv.sd)
            for L in ("reg", "uf"): self.ms.sd[L] *= np.sqrt(c["GEO"]); self.mv.sd[L] *= np.sqrt(c["GEO"])
        var = self.ms.var_lin(Ms, us, vh / Vt, vh) + self.mv.var_lin(Mv, uv, (yh - nat) * vh / Vt, vh)
        if c["GEO"] != 1: self.ms.sd, self.mv.sd = sds, sdv
        var += W ** 2 * (self.tau2 + self.pvar)
        se = np.sqrt(var); zq = tdist.ppf(0.95, c["NU"])
        self.last = dict(tau0=self.tau0, tau=np.sqrt(self.tau2), dint=self.dint, pshift=self.pshift, lam=getattr(self, 'ps', 0) / 100)
        return nat - zq * se, nat + zq * se
