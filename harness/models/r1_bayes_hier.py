"""r1_bayes_hier — modelo bayesiano hierárquico de VIRADA (swing) 1T->2T.

s_z = y_z - b_z,  b_z = c13/(c13+c22)  (fatia do Lula no 1T entre os dois)
s_z = x_z'beta + u_reg + u_uf + u_zona(uf,zona) + u_mun + e_z
  x_z: estrutura de TRANSFERÊNCIA  s = sum_k r_k (p_k - b)  => colunas r_k e r_k*b (k = Tebet, Ciro, outros),
       + brancos/nulos 1T, abstenção 1T (centradas). Priori estrutural: p_k ~ 0,5 (beta_rk=0,5; beta_rkb=-1).
  u_zona: zona eleitoral = bloco espacial de municípios vizinhos (a ideia "blocos" do usuário).
  e_z ~ N(0, s2e (1 + 6000/v_z))  (ruído maior em zona pequena)
Tudo gaussiano: beta e u integrados analiticamente (equações de modelo misto, scipy.sparse).
Hiperparâmetros (5 desvios-padrão) = MAP do log-verossimilhança marginal + priori log-normal.
Zonas não apuradas recebem os BLUPs dos grupos já vistos; grupos não vistos -> 0 com variância a priori,
que entra no intervalo nacional como erro CORRELACIONADO (soma dos pesos do grupo ao quadrado).
"""
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.optimize import minimize
from scipy.stats import norm, t as tdist

LEVELS = ["reg", "uf", "zon", "mun"]
# priori (mediana, sd em log) dos desvios-padrão por nível: escala típica de 2018 (fit no 2T/2018 completo), larga
PRIOR_SD = dict(reg=0.008, uf=0.008, zon=0.008, mun=0.006, e=0.006)
PRIOR_LOGSD = 1.0
C_SIZE = 6000.0

def feats(d):
    V = d.validos1.values.astype(float); b = d.c13.values / (d.c13.values + d.c22.values)
    r15 = d.c15.values / V; r12 = d.c12.values / V; ro = d.c_out.values / V
    bn = (d.brancos1.values + d.nulos1.values) / d.comp1.values; ab = d.abst1.values / d.a1.values
    X = np.column_stack([np.ones(len(d)), r15, r12, ro, r15 * b, r12 * b, ro * b, bn - 0.06, ab - 0.2])
    return X, b

B0 = np.array([0.0, 0.5, 0.5, 0.5, -1.0, -1.0, -1.0, 0.0, 0.0])
B0SD = np.array([0.05, 0.5, 0.5, 0.5, 0.7, 0.7, 0.7, 0.2, 0.2])

def keys(d):
    return dict(reg=d.regiao.astype(str).values, uf=d.uf.astype(str).values,
                zon=(d.uf.astype(str) + "-" + d.zona.astype(str)).values, mun=d.munid.astype(str).values)

class Model:
    name = "r1_bayes_hier"
    nu = 6  # graus de liberdade do t no intervalo (cauda p/ incerteza dos hiperparâmetros)

    def fit(self, R):
        X, b = feats(R); y = R.y.values - b; v = R.v2.values.astype(float)
        self.kR = keys(R)
        # códigos de grupo observados
        self.lev = {}; cols = []; off = X.shape[1]
        for L in LEVELS:
            u, inv = np.unique(self.kR[L], return_inverse=True)
            self.lev[L] = dict(names=u, idx={n: i for i, n in enumerate(u)}, off=off, n=len(u)); off += len(u)
            cols.append(sp.csr_matrix((np.ones(len(R)), (np.arange(len(R)), inv)), shape=(len(R), len(u))))
        M = sp.hstack([sp.csr_matrix(X)] + cols).tocsc()
        self.p = X.shape[1]; self.q = off
        winv = 1.0 / (1.0 + C_SIZE / np.maximum(v, 1.0))   # R^{-1} = winv / s2e
        r = y - X @ B0
        MtW = M.T.multiply(winv).tocsc()
        MtWM = (MtW @ M).tocsc(); MtWr = MtW @ r; rWr = (r * winv * r).sum(); logdetW = np.log(winv).sum()
        n = len(y)
        sizes = [self.lev[L]["n"] for L in LEVELS]

        # ordenação que reduz preenchimento: calculada uma vez, reaproveitada (fatoração ~20ms)
        P = splu((MtWM + sp.identity(self.q)).tocsc(), permc_spec="MMD_AT_PLUS_A").perm_c
        self.P = P; MtWMp = MtWM[P][:, P].tocsc(); MtWrp = MtWr[P]

        def build(th):
            sd = np.exp(th); s2e = sd[-1] ** 2
            kdiag = np.r_[B0SD ** 2, np.concatenate([np.full(nl, sd[i] ** 2) for i, nl in enumerate(sizes)])]
            C = (MtWMp / s2e + sp.diags(1.0 / kdiag[P])).tocsc()
            return C, kdiag, s2e

        def fact(C):
            return splu(C, permc_spec="NATURAL", diag_pivot_thresh=0, options=dict(SymmetricMode=True))

        def nll(th):
            C, kdiag, s2e = build(th)
            try: lu = fact(C)
            except Exception: return 1e10
            ld = np.log(np.abs(lu.U.diagonal())).sum()
            g = MtWrp / s2e; sol = lu.solve(g)
            quad = rWr / s2e - g @ sol
            logdetV = (n * np.log(s2e) - logdetW) + np.log(kdiag).sum() + ld
            ll = -0.5 * (logdetV + quad)
            lp = -0.5 * (((th - self.pm) / PRIOR_LOGSD) ** 2).sum()
            return -(ll + lp)

        self.pm = np.log([PRIOR_SD[k] for k in LEVELS + ["e"]])
        th0 = self.pm.copy()
        o = minimize(nll, th0, method="L-BFGS-B", bounds=[(np.log(1e-4), np.log(0.1))] * 5,
                     options=dict(maxiter=40, ftol=1e-7))
        self.th = o.x; self.sd = dict(zip(LEVELS + ["e"], np.exp(o.x))); self.nit = o.nfev
        C, kdiag, s2e = build(o.x); self._lu = fact(C); self.s2e = s2e
        B0_full = np.r_[B0, np.zeros(self.q - self.p)]
        self.post = B0_full + self.solve(MtWr / s2e)   # média a posteriori de (beta, u)
        # razão de votos válidos (para v2 das não apuradas)
        self.vr = R.v2.sum() / R.validos1.sum()
        return self

    def solve(self, g):
        # resolve C x = g na ordem original (C fatorada na ordem permutada P)
        x = np.empty_like(g); x[self.P] = self._lu.solve(g[self.P]); return x

    def _design(self, U):
        X, b = feats(U); kU = keys(U); n = len(U)
        rows, cols = [np.arange(n)] * 0, []
        Ai = []; Aj = []
        unobs = {}
        for L in LEVELS:
            idx = self.lev[L]["idx"]; off = self.lev[L]["off"]
            j = np.array([idx.get(k, -1) for k in kU[L]])
            m = j >= 0
            Ai.append(np.arange(n)[m]); Aj.append(off + j[m])
            unobs[L] = np.where(m, None, kU[L])
        Ai = np.concatenate(Ai); Aj = np.concatenate(Aj)
        Zs = sp.csr_matrix((np.ones(len(Ai)), (Ai, Aj)), shape=(n, self.q))
        M = sp.hstack([sp.csr_matrix(X), Zs[:, self.p:]]).tocsr()
        return M, b, unobs

    def predict(self, U):
        M, b, unobs = self._design(U)
        self._cache = (M, b, unobs)
        yh = b + M @ self.post
        return pd.DataFrame({"y": yh, "v2": U.validos1.values * self.vr}, index=U.index)

    def national_interval(self, R, U, pred):
        M, b, unobs = self._cache
        vh = np.asarray(pred["v2"], float); yh = np.clip(np.asarray(pred["y"], float), 0, 1)
        Vt = R.v2.sum() + vh.sum(); w = vh / Vt
        nat = (R.L2.sum() + (yh * vh).sum()) / Vt
        a = M.T @ w
        var = a @ self.solve(a)                                   # beta + efeitos já vistos (posterior conjunta)
        for L in LEVELS:                                             # grupos não vistos: erro correlacionado
            k = unobs[L]; m = np.array([x is not None for x in k])
            if m.any():
                sw = pd.Series(w[m]).groupby(k[m].astype(str)).sum().values
                var += self.sd[L] ** 2 * (sw ** 2).sum()
        var += self.s2e * (w ** 2 * (1 + C_SIZE / np.maximum(vh, 1))).sum()
        se = np.sqrt(var); zq = tdist.ppf(0.95, self.nu)
        return nat - zq * se, nat + zq * se
