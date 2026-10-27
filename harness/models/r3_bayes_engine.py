"""Motor r3: regressão gaussiana hierárquica (efeitos aleatórios reg/uf/zona/mun, beta com priori gaussiana),
integrada analiticamente, com VARIÂNCIA DE OBSERVAÇÃO POR LINHA dependente de hiperparâmetros:
    var_z = s2e * c_z + s2s * g_z
  c_z = 1 + C/tamanho (idiossincrasia da zona, maior em zona pequena)
  g_z = (1-f)/n_secoes_vistas  (amostragem de seções de uma zona PARCIAL; 0 se completa)
Hiperparâmetros (sd dos níveis, sd_e, sd_secao) = MAP da verossimilhança marginal com priori log-normal.
Também devolve o BLUP do efeito próprio da linha:  E[e_z | dados] = s2e c_z / var_z * resíduo_z."""
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.optimize import minimize
from scipy.sparse.csgraph import reverse_cuthill_mckee

LEVELS = ["reg", "uf", "zon", "mun"]

def keys(d):
    return dict(reg=d.regiao.astype(str).values, uf=d.uf.astype(str).values,
                zon=(d.uf.astype(str) + "-" + d.zona.astype(str)).values, mun=d.munid.astype(str).values)

class HG3:
    def __init__(self, b0, b0sd, prior_sd, prior_logsd=1.0, levels=LEVELS, maxiter=30):
        self.b0 = np.asarray(b0, float); self.b0sd = np.asarray(b0sd, float)
        self.levels = list(levels); self.prior_sd = prior_sd; self.plsd = prior_logsd; self.maxiter = maxiter

    def fit(self, y, X, k, c, g, th=None, th0=None):
        self.lev = {}; cols = []; off = X.shape[1]; n = len(y)
        for L in self.levels:
            u, inv = np.unique(k[L], return_inverse=True)
            self.lev[L] = dict(idx={nm: i for i, nm in enumerate(u)}, off=off, n=len(u)); off += len(u)
            cols.append(sp.csr_matrix((np.ones(n), (np.arange(n), inv)), shape=(n, len(u))))
        M = sp.hstack([sp.csr_matrix(X)] + cols).tocsc()
        self.p = X.shape[1]; self.q = off
        isd = np.zeros(self.q, bool); isd[:self.p] = True
        for L in self.levels:
            if self.lev[L]["n"] <= 60: o_ = self.lev[L]["off"]; isd[o_:o_ + self.lev[L]["n"]] = True
        sidx = np.where(~isd)[0]
        S0 = (M.T @ M).tocsc()
        S = (abs(S0[sidx][:, sidx]) + sp.identity(len(sidx))).tocsr()
        ps = reverse_cuthill_mckee(S, symmetric_mode=True) if len(sidx) else np.array([], int)
        P = np.r_[sidx[ps], np.where(isd)[0]]; self.P = P
        Mp = M[:, P].tocsc(); MpT = Mp.T.tocsr()
        r = y - X @ self.b0
        sizes = [self.lev[L]["n"] for L in self.levels]
        hasg = bool((g > 0).any()); self.hasg = hasg
        names = self.levels + ["e"] + (["s"] if hasg else [])
        pm = np.log([self.prior_sd[L] for L in names])
        b0v = self.b0sd ** 2
        fact = lambda C: splu(C, permc_spec="NATURAL", diag_pivot_thresh=0, options=dict(SymmetricMode=True))

        def build(th):
            sd = np.exp(th); s2e = sd[len(self.levels)] ** 2; s2s = sd[-1] ** 2 if hasg else 0.0
            var = s2e * c + s2s * g; W = 1.0 / var
            kdiag = np.r_[b0v, np.concatenate([np.full(nl, sd[i] ** 2) for i, nl in enumerate(sizes)])]
            MtW = MpT.multiply(W[None, :]).tocsr()
            C = (MtW @ Mp + sp.diags(1.0 / kdiag[P])).tocsc()
            return C, kdiag, W, MtW @ r

        def nll(th):
            C, kdiag, W, gg = build(th)
            try: lu = fact(C)
            except Exception: return 1e10
            ld = np.log(np.abs(lu.U.diagonal())).sum()
            quad = (r * W * r).sum() - gg @ lu.solve(gg)
            return 0.5 * (-np.log(W).sum() + np.log(kdiag).sum() + ld + quad) + 0.5 * (((th - pm) / self.plsd) ** 2).sum()

        if th is None:
            o = minimize(nll, pm.copy() if th0 is None or len(th0) != len(pm) else np.asarray(th0), method="L-BFGS-B", bounds=[(np.log(1e-4), np.log(0.5))] * len(pm),
                         options=dict(maxiter=self.maxiter, ftol=1e-6))
            th = o.x; self.nfev = o.nfev
        self.th = th; self.sd = dict(zip(names, np.exp(th)))
        C, kdiag, W, gg = build(th); self._lu = fact(C)
        self.s2e = self.sd["e"] ** 2; self.s2s = self.sd.get("s", 0.0) ** 2
        self.post = np.r_[self.b0, np.zeros(self.q - self.p)] + self.solve_p(gg)
        self.resid = y - M @ self.post                    # resíduo de cada linha observada
        self.kshr = (self.s2e * c) / (self.s2e * c + self.s2s * g)   # peso da própria observação
        return self

    def solve_p(self, gp):            # gp na ordem permutada -> x na ordem original
        x = np.empty_like(gp); x[self.P] = self._lu.solve(gp); return x

    def solve(self, g):
        x = np.empty_like(g); x[self.P] = self._lu.solve(g[self.P]); return x

    def design(self, X, k):
        n = X.shape[0]; Ai = []; Aj = []; unobs = {}
        for L in self.levels:
            idx = self.lev[L]["idx"]; off = self.lev[L]["off"]
            j = np.array([idx.get(x, -1) for x in k[L]], dtype=int); m = j >= 0
            Ai.append(np.arange(n)[m]); Aj.append(off + j[m]); unobs[L] = (~m, np.asarray(k[L]))
        Ai = np.concatenate(Ai); Aj = np.concatenate(Aj)
        Z = sp.csr_matrix((np.ones(len(Ai)), (Ai, Aj)), shape=(n, self.q))
        M = (sp.hstack([sp.csr_matrix(X), Z[:, self.p:]])).tocsr()
        return M, unobs

    def var_lin(self, a_vec, unobs, wgrp, resid_var_w2):
        """var de a'theta (posterior conjunta) + grupos não vistos (pesos wgrp) + ruído próprio (já * w^2)."""
        var = a_vec @ self.solve(a_vec)
        for L in self.levels:
            m, kk = unobs[L]
            if m.any():
                sw = pd.Series(wgrp[m]).groupby(kk[m]).sum().values; var += self.sd[L] ** 2 * (sw ** 2).sum()
        return var + resid_var_w2
