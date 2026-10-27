"""Motor comum (r1_bayes_*): regressão gaussiana hierárquica com efeitos aleatórios aninhados/cruzados,
beta e u integrados analiticamente; hiperparâmetros = MAP (verossimilhança marginal + priori log-normal).
Usa scipy.sparse (ordenação MMD calculada 1x, fatorações seguintes ~20 ms)."""
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.optimize import minimize
from scipy.sparse.csgraph import reverse_cuthill_mckee

LEVELS = ["reg", "uf", "zon", "mun"]

def keys(d):
    return dict(reg=d.regiao.astype(str).values, uf=d.uf.astype(str).values,
                zon=(d.uf.astype(str) + "-" + d.zona.astype(str)).values, mun=d.munid.astype(str).values)

class HG:
    def __init__(self, b0, b0sd, prior_sd, prior_logsd=1.0, csize=6000.0, levels=LEVELS):
        self.b0 = np.asarray(b0, float); self.b0sd = np.asarray(b0sd, float)
        self.levels = list(levels); self.prior_sd = prior_sd; self.plsd = prior_logsd; self.csize = csize

    def fit(self, y, X, k, v, th=None):
        """th=None: estima hiperparâmetros (MAP). th dado: só posterior condicional (1 fatoração)."""
        self.lev = {}; cols = []; off = X.shape[1]; n = len(y)
        for L in self.levels:
            u, inv = np.unique(k[L], return_inverse=True)
            self.lev[L] = dict(idx={nm: i for i, nm in enumerate(u)}, off=off, n=len(u)); off += len(u)
            cols.append(sp.csr_matrix((np.ones(n), (np.arange(n), inv)), shape=(n, len(u))))
        M = sp.hstack([sp.csr_matrix(X)] + cols).tocsc()
        self.p = X.shape[1]; self.q = off
        winv = 1.0 / (1.0 + self.csize / np.maximum(v, 1.0))
        r = y - X @ self.b0
        MtW = M.T.multiply(winv).tocsc()
        MtWM = (MtW @ M).tocsc(); MtWr = MtW @ r; rWr = (r * winv * r).sum(); logdetW = np.log(winv).sum()
        sizes = [self.lev[L]["n"] for L in self.levels]
        # ordenação: bloco esparso (zona, mun) por RCM, colunas densas (beta, reg, uf) no fim -> pouco preenchimento
        isd = np.zeros(self.q, bool); isd[:self.p] = True
        for L in self.levels:
            if self.lev[L]["n"] <= 60: o_ = self.lev[L]["off"]; isd[o_:o_ + self.lev[L]["n"]] = True
        sidx = np.where(~isd)[0]
        S = MtWM[sidx][:, sidx]; S = (abs(S) + sp.identity(len(sidx))).tocsr()
        ps = reverse_cuthill_mckee(S, symmetric_mode=True) if len(sidx) else np.array([], int)
        P = np.r_[sidx[ps], np.where(isd)[0]]
        self.P = P; MtWMp = MtWM[P][:, P].tocsc(); MtWrp = MtWr[P]
        b0v = self.b0sd ** 2

        def build(th):
            sd = np.exp(th); s2e = sd[-1] ** 2
            kdiag = np.r_[b0v, np.concatenate([np.full(nl, sd[i] ** 2) for i, nl in enumerate(sizes)])]
            return (MtWMp / s2e + sp.diags(1.0 / kdiag[P])).tocsc(), kdiag, s2e

        fact = lambda C: splu(C, permc_spec="NATURAL", diag_pivot_thresh=0, options=dict(SymmetricMode=True))
        pm = np.log([self.prior_sd[L] for L in self.levels + ["e"]])

        def nll(th):
            C, kdiag, s2e = build(th)
            try: lu = fact(C)
            except Exception: return 1e10
            ld = np.log(np.abs(lu.U.diagonal())).sum()
            g = MtWrp / s2e
            quad = rWr / s2e - g @ lu.solve(g)
            logdetV = (n * np.log(s2e) - logdetW) + np.log(kdiag).sum() + ld
            return 0.5 * (logdetV + quad) + 0.5 * (((th - pm) / self.plsd) ** 2).sum()

        if th is None:
            o = minimize(nll, pm.copy(), method="L-BFGS-B", bounds=[(np.log(1e-4), np.log(0.2))] * len(pm),
                         options=dict(maxiter=40, ftol=1e-7))
        else:
            class o: x = np.asarray(th); nfev = 0
        self.nfev = o.nfev; self.th = o.x; self.sd = dict(zip(self.levels + ["e"], np.exp(o.x)))
        C, kdiag, s2e = build(o.x); self._lu = fact(C); self.s2e = s2e
        self.post = np.r_[self.b0, np.zeros(self.q - self.p)] + self.solve(MtWr / s2e)
        return self

    def solve(self, g):
        x = np.empty_like(g); x[self.P] = self._lu.solve(g[self.P]); return x

    def design(self, X, k):
        n = X.shape[0]; Ai = []; Aj = []; unobs = {}
        for L in self.levels:
            idx = self.lev[L]["idx"]; off = self.lev[L]["off"]
            j = np.array([idx.get(x, -1) for x in k[L]]); m = j >= 0
            Ai.append(np.arange(n)[m]); Aj.append(off + j[m]); unobs[L] = (~m, k[L])
        Ai = np.concatenate(Ai); Aj = np.concatenate(Aj)
        Z = sp.csr_matrix((np.ones(len(Ai)), (Ai, Aj)), shape=(n, self.q))
        M = (sp.hstack([sp.csr_matrix(X), Z[:, self.p:]])).tocsr()
        return M, unobs

    def var_lin(self, M, unobs, w, size=None):
        """Variância a posteriori de sum_z w_z * eta_z (eta = preditor + efeitos não vistos + ruído)."""
        a = M.T @ w; var = a @ self.solve(a)
        for L in self.levels:
            m, kk = unobs[L]
            if m.any():
                sw = pd.Series(w[m]).groupby(kk[m]).sum().values; var += self.sd[L] ** 2 * (sw ** 2).sum()
        if size is not None:
            var += self.s2e * (w ** 2 * (1 + self.csize / np.maximum(size, 1))).sum()
        return var
