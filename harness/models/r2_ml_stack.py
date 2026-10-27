"""r2_ml_stack — stacking HONESTO sobre o r1_bayes_v5 (rodada 2, agente ML).
Base: r1_bayes_v5 (importado, não reescrito): virada hierárquica + comparecimento + shift de seleção + IC.
Correções de ML no RESÍDUO do v5:
  1. Previsões out-of-fold do v5 nas zonas JÁ apuradas: GroupKFold por município, refazendo o posterior do v5 em cada
     fold com os hiperparâmetros (th) do ajuste completo (1 fatoração esparsa por fold).
  2. r_oof = virada observada - v5_oof. Aprendizes de resíduo (XGB raso; MLP opcional) treinados em r_oof com features
     de transferência do 1T (+ histórico da eleição ANTERIOR, quando existir), com previsões OOF nos mesmos folds.
  3. Meta: NNLS  r_oof ~ a*g_xgb + c*g_mlp  (a,c >= 0, a+c <= 1), ponderado pela precisão do v5 (1/(1+6000/v2)).
     Com poucas zonas (n < NMIN) ou poucos municípios: a=c=0 => v5 puro.
  y(U) = y_v5(U) + a*g_xgb(U) + c*g_mlp(U);  v2 e intervalo nacional: do v5.
Histórico (regra R2): features de eleição anterior só valem se vierem da eleição ANTERIOR à avaliada.
  Avaliando 2022 -> usa 2018 (virada 2018 da zona, mudança de b); avaliando 2018 -> NADA (detectado: os votos do 1T das
  zonas apuradas coincidem com os de 2018). Em 2026 trocar a fonte para 2022. Sem p18 (era vazamento em --ano 2018).
Hiperparâmetros escolhidos SÓ em --ano 2018."""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "harness")
import numpy as np, pandas as pd, xgboost as xgb
from scipy.optimize import nnls
from sklearn.model_selection import GroupKFold
import r1_bayes_v5 as V5
from r1_bayes_common import HG, keys
import r1_bayes_v2 as B2, r1_bayes_v3 as B3

UFS = sorted("AC AM AP PA RO RR TO AL BA CE MA PB PE PI RN SE DF GO MS MT ES MG RJ SP PR RS SC ZZ".split())
REGS = ["N", "NE", "CO", "SE", "S", "EX"]; K = ["uf", "mun", "zona"]
_H = {}

def _hist_table():
    if "t" not in _H:
        from data import load_2018
        o = load_2018(); b = o.c13 / (o.c13 + o.c22).clip(lower=1)
        _H["t"] = pd.DataFrame({"sw_prev": (o.y - b).values, "b_prev": b.values, "c13_prev": o.c13.values},
                               index=pd.MultiIndex.from_frame(o[K]))
    return _H["t"]

def hist_ok(R):
    """False se a eleição avaliada É a do histórico (2018): aí não existe eleição anterior disponível."""
    h = _hist_table().reindex(pd.MultiIndex.from_frame(R[K]))
    same = (h.c13_prev.values == R.c13.values).mean()
    return same < 0.5

def feats(d, hist):
    v = d.validos1.values.astype(float); c = d.comp1.values.astype(float); a = d.a1.values.astype(float)
    b = d.c13.values / np.maximum(d.c13.values + d.c22.values, 1)
    X = {"b": b, "t": d.c15.values / v, "c": d.c12.values / v, "o": d.c_out.values / v,
         "bn": (d.brancos1.values + d.nulos1.values) / c, "ab": d.abst1.values / a, "lt": d.c13.values / v,
         "bt": d.c22.values / v, "la": np.log(a), "da": d.aptos2.values / a - 1,
         "ufc": d.uf.map({u: i for i, u in enumerate(UFS)}).values.astype(float),
         "reg": d.regiao.map({r: i for i, r in enumerate(REGS)}).values.astype(float)}
    if hist:
        h = _hist_table().reindex(pd.MultiIndex.from_frame(d[K]))
        sp_ = h.sw_prev.values; bp = h.b_prev.values
        X["sw_prev"] = np.where(np.isnan(sp_), np.nanmean(sp_) if np.isfinite(sp_).any() else 0.0, sp_)
        X["db_prev"] = np.where(np.isnan(bp), 0.0, b - bp)
    return pd.DataFrame(X, index=d.index)

class Model(V5.Model):
    name = "r2_ml_stack"
    NMIN = 300; NFOLD = 3; USE_MLP = True; USE_HIST = True; CAP = 1.0
    # CONGELADO (escolhido em --ano 2018 quick: d3/150 1.215 -> d4/300 1.207 -> +MLP(32) 1.199; v2 ridge pior que v2 do v5)
    XGB = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
               min_child_weight=10, reg_lambda=10.)
    MLP = dict(hidden_layer_sizes=(32,), alpha=1e-2, learning_rate_init=3e-3, max_iter=60)
    V2 = "v5"

    def _learners(self, F, r, w):
        out = {"x": xgb.XGBRegressor(n_jobs=1, verbosity=0, **self.XGB).fit(F, r, sample_weight=w)}
        if self.USE_MLP:
            from sklearn.neural_network import MLPRegressor
            mu = F.mean(0); sd = F.std(0) + 1e-9; rs = r.std() + 1e-9
            m = MLPRegressor(random_state=0, **self.MLP).fit((F - mu) / sd, r / rs, sample_weight=w)
            out["m"] = (m, mu, sd, rs)
        return out

    def _gpred(self, L, F):
        g = {"x": L["x"].predict(F)}
        if "m" in L:
            m, mu, sd, rs = L["m"]; g["m"] = m.predict((F - mu) / sd) * rs
        return g

    def fit(self, R):
        super().fit(R)                      # v5 completo (posterior, comparecimento, shift, tau)
        self.coef = {}; self.L = None
        if self.V2 == "ridge":              # alternativa: ridge (features 1T + dummies UF) em log(v2/validos1) — r1_ml_lib
            import r1_ml_lib as ML
            wv = R.v2.values / R.v2.mean()
            self.VR = ML.HRidge(ML.COLS, alpha=10., su=1., kmun=0, kzon=0).fit(ML.feats(R), R, np.log(R.v2 / R.validos1).values, wv)
        n = len(R)
        if n < self.NMIN or R.munid.nunique() < 4 * self.NFOLD:
            return self
        self.hist = self.USE_HIST and hist_ok(R)
        k = keys(R); v = R.v2.values.astype(float); X5, b = B3.feats_s(R); s = R.y.values - b
        w = 1.0 / (1.0 + 6000.0 / np.maximum(v, 1.0))
        folds = list(GroupKFold(self.NFOLD).split(X5, s, R.munid.values))
        p5 = np.zeros(n)
        sub = lambda kk, ii: {Lv: kk[Lv][ii] for Lv in kk}
        for tr, te in folds:
            h = HG(B3.S_B0, B3.S_SD, B2.S_PRIOR).fit(s[tr], X5[tr], sub(k, tr), v[tr], th=self.ms.th)
            M, _ = h.design(X5[te], sub(k, te)); p5[te] = M @ h.post
        r = s - p5
        F = feats(R, self.hist).values
        G = {}
        for tr, te in folds:
            g = self._gpred(self._learners(F[tr], r[tr], w[tr]), F[te])
            for nm, val in g.items(): G.setdefault(nm, np.zeros(n))[te] = val
        names = sorted(G); A = np.c_[[G[nm] for nm in names]].T; sw = np.sqrt(w)
        coef, _ = nnls(A * sw[:, None], r * sw)
        if coef.sum() > self.CAP: coef = coef * self.CAP / coef.sum()
        self.coef = dict(zip(names, coef))
        self.oof_gain = 1 - np.average((r - A @ coef) ** 2, weights=w) / np.average(r ** 2, weights=w)
        if coef.sum() > 0:
            self.L = self._learners(F, r, w)
        return self

    def predict(self, U):
        p = super().predict(U)
        if self.V2 == "ridge":
            import r1_ml_lib as ML
            p["v2"] = U.validos1.values * np.exp(self.VR.predict(ML.feats(U), U, re=False))
        if self.L is not None:
            g = self._gpred(self.L, feats(U, self.hist).values)
            p["y"] = p["y"].values + sum(self.coef[nm] * g[nm] for nm in g)
        return p
