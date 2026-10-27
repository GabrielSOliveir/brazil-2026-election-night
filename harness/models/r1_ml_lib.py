"""Biblioteca do agente ML (rodada 1). NÃO é um modelo para avaliar; os modelos r1_ml_*.py importam daqui.
Alvo sempre = virada aditiva  s = y - b,  b = c13/(c13+c22) do 1T (fatia do Lula nos dois primeiros do 1T).
Fatia de votos válidos v2 = validos1 * exp(r), r previsto por ridge hierárquico (corrige viés de comparecimento).
Modelos base sobre s (todos ponderados por v2):
  m0 swing uniforme | m1 ridge de transferência + efeitos UF/mun/zona (shrinkage) | m2 m1 + XGB raso no resíduo
  m3 m1 + MLP pequeno no resíduo | m4 XGB raso direto em s
Stacking: OOF GroupKFold por município DENTRO das zonas apuradas -> NNLS (pesos >=0, soma 1); fallback m1 se poucas zonas.
"""
import numpy as np, pandas as pd, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "harness")
from scipy.optimize import nnls
from sklearn.model_selection import GroupKFold
import xgboost as xgb

UFS = sorted("AC AM AP PA RO RR TO AL BA CE MA PB PE PI RN SE DF GO MS MT ES MG RJ SP PR RS SC ZZ".split())
REGS = ["N", "NE", "CO", "SE", "S", "EX"]
_K = ["uf", "mun", "zona"]
_O18 = None

def _o18():
    global _O18
    if _O18 is None:
        from data import load_2018
        o = load_2018()
        b18 = o.c13 / (o.c13 + o.c22).clip(lower=1)
        _O18 = pd.DataFrame({"sw18": o.y - b18, "b18": b18}).set_index(pd.MultiIndex.from_frame(o[_K]))
    return _O18

def feats(d, use18=False):
    v = d.validos1.astype(float); c = d.comp1.astype(float); a = d.a1.astype(float)
    b = d.c13 / (d.c13 + d.c22).clip(lower=1)
    X = pd.DataFrame({"b": b, "bb": b * (1 - b), "t": d.c15 / v, "c": d.c12 / v, "o": d.c_out / v,
                      "bn": (d.brancos1 + d.nulos1) / c, "ab": d.abst1 / a, "lt": d.c13 / v, "bt": d.c22 / v,
                      "ex": (d.uf == "ZZ").astype(float), "la": np.log(a), "da": d.aptos2 / a - 1}, index=d.index)
    if use18:
        o = _o18().reindex(pd.MultiIndex.from_frame(d[_K]))
        s18 = o.sw18.values; b18 = o.b18.values
        X["sw18"] = np.where(np.isnan(s18), np.nanmean(s18), s18)  # falta (0,6%) -> média
        X["db18"] = np.where(np.isnan(b18), 0.0, b.values - b18)
        X["m18"] = o.sw18.isna().values.astype(float)
        X["p18"] = _p18model().predict(X, d, re=False)  # virada prevista pelo modelo de transferência AJUSTADO EM 2018
    return X

_P18 = None
def _p18model():
    """Ridge de transferência (features 1T + UF) ajustado no 2018 completo (eleição passada => sem vazamento)."""
    global _P18
    if _P18 is None:
        from data import load_2018
        o = load_2018(); Xo = feats(o); to = (o.y - Xo.b).values; wo = o.v2.values / o.v2.mean()
        _P18 = HRidge(COLS, su=3., kmun=0, kzon=0).fit(Xo, o, to, wo)
    return _P18

def ufd(d): return np.c_[[(d.uf == u).values.astype(float) for u in UFS]].T

def wridge(X, t, w, alpha, pen):
    sw = w / w.sum(); mx = sw @ X; mt = sw @ t; Xc = X - mx; tc = t - mt
    A = (Xc * w[:, None]).T @ Xc + np.diag(alpha * pen) * w.sum() / len(w)
    beta = np.linalg.solve(A, (Xc * w[:, None]).T @ tc)
    return beta, mt - mx @ beta

def reff(keys, r, w, k):
    s = pd.DataFrame({"k": keys, "wr": w * r, "w": w}).groupby("k").sum()
    return s.wr / (s.w + k)

class HRidge:
    """ridge ponderado em features padronizadas + dummies de UF (escala su) + efeitos aleatórios mun/zona (k em 'zonas médias')."""
    def __init__(s, cols, alpha=10., su=3., kmun=2., kzon=2., iters=2):
        s.cols, s.alpha, s.su, s.kmun, s.kzon, s.iters = cols, alpha, su, kmun, kzon, iters
    def _X(s, X, d):
        Xs = ((X[s.cols] - s.mu) / s.sd).values
        return np.c_[Xs, ufd(d)] if s.su > 0 else Xs
    def fit(s, X, d, t, w):
        s.mu = X[s.cols].mean(); s.sd = X[s.cols].std() + 1e-9
        Xa = s._X(X, d); pen = np.r_[np.ones(len(s.cols)), np.ones(Xa.shape[1] - len(s.cols)) / max(s.su, 1e-9) ** 2]
        km = d.munid.values; kz = (d.uf + "-" + d.zona.astype(str)).values
        s.em = pd.Series(dtype=float); s.ez = pd.Series(dtype=float); off = np.zeros(len(d))
        for _ in range(s.iters if (s.kmun or s.kzon) else 1):
            s.beta, s.c = wridge(Xa, t - off, w, s.alpha, pen); f = Xa @ s.beta + s.c
            if s.kmun: s.em = reff(km, t - f - s.ez.reindex(kz).fillna(0).values, w, s.kmun)
            if s.kzon: s.ez = reff(kz, t - f - s.em.reindex(km).fillna(0).values, w, s.kzon)
            off = s.em.reindex(km).fillna(0).values + s.ez.reindex(kz).fillna(0).values
        return s
    def predict(s, X, d, re=True):
        p = s._X(X, d) @ s.beta + s.c
        if re:
            p = p + s.em.reindex(d.munid.values).fillna(0).values + s.ez.reindex((d.uf + "-" + d.zona.astype(str)).values).fillna(0).values
        return p

def tree_X(X, d, cols):
    A = X[cols].copy(); A["ufc"] = d.uf.map({u: i for i, u in enumerate(UFS)}).values; A["reg"] = d.regiao.map({r: i for i, r in enumerate(REGS)}).values
    return A.values

class Base:
    """Ajusta todos os modelos base num conjunto (X,d,t,w) e prevê em outro."""
    def __init__(s, cols, tcols, which, xgb_kw, mlp_kw, ridge_kw):
        s.cols, s.tcols, s.which, s.xgb_kw, s.mlp_kw, s.ridge_kw = cols, tcols, which, xgb_kw, mlp_kw, ridge_kw
    def fit(s, X, d, t, w):
        s.m0 = np.average(t, weights=w)
        s.r = HRidge(s.cols, **s.ridge_kw).fit(X, d, t, w)
        res = t - s.r.predict(X, d, re=False)  # resíduo sem RE (RE é aplicado depois, evita vazamento in-sample)
        TX = tree_X(X, d, s.tcols)
        if "m2" in s.which:
            s.x2 = xgb.XGBRegressor(n_jobs=1, verbosity=0, **s.xgb_kw).fit(TX, res, sample_weight=w)
        if "m4" in s.which:
            s.x4 = xgb.XGBRegressor(n_jobs=1, verbosity=0, **s.xgb_kw).fit(TX, t, sample_weight=w)
        if "m3" in s.which:
            from sklearn.neural_network import MLPRegressor
            s.mmu = X[s.tcols].mean(); s.msd = X[s.tcols].std() + 1e-9
            M = np.c_[((X[s.tcols] - s.mmu) / s.msd).values, ufd(d) ]
            s.rs = res.std() + 1e-9
            s.x3 = MLPRegressor(random_state=0, **s.mlp_kw).fit(M, res / s.rs, sample_weight=w)
        return s
    def predict(s, X, d):
        r = s.r.predict(X, d, re=False); reff_ = s.r.predict(X, d, re=True) - r
        out = {"m0": np.full(len(d), s.m0), "m1": r + reff_}
        TX = tree_X(X, d, s.tcols)
        if "m2" in s.which: out["m2"] = r + reff_ + s.x2.predict(TX)
        if "m4" in s.which: out["m4"] = s.x4.predict(TX)
        if "m3" in s.which:
            M = np.c_[((X[s.tcols] - s.mmu) / s.msd).values, ufd(d)]
            out["m3"] = r + reff_ + s.x3.predict(M) * s.rs
        return pd.DataFrame(out, index=d.index)

COLS = ["t", "c", "o", "bn", "ab", "b", "bb", "lt", "bt", "ex"]
TCOLS = ["t", "c", "o", "bn", "ab", "b", "lt", "bt", "la", "da"]

class StackModel:
    name = "r1_ml_stack"
    use18 = False; which = ("m2", "m3", "m4"); nfold = 3; nmin = 300; wpow_meta = 1.0; stack = True
    cols = COLS; tcols = TCOLS
    xgb_kw = dict(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=20, reg_lambda=10.)
    mlp_kw = dict(hidden_layer_sizes=(32, 16), alpha=1e-2, learning_rate_init=3e-3, max_iter=80, early_stopping=False)
    ridge_kw = dict(alpha=10., su=3., kmun=2., kzon=2.)
    wrec = 0.0
    fixed_w = None  # dict para pesos fixos (modelo adaptativo sem stacking)
    def _cols(s):
        e = ["sw18", "db18", "m18", "p18"] if s.use18 else []
        return list(s.cols) + e, list(s.tcols) + e
    alpha_n = None  # se setado: alpha do ridge = max(alpha, alpha_n/n) (mais shrinkage com poucas zonas)
    def _base(s, n=None):
        c, tc = s._cols(); rk = dict(s.ridge_kw)
        if s.alpha_n and n: rk["alpha"] = max(rk["alpha"], s.alpha_n / n)
        return Base(c, tc, s.which, s.xgb_kw, s.mlp_kw, rk)
    def fit(s, R):
        X = feats(R, s.use18); t = (R.y - X.b).values; w = R.v2.values.astype(float); w = w / w.mean()
        n = len(R)
        if s.wrec:  # peso de recência: zonas apuradas mais tarde parecem mais com as que faltam
            o = np.argsort(np.argsort(R.ordem.values)) if "ordem" in R else np.arange(n)
            u = (np.cumsum(R.v2.values[np.argsort(o)])[o]) / R.v2.sum()
            w = w * (0.05 + u) ** s.wrec; w = w / w.mean()
        s.B = s._base(n).fit(X, R, t, w)
        names = ["m0", "m1"] + [m for m in ("m2", "m3", "m4") if m in s.which]
        s.names = names
        if s.fixed_w is not None:
            s.W = s.fixed_w(n)
        elif not s.stack or n < s.nmin or R.munid.nunique() < 2 * s.nfold:
            s.W = {"m1": 1.0}
        else:
            P = pd.DataFrame(index=R.index, columns=names, dtype=float)
            for tr, te in GroupKFold(s.nfold).split(X, t, R.munid.values):
                b = s._base(n).fit(X.iloc[tr], R.iloc[tr], t[tr], w[tr])
                P.iloc[te] = b.predict(X.iloc[te], R.iloc[te])[names].values
            ww = np.sqrt(w ** s.wpow_meta)
            A = P.values * ww[:, None]; yv = t * ww
            # NNLS com soma=1 via linha de penalização forte
            lam = 1e3 * np.sqrt((A ** 2).sum() / A.shape[0])
            A2 = np.r_[A, lam * np.ones((1, A.shape[1]))]; y2 = np.r_[yv, lam]
            coef, _ = nnls(A2, y2); coef = coef / coef.sum()
            s.W = dict(zip(names, coef)); s.oof = P
        # v2: log(v2/validos1) por ridge hierárquico
        Xv = feats(R); s.V = HRidge(["t", "c", "o", "bn", "ab", "b", "bb", "lt", "bt", "ex"], alpha=10., su=1., kmun=0, kzon=0).fit(
            Xv, R, np.log(R.v2 / R.validos1).values, w)
        return s
    def predict(s, U):
        X = feats(U, s.use18); P = s.B.predict(X, U)
        sw = sum(wt * P[m].values for m, wt in s.W.items())
        y = X.b.values + sw
        v2 = U.validos1.values * np.exp(s.V.predict(feats(U), U, re=False))
        return pd.DataFrame({"y": y, "v2": v2}, index=U.index)
