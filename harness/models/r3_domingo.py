"""r3_domingo — modelo para o domingo (interface v3: zonas PARCIAIS). Hiperparâmetros escolhidos só em 2018.

Ideia: cada zona com seções apuradas é uma MEDIDA RUIDOSA E ENVIESADA da sua fatia final:
    y_parcial_z = b_z + virada_z + delta(f_z) + amostragem_z
  - virada_z: hierarquia reg > UF > zona eleitoral > município (+ efeito próprio da zona), transferência x b, curva em b,
    log(tamanho)  [o mesmo modelo do r2_bayes_final_dint];
  - delta(f) = sum_p d_p (1-f)^p, p em DELTA_POW: viés intra-zona das primeiras seções, ESTIMADO AO VIVO
    (zera em f=1; priori simétrica -> serve para direita-antes, esquerda-antes ou nenhum); identificado pelas
    zonas completas e quase completas;
  - amostragem: var = s_sec^2 (1-f)/n_secoes_vistas (s_sec estimado ao vivo).
  Previsão da fatia FINAL de uma zona incompleta = parte hierárquica + BLUP do efeito próprio (encolhe a parcial
  desenviesada para a previsão, com peso s2e/(s2e + amostragem)). Comparecimento: mesmo esquema em log(v2/validos1).
Exterior (ZZ): fora do modelo do Brasil; virada própria das zonas do exterior quase completas, encolhida para a
virada nacional. Intervalo 90% (t): posterior conjunta + grupos não vistos + ruído próprio + comparecimento
+ exterior + termo de discrepância W_restante^2 (KAPPA*sqrt(sd_reg^2+sd_uf^2))^2.
publish_ok: trava (completas+equivalente em parciais, nº de UFs, meia-largura do IC). Saneamento e fallback.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.stats import t as tdist
from r3_bayes_engine import HG3, keys

# ------------------------------------------------------------------ CONFIG (editar aqui para 2026)
CFG = dict(
    COL=dict(A="c13", B="c22", T3="c15", T4="c12", REST="c_out"),   # A = finalista "esquerda" (alvo), B = o outro
    EXTERIOR="ZZ", ELEIT_POR_SECAO=330.0, C_SIZE=6000.0,
    DELTA_POW=(10.0, 30.0), DELTA_SD=(0.10, 0.10),       # viés intra-zona concentrado nas primeiras seções
    S_PRIOR=dict(reg=0.008, uf=0.008, zon=0.008, mun=0.006, e=0.006, s=0.08),
    V_PRIOR=dict(reg=0.02, uf=0.02, zon=0.015, mun=0.015, e=0.015, s=0.1),
    KAPPA=0.08, NU=6.0, DMIS=0.30, DMIS_NOHIST=0.5,          # erro de forma do desvio (fração do viés estimado)
    EX_PRIOR_SD=0.05, EX_FMIN=0.5,                       # exterior: sd a priori da diferença p/ virada nacional
    PUB_MIN_EQ=500.0, PUB_MIN_UF=20, PUB_MAX_HALF=1.0,   # trava de publicação
    COMP_SD=0.01,                                        # sd a priori do desvio de nível das zonas COMPLETAS (seleção de quem termina antes)
    # prioris dos coeficientes (ordem das colunas em feats_s / feats_v). Transferência: p_k ~ 0,5 (neutra) com sd larga
    S_B0=[0.0, 0.5, 0.5, 0.5, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    S_SD=[0.05, 0.5, 0.5, 0.5, 0.7, 0.7, 0.7, 0.2, 0.2, 0.3, 0.5, 0.02],
    V_B0=[0.0, 0, 0, 0, 0, 1.0], V_SD=[0.05, 0.2, 0.2, 0.3, 0.3, 0.5],
    MAXITER=25, MIN_SEC=1.0, USE_HIST=True, HIST_MIN_PAIRS=200, HIST_SD_MULT=0.05,  # delta vem da história da zona                             # linhas parciais com < MIN_SEC seções vistas não entram
)
_WARM = {}   # warm start dos hiperparâmetros entre atualizações (ao vivo e no avaliador)
_HIST = {}   # última leitura parcial de cada zona (f, fatia) -> viés intra-zona pela EVOLUÇÃO da própria zona

# ------------------------------------------------------------------ saneamento
def _num(d, cols):
    for c in cols:
        if c in d: d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0).astype(float)
    return d

def _b(d):
    A = d[CFG["COL"]["A"]].values; B = d[CFG["COL"]["B"]].values
    return np.where(A + B > 0, A / np.maximum(A + B, 1e-9), 0.5)

def clean(d, is_R):
    d = d.copy()
    if "regiao" in d: d["regiao"] = d.regiao.fillna("NA")
    d = _num(d, ["a1", "comp1", "abst1", "brancos1", "nulos1", "validos1", "aptos2", "frac", "L2", "B2", "v2", "L2p", "B2p"] + list(CFG["COL"].values()))
    cand = sum(d[c] for c in CFG["COL"].values())
    d["validos1"] = np.maximum(d.validos1.values, cand.values)          # coerência 1T (validos >= soma dos candidatos)
    if is_R:
        d = d[(d.v2 > 0) & np.isfinite(d.v2) & (d.validos1 > 0)]
        if "frac" not in d: d["frac"] = 1.0
        d["frac"] = d.frac.clip(1e-6, 1.0)
        d = d.drop_duplicates(["uf", "mun", "zona"], keep="last")
    return d

# ------------------------------------------------------------------ features
def feats_s(d):
    c = CFG["COL"]; V = np.maximum(d.validos1.values, 1.0); b = _b(d)
    r3 = d[c["T3"]].values / V; r4 = d[c["T4"]].values / V; ro = d[c["REST"]].values / V
    bn = (d.brancos1.values + d.nulos1.values) / np.maximum(d.comp1.values, 1); ab = d.abst1.values / np.maximum(d.a1.values, 1)
    X = np.column_stack([np.ones(len(d)), r3, r4, ro, r3 * b, r4 * b, ro * b, bn - 0.06, ab - 0.2,
                         (b - 0.5) ** 2, (b - 0.5) ** 3, np.log(np.maximum(d.a1.values, 1)) - 9.0])
    return X, b
S_B0 = CFG["S_B0"]; S_SD = CFG["S_SD"]

def feats_v(d):
    c = CFG["COL"]; V = np.maximum(d.validos1.values, 1.0); b = _b(d)
    r = (d[c["T3"]].values + d[c["T4"]].values + d[c["REST"]].values) / V
    bn = (d.brancos1.values + d.nulos1.values) / np.maximum(d.comp1.values, 1); ab = d.abst1.values / np.maximum(d.a1.values, 1)
    da = np.log(np.maximum(d.aptos2.values, 1) / np.maximum(d.a1.values, 1))
    return np.column_stack([np.ones(len(d)), b - 0.5, r - 0.1, bn - 0.06, ab - 0.2, da])
V_B0 = CFG["V_B0"]; V_SD = CFG["V_SD"]

def delta_basis(f):   # viés intra-zona (zera em f=1) + indicador de zona completa (desvio de nível por seleção)
    return np.column_stack([(1.0 - f) ** p for p in CFG["DELTA_POW"]] + [(f >= 1).astype(float)])

def samp_g(d, f):
    nsec = np.maximum(d.a1.values / CFG["ELEIT_POR_SECAO"], 1.0)
    return np.where(f >= 1, 0.0, (1 - f) / np.maximum(f * nsec, 1.0))

def zkey(d): return (d.uf.astype(str) + "|" + d.mun.astype(str) + "|" + d.zona.astype(str)).values

# ------------------------------------------------------------------ modelo
class Model:
    name = "r3_domingo"; v3 = True; cfg = CFG

    def fit(self, R):
        self.ok = False; self.R = R
        try:
            self._fit(clean(R, True)); self.ok = True
        except Exception as e:      # fallback: swing uniforme das linhas completas (ou bruto)
            self.err = repr(e); self._fit_fallback(R)
        return self

    def _fit_fallback(self, R):
        R = clean(R, True); comp = R[R.frac >= 1] if len(R[R.frac >= 1]) else R
        b = _b(comp); w = comp.v2.values
        self.fb_s = float(np.average(comp.y.values - b, weights=w)) if len(comp) and w.sum() > 0 else 0.0
        self.fb_ratio = float(comp.v2.sum() / comp.validos1.sum()) if len(comp) and comp.validos1.sum() > 0 else 0.97

    def _fit(self, R):
        c = self.cfg; ex = R.uf.astype(str).values == c["EXTERIOR"]
        nsv = R.frac.values * np.maximum(R.a1.values / c["ELEIT_POR_SECAO"], 1.0)
        keep = (R.frac.values >= 1) | (nsv >= c["MIN_SEC"])
        D = R[~ex & keep]; E = R[ex]
        f = D.frac.values; b = _b(D)
        X, _ = feats_s(D); Db = delta_basis(f); k = keys(D)
        csz = 1.0 + c["C_SIZE"] / np.maximum(D.validos1.values, 1.0); g = samp_g(D, f)
        ys = D.L2.values / D.v2.values - b
        nd = len(c["DELTA_POW"]) + 1; dsd = list(c["DELTA_SD"]) + [c["COMP_SD"]]
        d0 = [0.0] * nd
        if c["USE_HIST"]:
            dh = self._delta_hist(D, f, ys + b, g)
            if dh is not None:
                m_, sd_ = dh; d0 = list(m_) + [0.0]; dsd = list(np.asarray(sd_) * c["HIST_SD_MULT"]) + [c["COMP_SD"]]
                self.dh_cov = np.diag(np.asarray(sd_) ** 2)
        self.ms = HG3(S_B0 + d0, S_SD + dsd, c["S_PRIOR"], maxiter=c["MAXITER"])
        self.ms.fit(ys, np.column_stack([X, Db]), k, csz, g, th0=_WARM.get("s")); _WARM["s"] = self.ms.th
        _WARM["s2s"] = self.ms.s2s if self.ms.s2s > 0 else 0.08 ** 2
        Xv = feats_v(D); yv = np.log(D.v2.values / (f * D.validos1.values))
        self.mv = HG3(V_B0 + [0.0] * nd, V_SD + dsd, c["V_PRIOR"], maxiter=c["MAXITER"])
        self.mv.fit(yv, np.column_stack([Xv, Db]), k, csz, g, th0=_WARM.get("v")); _WARM["v"] = self.mv.th
        self.rowR = pd.Series(np.arange(len(D)), index=zkey(D))
        self.fR = f; self.vpR = D.v2.values
        # virada nacional (média doméstica prevista, para o exterior) e razão de comparecimento
        self.ratio = float(np.exp(np.average(yv, weights=D.validos1.values * f)))
        sw_nat = float(np.average(ys - Db @ self.ms.post[X.shape[1]:X.shape[1] + nd], weights=D.validos1.values * f))
        # exterior: zonas quase completas
        Eq = E[E.frac >= c["EX_FMIN"]]
        if len(Eq):
            se = Eq.L2.values / Eq.v2.values - _b(Eq); w = Eq.v2.values / Eq.frac.values
            m = np.average(se, weights=w); n_eff = w.sum() ** 2 / (w ** 2).sum()
            vs = np.average((se - m) ** 2, weights=w) + 1e-4
            pv = 1 / (1 / c["EX_PRIOR_SD"] ** 2 + n_eff / vs)
            self.ex_s = pv * (sw_nat / c["EX_PRIOR_SD"] ** 2 + n_eff * m / vs); self.ex_var = pv
            self.ex_ratio = float(Eq.v2.sum() / (Eq.frac * Eq.validos1).sum())
        else:
            self.ex_s = sw_nat; self.ex_var = c["EX_PRIOR_SD"] ** 2; self.ex_ratio = self.ratio
        self.px = X.shape[1]; self.pvx = Xv.shape[1]

    def _delta_hist(self, D, f, yobs, g):
        """viés intra-zona estimado pela mudança da fatia parcial da MESMA zona entre leituras (cancela a virada
        da zona e a seleção entre zonas). Devolve (média, sd) para a priori de delta, ou None."""
        c = self.cfg; key = zkey(D); prev = _HIST.get("z")
        cur = pd.DataFrame({"f": f, "y": yobs, "g": g}, index=key)
        out = None; self.nhist = 0
        if prev is not None:
            j = prev.reindex(cur.index); mt = j.f.notna().values
            if mt.sum() > 0 and (cur.f.values[mt] + 1e-9 >= j.f.values[mt]).mean() < 0.95: prev = None   # novo fluxo
        if prev is not None:
            j = prev.reindex(cur.index); mt = (j.f.notna() & (cur.f > j.f + 1e-3)).values
            if mt.sum() >= c["HIST_MIN_PAIRS"]:
                f1 = j.f.values[mt]; f2 = cur.f.values[mt]
                dB = delta_basis(f2)[:, :-1] - delta_basis(f1)[:, :-1]; dy = cur.y.values[mt] - j.y.values[mt]
                s2 = _WARM.get("s2s", 0.08 ** 2)
                vv = s2 * np.maximum(j.g.values[mt] - cur.g.values[mt], 0.0) + 1e-4
                A = dB.T @ (dB / vv[:, None]) + np.diag(1.0 / np.asarray(c["DELTA_SD"]) ** 2)
                cov = np.linalg.inv(A); mu = cov @ (dB.T @ (dy / vv))
                # robustez: escala pelo qui-quadrado dos resíduos (superdispersão)
                rr = dy - dB @ mu; phi = max(1.0, float((rr ** 2 / vv).mean()))
                out = (mu, np.sqrt(np.diag(cov) * phi)); self.nhist = int(mt.sum())
        _HIST["z"] = cur
        return out

    def predict(self, U):
        idx = U.index; out = pd.DataFrame({"y": 0.5, "v2": 0.0}, index=idx)
        try:
            Uc = clean(U, False)
            if not self.ok: raise RuntimeError("fallback")
            y, v2 = self._predict(Uc)
        except Exception as e:
            self.ok = False; self.err = getattr(self, "err", repr(e))
            if not hasattr(self, "fb_s"): self._fit_fallback(self.R)
            Uc = clean(U, False); y = _b(Uc) + self.fb_s; v2 = Uc.validos1.values * self.fb_ratio
        y = np.nan_to_num(np.clip(y, 0.0, 1.0), nan=0.5); v2 = np.nan_to_num(np.maximum(v2, 0.0), nan=0.0)
        out["y"] = y; out["v2"] = v2; self._pred = out
        return out

    def _predict(self, U):
        c = self.cfg; n = len(U); ex = U.uf.astype(str).values == c["EXTERIOR"]
        f = U.frac.values if "frac" in U else np.zeros(n)
        vp = (U.L2p.values + U.B2p.values) if "L2p" in U else np.zeros(n)
        b = _b(U); y = np.zeros(n); v2 = np.zeros(n)
        # exterior
        y[ex] = b[ex] + self.ex_s; v2[ex] = np.maximum(U.validos1.values[ex] * self.ex_ratio, vp[ex])
        # Brasil
        dm = ~ex; D = U[dm]; nd = len(c["DELTA_POW"]) + 1
        X, _ = feats_s(D); Xv = feats_v(D); k = keys(D)
        Ms, us = self.ms.design(np.column_stack([X, np.zeros((len(D), nd))]), k)
        Mv, uv = self.mv.design(np.column_stack([Xv, np.zeros((len(D), nd))]), k)
        j = self.rowR.reindex(zkey(D)).values; obs = ~np.isnan(j); jj = np.where(obs, j, 0).astype(int)
        ks = np.where(obs, self.ms.kshr[jj], 0.0); kv = np.where(obs, self.mv.kshr[jj], 0.0)
        rs = np.where(obs, self.ms.resid[jj], 0.0); rv = np.where(obs, self.mv.resid[jj], 0.0)
        ys = b[dm] + Ms @ self.ms.post + ks * rs
        lv = Mv @ self.mv.post + kv * rv
        vd = np.maximum(D.validos1.values * np.exp(lv), vp[dm])
        y[dm] = ys; v2[dm] = vd
        fD = f[dm]; DbU = delta_basis(np.where(obs, fD, 1.0))       # base delta das linhas observadas
        self._c = dict(Ms=Ms, us=us, Mv=Mv, uv=uv, ks=ks, kv=kv, Db=DbU, dm=dm, ex=ex,
                       csz=1.0 + c["C_SIZE"] / np.maximum(np.maximum(D.validos1.values, vd), 1.0), vp=vp)
        return y, v2

    # -------------------------------------------------------------- incerteza
    def _nat_se(self, R, U, pred):
        c = self.cfg; C = self._c
        vh = np.asarray(pred["v2"], float); yh = np.asarray(pred["y"], float)
        comp = clean(R, True); comp = comp[comp.frac >= 1]
        Vt = comp.v2.sum() + vh.sum(); nat = (comp.L2.sum() + (yh * vh).sum()) / Vt
        dm = C["dm"]; w = vh[dm] / Vt; px = self.px; pvx = self.pvx; nd = len(c["DELTA_POW"]) + 1
        var = 0.0
        for mdl, M, us, kk, wz, pxx in [(self.ms, C["Ms"], C["us"], C["ks"], w, px),
                                         (self.mv, C["Mv"], C["uv"], C["kv"], (yh[dm] - nat) * w, pvx)]:
            a = M.T @ (wz * (1 - kk))
            a[pxx:pxx + nd] -= C["Db"].T @ (wz * kk)
            rv = (wz ** 2 * mdl.s2e * C["csz"] * (1 - kk)).sum()
            var += mdl.var_lin(a, us, wz * (1 - kk), rv)
            if mdl is self.ms and getattr(self, "dh_cov", None) is not None:
                ad = a[pxx:pxx + nd - 1]; var += ad @ self.dh_cov @ ad
        wex = vh[C["ex"]].sum() / Vt; var += wex ** 2 * self.ex_var
        # erro de especificação do desenviesamento: proporcional ao viés corrigido
        dcorr = C["Db"][:, :-1] @ self.ms.post[px:px + nd - 1]
        fr = c["DMIS"] if getattr(self, "dh_cov", None) is not None else c["DMIS_NOHIST"]
        Sd = (w * C["ks"] * np.abs(dcorr)).sum(); var0 = var
        var += (fr * Sd) ** 2
        Wrest = (vh.sum() - C["vp"].sum()) / Vt; hyp = np.hypot(self.ms.sd["reg"], self.ms.sd["uf"])
        var += (Wrest * c["KAPPA"] * hyp) ** 2
        self.last = dict(var0=var0, Sd=Sd, Wh=Wrest * hyp, hist=int(getattr(self, "dh_cov", None) is not None))
        return nat, np.sqrt(max(var, 1e-12))

    def national_interval(self, R, U, pred):
        try:
            if not self.ok: raise RuntimeError
            nat, se = self._nat_se(R, U, pred)
        except Exception:
            vh = np.asarray(pred["v2"], float); yh = np.asarray(pred["y"], float)
            nat = float((yh * vh).sum() / max(vh.sum(), 1)); se = 0.03
        zq = tdist.ppf(0.95, self.cfg["NU"]); self.se = se
        return float(nat - zq * se), float(nat + zq * se)

    def publish_ok(self, R, U):
        c = self.cfg
        try:
            Rc = clean(R, True); dom = Rc[Rc.uf.astype(str) != c["EXTERIOR"]]
            eq = float(dom.frac.sum()); nuf = dom[dom.frac >= 0.2].uf.nunique()
            if not hasattr(self, "se"): self.national_interval(R, U, self._pred)
            half = tdist.ppf(0.95, c["NU"]) * self.se * 100
            return bool(self.ok and getattr(self, "dh_cov", None) is not None and eq >= c["PUB_MIN_EQ"] and nuf >= c["PUB_MIN_UF"] and half <= c["PUB_MAX_HALF"])
        except Exception:
            return False
