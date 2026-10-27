"""Utilitários dos modelos temporais (rodada 1, agente ts). Não é um modelo.
Ideia central: swing s = y - b, b = c13/(c13+c22) do 1T. A base é o swing uniforme.
Os modelos temporais olham a SEQUÊNCIA de chegada do swing em R e tentam extrapolar o swing médio de U.
Tempo = fração acumulada dos votos válidos u (0..1) e sua transformação probit g = Phi^-1(u)."""
import numpy as np, pandas as pd
from scipy.stats import norm

def b1(d): return d.c13/(d.c13+d.c22)

def feats(d):
    v=d.validos1
    return np.c_[d.c13/v, d.c22/v, d.c15/v, d.c12/v, d.c_out/v, d.abst1/d.a1, (d.brancos1+d.nulos1)/d.comp1]

def residualize(R, lam=1e-3):
    """resíduo do swing após ridge nas proporções do 1T, ajustado só em R (remove composição, ex. como2022)."""
    s=(R.y-b1(R)).values; w=R.v2.values/R.v2.mean(); X=feats(R); mu=np.average(X,0,w); sd=X.std(0)+1e-9
    Xs=np.c_[np.ones(len(R)),(X-mu)/sd]; A=Xs.T@(Xs*w[:,None])+lam*len(R)*np.diag([0]+[1]*X.shape[1])
    beta=np.linalg.solve(A,Xs.T@(w*s)); return s-Xs@beta

def timeline(R, U):
    """u_i (meio do intervalo acumulado) de cada zona de R, fração apurada q e V total estimado."""
    VU=(U.validos1*R.v2.sum()/R.validos1.sum()).sum(); V=R.v2.sum()+VU
    c=R.v2.cumsum().values; u=(c-R.v2.values/2)/V; return u, R.v2.sum()/V, V

def batches(vals, w, K):
    """agrupa a sequência (ordenada) em K lotes de peso ~igual; devolve médias ponderadas e pesos."""
    cw=np.cumsum(w)/w.sum(); k=np.minimum((cw*K-1e-9).astype(int),K-1)
    W=np.bincount(k,w,K); out=[np.bincount(k,w*v,K)/np.maximum(W,1e-12) for v in vals]
    keep=W>0; return [o[keep] for o in out], W[keep]

def probit_shift(R, U, tau=0.003, K=None, resid=True, return_all=False, slab=None):
    """Modelo de seleção na ordem: s_k = a + gamma*(g_k - gbar) por lote; gamma encolhido com priori N(0,tau^2)
    (se de gamma estimado pela dispersão ENTRE lotes, que já absorve correlação espacial).
    Devolve o deslocamento do swing médio de U relativo à média de R: gamma*(E[g|u>q] - gbar_R)."""
    n=len(R)
    if n<30: return (0.0,0.0,0.0) if return_all else 0.0
    u,q,V=timeline(R,U); g=norm.ppf(np.clip(u,1e-6,1-1e-6))
    s=residualize(R) if resid else (R.y-b1(R)).values
    K=K or int(np.clip(n//15,6,40)); w=R.v2.values.astype(float)
    (sk,gk),Wk=batches([s,g],w,K); Wn=Wk/Wk.mean()
    gb=np.average(gk,weights=Wn); sb=np.average(sk,weights=Wn); dg=gk-gb
    Sxx=(Wn*dg**2).sum(); gam=(Wn*dg*(sk-sb)).sum()/Sxx
    r=sk-sb-gam*dg; s2=(Wn*r**2).sum()/max(len(sk)-2,1); se2=s2/Sxx
    post=gam*tau**2/(tau**2+se2) if tau is not None else gam
    if slab is not None:   # spike-and-slab: slab=(prob a priori de haver tendência, tau_slab)
        pi,ts=slab; t2=gam**2/se2; r=se2/(se2+ts**2)
        lbf=0.5*np.log(r)+0.5*t2*(1-r); ps=1/(1+(1-pi)/pi*np.exp(-min(lbf,50)))
        post=ps*gam*ts**2/(ts**2+se2)
    gU=norm.pdf(norm.ppf(q))/(1-q); gR=np.average(g,weights=w)
    sh=post*(gU-gR)
    return (sh,gam,np.sqrt(se2)) if return_all else sh
