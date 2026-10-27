# gera um snapshot FALSO de 2T a partir do 1T 2026 só para testar o código (não é previsão)
import pandas as pd, numpy as np, sys
t=pd.read_csv("data/derived/zones_2026_round1.csv",dtype={"mun":str,"zona":str}); rng=np.random.default_rng(int(sys.argv[2]))
f=float(sys.argv[1]); fr=np.clip(rng.beta(1,1/f-1 if f<1 else 1,len(t)) if f<1 else np.ones(len(t)),0,1)
b=(t.c13/(t.c13+t.c22).clip(lower=1)).fillna(0.5); y=np.clip(b+0.015+rng.normal(0,0.01,len(t)),0,1); vv=(t.c13+t.c22+t.c70+t.c14+t.c55+t.cout)*0.98
st=np.floor(t.ts*fr).astype(int); fr2=st/t.ts.clip(lower=1)
L=np.round(vv*fr2*np.clip(y-0.03*(1-fr2)**10,0,1)); B=np.round(vv*fr2)-L
out=t.copy(); out["st"]=st; out["c13"]=L.astype(int); out["c22"]=B.astype(int); out[["c70","c14","c55","cout"]]=0; out.to_csv(sys.argv[3],index=False)
