"""Simulação de apuração PARCIAL (seções chegando dentro de cada zona), calibrada na curva oficial do TSE de 2022.
f_i(τ) = fração apurada da zona i no horário τ = clip((τ-17h)/(c·(ts_i-17h)),0,1)^k  (ts_i = hora em que a zona terminou em 2022)
fatia parcial observada = y_i + β·(1-f_i)^m + ruído de seções  (β<0: seções pró-direita chegam antes dentro da zona, como em 2022)
"""
import numpy as np, pandas as pd
T0=pd.Timestamp("2022-10-30 17:00:00")
H="data/tse/2022/Historico_Totalizacao_Presidente_BR_2T_2022.csv"
def tse_curve():
    h=pd.read_csv(H,sep=";",decimal=",",skipinitialspace=True); h.columns=[c.strip() for c in h.columns]
    h["t"]=pd.to_datetime(h.DT_TOTALIZACAO.str.strip(),format="%d/%m/%Y %H:%M:%S"); h["pct"]=100*h.PE_SECOES_TOT_ACUMULADO
    L=h.LULA_QT_VOTOS_TOT_ACUMULADO; B=h.JAIR_BOLSONARO_QT_VOTOS_TOT_ACUMULADO; h["lula"]=L/(L+B)
    return h[["t","pct","lula"]]
def frac(ts_min,tau_min,c,k): return np.clip(tau_min/np.maximum(c*ts_min,1),0,1)**k
def simulate(z,tau_min,c,k,beta,s_sec,seed,m=1.0):
    rng=np.random.default_rng(seed); ts=((z.t-T0).dt.total_seconds()/60).values
    f=frac(ts,tau_min,c,k); f[ts<=tau_min]=1.0
    nsec=np.maximum(z.a1.values/330,1); ns=np.maximum(f*nsec,1e-9)
    sd=s_sec*np.sqrt(np.clip(1-f,0,1)/ns)
    yp=np.clip(z.y.values+beta*(1-f)**m+rng.normal(0,1,len(z))*sd,0.001,0.999)
    vp=np.round(f*z.v2.values); Lp=np.round(vp*yp)
    return f,Lp,vp-Lp
