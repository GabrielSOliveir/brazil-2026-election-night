"""ARIMA na série ACUMULADA por lote, extrapolada até 100% dos votos.
SERIE='swing': swing acumulado (ancorado no 1T).  SERIE='bruto': fatia bruta acumulada do Lula (sem âncora no 1T).
Ordem (p,d,q)=(1,1,0); TREND="t" (drift) ou "n" (sem drift). Horizonte = nº de lotes que faltam (lotes de igual peso)."""
import sys, os, warnings; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from r1_ts_common import b1, batches, timeline
SERIE=os.environ.get("TS_ASERIE","swing"); TREND=os.environ.get("TS_ATREND","t")
class Model:
    name="r1_ts_arima_"+SERIE+"_"+TREND
    def fit(self,R): self.R=R
    def predict(self,U):
        R=self.R; w=R.v2.values.astype(float); bR=b1(R).values
        x=(R.y.values-bR) if SERIE=="swing" else R.y.values
        _,q,V=timeline(R,U); VR=w.sum(); VU=V-VR
        K=int(np.clip(len(R)//15,6,40)); (xk,),Wk=batches([x],w,K)
        C=np.cumsum(xk*Wk)/np.cumsum(Wk)                    # série acumulada (o que aparece na tela da apuração)
        H=int(min(K*(1-q)/q,4000)); Cf=C[-1]
        try: Cf=float(ARIMA(C*100,order=(1,1,0),trend=TREND).fit().forecast(H)[-1])/100
        except Exception: pass
        xU=(Cf*V-C[-1]*VR)/VU                               # média de U implícita no acumulado final previsto
        vh=U.validos1*VR/R.validos1.sum()
        if SERIE=="swing": y=b1(U)+xU
        else: bU=b1(U); y=bU+(xU-np.average(bU,weights=vh))  # fatia média de U = xU, distribuída pelo perfil do 1T
        return pd.DataFrame({"y":y},index=U.index)
