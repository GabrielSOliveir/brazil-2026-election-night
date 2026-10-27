"""Kalman (statsmodels UnobservedComponents) na série de swing médio por LOTE de chegada.
MODE='llevel': nível local (random walk + ruído) -> swing de U = nível filtrado no último lote ("memória ajustada ao vivo").
MODE='lltrend': nível+tendência local -> swing de U = média das previsões nos lotes futuros até 100%."""
import sys, os, warnings; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents
from r1_ts_common import b1, batches, timeline
MODE=os.environ.get("TS_KMODE","llevel")
class Model:
    name="r1_ts_kalman_"+MODE
    def fit(self,R): self.R=R
    def predict(self,U):
        R=self.R; s=(R.y-b1(R)).values; w=R.v2.values.astype(float); su=np.average(s,weights=w)
        sh=su
        if len(R)>=60:
            _,q,V=timeline(R,U); K=int(np.clip(len(R)//15,6,40)); (sk,),Wk=batches([s],w,K)
            try:
                m=UnobservedComponents(sk*100,MODE).fit(disp=False,maxiter=200)
                if MODE=="llevel": sh=m.filtered_state[0,-1]/100     # nível no último lote
                else:
                    H=int(min(K*(1-q)/q,4000)); f=m.forecast(H)/100; sh=float(np.mean(f))
            except Exception: pass
        return pd.DataFrame({"y":b1(U)+sh},index=U.index)
