"""Swing uniforme + corretor temporal de SELEÇÃO NA ORDEM (o vencedor desta rodada no eixo temporal).
Por lote de chegada: s_k = a + gamma*(g_k - gbar), g = probit da fração acumulada de votos. Se a ordem de chegada
estiver correlacionada com o swing (cenário 'pior'), E[s | posição u] ~ mu + gamma*Phi^-1(u) (normal bivariada), e a
média do swing nas zonas que faltam é a + gamma*(E[g|u>q] - gbar_R) — extrapolação com forma fechada, sem explodir.
gamma tem priori spike-and-slab (pi=1% de haver tendência; slab N(0, 1pp^2)); erro-padrão pela dispersão ENTRE lotes.
Só dispara com t >~ 3,5; nos cenários normais fica ~= swing uniforme."""
import sys, os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from r1_ts_common import b1, probit_shift
PI=float(os.environ.get("TS_PI","0.01")); TAU=float(os.environ.get("TS_TAU","0.01"))
class Model:
    name="r1_ts_probit"
    def fit(self,R): self.R=R; self.s=np.average(R.y-b1(R),weights=R.v2)
    def predict(self,U):
        return pd.DataFrame({"y":b1(U)+self.s+probit_shift(self.R,U,slab=(PI,TAU),resid=False)},index=U.index)
