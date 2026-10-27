"""Combinação: swing uniforme + GRU pré-treinada em 2018 SEM ordens seletivas (mix 60,20,20,0,0)
+ corretor de tendência probit com spike-and-slab (pi=0,01, tau=1pp por unidade probit) sobre o resíduo da GRU."""
import sys, os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
os.environ["TS_GMIX"]="60,20,20,0,0"
import numpy as np, pandas as pd
import r1_ts_gru as G
from r1_ts_common import b1, probit_shift
class Model:
    name="r1_ts_combo"
    def fit(self,R): self.R=R
    def predict(self,U):
        R=self.R; p=G.Model(); p.fit(R); base=p.predict(U).y
        return pd.DataFrame({"y":base+probit_shift(R,U,slab=(0.01,0.01),resid=False)},index=U.index)
