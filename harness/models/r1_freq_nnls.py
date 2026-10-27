"""r1_freq_nnls — hipótese literal do usuário: matriz de transferência por regressão de contagens restrita.
L2 ≈ X·pL, B2 ≈ X·pB, X = [c13,c22,c15,c12,c_out,brancos1+nulos1,abst1]; retenção própria em [0.8,1.1],
transferências em [0,1]; WLS com var(contagem) ∝ a1^2 (pesos 1/a1^2). Prevê y=L/(L+B) e v2=L+B,
mais correção de nível (swing residual uniforme) e de escala de v2 estimadas nas zonas apuradas."""
import numpy as np, pandas as pd
from scipy.optimize import lsq_linear
SRC=["c13","c22","c15","c12","c_out","bn1","abst1"]
def X_(d): return d.assign(bn1=d.brancos1+d.nulos1)[SRC].values.astype(float)
class Model:
    name="r1_freq_nnls"
    lo=np.array([0.8,0,0,0,0,0,0.]); hi=np.array([1.1,0.1,1,1,1,1,1.])
    def fit(self,R):
        X=X_(R); w=1/R.a1.values.astype(float)
        self.pL=lsq_linear(X*w[:,None],R.L2.values*w,bounds=(self.lo,self.hi)).x
        o=[1,0,2,3,4,5,6]; self.pB=lsq_linear(X*w[:,None],R.B2.values*w,bounds=(self.lo[o],self.hi[o])).x
        p=self._raw(R); self.s=np.average(R.y-p[0],weights=R.v2); self.r=R.v2.sum()/p[1].sum()
    def _raw(self,d):
        X=X_(d); L=np.maximum(X@self.pL,1); B=np.maximum(X@self.pB,1); return L/(L+B),L+B
    def predict(self,U):
        y,v=self._raw(U); return pd.DataFrame({"y":y+self.s,"v2":v*self.r},index=U.index)
