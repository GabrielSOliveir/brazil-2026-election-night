"""GRU treinada AO VIVO só com o que já chegou (sem pré-treino): para prefixos de R (30%..90% de R) a rede lê os lotes
do prefixo e aprende a prever o swing médio do RESTO de R (relativo ao prefixo, normalizado). Na previsão, lê R inteiro
e o horizonte real (log q) -> correção do swing de U. Limitação estrutural: no treino o horizonte é no máx. ~3x o
prefixo; na hora H, com 1% apurado, é 99x. A rede não tem como extrapolar isso."""
import sys, os, warnings; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch, torch.nn as nn
from scipy.stats import norm
from r1_ts_common import b1, batches, timeline
torch.set_num_threads(1); KB=10
def fe(s,w,u,qq):
    sm=np.average(s,weights=w); sd=np.sqrt(np.average((s-sm)**2,weights=w))+1e-4
    (sk,gk),_=batches([s,norm.ppf(np.clip(u,1e-6,1-1e-6))],w,KB)
    if len(sk)<KB: return None,sm,sd
    return np.c_[(sk-sm)/sd,gk/3,np.full(KB,np.log(qq)/4)].astype(np.float32),sm,sd
class Net(nn.Module):
    def __init__(s,h=8): super().__init__(); s.r=nn.GRU(3,h,batch_first=True); s.o=nn.Linear(h,1)
    def forward(s,x): o,_=s.r(x); return s.o(o[:,-1]).squeeze(-1)
class Model:
    name="r1_ts_gru_online"
    def fit(self,R): self.R=R
    def predict(self,U):
        R=self.R; s=(R.y-b1(R)).values; w=R.v2.values.astype(float); sh=np.average(s,weights=w)
        if len(R)>=100:
            u,q,V=timeline(R,U); cw=np.cumsum(w); X=[];Y=[]
            for f in np.linspace(0.3,0.9,40):
                j=int(np.searchsorted(cw,f*cw[-1]))
                x,sm,sd=fe(s[:j],w[:j],u[:j],cw[j-1]/V)
                if x is None or j>=len(s)-5: continue
                X.append(x); Y.append(np.clip((np.average(s[j:],weights=w[j:])-sm)/sd,-3,3))
            xR,sm,sd=fe(s,w,u,q)
            if len(X)>=10 and xR is not None:
                torch.manual_seed(0); X=torch.tensor(np.stack(X)); Y=torch.tensor(np.array(Y,dtype=np.float32))
                net=Net(); opt=torch.optim.Adam(net.parameters(),1e-2,weight_decay=1e-3)
                for ep in range(150): opt.zero_grad(); l=nn.functional.mse_loss(net(X),Y); l.backward(); opt.step()
                with torch.no_grad(): sh=sm+sd*float(net(torch.tensor(xR[None])))
        return pd.DataFrame({"y":b1(U)+sh},index=U.index)
