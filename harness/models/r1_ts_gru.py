"""GRU (torch, 1 thread) que lê a SEQUÊNCIA de lotes já chegados e prevê a correção do swing das zonas que faltam:
    swing_U = swing_médio_R + sd_R * GRU(seq)        (corretor sobre o swing uniforme)
Entrada por lote k (K=20 lotes de igual peso dentro de R): (s_k - s̄_R)/sd_R, probit(u_k), (b_k - b̄_R)/sd_b, log q.
Pré-treino: 2018 (load_2018) com ORDENS SIMULADAS (a ordem de 2018 não existe): aleatória, como (1T bolsonarista
antes), geo (blocos por UF), pior/pior_inv (ordenada por swing + ruído de força aleatória). Alvo normalizado:
(s̄_U - s̄_R)/sd_R. Nada de 2022 entra no treino. Pesos em harness/weights/ts_gru.pt.
MIX (env TS_GMIX): pesos das ordens no pré-treino = priori de quão provável é uma ordem "seletiva"."""
import sys, os, warnings; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,"harness")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch, torch.nn as nn
from scipy.stats import norm
from r1_ts_common import b1, batches, timeline
torch.set_num_threads(1)
K=20
MIX=os.environ.get("TS_GMIX","30,20,20,15,15")   # aleat, como, geo, pior, pior_inv
WP=f"harness/weights/ts_gru_{MIX.replace(',','-')}.pt"

def seq_feats(R,U):
    s=(R.y-b1(R)).values; b=b1(R).values; w=R.v2.values.astype(float)
    u,q,V=timeline(R,U); g=norm.ppf(np.clip(u,1e-6,1-1e-6))
    sm=np.average(s,weights=w); sd=np.sqrt(np.average((s-sm)**2,weights=w))+1e-4
    bm=np.average(b,weights=w); sdb=np.sqrt(np.average((b-bm)**2,weights=w))+1e-4
    (sk,gk,bk),Wk=batches([s,g,b],w,K)
    X=np.c_[(sk-sm)/sd,gk/3,(bk-bm)/sdb,np.full(len(sk),np.log(q)/4)]
    return X.astype(np.float32), sm, sd

class Net(nn.Module):
    def __init__(s,h=24):
        super().__init__(); s.rnn=nn.GRU(4,h,batch_first=True); s.out=nn.Linear(h,1)
    def forward(s,x): o,_=s.rnn(x); return s.out(o[:,-1]).squeeze(-1)

def make_data(n=6000,seed=0):
    from data import load_2018
    z=load_2018().reset_index(drop=True); rng=np.random.default_rng(seed)
    sw=(z.y-b1(z)).values; b1v=(z.c22/z.validos1).values; sdsw=sw.std()
    ufs=z.uf.values; uu=np.unique(ufs); mix=np.array([float(x) for x in MIX.split(",")]); mix/=mix.sum()
    Xs=[];Ys=[]
    for i in range(n):
        typ=rng.choice(5,p=mix)
        if typ==0: o=rng.random(len(z))
        elif typ==1: o=-b1v+rng.normal(0,rng.uniform(0.2,1.0),len(z))
        elif typ==2:
            m=dict(zip(uu,rng.permutation(len(uu)))); o=np.vectorize(m.get)(ufs)/len(uu)+rng.normal(0,rng.uniform(0.1,1),len(z))
        else:
            sg=1 if typ==3 else -1; o=sg*sw/sdsw+rng.normal(0,rng.uniform(0.3,3),len(z))
        zz=z.iloc[np.argsort(o)].reset_index(drop=True); cum=zz.v2.cumsum().values/zz.v2.sum()
        q=np.exp(rng.uniform(np.log(0.01),np.log(0.85))); nn_=int((cum<=q).sum())
        if nn_<40: continue
        R=zz.iloc[:nn_]; U=zz.iloc[nn_:]
        X,sm,sd=seq_feats(R,U)
        if len(X)<K: continue
        sU=np.average((U.y-b1(U)).values,weights=U.v2.values)
        Xs.append(X); Ys.append(np.clip((sU-sm)/sd,-3,3))
    return torch.tensor(np.stack(Xs)), torch.tensor(np.array(Ys,dtype=np.float32))

def pretrain():
    torch.manual_seed(0); X,Y=make_data(); n=len(X); idx=torch.randperm(n); va=idx[:n//6]; tr=idx[n//6:]
    net=Net(); opt=torch.optim.Adam(net.parameters(),1e-3,weight_decay=1e-4); best=(1e9,None)
    for ep in range(60):
        net.train(); p=tr[torch.randperm(len(tr))]
        for j in range(0,len(p),128):
            bi=p[j:j+128]; opt.zero_grad(); l=nn.functional.huber_loss(net(X[bi]),Y[bi],delta=1.0); l.backward(); opt.step()
        net.eval()
        with torch.no_grad(): lv=nn.functional.mse_loss(net(X[va]),Y[va]).item()
        if lv<best[0]: best=(lv,{k:v.clone() for k,v in net.state_dict().items()})
    print("pretrain val mse",best[0],"var(Y)",Y[va].var().item(),"n",n)
    torch.save(best[1],WP)

_NET=None
def get_net():
    global _NET
    if _NET is None:
        if not os.path.exists(WP): pretrain()
        _NET=Net(); _NET.load_state_dict(torch.load(WP)); _NET.eval()
    return _NET

class Model:
    name="r1_ts_gru_"+MIX.replace(",","-")
    def fit(self,R): self.R=R
    def predict(self,U):
        R=self.R; s=(R.y-b1(R)).values; sh=np.average(s,weights=R.v2)
        if len(R)>=40:
            X,sm,sd=seq_feats(R,U)
            if len(X)==K:
                with torch.no_grad(): c=float(get_net()(torch.tensor(X[None])))
                sh=sm+sd*c
        return pd.DataFrame({"y":b1(U)+sh},index=U.index)

if __name__=="__main__": pretrain()
