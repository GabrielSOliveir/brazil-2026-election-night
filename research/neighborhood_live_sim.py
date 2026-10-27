import pandas as pd, numpy as np
from sklearn.neighbors import BallTree
d=pd.read_parquet("data/work/d.parquet").reset_index(drop=True)
def shrink_mean(R, key, prior, k):
    g=R.groupby(key).agg(s=("res","sum"),n=("res","size")); return g.s/(g.n+k)
def run(frac,seed,mode):
    rng=np.random.default_rng(seed)
    if mode=="aleatorio": rep=rng.random(len(d))<frac
    else:  # bairros inteiros chegam juntos (mais realista: uma zona/local reporta em bloco)
        b=d.bid.unique(); chosen=set(b[rng.random(len(b))<frac]); rep=d.bid.isin(chosen).values
    R=d[rep].copy(); U=d[~rep].copy(); w=R.v2
    nat=np.average(R.sw,weights=w); R["res"]=R.sw-nat; pred={}
    # A: hierarquia UF > município > zona (partial pooling)
    p=np.full(len(U),nat); Rr=R.copy(); Ur=U.copy()
    for key,k in (("uf",5),("munid",3),("zid",3)):
        eff=shrink_mean(Rr,key,0,k); Rr["res"]=Rr.res-Rr[key].map(eff).fillna(0).values; p=p+Ur[key].map(eff).fillna(0).values
    pred["A_hier_zona"]=p.copy()
    # B: + bairro
    effb=shrink_mean(Rr,"bid",0,2); pred["B_+bairro"]=p+U.bid.map(effb).fillna(0).values
    # C: + vizinhos geográficos (resíduo dos locais apurados a <2km, peso por distância), sobre A
    tree=BallTree(np.radians(R[["lat","lon"]].values),metric="haversine")
    ind,dist=tree.query_radius(np.radians(U[["lat","lon"]].values),r=2/6371,return_distance=True)
    rres=Rr.res.values; adj=np.zeros(len(U))
    for i,(ix,ds) in enumerate(zip(ind,dist)):
        if len(ix): wt=1/(ds*6371+0.3); adj[i]=(wt*rres[ix]).sum()/(wt.sum()+1.5)
    pred["C_+vizinhos2km"]=p+adj
    pred["D_+bairro+viz"]=pred["B_+bairro"]+0.5*adj
    out={m:np.average(np.abs(v-U.sw.values),weights=U.v2) for m,v in pred.items()}
    # agregado por zona (o que o modelo usa)
    for m,v in pred.items():
        z=pd.DataFrame({"z":U.zid.values,"e":(v-U.sw.values)*U.v2.values,"w":U.v2.values}).groupby("z").sum()
        out[m+"|zona"]=np.average(np.abs(z.e/z.w),weights=z.w)
    return out
rows=[]
for mode in ["aleatorio","bairro_em_bloco"]:
    for frac in [0.05,0.2,0.5]:
        for s in range(2):
            r=run(frac,s,mode); r.update(modo=mode,frac=frac); rows.append(r)
T=pd.DataFrame(rows).groupby(["modo","frac"]).mean()
cols=[c for c in T.columns if "|" not in c]; print("MAE da virada por LOCAL (pp):"); print(T[cols].round(3).to_string())
cols=[c for c in T.columns if "|" in c]; print("MAE agregado por ZONA (pp):"); print(T[cols].round(3).to_string())
