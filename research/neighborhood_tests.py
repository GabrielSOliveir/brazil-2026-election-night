import pandas as pd, numpy as np
from sklearn.neighbors import BallTree
v=pd.read_parquet("data/work/votos_local.parquet"); l=pd.read_parquet("data/work/locais.parquet")
k=["uf","mun","zona","local"]
t1=v[v.turno==1].set_index(k); t2=v[v.turno==2].set_index(k)
d=pd.DataFrame({"l1":t1["13"],"b1":t1["22"],"t1":t1["15"],"c1":t1["12"],"v1":t1[["13","22","15","12","out"]].sum(axis=1),
                "L2":t2["13"],"B2":t2["22"]}).dropna().join(l.set_index(k)).reset_index()
d=d[(d.uf!="ZZ")&(d.lat.between(-34,6))&(d.lon.between(-75,-28))&(d.L2+d.B2>=300)&(d.l1+d.b1>0)].copy()
d["v2"]=d.L2+d.B2; d["y"]=100*d.L2/d.v2; d["base"]=100*d.l1/(d.l1+d.b1); d["sw"]=d.y-d.base
d["munid"]=d.uf+d.mun.astype(str); d["zid"]=d.munid+"-"+d.zona.astype(str); d["bid"]=d.munid+"-"+d.bairro.fillna("").str.upper().str.strip()
print("locais:",len(d)," swing médio %.2f, dp %.2f | nível dp %.2f"%(np.average(d.sw,weights=d.v2),d.sw.std(),d.y.std()))
# T1: variância explicada por níveis aninhados
def expl(col):
    tot=np.average((d[col]-np.average(d[col],weights=d.v2))**2,weights=d.v2); out=[]
    for g in ["uf","munid","zid","bid"]:
        m=d.groupby(g)[col].transform(lambda s: s.mean()); out.append(round(100*(1-np.average((d[col]-m)**2,weights=d.v2)/tot),1))
    return out
print("T1 % da variância explicada por [UF, município, zona, bairro] — VIRADA:",expl("sw")," NÍVEL:",expl("y"))
# T2: semelhança vs distância (dentro de cidades grandes), após tirar a média do município
big=d.groupby("munid").filter(lambda g: len(g)>=150).copy()
for c in ["sw","y"]: big[c+"_r"]=big[c]-big.groupby("munid")[c].transform("mean")
rng=np.random.default_rng(0); rows=[]
for mid,g in big.groupby("munid"):
    X=np.radians(g[["lat","lon"]].values); n=len(g); i=rng.integers(0,n,4000); j=rng.integers(0,n,4000); ok=i!=j; i,j=i[ok],j[ok]
    dist=6371*2*np.arcsin(np.sqrt(np.sin((X[i,0]-X[j,0])/2)**2+np.cos(X[i,0])*np.cos(X[j,0])*np.sin((X[i,1]-X[j,1])/2)**2))
    rows.append(pd.DataFrame({"dist":dist,"sw_i":g.sw_r.values[i],"sw_j":g.sw_r.values[j],"y_i":g.y_r.values[i],"y_j":g.y_r.values[j],
        "mesmo_bairro":(g.bid.values[i]==g.bid.values[j])}))
P=pd.concat(rows); P["faixa"]=pd.cut(P.dist,[0,0.5,1,2,5,10,30,1e9],labels=["<0,5km","0,5-1","1-2","2-5","5-10","10-30",">30"])
t=P.groupby("faixa",observed=True).apply(lambda s: pd.Series({"pares":len(s),"corr_virada":np.corrcoef(s.sw_i,s.sw_j)[0,1],"corr_nivel":np.corrcoef(s.y_i,s.y_j)[0,1]}))
print("T2 (cidades com >=150 locais: %d cidades) correlação entre locais por distância, após tirar a média da cidade:"%big.munid.nunique()); print(t.round(2).to_string())
mb=P.groupby("mesmo_bairro").apply(lambda s: pd.Series({"corr_virada":np.corrcoef(s.sw_i,s.sw_j)[0,1],"corr_nivel":np.corrcoef(s.y_i,s.y_j)[0,1]}))
print("mesmo bairro vs bairros diferentes:"); print(mb.round(2).to_string())
d.to_parquet("data/work/d.parquet"); big.to_parquet("data/work/big.parquet")
