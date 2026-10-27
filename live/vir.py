import pandas as pd, numpy as np, sys
s=pd.read_csv(sys.argv[1],dtype={"mun":str,"zona":str}); t=pd.read_csv("data/derived/zones_2026_round1.csv",dtype={"mun":str,"zona":str})
m=s.merge(t[["uf","mun","zona","c13","c22"]],on=["uf","mun","zona"],suffixes=("","_1")); m=m[(m.st>0)&(m.c13+m.c22>0)]; m["f"]=m.st/m.ts
for uf in ["sp","rj","mg"]:
    d=m[m.uf==uf]; c=d[d.f>=1]
    vp=d.c13.sum()/(d.c13+d.c22).sum()-d.c13_1.sum()/(d.c13_1+d.c22_1).sum()
    vc=(c.c13.sum()/(c.c13+c.c22).sum()-c.c13_1.sum()/(c.c13_1+c.c22_1).sum()) if len(c) else np.nan
    print(f"{uf}: zonas c/ dados {len(d)}, completas {len(c)} | virada parciais {100*vp:+.1f} | completas {100*vc:+.1f}")
