import sys; sys.path.insert(0,"harness"); from data import UF2REG
import pandas as pd
s=pd.read_csv(sys.argv[1],dtype={"mun":str}); s["reg"]=s.uf.str.upper().map(UF2REG).fillna("EX")
g=s.groupby("reg").agg(st=("st","sum"),ts=("ts","sum"),L=("c13","sum"),F=("c22","sum")); print("apurado por região:",(100*g.st/g.ts).round(1).to_dict())
u=s.groupby("uf").agg(st=("st","sum"),ts=("ts","sum"),L=("c13","sum"),F=("c22","sum"))
for k in ["sp","rj","mg"]: print(k, "apurado %.1f%%"%(100*u.st[k]/u.ts[k]), "Lula contado %.1f%%"%(100*u.L[k]/max(u.L[k]+u.F[k],1)))
