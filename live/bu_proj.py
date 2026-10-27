import pandas as pd, numpy as np, sys
s=pd.read_csv(sys.argv[1],dtype={"mun":str,"zona":str,"secao":str}); s=s[(s.l1+s.f1>0)&(s.l2+s.f2>0)]
s["b"]=s.l1/(s.l1+s.f1); s["y"]=s.l2/(s.l2+s.f2); s["sw"]=s.y-s.b; s["w"]=s.l2+s.f2
s["v1"]=s.l1+s.f1+s.cury1+s.renan1+s.caiado1+s.out1; s["t3"]=(s.cury1+s.renan1+s.caiado1+s.out1)/s.v1
s["munid"]=s.uf+s.mun
print(f"seções {len(s)} | municípios {s.munid.nunique()} | UFs {s.uf.nunique()}")
print(f"mesmas seções: Lula 1T {100*s.l1.sum()/(s.l1+s.f1).sum():.2f}% -> 2T {100*s.l2.sum()/(s.l2+s.f2).sum():.2f}% (só Lula+Flávio)")
print(f"virada média ponderada {100*np.average(s.sw,weights=s.w):+.2f}pp | mediana {100*s.sw.median():+.2f} | seções com virada pró-Lula {100*(s.sw>0).mean():.0f}%")
print(f"comparecimento {s.comp1.sum()} -> {s.comp2.sum()} ({100*(s.comp2.sum()/s.comp1.sum()-1):+.1f}%) | válidos L+F 2T / válidos 1T = {s.w.sum()/s.v1.sum():.3f}")
# virada vs perfil
X=np.c_[np.ones(len(s)),s.b-0.5,s.t3]; beta=np.linalg.lstsq(X*np.sqrt(s.w.values)[:,None],s.sw.values*np.sqrt(s.w.values),rcond=None)[0]
print("virada = %.4f + %.4f*(b-0.5) + %.4f*fatia_terceiros"%tuple(beta))
# projeção com as zonas do 1T 2026
t=pd.read_csv("data/derived/zones_2026_round1.csv",dtype={"mun":str,"zona":str})
t=t[(t.c13+t.c22)>0].copy(); t["b"]=t.c13/(t.c13+t.c22); t["v1"]=t.c13+t.c22+t.c70+t.c14+t.c55+t.cout; t["t3"]=(t.c70+t.c14+t.c55+t.cout)/t.v1
t["zz"]=t.uf=="zz"; ratio=s.w.sum()/s.v1.sum()
def nat(yz): v=t.v1*ratio; return 100*(yz*v).sum()/v.sum()
b1n=100*t.c13.sum()/(t.c13+t.c22).sum(); print(f"base 1T nacional (Lula/(Lula+Flávio)) = {b1n:.2f}%")
p_unif=nat(np.clip(t.b+np.average(s.sw,weights=s.w),0,1))
p_reg=nat(np.clip(t.b+beta[0]+beta[1]*(t.b-0.5)+beta[2]*t.t3,0,1))
# swing por UF com encolhimento para a média nacional
g=s.groupby("uf").apply(lambda d: pd.Series({"sw":np.average(d.sw,weights=d.w),"n":len(d)})); k=20
nat_sw=np.average(s.sw,weights=s.w); g["shr"]=(g.sw*g.n+nat_sw*k)/(g.n+k)
p_uf=nat(np.clip(t.b+t.uf.map(g.shr).fillna(nat_sw),0,1))
# bootstrap por município (incerteza amostral; NÃO cobre a não-representatividade)
rng=np.random.default_rng(0); mids=s.munid.unique(); bs=[]
for _ in range(400):
    pick=rng.choice(mids,len(mids)); ss=pd.concat([s[s.munid==m] for m in pick]); bs.append(nat(np.clip(t.b+np.average(ss.sw,weights=ss.w),0,1)))
lo,hi=np.percentile(bs,[5,95])
print(f"PROJEÇÃO Lula (válidos): swing uniforme {p_unif:.2f}% [90% amostral {lo:.2f}–{hi:.2f}] | swing por perfil {p_reg:.2f}% | swing por UF (encolhido) {p_uf:.2f}%")
print("por UF (n, virada pp):", {u:(int(r.n),round(100*r.sw,1)) for u,r in g.iterrows()})
