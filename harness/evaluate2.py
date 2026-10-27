"""Avaliador v2 (a partir da rodada 2). Mesma interface Model da v1 (ver evaluate.py).
  python3 -I harness/evaluate2.py <modelo.py> --ano 2018|2022 [--quick] --round N --agent nome
REGRA DE OURO: ajuste/escolha de hiperparâmetros SÓ olhando --ano 2018 (desenvolvimento).
2022 é o TESTE FINAL; rode 2022 apenas com a versão congelada (o placar registra quantas vezes cada um rodou 2022).
Cenários: real (só 2022) | como (zonas pró-direita no 1T antes, 3 sementes) | aleatorio (3 sementes)
          | geo (cada UF com atraso aleatório + ruído, 3 sementes) | pior (zonas com mais virada pró-direita antes; estresse)
Métrica principal: nac_rmse = RMSE do erro nacional (pp) em todos os checkpoints dos cenários realistas (real, como, aleatorio, geo).
"""
import sys, time, importlib.util, argparse, json, os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0,"harness"); from data import load, load_2018, pre_view
CK=[0.01,0.02,0.05,0.10,0.20,0.40,0.70]; CKQ=[0.01,0.05,0.20,0.60]
def orders(z,ano,quick):
    out={}; seeds=[0] if quick else [0,1,2]
    if ano==2022: out["real"]=z.sort_values("t").index
    b1=z.c22/z.validos1
    for s in seeds:
        rng=np.random.default_rng(100+s)
        out[f"como{s}"]=z.assign(o=-b1+rng.normal(0,0.5,len(z))).sort_values("o").index
        out[f"aleatorio{s}"]=z.sample(frac=1,random_state=200+s).index
        ufs=z.uf.unique(); delay=dict(zip(ufs,rng.uniform(0,1,len(ufs))))
        out[f"geo{s}"]=z.assign(o=z.uf.map(delay)+rng.uniform(0,0.6,len(z))).sort_values("o").index
    if not quick:
        sw=z.y-z.c13/(z.c13+z.c22); rng=np.random.default_rng(9)
        out["pior"]=z.assign(o=sw+rng.normal(0,0.01,len(z))).sort_values("o").index
    return out
def run(path,ano,quick,rnd,agent):
    spec=importlib.util.spec_from_file_location("m",path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    z=load() if ano==2022 else load_2018()
    if "t" not in z: z=z.assign(t=pd.NaT)
    FINAL=z.L2.sum()/z.v2.sum(); res=[]; t0=time.time()
    for sc,idx in orders(z,ano,quick).items():
        zz=z.loc[idx].reset_index(drop=True); zz["ordem"]=np.arange(len(zz)); cum=zz.v2.cumsum()/zz.v2.sum()
        for q in (CKQ if quick else CK):
            n=int((cum<=q).sum()); R=zz.iloc[:n].copy(); Uf=zz.iloc[n:]; U=pre_view(Uf).copy()
            m=mod.Model(); m.fit(R); p=m.predict(U)
            yh=np.clip(np.asarray(p["y"],float),0,1)
            vh=np.asarray(p["v2"],float) if "v2" in p else U.validos1.values*R.v2.sum()/R.validos1.sum()
            Lnat=(R.L2.sum()+(yh*vh).sum())/(R.v2.sum()+vh.sum())
            g=pd.DataFrame({"uf":zz.uf,"L":np.r_[R.L2.values,yh*vh],"V":np.r_[R.v2.values,vh]}).groupby("uf").sum()
            gt=zz.groupby("uf")[["L2","v2"]].sum()
            row=dict(cenario=sc,tipo=''.join(ch for ch in sc if not ch.isdigit()),pct=round(100*q),zonas=n,erro_nac=100*(Lnat-FINAL),
                bruto=100*(R.L2.sum()/R.v2.sum()-FINAL),mae_zona=100*np.mean(np.abs(yh-Uf.y.values)),
                mae_uf=100*np.mean(np.abs(g.L/g.V-gt.L2/gt.v2)))
            if hasattr(m,"national_interval"):
                lo,hi=m.national_interval(R,U,p); row.update(ic_larg=100*(hi-lo),ic_cobre=int(lo<=FINAL<=hi))
            res.append(row)
    df=pd.DataFrame(res); real=df[df.tipo!="pior"]; name=getattr(mod.Model,"name",os.path.basename(path))
    s=dict(modelo=name,agente=agent,rodada=rnd,ano=ano,modo="quick" if quick else "full",seg=round(time.time()-t0),
        nac_rmse=float(np.sqrt((real.erro_nac**2).mean())),
        nac_ini_rmse=float(np.sqrt((real[real.pct<=5].erro_nac**2).mean())),
        nac_max=float(real.erro_nac.abs().max()),
        **{f"rmse_{t}":float(np.sqrt((g.erro_nac**2).mean())) for t,g in real.groupby("tipo")},
        mae_zona=float(real.mae_zona.mean()),mae_uf=float(real.mae_uf.mean()))
    if not quick: s["nac_pior"]=float(df[df.tipo=="pior"].erro_nac.abs().mean())
    if "ic_cobre" in df: s["ic_cob"]=float(real.ic_cobre.mean()); s["ic_cob_pior"]=float(df[df.tipo=="pior"].ic_cobre.mean()) if not quick else None; s["ic_larg"]=float(real.ic_larg.mean())
    print(df.round(3).to_string(index=False)); print("RESUMO",json.dumps({k:(round(v,3) if isinstance(v,float) else v) for k,v in s.items()},ensure_ascii=False))
    with open("harness/leaderboard2.jsonl","a") as f: f.write(json.dumps(s,ensure_ascii=False)+"\n")
    os.makedirs("harness/results2",exist_ok=True); df.to_csv(f"harness/results2/{name}_{ano}_{'q' if quick else 'f'}.csv",index=False)
if __name__=="__main__":
    a=argparse.ArgumentParser(); a.add_argument("path"); a.add_argument("--ano",type=int,required=True); a.add_argument("--quick",action="store_true")
    a.add_argument("--round",type=int,default=0); a.add_argument("--agent",default="chefe"); x=a.parse_args(); run(x.path,x.ano,x.quick,x.round,x.agent)
