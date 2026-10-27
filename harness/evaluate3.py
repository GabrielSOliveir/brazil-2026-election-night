"""Avaliador v3 (rodada 3): apuração PARCIAL por seções, como o TSE divulga ao vivo. Ver partial.py (calibrado na curva oficial de 2022).
  python3 -I harness/evaluate3.py <modelo.py> --ano 2018|2022 [--legado completas|parciais] --round N --agent nome
INTERFACE v3 (nova):
  class Model:
    v3 = True
    def fit(self, R):  R = zonas com alguma seção apurada (frac>0). L2,B2,v2,y são PARCIAIS (só o que já chegou); 'frac' = fração apurada da zona (1 = completa).
    def predict(self, U): U = zonas ainda incompletas (frac<1): colunas pré-eleição + 'frac' + 'L2p','B2p' (parciais observados; 0 se frac=0).
        -> DataFrame (índice de U): 'y' = fatia FINAL prevista da zona, 'v2' = válidos FINAIS previstos (opcional).
    def national_interval(self,R,U,pred): -> (lo,hi) 90%   (opcional)
    def publish_ok(self,R,U): -> bool  trava de publicação (opcional; padrão True)
  Modelos antigos (sem v3=True) rodam com --legado: 'completas' = só zonas 100% apuradas entram em R (ignora parciais);
  'parciais' = trata parciais como se fossem completas (ingênuo).
Nacional = soma real das zonas completas + previsão do modelo para todas as incompletas.
Cenários: dir (seções pró-direita chegam antes dentro da zona, como 2022; 3 sementes) | inv (o oposto) | neutro (sem viés dentro da zona).
Em 2018 usa os horários de 2022 por zona (ordem realista). Checkpoints = % de seções apuradas no TSE: 1,2,5,10,20,40,70.
"""
import sys, time, importlib.util, argparse, json, os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.path.insert(0,"harness")
from data import load, load_2018, pre_view
from partial import simulate
CAL=json.load(open("harness/partial_calib.json"))
def zones(ano):
    if ano==2022: return load()
    z=load_2018(); t=load()[["uf","mun","zona","t"]]; z=z.merge(t,on=["uf","mun","zona"],how="left")
    late=pd.Timestamp("2022-10-30 20:30:00"); z["t"]=z.t.fillna(late); return z
def run(path,ano,legado,rnd,agent):
    spec=importlib.util.spec_from_file_location("m",path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    z=zones(ano); FINAL=z.L2.sum()/z.v2.sum(); res=[]; t0=time.time(); isv3=getattr(mod.Model,"v3",False)
    if not isv3 and legado is None: raise SystemExit("modelo sem v3=True: use --legado completas|parciais")
    scen=[("dir",s,CAL["beta"]) for s in (0,1,2)]+[("inv",0,-CAL["beta"]),("neutro",0,0.0)]
    for tipo,seed,beta in scen:
        for p,tau in zip(CAL["targets"],CAL["taus"]):
            f,Lp,Bp=simulate(z,tau,CAL["c"],CAL["k"],beta,CAL["s_sec"],1000*ano+seed,CAL["m"])
            zz=z.copy(); zz["frac"]=f; zz["L2p"]=Lp; zz["B2p"]=Bp
            comp=f>=1; vis=f>0
            Robs=zz[vis].copy(); Robs["L2"]=Robs.L2p; Robs["B2"]=Robs.B2p; Robs["v2"]=Robs.L2+Robs.B2; Robs=Robs[Robs.v2>0]
            Robs["y"]=Robs.L2/Robs.v2; Robs=Robs.drop(columns=["L2p","B2p"])
            if isv3: R=Robs; Uidx=~comp
            elif legado=="completas": R=Robs[Robs.frac>=1]; Uidx=~comp
            else: R=Robs; Uidx=~vis
            Ufull=zz[Uidx]; U=pre_view(Ufull).copy()
            if isv3: U["frac"]=Ufull.frac; U["L2p"]=Ufull.L2p; U["B2p"]=Ufull.B2p
            elif "frac" in U: U=U.drop(columns=["frac","L2p","B2p"],errors="ignore")
            Rin=R.drop(columns=["frac"]) if not isv3 else R
            m=mod.Model(); m.fit(Rin.reset_index(drop=True).assign(ordem=np.arange(len(Rin)))); pr=m.predict(U)
            yh=np.clip(np.asarray(pr["y"],float),0,1)
            if "v2" in pr: vh=np.asarray(pr["v2"],float)
            else:
                ratio=R.v2.sum()/(R.validos1*(R.frac if "frac" in R else 1)).sum(); vh=U.validos1.values*ratio
            # zonas visíveis mas não em U (só no modo 'parciais'): conta parcial como final
            known=zz[~Uidx]
            if not isv3 and legado=="parciais": Lk=known.L2p.sum(); Vk=(known.L2p+known.B2p).sum()
            else: Lk=known.L2.sum(); Vk=known.v2.sum()
            Lnat=(Lk+(yh*vh).sum())/(Vk+vh.sum())
            pub=bool(m.publish_ok(R,U)) if hasattr(m,"publish_ok") else True
            row=dict(cen=f"{tipo}{seed}",tipo=tipo,pct_tse=p,completas=int(comp.sum()),erro_nac=100*(Lnat-FINAL),
                bruto=100*(Robs.L2.sum()/Robs.v2.sum()-FINAL),mae_zona=100*np.mean(np.abs(yh-Ufull.y.values)),publica=int(pub))
            if hasattr(m,"national_interval"):
                lo,hi=m.national_interval(R,U,pr); row.update(ic_larg=100*(hi-lo),ic_cobre=int(lo<=FINAL<=hi))
            res.append(row)
    df=pd.DataFrame(res); name=getattr(mod.Model,"name",os.path.basename(path))+("" if isv3 else f"[legado:{legado}]")
    pubd=df[df.publica==1]
    s=dict(modelo=name,agente=agent,rodada=rnd,ano=ano,seg=round(time.time()-t0),
        nac_rmse=float(np.sqrt((df.erro_nac**2).mean())),nac_rmse_ate5=float(np.sqrt((df[df.pct_tse<=5].erro_nac**2).mean())),
        nac_max=float(df.erro_nac.abs().max()),**{f"rmse_{t}":float(np.sqrt((g.erro_nac**2).mean())) for t,g in df.groupby("tipo")},
        mae_zona=float(df.mae_zona.mean()),frac_publicada=float(df.publica.mean()),
        nac_max_publicado=float(pubd.erro_nac.abs().max()) if len(pubd) else None)
    if "ic_cobre" in df: s.update(ic_cob=float(df.ic_cobre.mean()),ic_cob_publicado=float(pubd.ic_cobre.mean()) if len(pubd) else None,ic_larg=float(df.ic_larg.mean()))
    print(df.round(3).to_string(index=False)); print("RESUMO",json.dumps({k:(round(v,3) if isinstance(v,float) else v) for k,v in s.items()},ensure_ascii=False))
    with open("harness/leaderboard3.jsonl","a") as fh: fh.write(json.dumps(s,ensure_ascii=False)+"\n")
    os.makedirs("harness/results3",exist_ok=True); df.to_csv(f"harness/results3/{name.replace('[','_').replace(']','').replace(':','_')}_{ano}.csv",index=False)
if __name__=="__main__":
    a=argparse.ArgumentParser(); a.add_argument("path"); a.add_argument("--ano",type=int,required=True); a.add_argument("--legado",choices=["completas","parciais"])
    a.add_argument("--round",type=int,default=3); a.add_argument("--agent",default="chefe"); x=a.parse_args(); run(x.path,x.ano,x.legado,x.round,x.agent)
