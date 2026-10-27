"""Avaliador comum. Uso:  python3 -I harness/evaluate.py models/<arquivo>.py [--quick] [--round N] [--agent nome]
O arquivo do modelo define:
    class Model:
        name = "nome_curto"
        def fit(self, R): ...            # R = zonas JÁ apuradas (todas as colunas, inclusive 2T)
        def predict(self, U): -> DataFrame com índice de U e colunas:
              'y'  (obrigatório) fatia prevista do Lula nos válidos do 2T
              'v2' (opcional) votos válidos (Lula+Bolsonaro) previstos; se faltar usa validos1 * razão observada
        def national_interval(self, R, U, pred): -> (lo, hi)  OPCIONAL, intervalo de 90% da fatia nacional do Lula (0-1)
    U chega SEM as colunas do 2T (ver data.pre_view). R chega ordenado pela chegada, com coluna 'ordem'.
Cenários de chegada (sem acesso ao resultado do 2T fora do que já chegou):
  real   : hora em que cada zona terminou de ser apurada em 2022
  como2022: zonas mais bolsonaristas no 1T chegam antes (+ruído), calibrado p/ imitar a curva real (Lula ~7pp atrás no início)
  aleatorio: ordem aleatória
  pior   : zonas que mais viraram para o Bolsonaro chegam antes (estresse; nenhum modelo vê isso)
Métricas (pontos percentuais): erro nacional por checkpoint, MAE por zona (não apuradas), MAE por UF, cobertura do intervalo.
"""
import sys, time, importlib.util, argparse, json, os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0,"harness"); from data import load, pre_view
CK_FULL=[0.01,0.02,0.05,0.10,0.20,0.40,0.60,0.80]; CK_QUICK=[0.01,0.05,0.20,0.60]
def orders(z,quick):
    rng=np.random.default_rng(0); out={}
    out["real"]=z.sort_values("t").index
    b1=z.c22/z.validos1
    out["como2022"]=(z.assign(o=-b1+rng.normal(0,0.5,len(z)))).sort_values("o").index
    if not quick:
        out["aleatorio"]=z.sample(frac=1,random_state=1).index
        sw=z.y-z.c13/(z.c13+z.c22)
        out["pior"]=(z.assign(o=sw+rng.normal(0,0.01,len(z)))).sort_values("o").index
    return out
def run(path,quick=False,rnd=0,agent="?"):
    spec=importlib.util.spec_from_file_location("m",path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    z=load(); FINAL=z.L2.sum()/z.v2.sum(); res=[]; t0=time.time()
    for sc,idx in orders(z,quick).items():
        zz=z.loc[idx].reset_index(drop=True); zz["ordem"]=np.arange(len(zz)); cum=zz.v2.cumsum()/zz.v2.sum()
        for q in (CK_QUICK if quick else CK_FULL):
            n=int((cum<=q).sum()); R=zz.iloc[:n].copy(); Ufull=zz.iloc[n:]; U=pre_view(Ufull).copy()
            m=mod.Model(); m.fit(R); p=m.predict(U)
            yh=np.clip(np.asarray(p["y"],float),0,1)
            if "v2" in p: vh=np.asarray(p["v2"],float)
            else: vh=U.validos1.values*R.v2.sum()/R.validos1.sum()
            Lnat=(R.L2.sum()+(yh*vh).sum())/(R.v2.sum()+vh.sum())
            g=pd.DataFrame({"uf":zz.uf,"L":np.r_[R.L2.values,yh*vh],"V":np.r_[R.v2.values,vh]}).groupby("uf").sum()
            gt=zz.groupby("uf")[["L2","v2"]].sum()
            row=dict(cenario=sc,pct=round(100*q),zonas_apuradas=n,erro_nacional=100*(Lnat-FINAL),
                bruto=100*(R.L2.sum()/R.v2.sum()-FINAL),
                mae_zona=100*np.mean(np.abs(yh-Ufull.y.values)),
                mae_uf=100*np.mean(np.abs(g.L/g.V-gt.L2/gt.v2)))
            if hasattr(m,"national_interval"):
                lo,hi=m.national_interval(R,U,p); row.update(ic_largura=100*(hi-lo),ic_cobre=int(lo<=FINAL<=hi))
            res.append(row)
    df=pd.DataFrame(res); name=getattr(mod.Model,"name",os.path.basename(path))
    s=dict(modelo=name,agente=agent,rodada=rnd,arquivo=path,modo="quick" if quick else "full",segundos=round(time.time()-t0),
        nac_real=df[df.cenario=="real"].erro_nacional.abs().mean(),
        nac_como2022=df[df.cenario=="como2022"].erro_nacional.abs().mean(),
        nac_inicio_max=df[df.pct<=5].query("cenario!='pior'").erro_nacional.abs().max(),
        mae_zona=df[df.cenario!="pior"].mae_zona.mean(), mae_uf=df[df.cenario!="pior"].mae_uf.mean())
    if not quick:
        s["nac_aleatorio"]=df[df.cenario=="aleatorio"].erro_nacional.abs().mean(); s["nac_pior"]=df[df.cenario=="pior"].erro_nacional.abs().mean()
    if "ic_cobre" in df: s["ic_cobertura"]=df.ic_cobre.mean(); s["ic_largura"]=df.ic_largura.mean()
    print(df.round(3).to_string(index=False)); print("RESUMO",json.dumps({k:(round(v,3) if isinstance(v,float) else v) for k,v in s.items()},ensure_ascii=False))
    lb="harness/leaderboard.csv"; pd.DataFrame([s]).to_csv(lb,mode="a",header=not os.path.exists(lb),index=False)
    os.makedirs("harness/results",exist_ok=True); df.to_csv(f"harness/results/{name}_{'q' if quick else 'f'}.csv",index=False)
if __name__=="__main__":
    a=argparse.ArgumentParser(); a.add_argument("path"); a.add_argument("--quick",action="store_true"); a.add_argument("--round",type=int,default=0); a.add_argument("--agent",default="chefe")
    x=a.parse_args(); run(x.path,x.quick,x.round,x.agent)
