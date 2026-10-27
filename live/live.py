"""Runner ao vivo — 2º turno 2026 (Lula x Flávio). Entrada: CSV do feed do TSE por zona (mesmo formato do 1T).
Uso: python3 -I live/live.py <snapshot_t2.csv> [--hora HH:MM]
Formato genérico de entrada (any system that emits per-zone counts can feed it): uf,mun,zona,ts,st,te,comp,abst,vb,vn,vv,c13,c22,... (ts/st = seções totais/apuradas)."""
import sys, os, json, pickle, hashlib, argparse, importlib.util
import numpy as np, pandas as pd
sys.path.insert(0,"harness"); sys.path.insert(0,"harness/models")
from data import UF2REG
T1="data/derived/zones_2026_round1.csv"; ST="data/work/live_state.pkl"; LOG="data/work/live_log.csv"
def load_mod(p,name):
    s=importlib.util.spec_from_file_location(name,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def base():
    t=pd.read_csv(T1,dtype={"mun":str,"zona":str})
    z=pd.DataFrame({"uf":t.uf.str.upper(),"mun":t.mun.astype(int),"zona":t.zona.astype(int),
        "a1":t.te,"comp1":t.comp,"abst1":t.abst,"brancos1":t.vb,"nulos1":t.vn,"validos1":t.vv,
        "c13":t.c13,"c22":t.c22,"c15":t.c70,"c12":t.c14,"c_out":t.c55+t.cout})
    z["munid"]=z.uf+"-"+z.mun.astype(str); z["regiao"]=z.uf.map(UF2REG).fillna("EX"); return z[z.validos1>0]
def main():
    a=argparse.ArgumentParser(); a.add_argument("snap"); a.add_argument("--hora",default=""); x=a.parse_args()
    s=pd.read_csv(x.snap,dtype={"mun":str,"zona":str})
    s=pd.DataFrame({"uf":s.uf.str.upper(),"mun":s.mun.astype(int),"zona":s.zona.astype(int),"ts":s.ts,"st":s.st,"te2":s.te,"L":s.c13,"B":s.c22})
    s=s.drop_duplicates(["uf","mun","zona"])
    z=base().merge(s,on=["uf","mun","zona"],how="inner"); z["aptos2"]=z.te2.where(z.te2>0,z.a1)
    z["frac"]=np.where(z.ts>0,z.st/z.ts.clip(lower=1),0.0).clip(0,1)
    # trava de consistência: votos parciais sem seção ou seção sem voto
    z.loc[(z.L+z.B)<=0,"frac"]=0.0
    pct_sec=100*s.st.sum()/max(s.ts.sum(),1)
    raw=100*s.L.sum()/max((s.L+s.B).sum(),1)
    h=hashlib.md5(pd.util.hash_pandas_object(z[["uf","mun","zona","st","L","B"]],index=False).values.tobytes()).hexdigest()
    st=pickle.load(open(ST,"rb")) if os.path.exists(ST) else {}
    if st.get("hash")==h:
        print("LEITURA REPETIDA (feed não mudou) — mantendo a projeção anterior:"); print(json.dumps(st["last"],ensure_ascii=False)); return
    out=dict(hora=x.hora,pct_secoes=round(pct_sec,2),zonas_completas=int((z.frac>=1).sum()),zonas_com_dados=int((z.frac>0).sum()),bruto_lula=round(raw,3))
    vis=z[z.frac>0].copy(); R=vis.assign(L2=vis.L,B2=vis.B); R["v2"]=R.L2+R.B2; R=R[R.v2>0]; R["y"]=R.L2/R.v2
    Uf=z[z.frac<1].copy(); comp=z[z.frac>=1]
    pre=[c for c in Uf.columns if c not in ("L","B","ts","st","te2")]
    if len(R)>=30:
        M=load_mod("harness/models/r3_domingo.py","r3m")
        if "mod" in st: M._HIST.update(st["mod"].get("hist",{})); M._WARM.update(st["mod"].get("warm",{}))
        U=Uf[pre].assign(L2p=Uf.L,B2p=Uf.B)
        m=M.Model(); m.fit(R[pre+["L2","B2","v2","y"]].reset_index(drop=True).assign(ordem=np.arange(len(R)))); p=m.predict(U)
        yh=np.clip(np.asarray(p["y"],float),0,1); vh=np.asarray(p["v2"],float) if "v2" in p else U.validos1.values*R.v2.sum()/(R.validos1*R.frac).sum()
        Lnat=(comp.L.sum()+(yh*vh).sum())/((comp.L+comp.B).sum()+vh.sum())
        lo,hi=m.national_interval(R,U,p); pub=bool(m.publish_ok(R,U))
        part=R[R.frac<1].v2.sum()/max(R.v2.sum(),1); out["fracao_votos_de_parciais"]=round(float(part),2)
        if part>0.3:  # regra da auditoria: meia-largura mínima de 0,5pp enquanto >30% dos votos vierem de zonas parciais
            c=(lo+hi)/2; hw=max((hi-lo)/2,0.005); lo,hi=c-hw,c+hw
        out.update(lula=round(100*Lnat,2),flavio=round(100-100*Lnat,2),ic90=[round(100*lo,2),round(100*hi,2)],publica=pub)
        # por região (apurado + previsto)
        g=pd.DataFrame({"reg":np.r_[comp.regiao.values,U.regiao.values],"L":np.r_[comp.L.values,yh*vh],"V":np.r_[(comp.L+comp.B).values,vh]}).groupby("reg").sum()
        out["regioes_lula"]={k:round(100*v,1) for k,v in (g.L/g.V).items()}
        st["mod"]={"hist":dict(M._HIST),"warm":dict(M._WARM)}
        # checagem cruzada: modelo só-com-zonas-completas (r2_bayes_final_dint)
        Rc=R[R.frac>=1]
        if len(Rc)>=30:
            M2=load_mod("harness/models/r2_bayes_final_dint.py","r2m"); m2=M2.Model(); m2.fit(Rc.drop(columns=["frac"]).reset_index(drop=True).assign(ordem=np.arange(len(Rc))))
            U2=Uf[[c for c in pre if c!="frac"]]; p2=m2.predict(U2); y2=np.clip(np.asarray(p2["y"],float),0,1)
            v2=np.asarray(p2["v2"],float) if "v2" in p2 else U2.validos1.values*Rc.v2.sum()/Rc.validos1.sum()
            L2n=(comp.L.sum()+(y2*v2).sum())/((comp.L+comp.B).sum()+v2.sum()); out["checagem_so_completas"]=round(100*L2n,2)
            if hasattr(m2,"national_interval"):
                l2,h2=m2.national_interval(Rc.drop(columns=["frac"]),U2,p2); out["checagem_ic90"]=[round(100*l2,2),round(100*h2,2)]
                out["ics_se_sobrepoem"]=bool(max(lo,l2)<=min(hi,h2))
        out["pode_publicar"]=bool(pub and out.get("ics_se_sobrepoem",True))
        out["vencedor_definido"]=bool(out["pode_publicar"] and (lo>0.5 or hi<0.5))
    else: out["obs"]="poucas zonas com dados; sem projeção"
    st["hash"]=h; st["last"]=out; pickle.dump(st,open(ST,"wb"))
    pd.DataFrame([{k:(json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v) for k,v in out.items()}]).to_csv(LOG,mode="a",header=not os.path.exists(LOG),index=False)
    print(json.dumps(out,ensure_ascii=False))
if __name__=="__main__": main()
