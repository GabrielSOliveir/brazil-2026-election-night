"""r1_freq_re — swing frequentista com efeitos aleatórios aninhados (região > UF > município), BLUP empírico.
y = b1 + f(b1) + u_reg + u_uf + u_mun + e,  b1 = c13/(c13+c22) do 1T,  f(b1) = β0 + β1(b1-.5) + β2(b1-.5)^2 (WLS).
Componentes de variância: momentos nos resíduos das zonas apuradas, misturados com priori (pseudo-observações).
v2: razão v2/validos1 por UF encolhida para a nacional. Intervalo 90%: variância analítica (β cluster-robusto por
município + variância a posteriori dos efeitos de UF/região/mun das partes não apuradas) com t de cauda pesada."""
import numpy as np, pandas as pd
from scipy import stats
_C=None
def _ctx():
    global _C
    if _C is None:
        import sys; sys.path.insert(0,"harness"); from data import load, pre_view
        z=pre_view(load()); g=z.groupby("uf")[["c15","c12","c_out","validos1","brancos1","nulos1","comp1"]].sum()
        _C=pd.DataFrame({"t":g.c15/g.validos1,"c":g.c12/g.validos1}); _C=_C-_C.mean()
    return _C
def feats(d,kind):
    v=d.validos1.values.astype(float); b1=(d.c13/(d.c13+d.c22)).values; x=b1-.5; one=np.ones(len(d))
    if kind=="unif": F=np.c_[one]
    elif kind=="b1q": F=np.c_[one,x,x**2]
    elif kind=="b1c": F=np.c_[one,x,x**2,x**3]
    elif kind=="ctx":  # contexto da UF (1T de TODAS as zonas, conhecido antes): fatia Tebet/Ciro/outros/bn na UF
        c=_ctx().reindex(d.uf.values).values; F=np.c_[one,x,x**2,c]
    elif kind=="full":
        F=np.c_[one,x,x**2,d.c15/v,d.c12/v,d.c_out/v,(d.brancos1+d.nulos1)/d.comp1,d.abst1/d.a1]
    return np.asarray(F,float),b1
class Model:
    name="r1_freq_re"
    kind="b1q"; k=9000.; levels=("regiao","uf","munid")
    prior_sd=dict(regiao=0.4,uf=0.5,munid=1.0,e=0.6)   # pp; usados como priori p/ momentos
    prior_n=dict(regiao=3,uf=8,munid=50,e=50)          # força da priori (pseudo-graus de liberdade)
    live_vc=True; v2mode="uf"; v2_k=30.; ic_df=8; ic_infl=0.7; ic_tau=0.1
    def _w(self,d): return 1/(1+self.k/d.validos1.values.astype(float))
    # ---------- componentes de variância por momentos ----------
    def _vc(self,R,r,w):
        pr={k:v**2 for k,v in self.prior_sd.items()}
        if not self.live_vc or len(R)<30: return pr
        out={}
        d=pd.DataFrame({"r":r,"w":w,"mun":R.munid.values,"uf":R.uf.values,"reg":R.regiao.values})
        n=d.groupby("mun").r.transform("size"); dm=d[n>1]
        if len(dm)>0:
            ss=((dm.r-dm.groupby("mun").r.transform("mean"))**2).sum(); df=len(dm)-dm.mun.nunique()
        else: ss,df=0,0
        out["e"]=(ss+self.prior_n["e"]*pr["e"])/(df+self.prior_n["e"])
        # mun-means dentro de UF: var = vm + ve/n (média)
        g=d.groupby("mun").agg(r=("r","mean"),n=("r","size"),uf=("uf","first"),reg=("reg","first"))
        cnt=g.groupby("uf").r.transform("size"); gg=g[cnt>1]
        if len(gg)>0:
            ss=((gg.r-gg.groupby("uf").r.transform("mean"))**2).sum(); df=len(gg)-gg.uf.nunique()
            vm=max(ss/max(df,1)-out["e"]*np.mean(1/gg.n),0.05)
        else: vm,df=pr["munid"],0
        out["munid"]=(df*vm+self.prior_n["munid"]*pr["munid"])/(df+self.prior_n["munid"])
        # UF means dentro de região
        u=g.groupby("uf").agg(r=("r","mean"),n=("r","size"),reg=("reg","first"))
        cnt=u.groupby("reg").r.transform("size"); uu=u[(cnt>1)&(u.n>=5)]
        if len(uu)>1:
            ss=((uu.r-uu.groupby("reg").r.transform("mean"))**2).sum(); df=len(uu)-uu.reg.nunique()
            vu=max(ss/max(df,1)-np.mean((out["munid"]+out["e"])/uu.n),0.0)
        else: vu,df=pr["uf"],0
        out["uf"]=(df*vu+self.prior_n["uf"]*pr["uf"])/(df+self.prior_n["uf"])
        out["regiao"]=pr["regiao"]  # poucas regiões: fica na priori
        return out
    # ---------- BLUP hierárquico (aproximação top-down) ----------
    def _blup(self,R,r,w):
        vc=self.vc; d=pd.DataFrame({"r":r,"w":w,"regiao":R.regiao.values,"uf":R.uf.values,"munid":R.munid.values})
        self.eff={}; self.pvar={}; resid=d.r.values.copy()
        # nível do município é observado com ruído e; níveis acima com ruído (soma dos níveis abaixo)
        below={"regiao":vc["uf"]+vc["munid"],"uf":vc["munid"],"munid":0.0}
        for lev in self.levels:
            d["_r"]=resid
            # agregue por município primeiro p/ ruído correto
            mm=d.groupby("munid").agg(e=("_r","mean"),n=("_r","size"),key=(lev,"first"))
            mm["vk"]=below[lev]+vc["e"]/mm.n
            if lev=="munid": mm["vk"]=vc["e"]/mm.n
            a=pd.DataFrame({"num":mm.e/mm.vk,"prec":1/mm.vk,"key":mm.key}).groupby("key").sum()
            V=vc[lev]; est=V*a.num/(1+V*a.prec); pv=V/(1+V*a.prec)
            self.eff[lev]=est.to_dict(); self.pvar[lev]=pv.to_dict()
            resid=resid-d[lev].map(self.eff[lev]).values
    def fit(self,R):
        F,b1=feats(R,self.kind); y=100*(R.y.values-b1); w=self._w(R)
        self.used=F.shape[1] if len(R)>=8*F.shape[1] else 1; F=F[:,:self.used]
        sw=np.sqrt(w); self.beta=np.linalg.lstsq(F*sw[:,None],y*sw,rcond=None)[0]; r=y-F@self.beta
        # cov cluster-robusta (município) de beta
        A=np.linalg.pinv((F*w[:,None]).T@F); sc=pd.DataFrame(F*(w*r)[:,None]).groupby(R.munid.values).sum().values
        self.Vb=A@(sc.T@sc)@A
        self.vc=self._vc(R,r,w); self._blup(R,r,w)
        # v2
        self.rat=R.v2.sum()/R.validos1.sum(); g=R.groupby("uf")[["v2","validos1"]].sum(); n=R.groupby("uf").size()
        self.rat_uf=((g.v2+self.v2_k*self.rat*g.validos1/n)/(g.validos1+self.v2_k*g.validos1/n)).to_dict()
        self.Rv=R
    def predict(self,U):
        F,b1=feats(U,self.kind); F=F[:,:self.used]; s=F@self.beta
        for lev in self.levels: s=s+U[lev].map(self.eff[lev]).fillna(0).values
        v=U.validos1.values*(U.uf.map(self.rat_uf).fillna(self.rat).values if self.v2mode=="uf" else self.rat)
        return pd.DataFrame({"y":np.clip(b1+s/100,0.001,0.999),"v2":v},index=U.index)
    def national_interval(self,R,U,pred):
        vh=pred.v2.values; Vtot=R.v2.sum()+vh.sum(); wz=vh/Vtot  # peso de cada zona não apurada no total
        F,_=feats(U,self.kind); F=F[:,:self.used]; g=wz@F; var=g@self.Vb@g
        for lev in self.levels:
            V=self.vc[lev]; pv=U[lev].map(self.pvar[lev]).fillna(V).values
            agg=pd.Series(wz).groupby(U[lev].values).sum(); pvk=pd.Series(pv).groupby(U[lev].values).first()
            var+=float((agg**2*pvk).sum())
        var+=float((wz**2).sum()*self.vc["e"])+(self.ic_tau*wz.sum())**2
        sd=np.sqrt(var)/100*self.ic_infl; L=(R.L2.sum()+(pred.y.values*vh).sum())/Vtot
        q=stats.t.ppf(0.95,self.ic_df); return L-q*sd, L+q*sd
