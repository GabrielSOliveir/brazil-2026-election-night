"""Carrega a tabela por zona (1º e 2º turno 2022). Uma linha = (uf, mun, zona).
Colunas conhecidas ANTES da apuração do 2T (usáveis para prever qualquer zona):
  a1 (aptos 1T), comp1, abst1, brancos1, nulos1, validos1, c13,c22,c15,c12,c_out (votos 1T por candidato),
  aptos2 (aptos 2T, conhecido antes), uf, mun, munid, zona, nm_mun, regiao
Colunas do 2T (só existem para zonas JÁ APURADAS): L2, B2, v2 (=L2+B2), comp2, brancos2, nulos2, abst2
Alvo principal: y = L2/v2 (fatia do Lula nos votos válidos).
"""
import pandas as pd, numpy as np, os
D="data/tse/2022/"
CACHE="harness/cache/zonas2022.parquet"
REG={"N":"AC AM AP PA RO RR TO","NE":"AL BA CE MA PB PE PI RN SE","CO":"DF GO MS MT","SE":"ES MG RJ SP","S":"PR RS SC","EX":"ZZ"}
UF2REG={u:r for r,s in REG.items() for u in s.split()}
PRE_COLS=["a1","comp1","abst1","brancos1","nulos1","validos1","c13","c22","c15","c12","c_out","aptos2"]
POST_COLS=["L2","B2","v2","comp2","brancos2","nulos2","abst2","y","t","ordem"]
def load():
    if os.path.exists(CACHE): return pd.read_parquet(CACHE)
    c=pd.read_csv(D+"cand.csv"); d=pd.read_csv(D+"det.csv"); k=["uf","mun","zona"]
    nm=c.drop_duplicates(k).set_index(k).nm_mun
    p=c.pivot_table(index=["turno"]+k,columns="cand",values="votos",aggfunc="sum",fill_value=0)
    t1=p.loc[1]; t2=p.loc[2]
    d1=d[d.turno==1].set_index(k); d2=d[d.turno==2].set_index(k)
    z=pd.DataFrame({"a1":d1.QT_APTOS,"comp1":d1.QT_COMPARECIMENTO,"abst1":d1.QT_ABSTENCOES,
        "brancos1":d1.QT_VOTOS_BRANCOS,"nulos1":d1.QT_TOTAL_VOTOS_NULOS,"validos1":d1.QT_TOTAL_VOTOS_VALIDOS})
    z=z.join(pd.DataFrame({"c13":t1[13],"c22":t1[22],"c15":t1[15],"c12":t1[12],
        "c_out":t1.drop(columns=[13,22,15,12]).sum(axis=1)}),how="inner")
    z=z.join(pd.DataFrame({"aptos2":d2.QT_APTOS,"comp2":d2.QT_COMPARECIMENTO,"abst2":d2.QT_ABSTENCOES,
        "brancos2":d2.QT_VOTOS_BRANCOS,"nulos2":d2.QT_TOTAL_VOTOS_NULOS,"ts":d2.ts}),how="inner")
    z=z.join(pd.DataFrame({"L2":t2[13],"B2":t2[22]}),how="inner").join(nm)
    z=z.reset_index(); z["v2"]=z.L2+z.B2; z=z[(z.v2>0)&(z.validos1>0)].copy()
    z["y"]=z.L2/z.v2; z["munid"]=z.uf+"-"+z.mun.astype(str); z["regiao"]=z.uf.map(UF2REG)
    z["t"]=pd.to_datetime(z.ts,format="%d/%m/%Y %H:%M:%S"); z=z.drop(columns="ts").reset_index(drop=True)
    z.to_parquet(CACHE); return z
def pre_view(U):
    """O que um modelo pode ver de uma zona NÃO apurada."""
    return U[[c for c in U.columns if c not in POST_COLS]]

def load_2018():
    """Mesma estrutura de load() para 2018 (Haddad x Bolsonaro), com nomes de colunas GENÉRICOS:
    c13 = PT (Haddad), c22 = Bolsonaro (nº 17 em 2018), c15 = Alckmin (45, centro, análogo grosseiro da Tebet),
    c12 = Ciro, c_out = demais. L2 = Haddad 2T, B2 = Bolsonaro 2T, y = L2/v2. Tudo completo (eleição passada).
    Junte com 2022 por ['uf','mun','zona'] (zonas mudaram um pouco: ~nem todas casam)."""
    c18="harness/cache/zonas2018.parquet"
    if os.path.exists(c18): return pd.read_parquet(c18)
    D18="data/tse/2018/"
    c=pd.read_csv(D18+"cand.csv"); d=pd.read_csv(D18+"det.csv"); k=["uf","mun","zona"]
    p=c.pivot_table(index=["turno"]+k,columns="cand",values="votos",aggfunc="sum",fill_value=0)
    t1=p.loc[1]; t2=p.loc[2]
    d1=d[d.turno==1].set_index(k); d2=d[d.turno==2].set_index(k)
    z=pd.DataFrame({"a1":d1.QT_APTOS,"comp1":d1.QT_COMPARECIMENTO,"abst1":d1.QT_ABSTENCOES,
        "brancos1":d1.QT_VOTOS_BRANCOS,"nulos1":d1.QT_TOTAL_VOTOS_NULOS,"validos1":d1.QT_TOTAL_VOTOS_VALIDOS})
    z=z.join(pd.DataFrame({"c13":t1[13],"c22":t1[17],"c15":t1[45],"c12":t1[12],
        "c_out":t1.drop(columns=[13,17,45,12]).sum(axis=1)}),how="inner")
    z=z.join(pd.DataFrame({"aptos2":d2.QT_APTOS,"comp2":d2.QT_COMPARECIMENTO,"abst2":d2.QT_ABSTENCOES,
        "brancos2":d2.QT_VOTOS_BRANCOS,"nulos2":d2.QT_TOTAL_VOTOS_NULOS}),how="inner")
    z=z.join(pd.DataFrame({"L2":t2[13],"B2":t2[17]}),how="inner").reset_index()
    z["v2"]=z.L2+z.B2; z=z[(z.v2>0)&(z.validos1>0)].copy(); z["y"]=z.L2/z.v2
    z["munid"]=z.uf+"-"+z.mun.astype(str); z["regiao"]=z.uf.map(UF2REG)
    z=z.reset_index(drop=True); z.to_parquet(c18); return z
