import pandas as pd, zipfile, sys
D="data/raw/"; O="data/work/"
parts=[]
with zipfile.ZipFile(D+"votacao_secao_2022_BR.zip") as zf, zf.open("votacao_secao_2022_BR.csv") as f:
    for ch in pd.read_csv(f,sep=";",encoding="latin1",usecols=["NR_TURNO","SG_UF","CD_MUNICIPIO","NR_ZONA","NR_LOCAL_VOTACAO","NR_VOTAVEL","QT_VOTOS"],dtype=str,chunksize=2_000_000):
        ch["QT_VOTOS"]=ch.QT_VOTOS.astype(int); ch["c"]=ch.NR_VOTAVEL.where(ch.NR_VOTAVEL.isin(["13","22","15","12","95","96"]),"out")
        parts.append(ch.groupby(["NR_TURNO","SG_UF","CD_MUNICIPIO","NR_ZONA","NR_LOCAL_VOTACAO","c"]).QT_VOTOS.sum())
v=pd.concat(parts).groupby(level=[0,1,2,3,4,5]).sum().unstack("c",fill_value=0).reset_index()
v.columns=["turno","uf","mun","zona","local"]+list(v.columns[5:]); 
for c in ["mun","zona","local","turno"]: v[c]=v[c].astype(int)
v.to_parquet(O+"votos_local.parquet"); print(v.shape)
with zipfile.ZipFile(D+"eleitorado_local_votacao_2022.zip") as zf, zf.open("eleitorado_local_votacao_2022.csv") as f:
    l=pd.read_csv(f,sep=";",encoding="latin1",usecols=["NR_TURNO","SG_UF","CD_MUNICIPIO","NM_MUNICIPIO","NR_ZONA","NR_LOCAL_VOTACAO","NM_BAIRRO","NR_LATITUDE","NR_LONGITUDE","QT_ELEITOR_SECAO"])
l=l[l.NR_TURNO==2]
loc=l.groupby(["SG_UF","CD_MUNICIPIO","NR_ZONA","NR_LOCAL_VOTACAO"]).agg(nm_mun=("NM_MUNICIPIO","first"),bairro=("NM_BAIRRO","first"),lat=("NR_LATITUDE","first"),lon=("NR_LONGITUDE","first"),eleitores=("QT_ELEITOR_SECAO","sum")).reset_index()
loc.columns=["uf","mun","zona","local","nm_mun","bairro","lat","lon","eleitores"]; loc.to_parquet(O+"locais.parquet"); print(loc.shape, loc.lat.describe())
