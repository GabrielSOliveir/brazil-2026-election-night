import numpy as np, pandas as pd, xgboost as xgb
F=lambda d: pd.DataFrame({"l":d.c13/d.validos1,"b":d.c22/d.validos1,"t":d.c15/d.validos1,"c":d.c12/d.validos1,
   "o":d.c_out/d.validos1,"ab":d.abst1/d.a1,"bn":(d.brancos1+d.nulos1)/d.comp1,"la":np.log(d.a1),"uf":d.uf.astype("category").cat.codes})
class Model:
    name="base_xgb_nivel"
    def fit(self,R): self.m=xgb.XGBRegressor(n_estimators=300,max_depth=5,learning_rate=0.05,subsample=0.8,n_jobs=1,verbosity=0).fit(F(R),R.y)
    def predict(self,U): return pd.DataFrame({"y":self.m.predict(F(U))},index=U.index)
