import numpy as np, pandas as pd
class Model:
    name="base_swing_uniforme"
    def fit(self,R):
        b=R.c13/(R.c13+R.c22); self.s=np.average(R.y-b,weights=R.v2)
    def predict(self,U):
        return pd.DataFrame({"y":U.c13/(U.c13+U.c22)+self.s},index=U.index)
