# yielding
f=getdata()

# integrity check
import random
pp(f,vars=f[random.choices(f.columns,k=3)]),
plt.show()

# probplot
scipy.stats.probplot(f.ng.dropna(),dist="norm",plot=plt),
plt.show()

# examplar


# aft:: pp.428
from statsmodels.tsa.stattools import adfuller as adf
adf(f.ng.dropna())

#high-order uv
hoe=np.poly1d(np.polyfit(x,y,2))
hoe.c
hoe.variable
from sklearn.metrics import r2_score
r2_score(y0,hoe(x0))

# low-order mv
from sklearn import linear_model
fa=fa.interpolate("linear",limit=7).dropna().bfill()[:-2]
fb=fa.apply(scipy.stats.zscore)
x0=fa[["fr","cb"]]
y0=fa["yt"]

loe=linear_model.LinearRegression()
loe.fit(x0,y0)
loe.coef_
loe.score(x0,y0)
loe.predict(x0)

# cc
vars=["fr","iy","hs","si"]
fa=f[vars].resample("10d").mean().dropna(thresh=int(len(vars)*0.8))
fb=impt(fa,vars)
cx_rslt=cx_(fb,vars,d=9)

# gcts
from statsmodels.tsa.stattools import grangercausalitytests as gct
vars=["fert","cl"]
fa=f.fert.resample("10d").mean().ffill().dropna()
fb=f.cl.resample("10d").mean().dropna()
fc=pd.concat([fa,fb],axis=1,how="outer")
gct_tgts=f[vars].resample("10d").mean().diff().dropna()
gct_rslt=gct(gct_tgts,np.arange(1,11))
