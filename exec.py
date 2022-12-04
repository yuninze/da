f=getdata(5)

# integrity check
import random
pp(f,vars=f[random.choices(f.columns,k=3)]),
plt.show()

# probplot
scipy.stats.probplot(f.ng.dropna(),dist="norm",plot=plt),
plt.show()

# rolling std unsubstantial
roll_pct(f,"si")

# examplar 0
freq       ="2d"
f_cols     =["hs","ic","cb","ys"]
f_cols_name=["HSI","JC","CBYS","LSYS"]
f0=f[f_cols]
f1=f0.resample(freq).mean().dropna()
f2=f1[["hs","ic","cb"]].apply(lambda q:scipy.stats.zscore(np.log(q)))
f2["ys"]=scipy.stats.zscore(f1["ys"])
f3=f2.loc["2007":"2010"].set_axis(f_cols_name,axis=1)
f3.plot(xlabel=f"{freq}",ylabel=""),
plt.show()

# aft:: pp.428
from statsmodels.tsa.stattools import adfuller as adf_
def adf(f)->dict:
    f=f.dropna()
    enog=adf(f)
    if enog[1]>.05:
        print("adf failed")
    return enog

# low-order uv
import scipy.stats
slope,intercept,rv,pv,stderr=scipy.stats.linregress(x0,y0)
y1=[(slope*x)+intercept for x in x1]
plt.scatter(x0.values,y0.values)
plt.plot(x0,y1)

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
