# from da import *
f=getdata(10).interpolate("linear",limit=2)

# stdev or diff
data=f.ng.dropna().rolling(5).mean().pct_change().dropna()
data_std=np.std(data)
fg,ax=plt.subplots()
ax.plot(data.loc["2022-01"])
ax.axhline(y=data_std,color="red")
ax.axhline(y=-data_std,color="red")
ax.axhline(y=data_std*2,color="red")
ax.axhline(y=-data_std*2,color="red")
plt.show()
cache=scipy.stats.yeojohnson(f.si.dropna().diff().dropna())[0]
plt.hist(cache),plt.show()

# rr, pca::
residual plots

# gcts
vars=["fert","cl"]
gct_tgts=f[vars].resample("30d").mean().diff().dropna()
gct_rslt=gct(gct_tgts,np.arange(1,11))
vars=["ic","cb"]
gct_tgts=f[vars].resample("7d").mean().diff().dropna()
gct_rslt=gct(gct_tgts,np.arange(1,11))

# high-order
import scipy.stats
slope,intercept,rv,pv,stderr=scipy.stats.linregress(x0,y0)
y1=[(slope*x)+intercept for x in x1]
plt.scatter(x0.values,y0.values)
plt.plot(x0,y1)

hoe=np.poly1d(np.polyfit(x,y,2))
hoe.c
hoe.variable
from sklearn.metrics import r2_score
r2_score(y0,hoe(x0))

# low-order
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











