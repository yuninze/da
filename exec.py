f=getdata(10)
f=f.interpolate("quadratic",limit=2)

fa=f[["hs","yt"]]

slope,intercept,rv,pv,stderr=scipy.stats.linregress(x0,y0)
y1=[(slope*x)+intercept for x in x1]
plt.scatter(x0.values,y0.values)
plt.plot(x0,y1)

model_lin_poly=np.poly1d(np.polyfit(x,y,2))

from sklearn import linear_model
regressor=linear_model.LinearRegression()

vars=[f.ci.ffill(),f.ie.interpolate("quadratic",limit=7).ffill()]
fa=pd.concat(vars,axis=1).dropna()
fa=(fa.ci*(fa.ie*0.01+1)) / 100
pd.concat([fa,f.cl],axis=1).dropna()










