import os
import requests
import numpy as np
import pandas as pd
import scipy.stats
import scipy.signal
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from glob import glob
from time import time
from tqdm import tqdm
from bs4 import BeautifulSoup as bs
from full_fred.fred import Fred
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.model_selection import train_test_split,GridSearchCV


BD=251
PATH="c:/code/"
P="bright"
intern={
"cb":"BAMLH0A0HYM2",
"fs":"STLFSI3",
"ie":"T5YIE",
"ic":"ICSA",
"pr":"PAYEMS",
"ce":"PCETRIM12M159SFRBDAL",
"yt":"DGS30",
"ys":"T10Y2Y",
"ng":"DHHNGSP",
"cl":"DCOILWTICO",
"fr":"DFF",
"nk":"NIKKEI225",
"ci":"CPIAUCSL",
"pi":"PPIACO",
"hi":"CSUSHPISA",
"ue":"DEXUSEU",
"uy":"DEXCHUS",
"ua":"DEXUSNZ"
}
extern={
"zs":"https://www.investing.com/commodities/us-soybeans-historical-data",
"zc":"https://www.investing.com/commodities/us-corn-historical-data",
"zw":"https://www.investing.com/commodities/us-wheat-historical-data",
"hg":"https://www.investing.com/commodities/copper-historical-data",
"si":"https://www.investing.com/commodities/silver-historical-data",
"ng":"https://www.investing.com/commodities/natural-gas-historical-data",
"cl":"https://www.investing.com/commodities/crude-oil-historical-data",
"hs":"https://www.investing.com/indices/hang-sen-40-historical-data",
}
rnd=np.random.RandomState(0) # time
sns.set_theme(palette=P,font="monospace",rc={"lines.linestyle":"-"})
pd.options.display.min_rows=6
pd.options.display.float_format=lambda q:f"{q:.5f}"


def apnd(path:str)->pd.DataFrame:
	if not path.endswith("/"):path+="/"
	return pd.concat(
        [pd.read_csv(f"{path}{q.name}") for q in
         os.scandir(path) if ".csv" in q.name],axis=0)


def mon(f:pd.DataFrame,start,stop)->np.ndarray:
    return np.asarray(
        sum([f.index.month==q for q in np.arange(start,stop,1)]),
        dtype="bool")


def full_range_idx(f:pd.DataFrame):
    return (pd.DataFrame(
        pd.date_range(f.index.min(),f.index.max()),
        columns=["date"]).set_index("date"))


def upd(url:str,i:str):
    cnxt=bs(requests.get(url).text)
    # bs::parsables
    cnxt=cnxt.select("tbody")[1]
    # tbodies[1]
    cnxt=cnxt.find_all("tr",class_="datatable_row__2vgJl")
    # trs iterable
    cache=[]
    for a in range(len(cnxt)):
        # each-tr
        date=cnxt[a].select("time")[0]["datetime"]
        val =cnxt[a].select("td")[1].text
        cache.append((date,val))
    s=pd.DataFrame(cache,columns=["date",f"{i}"])
    s["date"]=pd.to_datetime(s["date"])
    s=s.set_index("date").iloc[:,0].str.replace(",","")
    return s


def getdata(
    local=True,update=True,roll=1)->pd.DataFrame:
    t0=time()
    if local:
        f=pd.read_csv(f"{PATH}data0.csv",
            index_col="date",
            converters={"date":pd.to_datetime},
            na_filter=False)
    else:
        fred=Fred(f"{PATH}fed")
        fs={i:fred.get_series_df(intern[i])
            .loc[:,"date":]
            .astype({"date":"datetime64[ns]"})
            .set_index("date")
            .rename(columns={"value":i}
            .copy()) for i in intern}
        fsincsv=[pd.read_csv(q,
            index_col="date",
            converters={"date":pd.to_datetime},
            na_filter=False) for q in sorted(glob(rf"{PATH}da_*.csv"))]
        for q in range(len((fsincsv))):
            fs[f"{fsincsv[q].columns[0]}"]=fsincsv[q]
        f=pd.concat(fs.values(),axis=1)
        f.to_csv(f"{PATH}data0.csv",encoding="utf-8-sig")
    if update:
        [f.update(upd(extern[i],i)) for i in tqdm(extern,desc="updating")]
    f=f.apply(pd.to_numeric,errors="coerce")
    if roll!=1:
            f=f.rolling(roll,min_periods=roll//2).median()
    print(f"getdata:: {time()-t0:.1f}s::{local=},{update=},{roll=}")
    return f


def dtr(a,
    o:int=3):
    if any(np.isnan(a)):raise TypeError(f"nan in the array")
    x=np.arange(len(a))
    q=np.polyval(np.polyfit(x,a,deg=o),x)
    a-=q
    return a


def zs(f:pd.DataFrame,
    intp=True,save=False)->pd.DataFrame:
    fcache=[]
    for i in tqdm(f.columns,desc="z-score"):
        if intp:
            q=f[i].interpolate("quadratic").dropna()
        elif not intp:
            q=f[i].dropna()
        if i=="yt":
            q=dtr(q)
        q_=(q-q.min())/(q.max()-q.min())
        lz=pd.DataFrame(scipy.stats.zscore(
            scipy.stats.yeojohnson(q_)[0]),
            index=q.index)
        zp=pd.DataFrame(scipy.stats.norm.cdf(
            lz,loc=np.mean(lz.to_numpy()),scale=np.std(lz)),
            index=q.index)
        w=(pd.concat([q,lz,zp],axis=1)
            .set_axis([f"{i}",f"{i}lz",f"{i}lzp"],axis=1))
        fcache.append(w)
    f=full_range_idx(f).join(pd.concat(fcache,axis=1),how="left")
    if save:
        f.to_csv(f"{PATH}data1.csv",encoding="utf-8-sig")
    return f


def rng(f:pd.DataFrame,i:str,
    rng=(.05,3),test=False)->pd.DataFrame:
    f=f.copy()
    if not i in f.columns:
        raise NameError(f"{i} does not exist")
    col=f"{i}lzp"
    if test:
        rng=np.delete(
            np.round(np.flip(
            np.geomspace(rng[0],1,rng[1])),
            2),2)
    else:
        rng=np.round(np.flip(
            np.percentile(f[col].dropna(),(2,15,30,100))),
            2)
    f.loc[:,f"{col}rng"]=None
    #heaviside
    for q in range(len(rng)):
        f.loc[:,f"{col}rng"]=np.where(
            (~pd.isna(f[col])) & (f[col]<=rng[q]),
            rng[q],f[f"{col}rng"])
    f_=f.loc[:,f"{col}rng"].astype("category").copy()
    f.update(f_)
    return f


def nav(f:pd.DataFrame,i:str,v:float):
    rowidx=np.abs(f[f"{i}"]-v).argmin()
    colidx=f.columns.get_indexer([f"{i}"])[0]
    q=f.iloc[rowidx,colidx:colidx+3]
    w=q[f"{i}lzp"]
    print(f"{w*100:.2f}%")
    return q.copy(),(rowidx,colidx)


def ns(f:pd.DataFrame,x:str,y:str):
    f=f[[x,y]].dropna()
    rowidx=np.amin((f.count()[x],f.count()[y]))
    return f[f.shape[0]-rowidx:]


def cx(f:pd.DataFrame,x:str,y:str,
    d=24,normed=True,save=True,test=False):
    f=ns(f,x,y)
    if save:
        plt.figure(figsize=(20,12))
        xc=plt.xcorr(f[x],f[y],
            detrend=scipy.signal.detrend,maxlags=d,
            normed=normed)
        plt.suptitle(f"{x},{y},{d}")
        plt.savefig(f"e:/capt/{x}_{y}_{d}.png")
        plt.cla()
        plt.clf()
        plt.close()
        idx=xc[1].argmax()
        return xc[0][idx],xc[1][idx]
    else:
        fg,ax=plt.subplots(1,2)
        ac=ax[0].acorr(f[x],
            detrend=scipy.signal.detrend,maxlags=d)
        xc=ax[1].xcorr(f[x],f[y],
            detrend=scipy.signal.detrend,maxlags=d,
            normed=normed)
        fg.suptitle(f"{x},{y},{d}")
        ac_=ac[0][ac[1].argmax()]
        xc_=xc[0][xc[1].argmax()]
        if test:
            return (ac[0],ac[1]),(xc[0],xc[1])
        return ac_,xc_


def cx_(f:pd.DataFrame,x,
    d=24,freq="m"):
    if not f.columns==x:raise ValueError(f"wrong frame")
    f=f[x].resample(freq).median().dropna()
    cache=[]
    for col in tqdm(list(product(x,x)),desc=f"cx-rel"):
        if not col[0] is col[1]:
            rslt=cx(f[[col[0],col[1]]],
                col[0],col[1],d=d,save=True)
            cache.append((col[0],col[1],rslt[0],np.round(rslt[1],2)))
    return (pd.DataFrame(cache,columns=["x","y","dur","coef"])
            .sort_values(by="coef",ascending=False))


def cx__(f:pd.DataFrame):
    from statsmodels.tsa.stattools import grangercausalitytests
    data=f[["yt","fr"]].ffill().diff().dropna()
    rslt=grangercausalitytests(data,[a for a in np.arange(12,21)])
    ...


def exec(i,
    local=False,intp=None,roll=1):
    f=getdata(local=local,roll=roll)
    ff=zs(f,intp=intp)
    fff=rng(ff,i)
    return f,ff,fff


def impt(f:pd.DataFrame,
    x:list=None,n=10)->pd.DataFrame:
    if x is None:x=f.columns
    return (pd.DataFrame(
        KNNImputer(n_neighbors=n,weights="distance").fit_transform(f.loc[:,x]),
        index=f.index,columns=x))


def regr(x,y,
    s=0.2,t="l",cv=5):
    xi,xt,yi,yt=train_test_split(x,y,test_size=s)
    params={"max_depth":
                [a for a in np.arange(8,65,8)],
            "n_estimators":
                [a for a in np.arange(8,65,8)],
            "max_features":
                [a for a in np.arange(
                    int(np.sqrt(x.shape[1])),x.shape[1])],
            "n_jobs":
                [-1],
            "random_state":
                [rnd]}
    r=GridSearchCV(rfr(),params,cv=cv,n_jobs=-1,verbose=3)
    r.fit(xi,yi)
    print(f"r2::{r.score(xt,yt)}")
    return r


def regr_(f:pd.DataFrame,
    cv=5):
    ff=f.copy()
    xo=["cblz","cllz","nglz","zclz","uylz","ualz","ytlz","yslz"]
    xo1=["fslz"]
    yo=["pi","ci"]
    x_=ff.loc[:,xo].copy()
    x_smallest_c=x_[xo].count().index[x_[xo].count().argmin()]
    x_smallest_n=x_[xo].count()[x_[xo].count().argmin()]
    x_=impt(x_.dropna(subset=[x_smallest_c]),x_.columns,10)
    m0=ff.loc[:,xo1].bfill().dropna(how="all").shift(-7,"D")
    m1=ff.loc[:,yo].bfill().dropna(how="all").shift(-30,"D")
    x__=pd.concat([x_,m0,m1],axis=1).dropna(thresh=6)
    x__.update(x__.fslz.ffill())
    x0=x__.dropna(subset=["pi"])[x__.columns[:-2]]
    y0=x__.dropna(subset=["pi"])["pi"]
    x1=x__.dropna(subset=["ci"])[x__.columns[:-2]]
    y1=x__.dropna(subset=["ci"])["ci"]
    ppi=regr(x0,y0,cv=cv)
    cpi=regr(x1,y1,cv=cv)
    xi0=x__[pd.isna(x__["pi"])].iloc[:,:-2]
    xi1=x__[pd.isna(x__["ci"])].iloc[:,:-2]
    return {"ppi":[ppi,
                xi0.assign(p_pi=ppi.best_estimator_.predict(xi0))],
            "cpi":[cpi,
                xi1.assign(p_ci=cpi.best_estimator_.predict(xi1))],
            "ppi_rslt":ppi.cv_results_,
            "cpi_rslt":cpi.cv_results_}


# innermost visualisations
def hm(f,
    mm=(-1,1),ax=None,cbar=False,title=None):
    if title is None:
        title=f"{', '.join(f.columns)}"
    corr=f.corr("spearman")
    mask=np.triu(np.ones_like(corr,dtype=bool))
    cmap=sns.diverging_palette(240,10,as_cmap=True)
    if ax is None:
        plt.subplots(figsize=(12,12))
    plt.title(title)
    sns.heatmap(corr,
        mask=mask,cmap=cmap,ax=ax,cbar=cbar,
        vmin=mm[0],vmax=mm[1],
        annot=True,center=0,square=True,linewidths=.5,fmt=".2f")


def hm_(f):
    if len(f.columns)>20:
        q=input(f"{len(f.columns)} columns:: ")
        if not q:raise ValueError(f"{len(f.columns)} is too much")
    _,ax=plt.subplots(1,3,figsize=(22,16))
    hm(f,
        title=f"",ax=ax[0]),ax[0].title.set_text("org")
    hm(f.dropna(),
        title=f"",ax=ax[1]),ax[1].title.set_text("dropna")
    hm(impt(f,f.columns),
        title=f"",ax=ax[2]),ax[2].title.set_text("impt")


def pp(f,
    vars=None,l=False,hue=None):
    (sns.pairplot(data=f,vars=vars,hue=hue,
        dropna=False,kind="scatter",diag_kind="hist",palette=P)
        .map_diag(sns.histplot,log_scale=l,
        multiple="stack",element="step"))


def vs(f,x:str,y:str):
    fg,ax=plt.subplots()
    ax0=ax.twinx()
    ax.plot(f.loc[:,x],label=x,color="red")
    ax0.plot(f.loc[:,y],label=y,color="blue")
    ax.set_ylabel(x)
    ax0.set_ylabel(y)
    handles,labels=ax.get_legend_handles_labels()
    fg.legend(handles,(),loc="upper center")


def rp(f,x,y,
    hue=None,size=None,sizes=(20,200)):
    sns.relplot(data=f,x=x,y=y,hue=hue,size=size,sizes=sizes)
