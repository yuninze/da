import os
import requests
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns

from glob import glob
from time import time as t
from bs4 import BeautifulSoup as bs
from full_fred.fred import Fred

from sklearn.linear_model import LinearRegression as lr
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.model_selection import train_test_split as tts,GridSearchCV as gs


BD=251
PATH="c:/code/"
int={
    "cys":"BAMLH0A0HYM2",
    "5yi":"T5YIE",
    "pce":"PCETRIM12M159SFRBDAL",
    "10yt":"DGS10",
    "ng":"DHHNGSP",
    "wti":"DCOILWTICO",
    "ffr":"DFF",
}
ext={
    "zs":"https://www.investing.com/commodities/us-soybeans-historical-data",
    "zc":"https://www.investing.com/commodities/us-corn-historical-data",
    "hg":"https://www.investing.com/commodities/copper-historical-data",
    "ng":"https://www.investing.com/commodities/natural-gas-historical-data",
    "wti":"https://www.investing.com/commodities/crude-oil-historical-data",
    "hsi":"https://www.investing.com/indices/hang-sen-40-historical-data",
}


sns.set_theme("poster","whitegrid","deep","monospace",
    font_scale=0.5,
    rc={"lines.linestyle":"-"})
pd.options.display.min_rows=6
pd.options.display.float_format=lambda q:f"{q:.5f}"


def truthy(*vals):
    for x in vals:
        if not x:
            raise SystemExit(f"{x}")


def full_range_idx(f:pd.DataFrame):
    return (pd.DataFrame(
        pd.date_range(f.index.min(),f.index.max()),
            columns=["date"]).set_index("date"))


def messij(f:pd.DataFrame)->pd.DataFrame:
    return f.apply(pd.to_numeric,errors="coerce")


def apnd(path:str)->pd.DataFrame:
	if not path.endswith("/"):path+="/"
	return pd.concat(
        [pd.read_csv(f"{path}{q.name}") for q in
         os.scandir(path) if ".csv" in q.name],axis=0)


def mon(f:pd.DataFrame,
    start,stop)->np.ndarray:
    a=sum([f.index.month==q for q in np.arange(start,stop,1)])
    return np.asarray(a,dtype="bool")


def upd(url:str,i:str):
    naiyou=bs(requests.get(url).text) # bs::parsable tags
    naiyou=naiyou.select("tbody")[1] # tbodies[1]
    naiyou=naiyou.find_all("tr",class_="datatable_row__2vgJl") # trs iter
    cache=[]
    for a in range(len(naiyou)): # each tr
        s=2 if a==0 else 1
        date=naiyou[a].select("time")[0]["datetime"]
        val =naiyou[a].select("td")[s].text
        cache.append((date,val))
    s=pd.DataFrame(cache,columns=["date",f"{i}"])
    s["date"]=pd.to_datetime(s["date"])
    s=s.set_index("date").iloc[:,0].str.replace(",","")
    return s


def getdata(local=True,update=True,roll=1)->pd.DataFrame:
    t0=t()

    if local:
        f=pd.read_csv(f"{PATH}data0.csv",
            index_col="date",
            converters={"date":pd.to_datetime},
            na_filter=False)
    else:
        fred=Fred(f"{PATH}fed")
        fs={i:fred.get_series_df(int[i])
            .loc[:,"date":]
            .astype({"date":"datetime64[ns]"})
            .set_index("date")
            .rename(columns={"value":i}
            .copy()) for i in int}
        fsincsv=[pd.read_csv(q,
            index_col="date",
            converters={"date":pd.to_datetime},
            na_filter=False) for q in sorted(glob(rf"{PATH}da_*.csv"))]
        for q in range(len((fsincsv))):
            fs[f"{fsincsv[q].columns[0]}"]=fsincsv[q]
        f=pd.concat(fs.values(),axis=1)
        if update:
            [f.update(upd(ext[i],i)) for i in ext]
        f.to_csv(f"{PATH}data0.csv",encoding="utf-8-sig")
    f=messij(f)

    if roll!=1:
            f=f.rolling(roll,min_periods=roll//2).mean()
            print(f"rolled::{roll//2}")
    
    print(f"getdata::elapsed {t()-t0:.1f}s::{local=},{update=},{roll=}")
    return f


def zs(f:pd.DataFrame,
    pctrng=(0.2,99.8),intp=False)->pd.DataFrame:
    t0=t()
    
    fcache=[]
    for i in f.columns:
        if intp:
            q=(f[i]
            .interpolate("cubic")
            .interpolate("index")
            .dropna())
        else:
            q=(f[i]
            .dropna())
        
        mm=np.percentile(q,pctrng)
        q[(q<=mm[0])|(q>=mm[1])]=np.nan
        q=q.dropna()

        nz=scipy.stats.zscore(q)
        
        q_=(q-q.min())/(q.max()-q.min())
        lz=pd.DataFrame(scipy.stats.zscore(
            scipy.stats.yeojohnson(q_)[0]),index=q.index)
        
        zp=pd.concat(
                [pd.DataFrame(w,index=q.index) for w in 
                [scipy.stats.norm.pdf(np.absolute(e)) for e in 
                [nz,lz]]]
                ,axis=1)
        
        w=(pd.concat([q,nz,lz,zp],axis=1)
            .set_axis([f"{i}",f"{i}nz",f"{i}lz",f"{i}nzp",f"{i}lzp"],
                axis=1))
        fcache.append(w)
        print(f"zs::col::{i}")
    
    f=full_range_idx(f).join(pd.concat(fcache,axis=1),how="left")
    f.to_csv(f"{PATH}data1.csv",encoding="utf-8-sig")

    print(f"zs::elapsed {t()-t0:.1f}s::{pctrng=},{intp=}")
    return f


def rng(f:pd.DataFrame,i:str,
    rng=(.05,5),test=False)->pd.DataFrame:
    t0=t()

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
            np.percentile(f[col].dropna(),(1,15,35,100))),
            2)
    
    f.loc[:,f"{col}rng"]=None
    for q in range(len(rng)):
        f.loc[:,f"{col}rng"]=np.where(
            (~pd.isna(f[col])) & (f[col]<=rng[q]),
            rng[q],f[f"{col}rng"])
    
    f_=f.loc[:,f"{col}rng"].astype("category").copy()
    f.update(f_)

    print(f"prng::elapsed {t()-t0:.1f}s::{rng}")
    return f


def locate(f:pd.DataFrame,i:str,
    v:float,test=True):
    rowidx=np.abs(f[f"{i}"]-v).argmin()
    colidx=f.columns.get_indexer([f"{i}"])[0]
    q=f.iloc[rowidx,colidx:colidx+5]
    print(f"{q[4]*100:.3f}%")
    if test:
        print(f"({rowidx=}, {colidx=}~)")
    return q


def hm(f:pd.DataFrame,
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
    annot=True,center=0,square=True,linewidths=.5,fmt=".3f")


def hm_(f):
    if len(f.columns)>20:
        raise ValueError(f"{len(f.columns)} columns are too much")
    sns.set_theme("poster","whitegrid","deep","monospace",
        font_scale=0.4,
        rc={"lines.linestyle":"-"})
    fg,ax=plt.subplots(1,3,figsize=(24,12))
    hm(f,
    title=f"",ax=ax[0]),ax[0].title.set_text("org")
    hm(f.dropna(),
    title=f"",ax=ax[1]),ax[1].title.set_text("dropna")
    hm(f.interpolate("time").dropna(),
    title=f"",ax=ax[2]),ax[2].title.set_text("intp, dropna")


def regr_rfr(x,y,s=0.2,v=False,gs_=True):
    x,x_,y,y_=tts(x,y,test_size=s)
    r=rfr(n_jobs=-1,random_state=5,verbose=1)
    r.fit(x,y)
    y__=r.predict(x_)
    print(f"r2::{r.score(x_,y_)}")
    if v:
        plt.figure(figsize=(15,8))
        plt.plot(np.asarray(y_),label="real")
        plt.plot(np.asarray(y__),label="intp")
        plt.legend(prop={"family":"monospace"})
        plt.grid(visible=True)
    if gs_:
        params={
            "n_estimators":[20,40,80,100],
            "max_features":[a for a in np.arange(1,x.shape[1])],
            "min_samples_split":[16,32,64,96,192],
            "max_depth":[64,96,192],
            "n_jobs":[1],
            "random_state":[0]
        }
        rslt=gs(rfr(),params,cv=3,n_jobs=-1,verbose=3)
        rslt.fit(x,y)
        return [x,x_,y,y_,rslt]
# rslt=regr_(f_[x__],f_[y__[0]])


def regr_lin(x,y,s):
    f__=pd.read_csv("c:/code/fx.csv") 
    x__=["10yt","pce","ffr","wti","ng","zs","hg"]
    y__=["5yi"]
    x,x_,y,y_=tts(f__[x__],f__[y__])
    r=lr(n_jobs=1)
    r.fit(x,y)
    r.coef_
    r.score(x,y)


def exec(i,local=False,roll=1,intp=False):
    f=getdata(local=local,roll=roll)
    ff=zs(f,intp=intp)
    fff=rng(ff,i)
    return f,ff,fff