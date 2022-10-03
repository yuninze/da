import os
import requests
import numpy as np
import pandas as pd
import scipy.stats
import scipy.signal
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from glob import glob
from time import time as t
from bs4 import BeautifulSoup as bs
from full_fred.fred import Fred

from sklearn.linear_model import LinearRegression as lr
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.model_selection import train_test_split as tts,GridSearchCV as gs


BD=251
PATH="c:/code/"
P="bright"
col=["asset","date","price","latency"]
int={
"cb":"BAMLH0A0HYM2",
"fs":"STLFSI3",
"ie":"T5YIE",
"ce":"PCETRIM12M159SFRBDAL",
"yt":"DGS30",
"ys":"T10Y2Y",
"ng":"DHHNGSP",
"cl":"DCOILWTICO",
"fr":"DFF",
"nk":"NIKKEI225"
}
ext={
"zs":"https://www.investing.com/commodities/us-soybeans-historical-data",
"zc":"https://www.investing.com/commodities/us-corn-historical-data",
"zw":"https://www.investing.com/commodities/us-wheat-historical-data",
"hg":"https://www.investing.com/commodities/copper-historical-data",
"si":"https://www.investing.com/commodities/silver-historical-data",
"ng":"https://www.investing.com/commodities/natural-gas-historical-data",
"cl":"https://www.investing.com/commodities/crude-oil-historical-data",
"hs":"https://www.investing.com/indices/hang-sen-40-historical-data",
"tp":"https://www.investing.com/indices/topix-historical-data",
}


sns.set_theme(style="whitegrid",palette=P,font="monospace",rc={"lines.linestyle":"-"})
pd.options.display.min_rows=6
pd.options.display.float_format=lambda q:f"{q:.5f}"


def full_range_idx(f:pd.DataFrame):
    return (pd.DataFrame(
        pd.date_range(f.index.min(),f.index.max()),
        columns=["date"]).set_index("date"))


def apnd(path:str)->pd.DataFrame:
	if not path.endswith("/"):path+="/"
	return pd.concat(
        [pd.read_csv(f"{path}{q.name}") for q in
         os.scandir(path) if ".csv" in q.name],axis=0)


def mon(f:pd.DataFrame,start,stop)->np.ndarray:
    return np.asarray(
        sum([f.index.month==q for q in np.arange(start,stop,1)]),
        dtype="bool")


def upd(url:str,i:str):
    cnxt=bs(requests.get(url).text) # bs::parsables
    cnxt=cnxt.select("tbody")[1] # tbodies[1]
    cnxt=cnxt.find_all("tr",class_="datatable_row__2vgJl") # trs iterable
    cache=[]
    for a in range(len(cnxt)): # each-tr
        date=cnxt[a].select("time")[0]["datetime"]
        val =cnxt[a].select("td")[1].text
        cache.append((date,val))
    s=pd.DataFrame(cache,columns=["date",f"{i}"])
    s["date"]=pd.to_datetime(s["date"])
    s=s.set_index("date").iloc[:,0].str.replace(",","")
    return s


def getdata(
    local=True,update=True,roll=1)->pd.DataFrame:
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
        f.to_csv(f"{PATH}data0.csv",encoding="utf-8-sig")
    if update:
        [f.update(upd(ext[i],i)) for i in ext]
    f=f.apply(pd.to_numeric,errors="coerce")
    if roll!=1:
            f=f.rolling(roll,min_periods=roll//2).mean()
            print(f"rolled::{roll//2}")
    print(f"getdata::elapsed {t()-t0:.1f}s::{local=},{update=},{roll=}")
    return f


def zs(f:pd.DataFrame,
    pctrng=(0.1,99.9),intp=False)->pd.DataFrame:
    t0=t()
    fcache=[]
    for i in f.columns:
        if intp:
            q=(f[i]
            .interpolate("cubic")
            .ffill()
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
                [scipy.stats.norm.cdf(e,loc=np.median(e),scale=np.std(e))
                 for e in [nz,lz]]],
                axis=1)
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


def l_(f:pd.DataFrame,i:str,v:float,
    dist="l",test=True):
    rowidx=np.abs(f[f"{i}"]-v).argmin()
    colidx=f.columns.get_indexer([f"{i}"])[0]
    q=f.iloc[rowidx,colidx:colidx+5]
    w=q[f"{i}lzp"] if dist=="l" else q[f"{i}nzp"]
    if test:
        print(f"({rowidx},{colidx})")
    print(f"{w*100:.2f}%")
    return q.copy()


def xxx(f:pd.DataFrame,x:str,y:str):
    f=f[[x,y]]
    rowmax=f.shape[0]
    rowidx=np.min(f.count()[x],f.count()[y])
    return f[rowmax-rowidx:].interpolate("index")


def xcr(f:pd.DataFrame,x:str,y:str,d,normed=True,test=False):
    f=xxx(f,x,y)
    if f.shape[0]<BD*10:
        raise UserWarning(f"{f.shape[0]} rows")
    fg,ax=plt.subplots(1,2)
    ac=ax[0].acorr(f[x],
        detrend=scipy.signal.detrend,maxlags=d)
    xc=ax[1].xcorr(f[x],f[y],
        detrend=scipy.signal.detrend,maxlags=d,normed=normed)
    ac_=ac[0][np.abs(ac[1]).argmax()]
    xc_=xc[0][np.abs(xc[1]).argmax()]
    if test:
        return (ac[0],ac[1]),(xc[0],xc[1])
    return ac_,xc_


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
        raise ValueError(f"{len(f.columns)} columns are too much")
    fg,ax=plt.subplots(1,3,figsize=(22,16))
    hm(f,
    title=f"",ax=ax[0]),ax[0].title.set_text("org")
    hm(f.dropna(),
    title=f"",ax=ax[1]),ax[1].title.set_text("dropna")
    hm(f.interpolate("time"),
    title=f"",ax=ax[2]),ax[2].title.set_text("intp")


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


def regr_(x,y,
    s=0.2,v=False,test=False,load=False,save=True):
    if test:
        x,x_,y,y_=tts(x,y,test_size=s)
        # x=["10yt","pce","ffr","wti","ng","zs","hg"],y=["5yi"]
        r0=lr(n_jobs=-1)
        r0.fit(x,y)
        r1=rfr(n_jobs=-1,random_state=5,verbose=1)
        r1.fit(x,y)
        print(f"r2::r1 {r0.score(x_,y_)}, r2 {r1.score(x_,y_)}")
        if v:
            r0_pred,r1_pred=[q.pred(x_) for q in (r0,r1)]
            plt.figure(figsize=(15,8))
            plt.plot(np.asarray(y_),label="org")
            plt.plot(np.asarray(r0_pred),label="r0_pred")
            plt.plot(np.asarray(r1_pred),label="r1_pred")
            plt.legend(prop={"family":"monospace"})
            plt.grid(visible=True)
            plt.show(block=True)
        return r0,r1
    if load:
        return joblib.load(f"{PATH}da_regr")
    else:
        params={
            "n_estimators":[20,40,80,100],
            "max_features":[a for a in np.arange(1,x.shape[1])],
            "min_samples_split":[16,32,64,96,192],
            "max_depth":[64,96,192],
            "n_jobs":[1],
            "random_state":[0]
        }
        r=gs(rfr(),params,cv=3,n_jobs=-1,verbose=3)
        r.fit(x,y)
        print(f"r2::r {r.score(x_,y_)}")
    if save:
        joblib.dump(r,f"{PATH}da_regr")
    return r


def exec(i,
    local=False,roll=1,intp=False):
    f=getdata(local=local,roll=roll)
    ff=zs(f,intp=intp)
    fff=rng(ff,i)
    return f,ff,fff
